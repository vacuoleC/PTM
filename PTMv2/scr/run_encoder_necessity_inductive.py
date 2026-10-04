"""B05 归纳式（inductive）评估：encoder/PCA/标准化仅用训练侧无标签样本拟合。

背景与泄漏点
------------
``run_encoder_necessity.evaluate_encoder()`` 是**转导式**（transductive）流程：
它在**全部 577 个无标签样本**上一次性拟合特征对齐、检测过滤、标准化、PCA(64)
与 encoder 预训练，然后才在各外层折上评估。由于 577 预训练池包含困难任务的
测试患者（LSCC 每个患者在残差矩阵中同时有肿瘤样本 ``C3L-xxxxx`` 与正常样本
``C3L-xxxxx.N``），这会**将测试患者的信息泄漏进表征拟合**。

本脚本（B05）实现**归纳式**（inductive）流程：对**每一个外层折**，
先从 577 预训练池中**剔除该折全部测试患者的所有样本**（肿瘤 + 正常），
再仅在剩余的训练侧无标签样本上拟合：

  1. 特征对齐（三队列列交集，intersection）
  2. 检测率过滤（``minimum_detection``）
  3. 标准化（``mean`` / ``scale``）
  4. PCA(64)（``pca_components``）
  5. ``MaskedPTMAutoencoder`` 自监督预训练

随后把任务矩阵（LSCC multi_ptm）的 train+test 用**同一套训练侧参数** transform
成 raw / pca / encoder 三种表征，下游 logistic 只在 train 折上 fit、在 test 折
上 predict。三者（raw_logistic / pca_logistic / encoder_logistic）共用**完全相同**
的外层折（与 v1 ``make_splits`` 一致）与 logistic 规则。

如何保证测试患者未进入拟合（机制）
----------------------------------
- 预训练池样本 ID 形如 ``{cohort}:{sample_id}``（例如 ``LSCC:C3L-00081.N``）。
- 患者 ID = 去掉结尾 ``.N`` 的 sample_id（``patient_of``）。
- 每个外层折取 ``test_patients = set(groups.iloc[test])``，把预训练池中
  ``patient_of(sample) ∈ test_patients`` 的**全部**样本剔除（``keep_mask``）。
- 池对象（列/mean/scale/PCA/encoder）**仅**由 ``keep_mask`` 保留的样本构造；
  任务矩阵的 test 样本只在 transform 与 predict 阶段以只读方式经过这些参数，
  从不参与任何 ``fit``。
- ``assert_leak_free`` 逐折断言：保留下来的预训练样本中没有一位属于测试患者。

Caveat（如实记录，未做）
-----------------------
- 残差化（stoich_resid）在**矩阵构建阶段**预建，本脚本**不**改为折内重拟合；
  这需要从原始三模态数据逐折重建残差矩阵，属更大改动。因此输入任务矩阵
  ``lscc_multi_ptm_resid.pkl.gz`` 与预训练残差矩阵仍沿用既有的全样本残差化，
  这一点属于已知 caveat，未假装已消除。
- 本脚本只排除“训练侧无标签池”对测试患者的访问；下游 logistic 与折划分
  与 v1 完全一致。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.decomposition import PCA

from project_config import CONFIG, configured_path, configured_template_path
from run_encoder_necessity import (
    MaskedPTMAutoencoder,
    PretrainingData,
    build_model,
    encoder_config,
    make_logistic_regression,
    make_splits,
    metric_row,
    report_progress,
    resolve_device,
    set_random_seed,
)
from run_hard_task_ablation import load_hard_task_data, select_feature_set

# 归纳式流程里 encoder 表征的固定名称（不再叫 frozen，因为每折都重新预训练）。
ENCODER_MODEL_NAME = "encoder_logistic"
COMPARISON_BASELINE = "pca_logistic"
EVALUATION_MODELS = ["raw_logistic", "pca_logistic", "encoder_logistic"]


def patient_of(sample_id: str) -> str:
    """把样本 ID 归一化为患者 ID：LSCC 正常样本以 ``.N`` 结尾，去掉后缀。"""

    text = str(sample_id)
    return text[:-2] if text.endswith(".N") else text


def load_aligned_pretraining() -> tuple[pd.DataFrame, pd.Series]:
    """读取三队列残差矩阵并做列交集对齐，返回**未过滤、未标准化**的合并矩阵。

    这是 ``run_encoder_necessity.load_pretraining_data`` 的前半段（对齐部分）的
    忠实复制，以便在折内按训练侧子集重新拟合检测过滤与标准化参数。
    """

    configuration = encoder_config()
    frames: dict[str, pd.DataFrame] = {}
    shared_columns: pd.MultiIndex | None = None
    for cohort in configuration["pretraining_cohorts"]:
        path = configured_template_path("residual_matrix_template", cohort=cohort.lower())
        frame = pd.read_pickle(path)
        if not isinstance(frame.columns, pd.MultiIndex):
            raise TypeError(f"{cohort} 残差矩阵必须保留 PTM MultiIndex 列。")
        frames[cohort] = frame
        shared_columns = (
            frame.columns
            if shared_columns is None
            else shared_columns.intersection(frame.columns, sort=False)
        )

    if configuration["feature_alignment"] != "intersection":
        raise ValueError("当前只实现 config.yml 声明的 intersection 特征对齐策略。")
    if shared_columns is None or len(shared_columns) == 0:
        raise ValueError("预训练队列之间没有共同 PTM 特征。")

    aligned: list[pd.DataFrame] = []
    cohort_rows: list[pd.Series] = []
    for cohort, frame in frames.items():
        selected = frame.loc[:, shared_columns].copy()
        selected.index = pd.Index(
            [f"{cohort}:{sample_id}" for sample_id in selected.index.astype(str)],
            name="pretrain_sample_id",
        )
        aligned.append(selected)
        cohort_rows.append(pd.Series(cohort, index=selected.index, name="cohort"))

    combined = pd.concat(aligned, axis=0)
    cohorts = pd.concat(cohort_rows).reindex(combined.index)
    report_progress(
        "aligned pretraining pool loaded; "
        f"samples={combined.shape[0]}, shared_features={len(shared_columns)}"
    )
    return combined, cohorts


def prepare_pool(combined: pd.DataFrame, cohorts: pd.Series, keep_mask: np.ndarray) -> PretrainingData:
    """在训练侧样本子集上重新拟合检测过滤与标准化，返回 ``PretrainingData``。

    ``keep_mask`` 是与 ``combined`` 行对齐的布尔数组；只有为 True 的样本参与
    检测率、mean、scale 的拟合。列/参数随后用于 transform 任务矩阵。
    """

    configuration = encoder_config()
    subset = combined.loc[keep_mask]
    if subset.shape[0] == 0:
        raise ValueError("训练侧预训练池为空，无法拟合 encoder。")

    detection_rate = subset.notna().mean(axis=0).to_numpy(dtype=np.float32)
    keep = detection_rate >= configuration["minimum_detection"]
    if not keep.any():
        raise ValueError("minimum_detection 过滤后没有可供 encoder 训练的特征。")

    kept = subset.loc[:, keep]
    detection_rate = detection_rate[keep]
    raw_values = kept.to_numpy(dtype=np.float32)
    observed = ~np.isnan(raw_values)
    mean = np.nanmean(raw_values, axis=0)
    scale = np.nanstd(raw_values, axis=0)
    scale[scale == 0] = 1.0
    values = ((np.where(observed, raw_values, mean) - mean) / scale).astype(np.float32, copy=False)

    return PretrainingData(
        values=values,
        observed=observed,
        columns=kept.columns,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        sample_ids=kept.index,
        cohorts=cohorts.reindex(kept.index),
        detection_rate=detection_rate,
    )


def train_pool_encoder(data: PretrainingData, seed: int, epochs: int) -> MaskedPTMAutoencoder:
    """仅在给定训练侧无标签池上做随机遮蔽重建预训练，返回冻结的 encoder。"""

    configuration = encoder_config()
    set_random_seed(seed)
    device = resolve_device()
    model = build_model(data.values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    values = torch.from_numpy(data.values).to(device)
    observed = torch.from_numpy(data.observed).to(device)
    masking_probability = configuration["masking_probability"]
    batch_size = configuration["batch_size"]

    for _ in range(epochs):
        model.train()
        ordering = torch.randperm(values.shape[0], device=device)
        for start in range(0, len(ordering), batch_size):
            batch_indices = ordering[start : start + batch_size]
            batch = values[batch_indices]
            batch_observed = observed[batch_indices]
            mask = (torch.rand_like(batch) < masking_probability) & batch_observed
            corrupted = batch.masked_fill(mask, 0.0)
            _, reconstruction = model(corrupted)
            loss = torch.mean((reconstruction[mask] - batch[mask]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def transform_task(
    frame: pd.DataFrame,
    columns: pd.MultiIndex,
    mean: np.ndarray,
    scale: np.ndarray,
    model: MaskedPTMAutoencoder,
) -> tuple[np.ndarray, np.ndarray]:
    """用**训练侧**参数把任务矩阵标准化，并前向得到 encoder 嵌入。

    与 v1 ``transform_with_encoder`` 一致；区别仅在 mean/scale/columns/model 来自
    当折训练侧池，而非全体预训练池。
    """

    if not columns.isin(frame.columns).all():
        raise ValueError("困难任务矩阵缺少本轮训练侧池所需的共同 PTM 特征。")
    raw = frame.loc[:, columns].to_numpy(dtype=np.float32)
    standardized = ((np.where(np.isnan(raw), mean, raw) - mean) / scale).astype(np.float32, copy=False)
    device = resolve_device()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(standardized).to(device))[0].cpu().numpy()
    return standardized, embeddings


def corrected_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    """Nadeau--Bengio 校正的配对 t 检验：encoder 相对 PCA 的 AP 差异。

    逻辑与 ``run_encoder_necessity.corrected_comparison`` 相同（按其允许的
    “复制必要部分”而参数化模型名，以匹配归纳式的 encoder_logistic 命名）。
    """

    encoder_scores = scores.loc[scores["model"] == ENCODER_MODEL_NAME].set_index(["repeat", "fold"])
    baseline_scores = scores.loc[scores["model"] == COMPARISON_BASELINE].set_index(["repeat", "fold"])
    delta = encoder_scores["average_precision"] - baseline_scores["average_precision"]
    correction = 1 / len(delta) + (encoder_scores["n_test"] / encoder_scores["n_train"]).mean()
    standard_error = np.sqrt(delta.var(ddof=1) * correction)
    statistic = delta.mean() / standard_error if standard_error else np.nan
    return pd.DataFrame(
        [
            {
                "row_type": "corrected_comparison",
                "model": ENCODER_MODEL_NAME,
                "comparison_baseline": COMPARISON_BASELINE,
                "n_paired_folds": len(delta),
                "average_precision_delta_mean": delta.mean(),
                "average_precision_delta_std": delta.std(ddof=1),
                "nadeau_bengio_correction": correction,
                "nadeau_bengio_t": statistic,
                "nadeau_bengio_p_one_sided": student_t.sf(statistic, df=len(delta) - 1),
            }
        ]
    )


def assert_leak_free(kept_ids: pd.Index, test_patients: set[str]) -> None:
    """逐折断言：训练侧池中不含任何一位测试患者的样本。"""

    kept_patients = {patient_of(str(sample_id).split(":", 1)[-1]) for sample_id in kept_ids}
    overlap = kept_patients & test_patients
    if overlap:
        raise AssertionError(f"检测到测试患者泄漏进训练侧预训练池：{sorted(overlap)}")


def output_paths() -> tuple:
    """归纳式输出的 scores / summary / fold-pool 路径（与转导式输出区分开）。"""

    task_name = encoder_config()["task_name"]
    output_dir = configured_path("output_dir")
    scores_path = output_dir / f"{task_name}_encoder_necessity_inductive_scores.csv"
    summary_path = output_dir / f"{task_name}_encoder_necessity_inductive_summary.csv"
    folds_path = output_dir / f"{task_name}_encoder_necessity_inductive_fold_pools.csv"
    return scores_path, summary_path, folds_path


def evaluate_encoder_inductive(repeats: int, epochs: int) -> pd.DataFrame:
    """执行归纳式 encoder 必要性评估并保存逐折、逐折池与汇总结果。"""

    configuration = encoder_config()
    X, y, groups = load_hard_task_data()
    selected = select_feature_set(X, configuration["evaluation_feature_set"])
    combined, cohorts = load_aligned_pretraining()

    # 预训练池每行的患者 ID（用于按测试患者剔除样本）。
    pool_patients = np.array(
        [patient_of(str(sample_id).split(":", 1)[-1]) for sample_id in combined.index]
    )
    n_pool_total = combined.shape[0]

    report_progress(
        "inductive encoder necessity evaluation started; "
        f"task_samples={len(y)}, task_features={selected.shape[1]}, "
        f"pretraining_pool={n_pool_total}, repeats={repeats}, epochs={epochs}"
    )

    score_rows: list[dict[str, object]] = []
    fold_pool_rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        for fold, (train, test) in enumerate(make_splits(y, groups, repeat)):
            test_patients = set(groups.iloc[test].astype(str))
            keep_mask = ~np.isin(pool_patients, np.array(sorted(test_patients)))
            n_excluded = int((~keep_mask).sum())

            data = prepare_pool(combined, cohorts, keep_mask)
            assert_leak_free(data.sample_ids, test_patients)

            seed = configuration["random_seed"] + 1000 * repeat + fold
            model = train_pool_encoder(data, seed=seed, epochs=epochs)

            standardized, embeddings = transform_task(
                selected, data.columns, data.mean, data.scale, model
            )
            pca = PCA(
                n_components=configuration["pca_components"],
                svd_solver=CONFIG["model"]["pca_svd_solver"],
                random_state=configuration["random_seed"],
            )
            pca.fit(data.values)
            pca_representation = pca.transform(standardized)

            representations = {
                "raw_logistic": standardized,
                "pca_logistic": pca_representation,
                ENCODER_MODEL_NAME: embeddings,
            }

            fold_pool_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_test_patients": len(test_patients),
                    "n_pool_total": n_pool_total,
                    "n_excluded": n_excluded,
                    "n_pool_after_exclusion": data.values.shape[0],
                    "n_retained_features": data.values.shape[1],
                }
            )

            for model_name in EVALUATION_MODELS:
                classifier = make_logistic_regression()
                representation = representations[model_name]
                classifier.fit(representation[train], y.iloc[train])
                probability = classifier.predict_proba(representation[test])[:, 1]
                score_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "n_features": representation.shape[1],
                        "n_pretrain": data.values.shape[0],
                        "n_excluded": n_excluded,
                        "n_train": len(train),
                        "n_test": len(test),
                        **metric_row(y.iloc[test].to_numpy(), probability),
                    }
                )

            report_progress(
                f"inductive fold done; repeat={repeat}, fold={fold}, "
                f"test_patients={len(test_patients)}, pool={data.values.shape[0]} "
                f"(excluded={n_excluded}), retained_features={data.values.shape[1]}"
            )

    scores = pd.DataFrame(score_rows)
    scores_path, summary_path, folds_path = output_paths()
    scores.to_csv(scores_path, index=False)
    pd.DataFrame(fold_pool_rows).to_csv(folds_path, index=False)

    metrics = CONFIG["model"]["scoring"]
    summary = scores.groupby("model", sort=False).agg(
        n_paired_folds=("fold", "count"),
        n_features=("n_features", "first"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    ).reset_index()
    summary.insert(0, "row_type", "model_summary")
    summary = pd.concat([summary, corrected_comparison(scores)], ignore_index=True, sort=False)
    summary.to_csv(summary_path, index=False)

    report_progress(
        f"inductive encoder necessity evaluation completed; "
        f"scores={scores_path}, summary={summary_path}, fold_pools={folds_path}"
    )
    print(summary.to_string(index=False), flush=True)
    return scores


def main() -> None:
    """命令行入口；``--stage`` 仅为与 v1 接口兼容，``all`` 与 ``evaluate`` 等价。"""

    parser = argparse.ArgumentParser(description="B05 归纳式 encoder 必要性评估")
    parser.add_argument("--stage", choices=["all", "evaluate"], default="all")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="覆盖 config.yml 的 encoder.evaluation_repeats（用于冒烟测试）。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖 config.yml 的 encoder.epochs（用于冒烟测试）。",
    )
    arguments = parser.parse_args()
    configuration = encoder_config()
    repeats = arguments.repeats or configuration["evaluation_repeats"]
    epochs = arguments.epochs or configuration["epochs"]
    evaluate_encoder_inductive(repeats=repeats, epochs=epochs)


if __name__ == "__main__":
    main()
