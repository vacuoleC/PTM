"""B05-caveat 消除：逐外层折内的残差化（fold-internal residualization）。

要消除的 caveat
---------------
B05 归纳式脚本（``run_encoder_necessity_inductive``）解决了**表征拟合**的泄漏：
特征对齐 / 检测过滤 / 标准化 / PCA / encoder 预训练都只在训练侧无标签样本上拟合。
但它**如实保留了一个 caveat**：母蛋白残差化（``stoich_resid``，即逐位点
``PTM 丰度 ~ 母蛋白丰度`` 的 OLS 残差）仍是**矩阵构建阶段预建**的——用**全队列
212 个 LSCC 样本**（含测试折患者）一次性拟合系数，再作用于所有样本。因此测试折
患者的 PTM / 母蛋白丰度**参与了残差系数的估计**。

本脚本把残差化也改到**每个外层折内部**：
  - 任务矩阵：对 LSCC G2/G3 的 106 位患者，在每个外层折内，**仅用该折训练折
    肿瘤样本**（其 PTM 与母蛋白丰度）拟合逐位点 OLS 系数，再用该系数计算**全部
    106 位患者（含测试折）**的残差。
  - 预训练池：每个外层折从 **raw_lscc.pkl.gz 的原始三模态**重建 LSCC 残差，
    拟合样本为**该折训练侧（非测试患者）的全部 LSCC 样本**（肿瘤 + 正常），
    测试折患者（及其正常样本）完全不参与系数估计。

之后完全接既有归纳式流程：训练侧样本拟合 特征对齐 / 检测过滤 / 标准化 /
PCA(64) / encoder 预训练 → 对任务矩阵 transform → logistic（只 fit train 折）。
本脚本使用与 B05 归纳式**完全相同**的 50 折（10 repeat × 5 fold，同 ``make_splits``）
以便直接对比。

如何保证「残差系数只用训练折样本拟合」（机制）
--------------------------------------------
1. **拟合与应用分离**。``fold_residualize`` 接收 ``fit_rows`` 与 ``apply_rows``
   两组行号：OLS 的均值 / 斜率 / 截距**只在 ``fit_rows`` 上估计**，随后仅把系数
   作用到 ``apply_rows``。系数绝不接触 ``fit_rows`` 之外的样本。
2. **任务折**：``fit_rows`` = 该折训练折 106 位患者中的肿瘤样本；``apply_rows``
   = 全部 106 位患者的肿瘤样本（train + test）。测试折患者的残差由此系数算出，
   但从未参与其估计。
3. **预训练折**：``fit_rows`` = 该折非测试患者的全部 LSCC 样本；测试折患者
   （肿瘤 + 正常）被整体排除，既不进 ``fit_rows`` 也不进池。``assert_leak_free``
   在池层再次断言训练侧池中不含任何测试患者样本。
4. 逐位点 OLS 与 ``build_matrix.stoich_resid`` 语义一致（``min_valid`` 缺省沿用
   ``phase0.residual_minimum_valid_samples`` = 10；某位点在拟合样本上有效值
   不足 10 则整列输出 NaN；分母为 0 时退化为减去均值）。已用「fit=全部 212 样本」
   对照既有 ``lscc_multi_ptm_resid.pkl.gz`` 逐值核验：共同有效值 max|diff| ≈ 3e-14，
   NaN 掩码完全一致，确认向量化实现与建矩阵脚本等价。

如实记录的简化（未做）
----------------------
- **预训练池仅对 LSCC 做折内残差化**；LUAD / UCEC 的池样本沿用既有全队列残差
  矩阵（``luad_multi_ptm_resid.pkl.gz`` / ``ucec_multi_ptm_resid.pkl.gz``）。原因：
  从 umich TSV 重建 LUAD/UCEC 的原始三模态代价过大，且本 caveat 关注的是**任务
  队列（LSCC）**的测试折患者泄漏。**LUAD/UCEC 的残差化仍是全队列拟合**，属已知
  保留项。
- 保留的 ``min_valid`` 阈值、去重 / 母蛋白匹配等步骤不在折内重做（``raw_lscc``
  已是去重且母蛋白匹配后的三模态，与建矩阵脚本第 9 步产物一致）。
- 下游 logistic / 折划分 / 指标与 B05 归纳式完全一致。
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from project_config import CONFIG, PROJECT_ROOT, configured_path
from run_encoder_necessity import (
    encoder_config,
    make_logistic_regression,
    make_splits,
    metric_row,
    report_progress,
)
from run_encoder_necessity_inductive import (
    COMPARISON_BASELINE,
    ENCODER_MODEL_NAME,
    EVALUATION_MODELS,
    assert_leak_free,
    corrected_comparison,
    patient_of,
    prepare_pool,
    train_pool_encoder,
    transform_task,
)
from run_hard_task_ablation import load_hard_task_data, select_feature_set

DEFAULT_RAW_LSCC = PROJECT_ROOT / "raw_lscc.pkl.gz"
MIN_VALID = int(CONFIG["phase0"]["residual_minimum_valid_samples"])


def _add_modification_level(frame: pd.DataFrame, modification: str) -> pd.DataFrame:
    """给 PTM 特征列加上 Modification 层级，与建矩阵脚本 add_modification_level 一致。"""

    columns = pd.MultiIndex.from_arrays(
        [
            [modification] * frame.shape[1],
            frame.columns.get_level_values("Name"),
            frame.columns.get_level_values("Site"),
        ],
        names=["Modification", "Name", "Site"],
    )
    result = frame.copy()
    result.columns = columns
    return result


def load_raw_lscc(path) -> tuple[pd.Index, pd.MultiIndex, np.ndarray, np.ndarray]:
    """读取 LSCC 原始三模态，拼成 multi_ptm 值矩阵并把每个位点对齐到母蛋白值。

    返回 ``(sample_index, columns, ptm_values, parent_values)``：

    - ``ptm_values[sample, site]``：磷酸化在前、乙酰化在后的 PTM 丰度；
    - ``parent_values[sample, site]``：该位点对应母蛋白在同一病人上的丰度
      （按列 ``Name`` 查 protein 表；列顺序与 ``ptm_values`` 一一对应）。
    两矩阵行索引均为 LSCC 患者样本（含 ``.N`` 正常样本）。
    """

    phospho, acetyl, protein = pd.read_pickle(path)
    ptm = pd.concat(
        [
            _add_modification_level(phospho, "phosphorylation"),
            _add_modification_level(acetyl, "acetylation"),
        ],
        axis=1,
    )
    if ptm.columns.duplicated().sum() != 0:
        raise ValueError("raw_lscc 拼接后的 PTM 列存在重复。")

    genes = ptm.columns.get_level_values("Name")
    protein_columns = pd.Index(protein.columns.astype(str))
    gene_position = protein_columns.get_indexer(genes)
    if (gene_position < 0).any():
        missing = sorted(set(np.asarray(genes)[gene_position < 0]))
        raise ValueError(f"以下位点找不到母蛋白列：{missing[:5]} …")

    ptm_values = ptm.to_numpy(dtype=float)
    parent_values = protein.to_numpy(dtype=float)[:, gene_position]
    report_progress(
        "raw LSCC loaded; "
        f"samples={ptm_values.shape[0]}, sites={ptm_values.shape[1]}, "
        f"parents_matched={int((gene_position >= 0).sum())}"
    )
    return ptm.index, ptm.columns, ptm_values, parent_values


def fold_residualize(
    values: np.ndarray,
    parent: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    min_valid: int = MIN_VALID,
) -> np.ndarray:
    """逐位点 OLS（``PTM ~ 母蛋白``）：仅在 ``fit_rows`` 上估计系数，作用于 ``apply_rows``。

    与 ``build_matrix.stoich_resid`` 语义等价但按位点向量化：
    - 有效样本 = 该位点 PTM 与母蛋白皆非缺失的样本；
    - ``fit_rows`` 中有效样本数 < ``min_valid`` 的位点整列输出 NaN；
    - 母蛋白方差为 0 时退化为「减去 PTM 均值」；
    - 残差仅在 ``apply_rows`` 中 PTM 与母蛋白皆非缺失处有值，其余为 NaN。
    """

    x_fit = parent[fit_rows].astype(float, copy=False)
    y_fit = values[fit_rows].astype(float, copy=False)
    valid = np.isfinite(x_fit) & np.isfinite(y_fit)
    counts = valid.sum(axis=0)

    x_masked = np.where(valid, x_fit, np.nan)
    y_masked = np.where(valid, y_fit, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x_mean = np.nanmean(x_masked, axis=0)
        y_mean = np.nanmean(y_masked, axis=0)
    x_centered = x_masked - x_mean
    y_centered = y_masked - y_mean
    denominator = np.nansum(x_centered * x_centered, axis=0)
    numerator = np.nansum(x_centered * y_centered, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = numerator / denominator
    zero_variance = denominator == 0
    slope = np.where(zero_variance, 0.0, slope)
    intercept = np.where(zero_variance, y_mean, y_mean - slope * x_mean)

    x_apply = parent[apply_rows].astype(float, copy=False)
    y_apply = values[apply_rows].astype(float, copy=False)
    apply_valid = np.isfinite(x_apply) & np.isfinite(y_apply)
    residual = y_apply - (slope[None, :] * x_apply + intercept[None, :])
    residual = np.where(apply_valid, residual, np.nan)
    residual[:, counts < min_valid] = np.nan
    return residual


def build_pool_frame(
    raw_index: pd.Index,
    raw_columns: pd.MultiIndex,
    shared_parent: np.ndarray,
    shared_values: np.ndarray,
    shared_columns: pd.MultiIndex,
    lssc_fit_rows: np.ndarray,
    luad: pd.DataFrame,
    ucec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造某一折的预训练池：LSCC 行折内残差化，LUAD/UCEC 行沿用既有全队列残差。

    返回 ``(combined, cohorts)``，索引形如 ``{cohort}:{sample_id}``，与
    ``run_encoder_necessity_inductive.load_aligned_pretraining`` 的输出对齐。
    """

    lssc_residual = fold_residualize(
        shared_values, shared_parent, lssc_fit_rows, np.arange(len(raw_index))
    )
    lssc_frame = pd.DataFrame(lssc_residual, index=raw_index, columns=shared_columns)
    lssc_frame.index = pd.Index(
        [f"LSCC:{sample}" for sample in raw_index.astype(str)], name="pretrain_sample_id"
    )

    pieces = [lssc_frame]
    cohort_series = [pd.Series("LSCC", index=lssc_frame.index, name="cohort")]
    for cohort, frame in (("LUAD", luad), ("UCEC", ucec)):
        selected = frame.loc[:, shared_columns].copy()
        selected.index = pd.Index(
            [f"{cohort}:{sample}" for sample in selected.index.astype(str)],
            name="pretrain_sample_id",
        )
        pieces.append(selected)
        cohort_series.append(pd.Series(cohort, index=selected.index, name="cohort"))

    combined = pd.concat(pieces, axis=0)
    cohorts = pd.concat(cohort_series).reindex(combined.index)
    return combined, cohorts


def output_paths() -> tuple:
    """折内残差化输出的 scores / summary / fold-pool 路径（与其它 B05 输出区分）。"""

    task_name = encoder_config()["task_name"]
    output_dir = configured_path("output_dir")
    scores_path = output_dir / f"{task_name}_encoder_necessity_foldresid_scores.csv"
    summary_path = output_dir / f"{task_name}_encoder_necessity_foldresid_summary.csv"
    folds_path = output_dir / f"{task_name}_encoder_necessity_foldresid_fold_pools.csv"
    return scores_path, summary_path, folds_path


def evaluate_foldresid(repeats: int, epochs: int, raw_lscc_path) -> pd.DataFrame:
    """执行折内残差化的归纳式 encoder 必要性评估并保存逐折、逐折池与汇总结果。"""

    from sklearn.decomposition import PCA  # 局部导入以匹配既有脚本的延迟加载习惯

    configuration = encoder_config()
    X_reference, y, groups = load_hard_task_data()
    task_sample_ids = X_reference.index.astype(str)

    raw_index, raw_columns, raw_values, raw_parent = load_raw_lscc(raw_lscc_path)
    raw_index = raw_index.astype(str)
    # 任务样本 → raw_lscc 行号（任务肿瘤样本 ID 即患者 ID，且在 raw 索引中唯一）。
    task_raw_rows = raw_index.get_indexer(task_sample_ids)
    if (task_raw_rows < 0).any():
        missing = sorted(set(task_sample_ids[task_raw_rows < 0]))
        raise ValueError(f"任务样本在 raw_lscc 缺失：{missing[:5]} …")
    lscc_patients = np.array([patient_of(sample) for sample in raw_index])

    luad = pd.read_pickle(configured_path("output_dir") / "luad_multi_ptm_resid.pkl.gz")
    ucec = pd.read_pickle(configured_path("output_dir") / "ucec_multi_ptm_resid.pkl.gz")
    shared_columns = raw_columns.intersection(luad.columns, sort=False).intersection(
        ucec.columns, sort=False
    )
    if len(shared_columns) == 0:
        raise ValueError("LSCC 与 LUAD/UCEC 没有共同 PTM 特征。")
    shared_position = raw_columns.get_indexer(shared_columns)
    shared_values = raw_values[:, shared_position]
    shared_parent = raw_parent[:, shared_position]
    report_progress(
        f"pretraining alignment ready; shared_features={len(shared_columns)}, "
        f"lscc_samples={len(raw_index)}"
    )

    n_pool_total = len(raw_index) + len(luad) + len(ucec)
    report_progress(
        "fold-internal residualization evaluation started; "
        f"task_samples={len(y)}, task_sites={raw_columns.shape[0]}, "
        f"pretraining_pool={n_pool_total}, repeats={repeats}, epochs={epochs}, "
        f"min_valid={MIN_VALID}"
    )

    score_rows: list[dict[str, object]] = []
    fold_pool_rows: list[dict[str, object]] = []

    for repeat in range(repeats):
        for fold, (train, test) in enumerate(make_splits(y, groups, repeat)):
            test_patients = set(groups.iloc[test].astype(str))

            # --- 预训练池：LSCC 折内残差化（仅非测试患者参与拟合），LUAD/UCEC 沿用既有矩阵 ---
            retained = ~np.isin(lscc_patients, np.array(sorted(test_patients)))
            lscc_fit_rows = np.where(retained)[0]
            combined, cohorts = build_pool_frame(
                raw_index, raw_columns, shared_parent, shared_values,
                shared_columns, lscc_fit_rows, luad, ucec,
            )
            pool_patients = np.array(
                [patient_of(str(sample).split(":", 1)[-1]) for sample in combined.index]
            )
            keep_mask = ~np.isin(pool_patients, np.array(sorted(test_patients)))
            n_excluded = int((~keep_mask).sum())

            data = prepare_pool(combined, cohorts, keep_mask)
            assert_leak_free(data.sample_ids, test_patients)

            seed = configuration["random_seed"] + 1000 * repeat + fold
            model = train_pool_encoder(data, seed=seed, epochs=epochs)

            # --- 任务矩阵：折内残差化（仅训练折肿瘤样本参与拟合）---
            task_fit_rows = task_raw_rows[train]
            task_residual = fold_residualize(
                raw_values, raw_parent, task_fit_rows, task_raw_rows
            )
            task_frame = pd.DataFrame(
                task_residual, index=task_sample_ids, columns=raw_columns
            )
            selected = select_feature_set(task_frame, configuration["evaluation_feature_set"])

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
                    "n_task_fit_samples": len(train),
                    "n_lscc_pool_fit_samples": int(retained.sum()),
                    "n_lscc_pool_fit_patients": int(len(set(lscc_patients[retained]))),
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
                f"fold-resid fold done; repeat={repeat}, fold={fold}, "
                f"test_patients={len(test_patients)}, pool={data.values.shape[0]} "
                f"(excluded={n_excluded}), retained_features={data.values.shape[1]}, "
                f"task_fit={len(train)}, lscc_pool_fit={int(retained.sum())}"
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
        f"fold-internal residualization evaluation completed; "
        f"scores={scores_path}, summary={summary_path}, fold_pools={folds_path}"
    )
    print(summary.to_string(index=False), flush=True)
    return scores


def main() -> None:
    """命令行入口；``--repeats`` / ``--epochs`` 用于覆盖配置以便冒烟或全量运行。"""

    parser = argparse.ArgumentParser(description="B05 折内残差化 encoder 必要性评估")
    parser.add_argument("--stage", choices=["all", "evaluate"], default="all")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--raw-lscc", type=str, default=str(DEFAULT_RAW_LSCC))
    parser.add_argument("--tag", type=str, default="")
    arguments = parser.parse_args()
    configuration = encoder_config()
    repeats = arguments.repeats or configuration["evaluation_repeats"]
    epochs = arguments.epochs or configuration["epochs"]
    evaluate_foldresid(repeats=repeats, epochs=epochs, raw_lscc_path=arguments.raw_lscc)


if __name__ == "__main__":
    main()
