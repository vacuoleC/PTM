"""训练标签冻结的定量 PTM encoder，并在困难任务上公平评估其必要性。"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold

from project_config import CONFIG, configured_template_path
from run_hard_task_ablation import load_hard_task_data, select_feature_set


@dataclass
class PretrainingData:
    """保存 encoder 自监督预训练所需的特征、缺失掩码和样本来源。"""

    values: np.ndarray
    observed: np.ndarray
    columns: pd.MultiIndex
    mean: np.ndarray
    scale: np.ndarray
    sample_ids: pd.Index
    cohorts: pd.Series
    detection_rate: np.ndarray


class MaskedPTMAutoencoder(torch.nn.Module):
    """以随机遮蔽位点重建为目标的紧凑 PTM 表征模型。"""

    def __init__(self, input_dimension: int, hidden_dimension: int, latent_dimension: int, dropout: float):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dimension, latent_dimension),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, input_dimension),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 latent 表征和对输入特征的重建。"""

        latent = self.encoder(values)
        return latent, self.decoder(latent)


def report_progress(message: str) -> None:
    """输出带时间戳且立刻刷新的进度行，供远端日志监控器读取。"""

    timestamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def encoder_config() -> dict[str, Any]:
    """读取 encoder 专用配置并检查其任务名称与困难任务一致。"""

    configuration = CONFIG["encoder"]
    task_name = CONFIG["hard_task"]["primary"]["task_name"]
    if configuration["task_name"] != task_name:
        raise ValueError("encoder.task_name 必须与 hard_task.primary.task_name 一致。")
    return configuration


def resolve_device() -> torch.device:
    """按配置解析训练设备；显式请求 CUDA 时不允许静默回退到 CPU。"""

    requested = encoder_config()["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.yml 请求 CUDA，但当前 PyTorch 未检测到可用 CUDA 设备。")
    return torch.device(requested)


def set_random_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机源以便复现实验。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretraining_data() -> PretrainingData:
    """对齐配置队列的共同 PTM 位点，并仅以无标签数据拟合标准化参数。"""

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
    detection_rate = combined.notna().mean(axis=0).to_numpy(dtype=np.float32)
    keep = detection_rate >= configuration["minimum_detection"]
    if not keep.any():
        raise ValueError("minimum_detection 过滤后没有可供 encoder 训练的特征。")

    combined = combined.loc[:, keep]
    detection_rate = detection_rate[keep]
    raw_values = combined.to_numpy(dtype=np.float32)
    observed = ~np.isnan(raw_values)
    mean = np.nanmean(raw_values, axis=0)
    scale = np.nanstd(raw_values, axis=0)
    scale[scale == 0] = 1.0
    values = (np.where(observed, raw_values, mean) - mean) / scale
    values = values.astype(np.float32, copy=False)

    cohorts = pd.concat(cohort_rows).reindex(combined.index)
    report_progress(
        "encoder data prepared; "
        f"samples={values.shape[0]}, shared_features={len(shared_columns)}, "
        f"retained_features={values.shape[1]}"
    )
    return PretrainingData(
        values=values,
        observed=observed,
        columns=combined.columns,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        sample_ids=combined.index,
        cohorts=cohorts,
        detection_rate=detection_rate,
    )


def build_model(input_dimension: int) -> MaskedPTMAutoencoder:
    """从统一配置构造 encoder 与 decoder。"""

    configuration = encoder_config()
    return MaskedPTMAutoencoder(
        input_dimension=input_dimension,
        hidden_dimension=configuration["hidden_dimension"],
        latent_dimension=configuration["latent_dimension"],
        dropout=configuration["dropout"],
    )


def save_pretraining_artifacts(data: PretrainingData, history: pd.DataFrame, model: MaskedPTMAutoencoder) -> None:
    """保存可审计的特征清单、训练曲线、嵌入和可复用的冻结 checkpoint。"""

    configuration = encoder_config()
    task_name = configuration["task_name"]
    device = resolve_device()
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(data.values).to(device))[0].cpu().numpy()

    feature_manifest = data.columns.to_frame(index=False)
    feature_manifest["detection_rate"] = data.detection_rate
    feature_manifest_path = configured_template_path("encoder_feature_manifest_template", task=task_name)
    feature_manifest.to_csv(feature_manifest_path, index=False)

    history_path = configured_template_path("encoder_training_history_template", task=task_name)
    history.to_csv(history_path, index=False)

    embedding_frame = pd.DataFrame(embeddings, index=data.sample_ids)
    embedding_frame.index.name = "pretrain_sample_id"
    embedding_frame.insert(0, "cohort", data.cohorts)
    embedding_frame.columns = [
        column if column == "cohort" else f"latent_{column}"
        for column in embedding_frame.columns
    ]
    embedding_path = configured_template_path("encoder_embeddings_template", task=task_name)
    embedding_frame.to_csv(embedding_path)

    checkpoint_path = configured_template_path("encoder_checkpoint_template", task=task_name)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dimension": data.values.shape[1],
            "feature_columns": list(data.columns.to_list()),
            "column_names": list(data.columns.names),
            "mean": data.mean,
            "scale": data.scale,
            "encoder_config": configuration,
        },
        checkpoint_path,
    )
    report_progress(
        "encoder artifacts saved; "
        f"checkpoint={checkpoint_path}, history={history_path}, embeddings={embedding_path}"
    )


def train_encoder() -> None:
    """只用所有 CPTAC 无标签 PTM 残差数据进行随机遮蔽重建预训练。"""

    configuration = encoder_config()
    set_random_seed(configuration["random_seed"])
    device = resolve_device()
    data = load_pretraining_data()
    model = build_model(data.values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    values = torch.from_numpy(data.values).to(device)
    observed = torch.from_numpy(data.observed).to(device)
    history_rows: list[dict[str, float]] = []

    report_progress(f"encoder pretraining started; device={device}, epochs={configuration['epochs']}")
    for epoch in range(1, configuration["epochs"] + 1):
        model.train()
        ordering = torch.randperm(values.shape[0], device=device)
        epoch_loss = 0.0
        masked_count = 0
        for start in range(0, len(ordering), configuration["batch_size"]):
            batch_indices = ordering[start : start + configuration["batch_size"]]
            batch = values[batch_indices]
            batch_observed = observed[batch_indices]
            mask = (torch.rand_like(batch) < configuration["masking_probability"]) & batch_observed
            corrupted = batch.masked_fill(mask, 0.0)
            _, reconstruction = model(corrupted)
            loss = torch.mean((reconstruction[mask] - batch[mask]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(mask.sum().item())
            epoch_loss += loss.item() * count
            masked_count += count

        mean_loss = epoch_loss / masked_count
        history_rows.append({"epoch": epoch, "masked_mse": mean_loss, "masked_values": masked_count})
        if epoch % configuration["progress_every_epochs"] == 0 or epoch == 1 or epoch == configuration["epochs"]:
            report_progress(
                f"encoder pretraining epoch {epoch} / {configuration['epochs']}; "
                f"masked_mse={mean_loss:.6f}"
            )

    save_pretraining_artifacts(data, pd.DataFrame(history_rows), model)
    report_progress("encoder pretraining completed")


def load_frozen_encoder() -> tuple[MaskedPTMAutoencoder, pd.MultiIndex, np.ndarray, np.ndarray]:
    """加载已完成的 checkpoint，并将 encoder 固定为推理模式。"""

    task_name = encoder_config()["task_name"]
    checkpoint_path = configured_template_path("encoder_checkpoint_template", task=task_name)
    checkpoint = torch.load(checkpoint_path, map_location=resolve_device(), weights_only=False)
    model = build_model(checkpoint["input_dimension"]).to(resolve_device())
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    columns = pd.MultiIndex.from_tuples(checkpoint["feature_columns"], names=checkpoint["column_names"])
    return model, columns, checkpoint["mean"], checkpoint["scale"]


def transform_with_encoder(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """用预训练时的无标签标准化参数生成目标任务的原始与冻结嵌入表示。"""

    model, columns, mean, scale = load_frozen_encoder()
    if not columns.isin(frame.columns).all():
        raise ValueError("困难任务矩阵缺少 checkpoint 所需的共同 PTM 特征。")
    raw = frame.loc[:, columns].to_numpy(dtype=np.float32)
    standardized = (np.where(np.isnan(raw), mean, raw) - mean) / scale
    standardized = standardized.astype(np.float32, copy=False)
    device = resolve_device()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(standardized).to(device))[0].cpu().numpy()
    return standardized, embeddings


def make_splits(y: pd.Series, groups: pd.Series, repeat: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成可复现、分层且按病人分组的重复交叉验证划分。"""

    splitter = StratifiedGroupKFold(
        n_splits=CONFIG["model"]["cv_splits"],
        shuffle=True,
        random_state=encoder_config()["random_seed"] + repeat,
    )
    indices = np.arange(len(y))
    return list(splitter.split(indices, y, groups))


def make_logistic_regression() -> LogisticRegression:
    """构造只在每个训练折上学习的下游分类头。"""

    model_config = CONFIG["model"]
    return LogisticRegression(
        max_iter=model_config["logistic_max_iterations"],
        class_weight=model_config["logistic_class_weight"],
        solver=model_config["logistic_solver"],
    )


def metric_row(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """计算项目预注册的二分类指标。"""

    predicted = (probability >= 0.5).astype(int)
    return {
        "average_precision": average_precision_score(y_true, probability),
        "matthews_corrcoef": matthews_corrcoef(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted),
    }


def corrected_comparison(scores: pd.DataFrame) -> pd.DataFrame:
    """以 Nadeau--Bengio corrected resampled t 描述 encoder 相对 PCA 的配对差异。"""

    configuration = encoder_config()
    encoder_scores = scores.loc[scores["model"] == "frozen_encoder_logistic"].set_index(["repeat", "fold"])
    baseline_scores = scores.loc[scores["model"] == configuration["comparison_baseline"]].set_index(["repeat", "fold"])
    delta = encoder_scores["average_precision"] - baseline_scores["average_precision"]
    correction = 1 / len(delta) + (encoder_scores["n_test"] / encoder_scores["n_train"]).mean()
    standard_error = np.sqrt(delta.var(ddof=1) * correction)
    statistic = delta.mean() / standard_error if standard_error else np.nan
    return pd.DataFrame(
        [
            {
                "row_type": "corrected_comparison",
                "model": "frozen_encoder_logistic",
                "comparison_baseline": configuration["comparison_baseline"],
                "n_paired_folds": len(delta),
                "average_precision_delta_mean": delta.mean(),
                "average_precision_delta_std": delta.std(ddof=1),
                "nadeau_bengio_correction": correction,
                "nadeau_bengio_t": statistic,
                "nadeau_bengio_p_one_sided": student_t.sf(statistic, df=len(delta) - 1),
            }
        ]
    )


def evaluate_encoder() -> None:
    """在同一困难任务折上比较原始特征、transductive PCA 与冻结 encoder。"""

    configuration = encoder_config()
    X, y, groups = load_hard_task_data()
    selected = select_feature_set(X, configuration["evaluation_feature_set"])
    standardized, embeddings = transform_with_encoder(selected)
    if not configuration["transductive_unlabeled_fit"]:
        raise ValueError("本评估设计要求 PCA 与 encoder 都只使用全体无标签样本进行 transductive 拟合。")
    pretraining_values = load_pretraining_data().values
    pca = PCA(
        n_components=configuration["pca_components"],
        svd_solver=CONFIG["model"]["pca_svd_solver"],
        random_state=configuration["random_seed"],
    )
    pca.fit(pretraining_values)
    pca_representation = pca.transform(standardized)
    representations = {
        "raw_logistic": standardized,
        "pca_logistic": pca_representation,
        "frozen_encoder_logistic": embeddings,
    }
    unknown = set(configuration["evaluation_models"]) - set(representations)
    if unknown:
        raise ValueError(f"config.yml 包含未实现的 encoder evaluation models: {sorted(unknown)}")
    report_progress(
        "encoder necessity evaluation started; "
        f"task_samples={len(y)}, raw_features={standardized.shape[1]}, "
        f"pca_components={pca_representation.shape[1]}, latent_dimension={embeddings.shape[1]}"
    )

    score_rows: list[dict[str, object]] = []
    for repeat in range(configuration["evaluation_repeats"]):
        for fold, (train, test) in enumerate(make_splits(y, groups, repeat)):
            for model_name in configuration["evaluation_models"]:
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
                        "n_train": len(train),
                        "n_test": len(test),
                        **metric_row(y.iloc[test].to_numpy(), probability),
                    }
                )
        if (repeat + 1) % configuration["progress_every_repeats"] == 0:
            report_progress(
                f"encoder necessity evaluation completed {repeat + 1} / {configuration['evaluation_repeats']} repeats"
            )

    scores = pd.DataFrame(score_rows)
    task_name = configuration["task_name"]
    score_path = configured_template_path("encoder_scores_template", task=task_name)
    scores.to_csv(score_path, index=False)

    metrics = CONFIG["model"]["scoring"]
    summary = scores.groupby("model", sort=False).agg(
        n_paired_folds=("fold", "count"),
        n_features=("n_features", "first"),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in metrics
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in metrics
        },
    ).reset_index()
    summary.insert(0, "row_type", "model_summary")
    summary = pd.concat([summary, corrected_comparison(scores)], ignore_index=True, sort=False)
    summary_path = configured_template_path("encoder_summary_template", task=task_name)
    summary.to_csv(summary_path, index=False)
    report_progress(f"encoder necessity evaluation completed; scores={score_path}, summary={summary_path}")
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    """按阶段执行无标签预训练、冻结表征评估，或两者依次执行。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    arguments = parser.parse_args()
    if arguments.stage in {"train", "all"}:
        train_encoder()
    if arguments.stage in {"evaluate", "all"}:
        evaluate_encoder()


if __name__ == "__main__":
    main()
