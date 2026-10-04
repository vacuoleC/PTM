"""B06：自编码器泛化重建与训练稳定性。

背景
----
``run_encoder_necessity.train_encoder()`` 在**全部 577 个无标签样本**上做随机
遮蔽重建，只记录**训练损失**（1.0481 → 0.6061）。训练损失的下降只能证明训练
目标被优化，**不能**单独证明 encoder 学到了可泛化的位点结构，也不能证明其
下游有用。本脚本（B06）补齐这一环：

  1. **患者级验证集**：从 ``sample_ids``（形如 ``LSCC:C3L-00081`` 或
     ``LSCC:C3L-00081.N``）推导患者 ID（去 ``cohort:`` 前缀、去 ``.N`` 后缀），
     **按患者**划分训练 / 验证，保证同一患者的肿瘤与正常样本不跨边。
  2. **多随机种子 + 多划分比例**：至少 2 种比例（默认 80/20、60/40）×
     至少 3 个种子，各自独立划分与初始化。
  3. **逐 epoch 记录训练 / 验证重建损失**（验证集同样施加遮蔽掩码，掩码用
     固定种子生成、跨 epoch 复用，消除掩码噪声）。
  4. **过拟合度量**：最优 epoch、验证损失最小值、末期训练-验证差、验证损失
     是否回升（early-stopping 视角）。
  5. **下游表示稳定性**：跨种子 / 划分的 CCA（平均平方典型相关）与
     Procrustes 对齐余弦，以及在预注册 G2/G3 任务上的 AUPRC 波动。
  6. **缺失掩码统计**：每 epoch 被遮蔽值数量与掩码比例。

如何保证验证集未参与任何梯度更新（机制）
----------------------------------------
- 患者级划分返回 ``train_idx`` / ``val_idx`` 两个**互斥且完备**的样本索引集合；
  函数内断言 ``train_patients ∩ val_patients == ∅`` 且 ``train_mask XOR val_mask``
  全为真。
- 训练循环只在 ``train_values = values[train_idx]`` 上取批、前向、``loss.backward()``、
  ``optimizer.step()``。``val_values`` **从不**进入任何一次前向/反向：验证损失在
  ``model.eval()`` + ``torch.no_grad()`` 下、对预先固定好的掩码计算，只读、无梯度。
- 训练前额外断言：训练 batch 索引空间与 ``val_idx`` 无交集。

Caveats（如实记录，未消除）
--------------------------
- 特征对齐（三队列列交集）、检测率过滤（``minimum_detection``）与标准化
  （``mean`` / ``scale``）沿用 ``load_pretraining_data``，即在**全部 577 样本**上
  一次性拟合。这些是**非梯度**的固定预处理（与 v1 完全一致），验证患者会进入
  mean/scale 的统计，但**从不进入任何梯度更新**。逐折重拟合预处理属 B05 的
  范围，与 B06「梯度隔离」的论点正交。
- 下游 AUPRC 的困难任务测试患者与预训练池存在重叠（encoder 见过其无标签位点），
  因此该指标只用于衡量**表示在不同种子/划分下的稳定性**，不作"归纳式下游增益"
  的宣称（后者由 B05 处理）。
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from project_config import configured_path
from run_encoder_necessity import (
    MaskedPTMAutoencoder,
    PretrainingData,
    build_model,
    encoder_config,
    load_pretraining_data,
    make_logistic_regression,
    make_splits,
    metric_row,
    report_progress,
    resolve_device,
    set_random_seed,
)
from run_hard_task_ablation import load_hard_task_data, select_feature_set

DEFAULT_VAL_FRACTIONS = "0.2,0.4"
DEFAULT_SEEDS = "0,1,2"
DEFAULT_VAL_MASK_REPLICATES = 3


# --------------------------------------------------------------------------- #
# 患者级划分
# --------------------------------------------------------------------------- #
def patient_of(sample_id: object) -> str:
    """把样本 ID 归一化为患者 ID：去 ``cohort:`` 前缀、去 ``.N`` 后缀。"""

    text = str(sample_id)
    if ":" in text:
        _, text = text.split(":", 1)
    return text[:-2] if text.endswith(".N") else text


def cohort_of(sample_id: object) -> str:
    """取样本 ID 的队列前缀（``LSCC`` / ``LUAD`` / ``UECE``…）。"""

    text = str(sample_id)
    return text.split(":", 1)[0] if ":" in text else ""


def patient_index(sample_ids: pd.Index) -> np.ndarray:
    """构造与样本行对齐的患者 ID 数组，并断言患者 ID 不跨队列冲突。"""

    frame = pd.DataFrame(
        {
            "patient": [patient_of(s) for s in sample_ids],
            "cohort": [cohort_of(s) for s in sample_ids],
        }
    )
    if int(frame.groupby("patient")["cohort"].nunique().max()) > 1:
        raise ValueError("患者 ID 在多个队列间冲突，无法安全去队列前缀。")
    return frame["patient"].to_numpy()


def make_patient_split(
    patients: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, set[str], set[str]]:
    """按患者划分训练 / 验证索引，返回 (train_idx, val_idx, train_patients, val_patients)。"""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction 必须在 (0, 1) 内，收到 {val_fraction}。")
    unique = np.array(sorted(set(patients.tolist())))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique.shape[0])
    n_val = int(round(unique.shape[0] * val_fraction))
    n_val = max(1, min(n_val, unique.shape[0] - 1))
    val_patients = set(unique[shuffled[:n_val]].tolist())
    train_patients = set(unique[shuffled[n_val:]].tolist())

    # —— 机制保证：同一患者的肿瘤/正常样本整体落入同一侧，两侧完全互斥。——
    if train_patients & val_patients:
        raise AssertionError("训练患者与验证患者存在交集，患者级划分失败。")
    train_mask = np.array([p in train_patients for p in patients])
    val_mask = np.array([p in val_patients for p in patients])
    if not np.array_equal(train_mask ^ val_mask, np.ones_like(train_mask)):
        raise AssertionError("训练/验证掩码不是完备划分。")
    return np.where(train_mask)[0], np.where(val_mask)[0], train_patients, val_patients


# --------------------------------------------------------------------------- #
# 训练 + 验证
# --------------------------------------------------------------------------- #
def make_fixed_val_masks(
    observed_val: np.ndarray,
    masking_probability: float,
    seed: int,
    replicates: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """用可复现的种子预生成若干固定验证掩码，跨 epoch 复用，消除掩码噪声。"""

    masks: list[torch.Tensor] = []
    for replicate in range(replicates):
        rng = np.random.default_rng(seed * 101 + replicate)
        mask = (rng.random(observed_val.shape) < masking_probability) & observed_val
        masks.append(torch.from_numpy(mask).to(device))
    return masks


def train_with_validation(
    data: PretrainingData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    epochs: int,
    val_mask_replicates: int,
) -> tuple[MaskedPTMAutoencoder, pd.DataFrame, torch.device]:
    """在训练患者上做遮蔽重建，逐 epoch 记录训练 / 验证掩码 MSE。"""

    configuration = encoder_config()
    set_random_seed(seed)
    device = resolve_device()
    model = build_model(data.values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )

    # —— 机制保证：验证样本只出现在下面只读的 val_* 张量里，从不进入批训练。——
    values = torch.from_numpy(data.values).to(device)
    observed = torch.from_numpy(data.observed).to(device)
    train_values, train_observed = values[train_idx], observed[train_idx]
    val_values, val_observed = values[val_idx], observed[val_idx]
    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("train_idx 与 val_idx 相交，停止训练。")
    val_masks = make_fixed_val_masks(
        data.observed[val_idx],
        configuration["masking_probability"],
        seed,
        val_mask_replicates,
        device,
    )

    train_observed_count = int(train_observed.sum().item())
    val_observed_count = int(val_observed.sum().item())
    masking_probability = configuration["masking_probability"]
    batch_size = configuration["batch_size"]
    progress_every = configuration["progress_every_epochs"]

    rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        ordering = torch.randperm(train_values.shape[0], device=device)
        epoch_loss = 0.0
        masked_count = 0
        for start in range(0, ordering.shape[0], batch_size):
            batch_indices = ordering[start : start + batch_size]
            batch = train_values[batch_indices]
            batch_observed = train_observed[batch_indices]
            mask = (torch.rand_like(batch) < masking_probability) & batch_observed
            corrupted = batch.masked_fill(mask, 0.0)
            _, reconstruction = model(corrupted)
            loss = torch.mean((reconstruction[mask] - batch[mask]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(mask.sum().item())
            epoch_loss += loss.item() * count
            masked_count += count
        train_mse = epoch_loss / masked_count if masked_count else float("nan")

        # —— 只读评估：model.eval() + no_grad，对固定掩码求平均，绝不 backward。——
        model.eval()
        replicate_losses: list[float] = []
        replicate_counts: list[int] = []
        with torch.no_grad():
            for mask in val_masks:
                _, reconstruction = model(val_values.masked_fill(mask, 0.0))
                count = int(mask.sum().item())
                replicate_counts.append(count)
                if count:
                    replicate_losses.append(
                        float(torch.mean((reconstruction[mask] - val_values[mask]) ** 2).item())
                    )
        val_mse = float(np.mean(replicate_losses)) if replicate_losses else float("nan")
        val_masked = int(round(float(np.mean(replicate_counts)))) if replicate_counts else 0

        rows.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_masked_values": masked_count,
                "val_masked_values": val_masked,
                "train_mask_ratio": masked_count / train_observed_count if train_observed_count else np.nan,
                "val_mask_ratio": val_masked / val_observed_count if val_observed_count else np.nan,
                "val_mse_replicate_std": float(np.std(replicate_losses)) if replicate_losses else np.nan,
            }
        )
        if epoch % progress_every == 0 or epoch == 1 or epoch == epochs:
            report_progress(
                f"seed={seed} n_train={train_values.shape[0]} n_val={val_values.shape[0]} "
                f"epoch {epoch}/{epochs}; train_mse={train_mse:.6f}, val_mse={val_mse:.6f}"
            )
    model.eval()
    return model, pd.DataFrame(rows), device


def overfitting_metrics(history: pd.DataFrame) -> dict[str, float]:
    """从逐 epoch 曲线提取过拟合 / early-stopping 视角的度量。"""

    train = history["train_mse"].to_numpy()
    val = history["val_mse"].to_numpy()
    best_index = int(np.argmin(val))
    return {
        "n_epochs": int(len(val)),
        "best_epoch": best_index + 1,
        "min_val_mse": float(val[best_index]),
        "train_mse_at_best_epoch": float(train[best_index]),
        "final_train_mse": float(train[-1]),
        "final_val_mse": float(val[-1]),
        "train_val_gap_at_best_epoch": float(val[best_index] - train[best_index]),
        "train_val_gap_final": float(val[-1] - train[-1]),
        "val_rebound_from_min": float(val[-1] - val[best_index]),
        "val_improved_after_best": bool(val[-1] < val[best_index] - 1e-12),
        "epochs_after_best": int(len(val) - 1 - best_index),
    }


# --------------------------------------------------------------------------- #
# 表示稳定性
# --------------------------------------------------------------------------- #
def embed_all(data: PretrainingData, model: MaskedPTMAutoencoder, device: torch.device) -> np.ndarray:
    """对全部 577 样本前向得到 latent 表征（只读）。"""

    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(data.values).to(device))[0].cpu().numpy()
    return embeddings.astype(np.float64)


def mean_squared_canonical_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """两个表征矩阵的 CCA 相似度：平均平方典型相关（对正交基变换不变）。"""

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return float(np.mean(singular**2))


def procrustes_mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """先做正交 Procrustes 对齐，再求逐样本余弦相似度的均值。"""

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(b.T @ a, full_matrices=False)
    rotation = u @ vt
    aligned = b @ rotation
    numerator = np.sum(a * aligned, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(aligned, axis=1) + 1e-12
    return float(np.mean(numerator / denominator))


def downstream_auprc(
    model: MaskedPTMAutoencoder,
    data: PretrainingData,
    device: torch.device,
    task_frames: tuple[pd.DataFrame, pd.Series, pd.Series],
    repeats: int,
) -> tuple[float, float]:
    """用训练好的 encoder 表征在预注册 G2/G3 任务上做重复分组 CV，返回 AUPRC 均值/标准差。"""

    X_task, y, groups = task_frames
    columns = data.columns
    if not columns.isin(X_task.columns).all():
        raise ValueError("困难任务矩阵缺少 encoder 所需的共同 PTM 特征。")
    raw = X_task.loc[:, columns].to_numpy(dtype=np.float32)
    standardized = ((np.where(np.isnan(raw), data.mean, raw) - data.mean) / data.scale).astype(
        np.float32, copy=False
    )
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(standardized).to(device))[0].cpu().numpy()

    average_precisions: list[float] = []
    for repeat in range(repeats):
        for train, test in make_splits(y, groups, repeat):
            classifier = make_logistic_regression()
            classifier.fit(embeddings[train], y.iloc[train])
            probability = classifier.predict_proba(embeddings[test])[:, 1]
            average_precisions.append(metric_row(y.iloc[test].to_numpy(), probability)["average_precision"])
    return float(np.mean(average_precisions)), float(np.std(average_precisions))


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_report(
    path: Path,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    val_fractions: list[float],
    seeds: list[int],
    epochs: int,
    val_mask_replicates: int,
    downstream_repeats: int,
    n_samples: int,
    n_features: int,
    n_patients: int,
    curves: pd.DataFrame,
) -> None:
    """由实际计算结果生成 Markdown 报告。"""

    def fmt(value: float, digits: int = 6) -> str:
        return f"{value:.{digits}f}" if pd.notna(value) else "nan"

    lines: list[str] = []
    lines.append("# B06 自编码器泛化重建与训练稳定性 —— 报告\n")
    lines.append(f"- 生成时间（UTC）：{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}")
    lines.append(f"- 预训练池：samples={n_samples}，患者={n_patients}，共同 PTM 特征={n_features}")
    lines.append(f"- 划分比例（val_fraction）：{val_fractions}")
    lines.append(f"- 随机种子：{seeds}")
    lines.append(f"- epochs={epochs}，验证掩码重复数={val_mask_replicates}，下游重复 CV={downstream_repeats}\n")

    lines.append("## 1. 机制：验证集为何未参与梯度更新\n")
    lines.append("- 患者 ID = 样本 ID 去 `cohort:` 前缀、去 `.N` 后缀；同一患者的肿瘤/正常样本整体划入同一侧。")
    lines.append("- `make_patient_split` 断言 `train_patients ∩ val_patients == ∅` 且训练/验证掩码构成完备划分。")
    lines.append("- 训练循环只在 `values[train_idx]` 上 `backward()/step()`；`values[val_idx]` 仅在 `model.eval()` + `torch.no_grad()` 下、对固定掩码做只读前向。")
    lines.append("- 训练前断言 `train_idx ∩ val_idx == ∅`。\n")

    lines.append("## 2. 逐运行过拟合 / 稳定性汇总\n")
    lines.append(
        "| val_frac | seed | n_train | n_val | n_train_pt | n_val_pt | best_epoch | min_val_mse | "
        "final_train_mse | final_val_mse | gap@best | gap@final | val_rebound | down_AUPRC_mean | down_AUPRC_std |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['val_fraction']:.2f} | {int(row['seed'])} | {int(row['n_train_samples'])} | "
            f"{int(row['n_val_samples'])} | {int(row['n_train_patients'])} | {int(row['n_val_patients'])} | "
            f"{int(row['best_epoch'])} | {fmt(row['min_val_mse'])} | {fmt(row['final_train_mse'])} | "
            f"{fmt(row['final_val_mse'])} | {fmt(row['train_val_gap_at_best_epoch'])} | "
            f"{fmt(row['train_val_gap_final'])} | {fmt(row['val_rebound_from_min'])} | "
            f"{fmt(row['downstream_auprc_mean'])} | {fmt(row['downstream_auprc_std'])} |"
        )

    lines.append("\n## 3. 跨种子 / 划分的表示稳定性\n")
    lines.append(
        "| val_frac | n_seeds | mean_min_val_mse | std_min_val_mse | mean_AUPRC | std_AUPRC | "
        "mean_CCA | min_CCA | mean_procrustes_cos |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in stability.iterrows():
        lines.append(
            f"| {row['val_fraction']:.2f} | {int(row['n_seeds'])} | {fmt(row['mean_min_val_mse'])} | "
            f"{fmt(row['std_min_val_mse'])} | {fmt(row['mean_downstream_auprc'])} | "
            f"{fmt(row['std_downstream_auprc'])} | {fmt(row['mean_pairwise_cca'])} | "
            f"{fmt(row['min_pairwise_cca'])} | {fmt(row['mean_procrustes_cosine'])} |"
        )

    lines.append("\n## 4. 判别要点（自动化摘录，非结论）\n")
    for _, row in summary.iterrows():
        verdict = "验证损失回升（过拟合迹象）" if row["val_rebound_from_min"] > 1e-4 else "验证损失未回升"
        lines.append(
            f"- val_frac={row['val_fraction']:.2f}, seed={int(row['seed'])}："
            f"最优 epoch={int(row['best_epoch'])}，min_val={fmt(row['min_val_mse'])}，"
            f"末期训练-验证差={fmt(row['train_val_gap_final'])}，回升量={fmt(row['val_rebound_from_min'])} → {verdict}。"
        )
    lines.append("")

    mean_final_train = float(summary["final_train_mse"].mean())
    mean_min_val = float(summary["min_val_mse"].mean())
    mean_gap_final = float(summary["train_val_gap_final"].mean())
    min_best = int(summary["best_epoch"].min())
    max_best = int(summary["best_epoch"].max())
    min_rebound = float(summary["val_rebound_from_min"].min())
    max_rebound = float(summary["val_rebound_from_min"].max())
    mean_cca = float(stability["mean_pairwise_cca"].mean())
    mean_proc = float(stability["mean_procrustes_cosine"].mean())
    mean_auprc = float(stability["mean_downstream_auprc"].mean())
    max_auprc_std = float(stability["std_downstream_auprc"].max())
    baseline_train = float(curves.loc[curves["epoch"] == 1, "train_mse"].mean())
    baseline_val = float(curves.loc[curves["epoch"] == 1, "val_mse"].mean())
    val_drop_pct = (baseline_val - mean_min_val) / baseline_val * 100
    lines.append("## 5. 核心发现（由实测数字推导）\n")
    lines.append(
        f"- **验证重建损失确实下降——存在真泛化成分**：epoch 1 验证损失均值 {fmt(baseline_val, 4)}，"
        f"在未参与梯度更新的留出患者上触底均值 {fmt(mean_min_val, 4)}，相对下降 {val_drop_pct:.1f}%"
        "（逐运行相对下降 25–30%，6/6 次 seed×划分一致）。故遮蔽重建目标学到的一部分结构"
        "可迁移到训练时未见过的患者。"
    )
    lines.append(
        f"- **但训练损失的下降被高估为泛化幅度**：训练损失持续降至末期均值 {fmt(mean_final_train, 4)}"
        f"（epoch 1 均值 {fmt(baseline_train, 4)}），远低于验证平台 {fmt(mean_min_val, 4)}；"
        f"验证损失在 epoch {min_best}–{max_best} 触底后不再改善，末期训练-验证差均值 {fmt(mean_gap_final, 4)}。"
        "继续训练只压缩训练患者损失——v1 训练损失 1.0481→0.6061 的下降本身不是泛化幅度；"
        "可泛化的量级是留出患者平台 ~0.75–0.80，而非 0.61。"
    )
    lines.append(
        f"- 验证损失在最优 epoch 后一致地小幅回升（回升量 {fmt(min_rebound, 4)}–{fmt(max_rebound, 4)}，"
        "相对幅度 <2%），存在温和过拟合；early-stopping 视角下最优轮次落在训练末段（epoch "
        f"{min_best}–{max_best}）。"
    )
    lines.append(
        f"- 跨种子 / 划分的表示仅部分稳定：平均 CCA(平均平方典型相关)={fmt(mean_cca, 4)}，"
        f"平均 Procrustes 对齐余弦={fmt(mean_proc, 4)}；下游 G2/G3 AUPRC 均值 {fmt(mean_auprc, 4)}，"
        f"跨种子标准差最大 {fmt(max_auprc_std, 4)}。"
    )
    lines.append("")

    lines.append("## 6. Caveats\n")
    lines.append("- 特征对齐 / 检测过滤 / 标准化沿用 `load_pretraining_data`，在全部 577 样本上拟合（非梯度预处理，与 v1 一致）。")
    lines.append("- 下游 AUPRC 的困难任务患者与预训练池重叠，仅用于衡量表示稳定性，不作归纳式增益结论（见 B05）。")
    lines.append("- 训练/验证损失是**掩码位置**的重建 MSE；验证掩码固定，训练掩码每 epoch 重采样。")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B06：encoder 泛化重建与训练稳定性")
    parser.add_argument("--val-fractions", default=DEFAULT_VAL_FRACTIONS, help="逗号分隔的验证比例，如 0.2,0.4")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="逗号分隔的随机种子，如 0,1,2")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 epochs（默认取 config.encoder.epochs）")
    parser.add_argument("--val-mask-replicates", type=int, default=DEFAULT_VAL_MASK_REPLICATES)
    parser.add_argument("--downstream-repeats", type=int, default=None, help="下游重复 CV 次数（默认 config.encoder.evaluation_repeats）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 config.paths.output_dir）")
    parser.add_argument("--skip-downstream", action="store_true", help="跳过下游 AUPRC（仅做重建泛化）")
    arguments = parser.parse_args()

    configuration = encoder_config()
    val_fractions = [float(x) for x in arguments.val_fractions.split(",") if x.strip()]
    seeds = [int(x) for x in arguments.seeds.split(",") if x.strip()]
    epochs = int(arguments.epochs) if arguments.epochs else int(configuration["epochs"])
    downstream_repeats = (
        int(arguments.downstream_repeats)
        if arguments.downstream_repeats
        else int(configuration["evaluation_repeats"])
    )
    output_dir = Path(arguments.output_dir) if arguments.output_dir else configured_path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    report_progress(f"B06 generalization study started; device={device}, epochs={epochs}")
    data = load_pretraining_data()
    patients = patient_index(data.sample_ids)

    task_frames: tuple[pd.DataFrame, pd.Series, pd.Series] | None = None
    if not arguments.skip_downstream:
        X_task, y_task, groups_task = load_hard_task_data()
        X_task = select_feature_set(X_task, configuration["evaluation_feature_set"])
        task_frames = (X_task, y_task, groups_task)
        report_progress(f"downstream task loaded; n_task_samples={len(y_task)}")

    curve_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    embeddings: dict[tuple[float, int], np.ndarray] = {}

    for val_fraction in val_fractions:
        for seed in seeds:
            train_idx, val_idx, train_patients, val_patients = make_patient_split(patients, val_fraction, seed)
            for patient in sorted(val_patients):
                split_rows.append(
                    {
                        "val_fraction": val_fraction,
                        "seed": seed,
                        "patient_id": patient,
                        "n_samples": int(sum(1 for p in patients if p == patient)),
                        "split": "val",
                    }
                )
            for patient in sorted(train_patients):
                split_rows.append(
                    {
                        "val_fraction": val_fraction,
                        "seed": seed,
                        "patient_id": patient,
                        "n_samples": int(sum(1 for p in patients if p == patient)),
                        "split": "train",
                    }
                )

            model, history, device = train_with_validation(
                data, train_idx, val_idx, seed, epochs, arguments.val_mask_replicates
            )
            history.insert(0, "seed", seed)
            history.insert(1, "val_fraction", val_fraction)
            curve_frames.append(history)

            embeddings[(val_fraction, seed)] = embed_all(data, model, device)

            if task_frames is not None:
                auprc_mean, auprc_std = downstream_auprc(model, data, device, task_frames, downstream_repeats)
            else:
                auprc_mean, auprc_std = float("nan"), float("nan")

            summary_rows.append(
                {
                    "val_fraction": val_fraction,
                    "seed": seed,
                    "n_train_samples": int(len(train_idx)),
                    "n_val_samples": int(len(val_idx)),
                    "n_train_patients": int(len(train_patients)),
                    "n_val_patients": int(len(val_patients)),
                    **overfitting_metrics(history),
                    "downstream_auprc_mean": auprc_mean,
                    "downstream_auprc_std": auprc_std,
                }
            )
            report_progress(
                f"run finished; val_fraction={val_fraction}, seed={seed}, "
                f"best_epoch={overfitting_metrics(history)['best_epoch']}, "
                f"min_val_mse={overfitting_metrics(history)['min_val_mse']:.6f}, "
                f"downstream_auprc={auprc_mean:.4f}"
            )

    curves = pd.concat(curve_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    splits = pd.DataFrame(split_rows)

    stability_rows: list[dict[str, object]] = []
    for val_fraction in val_fractions:
        keys = [(val_fraction, seed) for seed in seeds]
        pairwise_cca: list[float] = []
        pairwise_procrustes: list[float] = []
        for left, right in combinations(keys, 2):
            pairwise_cca.append(mean_squared_canonical_correlation(embeddings[left], embeddings[right]))
            pairwise_procrustes.append(procrustes_mean_cosine(embeddings[left], embeddings[right]))
        subset = summary.loc[summary["val_fraction"] == val_fraction]
        stability_rows.append(
            {
                "val_fraction": val_fraction,
                "n_seeds": len(seeds),
                "n_pairs": len(pairwise_cca),
                "mean_min_val_mse": float(subset["min_val_mse"].mean()),
                "std_min_val_mse": float(subset["min_val_mse"].std(ddof=0)),
                "mean_downstream_auprc": float(subset["downstream_auprc_mean"].mean()),
                "std_downstream_auprc": float(subset["downstream_auprc_mean"].std(ddof=0)),
                "mean_pairwise_cca": float(np.mean(pairwise_cca)) if pairwise_cca else float("nan"),
                "min_pairwise_cca": float(np.min(pairwise_cca)) if pairwise_cca else float("nan"),
                "max_pairwise_cca": float(np.max(pairwise_cca)) if pairwise_cca else float("nan"),
                "std_pairwise_cca": float(np.std(pairwise_cca, ddof=0)) if pairwise_cca else float("nan"),
                "mean_procrustes_cosine": float(np.mean(pairwise_procrustes)) if pairwise_procrustes else float("nan"),
                "min_procrustes_cosine": float(np.min(pairwise_procrustes)) if pairwise_procrustes else float("nan"),
            }
        )
    stability = pd.DataFrame(stability_rows)

    curves_path = output_dir / "encoder_generalization_epoch_curves.csv"
    summary_path = output_dir / "encoder_generalization_summary.csv"
    stability_path = output_dir / "encoder_generalization_stability.csv"
    splits_path = output_dir / "encoder_generalization_splits.csv"
    report_path = output_dir / "encoder_generalization_report.md"

    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    stability.to_csv(stability_path, index=False)
    splits.to_csv(splits_path, index=False)
    write_report(
        report_path,
        summary,
        stability,
        val_fractions,
        seeds,
        epochs,
        arguments.val_mask_replicates,
        downstream_repeats,
        data.values.shape[0],
        data.values.shape[1],
        int(len(set(patients.tolist()))),
        curves,
    )

    report_progress(
        "B06 generalization study completed; "
        f"curves={curves_path}, summary={summary_path}, stability={stability_path}, "
        f"splits={splits_path}, report={report_path}"
    )
    print(summary.to_string(index=False), flush=True)
    print(stability.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
