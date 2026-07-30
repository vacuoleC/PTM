"""Audit PTM measurement completeness for the frozen labelled cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import yaml


def resolve_path(task_root: Path, relative_path: str) -> Path:
    """Resolve a configured path relative to the PTMv2 directory."""
    return (task_root / relative_path).resolve()


def summarise_measurement(matrix: pd.DataFrame, labels: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise labelled-patient nonmissing fractions by PTM modality and threshold."""
    if not isinstance(matrix.columns, pd.MultiIndex) or "Modification" not in matrix.columns.names:
        raise ValueError("Feature matrix must have a MultiIndex with a Modification level.")
    patient_ids = labels["patient_id"].astype(str)
    if patient_ids.duplicated().any() or not patient_ids.isin(matrix.index.astype(str)).all():
        raise ValueError("Labelled patient IDs must be unique and present in the feature matrix.")
    labelled = matrix.loc[patient_ids]
    detection = labelled.notna().mean(axis=0)
    modifications = pd.Index(matrix.columns.get_level_values("Modification"), name="Modification")
    summary_rows = []
    threshold_rows = []
    for modification in modifications.unique():
        mask = modifications == modification
        rates = detection.loc[mask]
        summary_rows.append(
            {
                "modification": modification,
                "features": int(mask.sum()),
                "labelled_patients": len(labelled),
                "mean_detection_rate": rates.mean(),
                "median_detection_rate": rates.median(),
                "feature_detection_q25": rates.quantile(0.25),
                "feature_detection_q75": rates.quantile(0.75),
                "all_missing_features": int((rates == 0).sum()),
            }
        )
        for threshold in thresholds:
            retained = int((rates >= threshold).sum())
            threshold_rows.append(
                {
                    "modification": modification,
                    "detection_threshold": threshold,
                    "retained_features": retained,
                    "retained_percent": 100 * retained / len(rates),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def write_threshold_svg(counts: pd.DataFrame, output_path: Path) -> None:
    """Write a compact SVG showing retained-feature percentage by fixed threshold."""
    width, height = 760, 470
    left, right, top, bottom = 92, 35, 90, 90
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = {"phosphorylation": "#4C78A8", "acetylation": "#E45756"}
    thresholds = sorted(counts["detection_threshold"].unique())
    modifications = counts["modification"].drop_duplicates().tolist()
    group_width = plot_width / len(thresholds)
    bar_width = min(68, group_width / (len(modifications) + 1))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:600}.axis{font-size:14px}.tick{font-size:13px}.value{font-size:12px;font-weight:600}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" class="title">Feature retention across frozen detection thresholds</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" class="axis">Nonmissing fraction in 106 labelled LSCC patients</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_height - plot_height * tick / 100
        elements.extend([f'<line x1="{left - 5}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D9D9D9" stroke-dasharray="3 3"/>', f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}%</text>'])
    for threshold_index, threshold in enumerate(thresholds):
        group_center = left + group_width * (threshold_index + 0.5)
        for mod_index, modification in enumerate(modifications):
            row = counts.loc[(counts["detection_threshold"] == threshold) & (counts["modification"] == modification)].iloc[0]
            value = row["retained_percent"]
            x = group_center - (len(modifications) - 1) * bar_width / 2 + mod_index * bar_width
            bar_height = plot_height * value / 100
            y = top + plot_height - bar_height
            elements.extend([f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width - 5:.1f}" height="{bar_height:.1f}" fill="{colors.get(modification, "#777")}"/>', f'<text x="{x:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{value:.1f}</text>'])
        elements.append(f'<text x="{group_center:.1f}" y="{top + plot_height + 25}" text-anchor="middle" class="axis">≥ {threshold:.1f}</text>')
    legend_x = width - 245
    for index, modification in enumerate(modifications):
        y = 78 + index * 20
        elements.extend([f'<rect x="{legend_x}" y="{y - 11}" width="12" height="12" fill="{colors.get(modification, "#777")}"/>', f'<text x="{legend_x + 18}" y="{y}" class="tick">{escape(str(modification))}</text>'])
    elements.extend([f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" class="axis">Pre-registered detection threshold</text>', f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top + plot_height / 2})">Features retained (%)</text>', "</svg>"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def write_report(summary: pd.DataFrame, counts: pd.DataFrame, figure_path: Path, output_path: Path) -> None:
    """Write a data-only E1.2 report; no label-informed feature selection occurs here."""
    lines = ["# E1.2 测量缺失与候选特征审计", "", "## 为什么做", "", "后续检测率过滤必须仅在训练折拟合；本审计只描述冻结标签子集的测量结构与预注册阈值的规模，不使用标签值选择特征。", "", "## 模态摘要", "", summary.to_markdown(index=False, floatfmt=".4f"), "", "## 固定阈值下的候选特征数", "", counts.to_markdown(index=False, floatfmt=".2f"), "", "## 证据图", "", f"![检测率阈值下的特征保留比例]({figure_path.as_posix()})", "", "该描述性审计不替代训练折内的过滤；E2 将把同一阈值候选封装在 Pipeline 中，避免测试数据影响特征可用性。"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config_path: Path) -> None:
    """Run E1.2 from the frozen manifest paths and threshold candidates."""
    task_root = config_path.parent.parent.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    design_path = resolve_path(task_root, config["paths"]["study_design"])
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    matrix = pd.read_pickle(resolve_path(task_root, config["paths"]["source_matrix"]))
    labels = pd.read_csv(resolve_path(task_root, config["paths"]["source_labels"]))
    summary, counts = summarise_measurement(matrix, labels, design["preprocessing"]["detection_threshold_candidates"])
    outputs = config["audit_outputs"]
    summary_path = resolve_path(task_root, outputs["measurement_summary"])
    counts_path = resolve_path(task_root, outputs["threshold_counts"])
    figure_path = resolve_path(task_root, outputs["measurement_figure"])
    report_path = resolve_path(task_root, outputs["measurement_report"])
    for path in (summary_path, counts_path): path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)
    write_threshold_svg(counts, figure_path)
    write_report(summary, counts, figure_path, report_path)
    print(f"[E1.2] Audit complete: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    main(parser.parse_args().config)
