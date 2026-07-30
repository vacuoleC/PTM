"""Audit frozen LSCC patient identifiers and grade labels without model fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import yaml


REQUIRED_LABEL_COLUMNS = {
    "sample_id",
    "patient_id",
    "raw_label",
    "class_name",
    "target",
}


def resolve_path(task_root: Path, relative_path: str) -> Path:
    """Resolve a configured path relative to the PTMv2 task directory."""
    return (task_root / relative_path).resolve()


def summarise_alignment(feature_index: pd.Index, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return auditable cohort counts and class counts, rejecting invalid labels."""
    missing_columns = REQUIRED_LABEL_COLUMNS.difference(labels.columns)
    if missing_columns:
        raise ValueError(f"Label table is missing columns: {sorted(missing_columns)}")
    if not feature_index.is_unique:
        raise ValueError("Feature-matrix sample IDs are not unique.")
    if labels["patient_id"].isna().any() or labels["target"].isna().any():
        raise ValueError("Patient IDs and targets must be complete.")
    if labels["patient_id"].duplicated().any():
        raise ValueError("Each labelled patient must occur exactly once.")
    if not labels["target"].isin([0, 1]).all():
        raise ValueError("The frozen target must be binary (0 or 1).")

    feature_ids = pd.Index(feature_index.astype(str), name="sample_id")
    label_ids = pd.Index(labels["patient_id"].astype(str), name="patient_id")
    label_sample_ids = pd.Index(labels["sample_id"].astype(str), name="label_sample_id")
    labelled_in_matrix = label_ids.intersection(feature_ids)
    missing_from_matrix = label_ids.difference(feature_ids)
    matrix_without_label = feature_ids.difference(label_ids)

    summary = pd.DataFrame(
        [
            ("matrix_samples", len(feature_ids)),
            ("matrix_unique_sample_ids", feature_ids.nunique()),
            ("label_rows", len(labels)),
            ("label_unique_patient_ids", label_ids.nunique()),
            ("label_sample_patient_id_mismatches", int((labels["sample_id"] != labels["patient_id"]).sum())),
            ("labelled_patients_present_in_matrix", len(labelled_in_matrix)),
            ("labelled_patients_missing_from_matrix", len(missing_from_matrix)),
            ("matrix_samples_without_grade_label", len(matrix_without_label)),
            ("label_target_missing", int(labels["target"].isna().sum())),
            ("normal_like_labelled_sample_ids", int(labels["sample_id"].str.endswith(".N").sum())),
        ],
        columns=["metric", "value"],
    )
    class_counts = (
        labels.groupby(["target", "class_name", "raw_label"], dropna=False)
        .size()
        .rename("patients")
        .reset_index()
        .sort_values("target")
        .reset_index(drop=True)
    )
    return summary, class_counts


def write_class_distribution_svg(class_counts: pd.DataFrame, output_path: Path) -> None:
    """Write a dependency-free SVG bar chart for the frozen target's class balance."""
    labels = class_counts["raw_label"].astype(str).tolist()
    counts = class_counts["patients"].tolist()
    colors = ["#4C78A8", "#E45756"]
    width, height = 720, 460
    left, right, top, bottom = 100, 40, 95, 90
    plot_width, plot_height = width - left - right, height - top - bottom
    max_count = max(counts) + 12
    bar_width = 150
    gap = (plot_width - bar_width * len(counts)) / (len(counts) + 1)
    prevalence = class_counts.loc[class_counts["target"] == 1, "patients"].sum() / sum(counts)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:21px;font-weight:600}.axis{font-size:14px}.tick{font-size:13px}.count{font-size:16px;font-weight:600}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" class="title">LSCC G2 versus G3: frozen patient-level target</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" class="axis">n = {sum(counts)}; positive-class prevalence = {prevalence:.3f}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for tick in range(0, max_count + 1, 10):
        y = top + plot_height - plot_height * tick / max_count
        elements.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D9D9D9" stroke-dasharray="3 3"/>',
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}</text>',
            ]
        )
    for index, (label, count) in enumerate(zip(labels, counts, strict=True)):
        x = left + gap + index * (bar_width + gap)
        bar_height = plot_height * count / max_count
        y = top + plot_height - bar_height
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[index]}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="count">{count}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 25}" text-anchor="middle" class="axis">{escape(label.split()[0])}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" class="axis">Histological grade</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top + plot_height / 2})">Independent patients</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def write_report(summary: pd.DataFrame, class_counts: pd.DataFrame, figure_path: Path, output_path: Path) -> None:
    """Write the human-readable E1.1 evidence report from the audit tables."""
    summary_values = summary.set_index("metric")["value"]
    prevalence = class_counts.loc[class_counts["target"] == 1, "patients"].sum() / class_counts["patients"].sum()
    lines = [
        "# E1.1 病人独立性、标签完整性与类别结构审计",
        "",
        "## 为什么做",
        "",
        "在任何模型训练前，验证冻结的 LSCC G2/G3 标签与特征矩阵是否按病人一一对应，避免样本重复、缺失标签或正常样本混入造成虚假的折外性能。",
        "",
        "## 结果",
        "",
        f"- 特征矩阵含 {summary_values['matrix_samples']} 个样本；其中 {summary_values['labelled_patients_present_in_matrix']} 个带冻结分级标签。",
        f"- 标签患者缺失于矩阵：{summary_values['labelled_patients_missing_from_matrix']}；标签 sample_id 与 patient_id 不一致：{summary_values['label_sample_patient_id_mismatches']}。",
        f"- 带标签的正常样本：{summary_values['normal_like_labelled_sample_ids']}；缺失 target：{summary_values['label_target_missing']}。",
        f"- 阳性类（G3）比例为 {prevalence:.4f}，这是随机分类器的 AUPRC 基线。",
        "",
        "## 类别计数",
        "",
        class_counts.to_markdown(index=False),
        "",
        "## 证据图",
        "",
        f"![冻结标签类别结构]({figure_path.as_posix()})",
        "",
        "该图只描述冻结标签的组成，不用于证明模型可泛化；后续 E2 将以患者级折分明确防止训练/测试重叠。",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(config_path: Path) -> None:
    """Run the configured cohort audit and write all configured E1.1 artifacts."""
    task_root = config_path.parent.parent.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix_path = resolve_path(task_root, config["paths"]["source_matrix"])
    label_path = resolve_path(task_root, config["paths"]["source_labels"])
    output_config = config["audit_outputs"]
    matrix = pd.read_pickle(matrix_path)
    labels = pd.read_csv(label_path)
    summary, class_counts = summarise_alignment(matrix.index, labels)

    summary_path = resolve_path(task_root, output_config["cohort_summary"])
    class_count_path = resolve_path(task_root, output_config["class_counts"])
    figure_path = resolve_path(task_root, output_config["cohort_figure"])
    report_path = resolve_path(task_root, output_config["cohort_report"])
    for output_path in (summary_path, class_count_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    class_counts.to_csv(class_count_path, index=False)
    write_class_distribution_svg(class_counts, figure_path)
    write_report(summary, class_counts, figure_path, report_path)
    print(f"[E1.1] Audit complete: {summary_path}", flush=True)
    print(f"[E1.1] Figure complete: {figure_path}", flush=True)
    print(f"[E1.1] Report complete: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/project.yaml"))
    args = parser.parse_args()
    main(args.config)
