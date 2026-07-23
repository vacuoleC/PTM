"""审计肿瘤内标签，并生成预注册困难任务的可复现标签文件。"""

from __future__ import annotations

import json
from pathlib import Path

import cptac
import pandas as pd

from cptac_setup import configure_cptac
from project_config import (
    CONFIG,
    configured_path,
    configured_template_path,
    get_cohort_class,
)


def load_tumor_sample_ids(cohort_name: str) -> pd.Index:
    """从已保存的残差矩阵读取该队列的肿瘤样本 ID。"""

    matrix_path = configured_template_path(
        "residual_matrix_template",
        cohort=cohort_name.lower(),
    )
    sample_ids = pd.read_pickle(matrix_path).index.astype(str)
    normal_suffix = CONFIG["phase0"]["normal_sample_suffix"]
    tumor_ids = sample_ids[~sample_ids.str.endswith(normal_suffix)]

    if tumor_ids.has_duplicates:
        raise ValueError(f"{cohort_name} 的肿瘤样本 ID 不唯一。")
    return tumor_ids


def patient_ids_from_samples(sample_ids: pd.Index) -> pd.Index:
    """将样本 ID 映射为病人 ID，并验证每位病人只有一个肿瘤样本。"""

    normal_suffix = CONFIG["phase0"]["normal_sample_suffix"]
    patient_ids = sample_ids.str.removesuffix(normal_suffix)
    if patient_ids.has_duplicates:
        raise ValueError("同一病人对应多个肿瘤样本，不能直接构造病人分组任务。")
    return patient_ids


def load_aligned_clinical(cohort_name: str, patient_ids: pd.Index) -> pd.DataFrame:
    """读取配置的 clinical 来源，并按 PTM 肿瘤病人顺序对齐。"""

    cohort = get_cohort_class(cptac, cohort_name)()
    clinical = cohort.get_clinical(CONFIG["cptac"]["clinical_source"]).copy()
    clinical.index = clinical.index.astype(str)
    return clinical.reindex(patient_ids)


def single_clinical_column(clinical: pd.DataFrame, column_name: str) -> pd.Series:
    """按名字取唯一 clinical 列，拒绝重名列带来的不确定性。"""

    positions = [
        position
        for position, current_name in enumerate(clinical.columns)
        if current_name == column_name
    ]
    if len(positions) != 1:
        raise ValueError(
            f"clinical 列 {column_name!r} 应唯一，实际找到 {len(positions)} 列。"
        )
    return clinical.iloc[:, positions[0]]


def summarize_candidate_labels(cohort_name: str) -> list[dict[str, object]]:
    """按配置的规模阈值审计一个队列的候选临床标签。"""

    sample_ids = load_tumor_sample_ids(cohort_name)
    patient_ids = patient_ids_from_samples(sample_ids)
    clinical = load_aligned_clinical(cohort_name, patient_ids)
    task_config = CONFIG["hard_task"]
    records: list[dict[str, object]] = []

    for column_name in task_config["candidate_clinical_columns"]:
        if column_name not in clinical.columns:
            records.append(
                {
                    "cohort": cohort_name,
                    "label_column": column_name,
                    "status": "missing_column",
                }
            )
            continue

        values = single_clinical_column(clinical, column_name).dropna().astype(str).str.strip()
        counts = values.value_counts()
        eligible = (
            len(values) >= task_config["candidate_minimum_labeled_samples"]
            and len(counts) <= task_config["candidate_maximum_classes"]
            and len(counts) >= 2
            and counts.min() >= task_config["candidate_minimum_class_size"]
        )
        records.append(
            {
                "cohort": cohort_name,
                "label_column": column_name,
                "status": "eligible" if eligible else "not_eligible",
                "tumor_samples": len(sample_ids),
                "labeled_samples": len(values),
                "class_count": len(counts),
                "smallest_class": counts.min() if len(counts) else 0,
                "class_sizes": json.dumps(counts.to_dict(), ensure_ascii=False),
            }
        )
    return records


def build_primary_labels() -> pd.DataFrame:
    """构造配置指定的二分类困难任务标签，并保留原始临床值。"""

    primary = CONFIG["hard_task"]["primary"]
    if primary["label_source"] != "clinical":
        raise ValueError("当前只支持 clinical 作为困难任务标签来源。")

    cohort_name = primary["cohort"]
    sample_ids = load_tumor_sample_ids(cohort_name)
    patient_ids = patient_ids_from_samples(sample_ids)
    clinical = load_aligned_clinical(cohort_name, patient_ids)
    raw_labels = single_clinical_column(clinical, primary["label_column"])

    prefix_map = primary["label_prefix_map"]
    class_names = raw_labels.astype("string").map(
        lambda value: next(
            (
                class_name
                for prefix, class_name in prefix_map.items()
                if value.startswith(prefix)
            ),
            pd.NA,
        )
    )
    target_map = {class_name: target for target, class_name in enumerate(prefix_map.values())}

    labels = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "raw_label": raw_labels.to_numpy(),
            "class_name": class_names.to_numpy(),
        },
        index=sample_ids,
    )
    labels.index.name = "sample_id"
    labels = labels.dropna(subset=["class_name"])
    labels["target"] = labels["class_name"].map(target_map).astype(int)

    class_counts = labels["class_name"].value_counts()
    if len(class_counts) != len(prefix_map):
        raise ValueError("预注册类别未全部出现在可用 PTM 肿瘤样本中。")
    if class_counts.min() < CONFIG["hard_task"]["candidate_minimum_class_size"]:
        raise ValueError("预注册类别中存在小于配置最小样本数的类别。")
    return labels


def save_audit_and_labels() -> tuple[Path, Path]:
    """保存跨队列候选标签审计和当前主任务标签。"""

    audit_records: list[dict[str, object]] = []
    for cohort_name in CONFIG["phase0"]["analysis_cohorts"]:
        audit_records.extend(summarize_candidate_labels(cohort_name))
    audit = pd.DataFrame(audit_records)
    audit_path = configured_path("hard_task_label_audit")
    audit.to_csv(audit_path, index=False)

    primary = CONFIG["hard_task"]["primary"]
    labels = build_primary_labels()
    labels_path = configured_template_path(
        "hard_task_labels_template",
        task=primary["task_name"],
    )
    labels.to_csv(labels_path)

    print("已保存候选标签审计：", audit_path)
    print("已保存主任务标签：", labels_path)
    print("主任务：", primary["task_name"])
    print(labels["class_name"].value_counts().to_string())
    return audit_path, labels_path


def main() -> None:
    """初始化 CPTAC 后生成困难任务标签审计产物。"""

    configure_cptac()
    save_audit_and_labels()


if __name__ == "__main__":
    main()
