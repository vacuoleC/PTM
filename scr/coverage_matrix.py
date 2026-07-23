"""生成 Phase 0 的三模态样本覆盖矩阵。

本脚本只回答一个问题：每个队列中有多少样本同时拥有磷酸化、乙酰化和
母蛋白丰度数据？它不训练模型，也不读取临床 Grade/Stage 标签。

运行方式（在项目根目录）：
    python scr/coverage_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cptac_setup import configure_cptac


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "coverage_matrix.csv"
SOURCE = "umich"
POWER_FLOOR = 20


def split_sample_ids(index: pd.Index) -> tuple[set[str], set[str]]:
    """按 CPTAC 的 ``.N`` 后缀把样本编号分成正常与肿瘤。"""
    sample_ids = {str(sample_id) for sample_id in index}
    normal_ids = {sample_id for sample_id in sample_ids if sample_id.endswith(".N")}
    tumor_ids = sample_ids - normal_ids
    return normal_ids, tumor_ids


def cohort_coverage(name: str, cohort_class) -> dict[str, object]:
    """读取一个队列的三种组学表，并统计三者共同拥有的样本。"""
    cohort = cohort_class()
    phospho = cohort.get_phosphoproteomics(SOURCE)
    acetyl = cohort.get_acetylproteomics(SOURCE)
    protein = cohort.get_proteomics(SOURCE)

    ph_normal, ph_tumor = split_sample_ids(phospho.index)
    ac_normal, ac_tumor = split_sample_ids(acetyl.index)
    pr_normal, pr_tumor = split_sample_ids(protein.index)

    common_normal = ph_normal & ac_normal & pr_normal
    common_tumor = ph_tumor & ac_tumor & pr_tumor

    return {
        "cancer": name,
        "phospho_samples": len(phospho.index),
        "acetyl_samples": len(acetyl.index),
        "protein_samples": len(protein.index),
        "common_normal": len(common_normal),
        "common_tumor": len(common_tumor),
        "normal_meets_power_floor": len(common_normal) >= POWER_FLOOR,
        "common_sample_count": len(common_normal | common_tumor),
    }


def main() -> None:
    configure_cptac(PROJECT_ROOT)

    # 在这里集中定义 Phase 0 允许使用的三支队列，后续建矩阵时也复用此名单。
    import cptac

    cohorts = {
        "LSCC": cptac.Lscc,
        "LUAD": cptac.Luad,
        "UCEC": cptac.Ucec,
    }
    rows = []
    for name, cohort_class in cohorts.items():
        print(f"\n读取 {name} …", flush=True)
        rows.append(cohort_coverage(name, cohort_class))

    coverage = pd.DataFrame(rows).set_index("cancer")
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    coverage.to_csv(OUTPUT_PATH)

    print("\n三模态样本覆盖矩阵：")
    print(coverage.to_string())
    print(f"\n已写入：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
