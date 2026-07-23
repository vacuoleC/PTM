"""Phase 0 的第一步：整理 LSCC 的三模态 PTM 原始数据。

本文件暂不训练模型；只对齐样本、统一特征身份、清理未注释 PTM，
并报告后续需要合并的重复特征。
"""

from __future__ import annotations

import cptac
import pandas as pd
import numpy as np
from cptac.utils import reduce_multiindex

from cptac_setup import configure_cptac
from project_config import (
    CONFIG,
    configured_path,
    configured_template_path,
    get_cohort_class,
)


SOURCE = CONFIG["cptac"]["omics_source"]


def align_three_modalities(
    ph: pd.DataFrame,
    ac: pd.DataFrame,
    pr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """只保留三张表共有的样本，并使行顺序完全一致。"""
    common_samples = ph.index[ph.index.isin(ac.index) & ph.index.isin(pr.index)]
    ph = ph.loc[common_samples]
    ac = ac.loc[common_samples]
    pr = pr.loc[common_samples]

    assert ph.index.equals(ac.index)
    assert ph.index.equals(pr.index)
    return ph, ac, pr


def simplify_feature_columns(
    ph: pd.DataFrame,
    ac: pd.DataFrame,
    pr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """保留 PTM 的 (Name, Site)，以及蛋白组的 Name。"""
    ph = reduce_multiindex(ph, levels_to_drop=["Peptide", "Database_ID"], quiet=True)
    ac = reduce_multiindex(ac, levels_to_drop=["Peptide", "Database_ID"], quiet=True)
    pr = reduce_multiindex(pr, levels_to_drop="Database_ID", quiet=True)
    return ph, ac, pr


def report_missing_ptm_annotation(df: pd.DataFrame, label: str) -> None:
    """在删除前记录无法映射母蛋白的 PTM 特征数量。"""
    column_info = df.columns.to_frame(index=False)
    missing = column_info["Name"].isna() | column_info["Site"].isna()

    print(f"{label}缺少 Name 或 Site 的位点数：{missing.sum()}")
    if missing.any():
        print(f"{label}缺失数（按列层级）：")
        print(column_info.isna().sum())


def drop_unannotated_ptm(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """删除缺少 Name 或 Site、不能做母蛋白校正的 PTM 特征。"""
    column_info = df.columns.to_frame(index=False)
    keep = column_info["Name"].notna() & column_info["Site"].notna()
    result = df.loc[:, keep.to_numpy()]

    assert not result.columns.to_frame(index=False).isna().any().any()
    print(f"{label}：删除 {(~keep).sum()} 个无法进行母蛋白校正的 PTM 特征。")
    return result


def inspect_duplicates(df: pd.DataFrame, label: str) -> None:
    """报告同一生物学特征被重复测量的情况。"""
    counts = df.columns.to_frame(index=False).value_counts(dropna=False)
    duplicate_groups = counts[counts > 1]

    print(f"\n{label}重复特征报告")
    print("重复特征组数：", len(duplicate_groups))
    print("重复列数（除首次外）：", df.columns.duplicated().sum())
    if not duplicate_groups.empty:
        print("每组重复次数摘要：")
        print(duplicate_groups.describe())
        print("重复次数最多的前 10 组：")
        print(duplicate_groups.head(10))


def collapse_duplicate_features(df, label):
    """把同一生物学特征的重复测量，按病人取中位数。"""
    before = df.shape[1]

    levels = list(range(df.columns.nlevels))
    aggregation = CONFIG["phase0"]["duplicate_aggregation"]
    collapsed = df.T.groupby(level=levels, sort=False).agg(aggregation).T

    after = collapsed.shape[1]

    print(
        f"{label}：使用 {aggregation} 合并重复特征，"
        f"从 {before} 列减少到 {after} 列。"
    )

    assert collapsed.columns.duplicated().sum() == 0
    return collapsed


def report_parent_protein_coverage(ptm, protein, label):
    """统计 PTM 位点中有多少能匹配到同名母蛋白。"""
    ptm_genes = ptm.columns.get_level_values("Name")
    protein_genes = set(protein.columns.astype(str))

    matched = ptm_genes.isin(protein_genes)

    print(f"{label}母蛋白匹配率：{matched.sum()} / {len(matched)} = {matched.mean():.2%}")

    return matched


def keep_parent_matched_sites(ptm, matched, label):
    """只保留能够匹配到母蛋白的 PTM 位点。"""

    result = ptm.loc[:, matched]

    print(
        f"{label}：保留 {result.shape[1]} 个可做母蛋白校正的位点，"
        f"排除 {(~matched).sum()} 个无法匹配的位点。"
    )

    return result


def residual_for_one_site(ptm, protein, feature):
    """对一个 PTM 位点做：PTM 丰度 ~ 母蛋白丰度，并返回残差。"""

    gene, site = feature

    # 该PTM位点在212位病人中的丰度
    y = ptm.loc[:, feature].to_numpy(dtype=float)

    # 该PTM位点对应的母蛋白在212位病人中的丰度
    x = protein.loc[:, gene].to_numpy(dtype=float)

    # 拟合线性回归模型
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    assert valid.sum() >= CONFIG["phase0"]["residual_minimum_valid_samples"]

    x_mean = x_valid.mean()
    y_mean = y_valid.mean()

    denominator = ((x_valid - x_mean) ** 2).sum()
    slope = ((x_valid - x_mean) * (y_valid - y_mean)).sum() / denominator
    intercept = y_mean - slope * x_mean

    residual = np.full(y.shape, np.nan)
    residual[valid] = y_valid - (slope * x_valid + intercept)

    print(f"测试位点：{gene} {site}")
    print(f"有效样本数：{valid.sum()}")
    print(f"回归方程：PTM = {intercept:.4f} + {slope:.4f} × 母蛋白")
    residual_valid = residual[valid]
    residual_centered = residual_valid - residual_valid.mean()
    protein_centered = x_valid - x_valid.mean()
    correlation = (
        (residual_centered * protein_centered).sum()
        / np.sqrt(
            (residual_centered**2).sum() * (protein_centered**2).sum()
        )
    )
    print(f"残差与母蛋白的相关系数：{correlation:.6f}")

    return pd.Series(residual, index=ptm.index, name=feature)


def stoich_resid(ptm, protein, min_n=None):
    """逐位点回归 PTM ~ 母蛋白丰度，返回所有位点的回归残差矩阵。"""

    min_n = min_n or CONFIG["phase0"]["residual_minimum_valid_samples"]

    assert ptm.index.equals(protein.index)

    protein_by_gene = {
        gene: protein.loc[:, gene].to_numpy(dtype=float) for gene in protein.columns
    }

    ptm_values = ptm.to_numpy(dtype=float)
    genes = ptm.columns.get_level_values("Name")

    residual_values = np.full(ptm_values.shape, np.nan)
    skipped = 0

    for j, gene in enumerate(genes):
        x = protein_by_gene[gene]
        y = ptm_values[:, j]

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < min_n:
            skipped += 1
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        x_mean = x_valid.mean()
        y_mean = y_valid.mean()

        denominator = ((x_valid - x_mean) ** 2).sum()

        if denominator == 0:
            residual_values[valid, j] = y_valid - y_mean
        else:
            slope = (
                ((x_valid - x_mean) * (y_valid - y_mean)).sum() / denominator
            )
            intercept = y_mean - slope * x_mean
            residual_values[valid, j] = y_valid - (slope * x_valid + intercept)

        if (j + 1) % CONFIG["phase0"]["residual_progress_every"] == 0:
            print(f"已完成 {j + 1} / {ptm.shape[1]} 个位点")

    result = pd.DataFrame(
        residual_values,
        index=ptm.index,
        columns=ptm.columns,
    )

    print(f"因有效样本少于 {min_n} 而跳过的位点数：{skipped}")
    return result


def report_detection_rate(df, label):
    """仅报告每个位点的非缺失比例，不在这里删除任何位点。"""

    detection_rate = df.notna().mean(axis=0)

    print(f"\n{label}残差矩阵的检测率摘要：")
    print(detection_rate.describe())

    for threshold in CONFIG["phase0"]["detection_report_thresholds"]:
        count = (detection_rate >= threshold).sum()
        print(f"检测率 ≥ {threshold:.0%} 的位点数：{count}")


def add_modification_level(df, modification):
    """给 PTM 特征列增加修饰类型层级。"""

    new_columns = pd.MultiIndex.from_arrays(
        [
            [modification] * df.shape[1],
            df.columns.get_level_values("Name"),
            df.columns.get_level_values("Site"),
        ],
        names=["Modification", "Name", "Site"],
    )

    result = df.copy()
    result.columns = new_columns
    return result


def make_tumor_labels(sample_index):
    """由 CPTAC 样本 ID 生成肿瘤/正常标签。"""

    labels = pd.Series(
        (
            ~sample_index.astype(str).str.endswith(
                CONFIG["phase0"]["normal_sample_suffix"]
            )
        ).astype(int),
        index=sample_index,
        name="is_tumor",
    )

    return labels


def save_phase0_artifacts(matrix, labels, cohort_name):
    """保存配置指定队列的 Phase 0 输入矩阵和样本标签。"""

    output_dir = configured_path("output_dir")
    output_dir.mkdir(exist_ok=True)

    artifact_cohort = cohort_name.lower()
    matrix_path = configured_template_path(
        "residual_matrix_template",
        cohort=artifact_cohort,
    )
    labels_path = configured_template_path(
        "tumor_normal_labels_template",
        cohort=artifact_cohort,
    )

    # pickle 能完整保留 MultiIndex 列名；gzip 可以减小文件体积。
    matrix.to_pickle(matrix_path, compression="gzip")
    labels.to_csv(labels_path, header=True)

    print(f"已保存残差矩阵：{matrix_path}")
    print(f"已保存肿瘤/正常标签：{labels_path}")


def main() -> None:
    # 1. 加载原始三模态数据。
    configure_cptac()
    cohort_name = CONFIG["datasets"]["matrix_cohort"]
    cohort = get_cohort_class(cptac, cohort_name)()
    ph = cohort.get_phosphoproteomics(SOURCE)
    ac = cohort.get_acetylproteomics(SOURCE)
    pr = cohort.get_proteomics(SOURCE)
    print("原始形状：", ph.shape, ac.shape, pr.shape)

    # 2. 对齐同一批病人。
    ph, ac, pr = align_three_modalities(ph, ac, pr)
    print("共同样本后的形状：", ph.shape, ac.shape, pr.shape)

    # 3. 简化列身份，但还不删除任何特征。
    ph, ac, pr = simplify_feature_columns(ph, ac, pr)
    print("简化后的列层级：")
    print("  磷酸化：", ph.columns.names)
    print("  乙酰化：", ac.columns.names)
    print("  蛋白组：", pr.columns.names)

    # 4. 清理前先记录缺失注释数量，保证实验过程可追溯。
    report_missing_ptm_annotation(ph, "磷酸化")
    report_missing_ptm_annotation(ac, "乙酰化")

    # 5. 删除无法匹配母蛋白的 PTM 位点。
    ph = drop_unannotated_ptm(ph, "磷酸化")
    ac = drop_unannotated_ptm(ac, "乙酰化")
    print("清理后的形状：", ph.shape, ac.shape, pr.shape)

    # 6. 先报告清理后、合并前的重复特征。
    inspect_duplicates(ph, "磷酸化")
    inspect_duplicates(ac, "乙酰化")
    inspect_duplicates(pr, "蛋白组")

    # 7. 合并同一生物学特征的重复测量。
    ph = collapse_duplicate_features(ph, "磷酸化")
    ac = collapse_duplicate_features(ac, "乙酰化")
    pr = collapse_duplicate_features(pr, "蛋白组")

    print("合并重复特征后的形状：", ph.shape, ac.shape, pr.shape)

    # 8. 检查母蛋白匹配率。
    ph_match = report_parent_protein_coverage(ph, pr, "磷酸化")
    ac_match = report_parent_protein_coverage(ac, pr, "乙酰化")

    # 9. 只保留匹配到母蛋白的 PTM 位点。
    ph = keep_parent_matched_sites(ph, ph_match, "磷酸化")
    ac = keep_parent_matched_sites(ac, ac_match, "乙酰化")

    print("匹配母蛋白后的形状：", ph.shape, ac.shape, pr.shape)


    # 10. 对每个 PTM 位点做：PTM 丰度 ~ 母蛋白丰度，并返回残差。
    test_feature = ph.columns[CONFIG["phase0"]["diagnostic_feature_index"]]
    test_residual = residual_for_one_site(ph, pr, test_feature)

    print(test_residual.head())

    # 11. 对所有 PTM 位点做：PTM 丰度 ~ 母蛋白丰度，并返回残差矩阵。
    ph_resid = stoich_resid(ph, pr)
    report_detection_rate(ph_resid, "磷酸化")

    print("磷酸化残差矩阵形状：", ph_resid.shape)
    print("残差矩阵缺失值比例：", ph_resid.isna().mean().mean())

    ac_resid = stoich_resid(ac, pr)

    print("乙酰化残差矩阵形状：", ac_resid.shape)
    print("乙酰化残差矩阵缺失值比例：", ac_resid.isna().mean().mean())

    report_detection_rate(ac_resid, "乙酰化")

    # 12. 将残差矩阵的列名增加修饰类型层级。
    ph_resid = add_modification_level(ph_resid, "phosphorylation")
    ac_resid = add_modification_level(ac_resid, "acetylation")

    multi_ptm_resid = pd.concat([ph_resid, ac_resid], axis=1)

    assert ph_resid.index.equals(ac_resid.index)
    assert multi_ptm_resid.columns.duplicated().sum() == 0

    print("多类 PTM 残差矩阵形状：", multi_ptm_resid.shape)
    print("列层级：", multi_ptm_resid.columns.names)
    print("前 3 个特征：", multi_ptm_resid.columns[:3])

    # 13. 生成肿瘤/正常标签。
    labels = make_tumor_labels(multi_ptm_resid.index)

    assert labels.index.equals(multi_ptm_resid.index)

    print("肿瘤样本数：", labels.sum())
    print("正常样本数：", (labels == 0).sum())

    save_phase0_artifacts(multi_ptm_resid, labels, cohort_name)


if __name__ == "__main__":
    main()
