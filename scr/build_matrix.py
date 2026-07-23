"""Phase 0 的第一步：整理 LSCC 的三模态 PTM 原始数据。

本文件暂不训练模型；只对齐样本、统一特征身份、清理未注释 PTM，
并报告后续需要合并的重复特征。
"""

from __future__ import annotations

import cptac
import pandas as pd
from cptac.utils import reduce_multiindex

from cptac_setup import configure_cptac


SOURCE = "umich"


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
    collapsed = df.T.groupby(level=levels, sort=False).median().T

    after = collapsed.shape[1]

    print(f"{label}：合并重复特征，从 {before} 列减少到 {after} 列。")

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


def main() -> None:
    # 1. 加载原始三模态数据。
    configure_cptac()
    lscc = cptac.Lscc()
    ph = lscc.get_phosphoproteomics(SOURCE)
    ac = lscc.get_acetylproteomics(SOURCE)
    pr = lscc.get_proteomics(SOURCE)
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


if __name__ == "__main__":
    main()
