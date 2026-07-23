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

    # 6. 只检查清理后的可用特征；下一步将据此合并重复列。
    inspect_duplicates(ph, "磷酸化")
    inspect_duplicates(ac, "乙酰化")
    inspect_duplicates(pr, "蛋白组")


if __name__ == "__main__":
    main()
