from __future__ import annotations
import re
import pandas as pd
from project_config import CONFIG, configured_path

def load_phase0_data():
    """读取 LSCC 残差矩阵、肿瘤标签，并构造病人分组。"""

    X = pd.read_pickle(
        configured_path("lscc_residual_matrix")
    )

    y = pd.read_csv(
        configured_path("lscc_tumor_normal_labels"),
        index_col=0,
    )["is_tumor"]

    # CSV 读入后显式转成字符串，确保能与矩阵样本 ID 严格比较。
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)

    assert X.index.equals(y.index)

    normal_suffix = CONFIG["phase0"]["normal_sample_suffix"]

    # C3L-00081 与 C3L-00081.N 是同一病人的肿瘤/正常样本。
    groups = X.index.to_series().str.replace(
        f"{re.escape(normal_suffix)}$",
        "",
        regex=True,
    )
    groups.name = "patient_id"

    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_phase0_data()

    print("特征矩阵形状：", X.shape)
    print("肿瘤样本数：", y.sum())
    print("正常样本数：", (y == 0).sum())
    print("独立病人数：", groups.nunique())