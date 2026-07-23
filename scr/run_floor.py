from __future__ import annotations
import re
import pandas as pd
from project_config import CONFIG, configured_path

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_validate

from xgboost import XGBClassifier

from transformers import DetectionFilter, MedianImputer

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


def make_linear_pipeline():
    """创建一个只在训练折拟合预处理器的线性基线模型。"""

    model_config = CONFIG["model"]

    return Pipeline(
        steps=[
            ("filter", DetectionFilter()),
            ("impute", MedianImputer()),
            ("scale", StandardScaler()),
            ("pca", PCA(
                n_components=model_config["pca_components"]
            )),
            ("head", LogisticRegression(
                max_iter=model_config["logistic_max_iterations"],
                class_weight=model_config["logistic_class_weight"],
                solver=model_config["logistic_solver"],
            )),
        ]
    )


def run_linear_floor(X, y, groups):
    """运行按病人分组的 5 折线性 floor 实验。"""

    model_config = CONFIG["model"]

    cv = GroupKFold(
        n_splits=model_config["cv_splits"]
    )

    scores = cross_validate(
        estimator=make_linear_pipeline(),
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        scoring=model_config["scoring"],
        return_train_score=False,
        error_score="raise",
    )

    results = pd.DataFrame(
        {
            metric: scores[f"test_{metric}"]
            for metric in model_config["scoring"]
        }
    )
    results.index.name = "fold"

    print("\n每折结果：")
    print(results.to_string())

    primary_metric = model_config["primary_metric"]
    print(
        f"\n{primary_metric}："
        f"{results[primary_metric].mean():.4f} "
        f"± {results[primary_metric].std():.4f}"
    )

    output_path = configured_path(
        "lscc_floor_linear_scores"
    )
    results.to_csv(output_path)

    print(f"已保存每折结果：{output_path}")

    return results


def make_xgboost_pipeline():
    """创建与线性基线使用同一预处理规则的 XGBoost floor 模型。"""

    model_config = CONFIG["model"]
    xgb_config = model_config["xgboost"]

    return Pipeline(
        steps=[
            ("filter", DetectionFilter()),
            ("impute", MedianImputer()),
            ("scale", StandardScaler()),
            ("head", XGBClassifier(
                objective=xgb_config["objective"],
                n_estimators=xgb_config["n_estimators"],
                max_depth=xgb_config["max_depth"],
                eval_metric=xgb_config["eval_metric"],
                random_state=xgb_config["random_state"],
                n_jobs=xgb_config["n_jobs"],
            )),
        ]
    )


if __name__ == "__main__":
    X, y, groups = load_phase0_data()

    print("特征矩阵形状：", X.shape)
    print("肿瘤样本数：", y.sum())
    print("正常样本数：", (y == 0).sum())
    print("独立病人数：", groups.nunique())

    results = run_linear_floor(X, y, groups)

    cv = GroupKFold(
        n_splits=CONFIG["model"]["cv_splits"]
    )

    train_index, test_index = next(cv.split(X, y, groups))

    xgb_pipeline = make_xgboost_pipeline()

    xgb_pipeline.fit(
        X.iloc[train_index],
        y.iloc[train_index],
    )

    probability = xgb_pipeline.predict_proba(
        X.iloc[test_index]
    )[:, 1]

    print("训练样本数：", len(train_index))
    print("测试样本数：", len(test_index))
    print(
        "训练折保留的特征数：",
        xgb_pipeline.named_steps["filter"].keep_.sum(),
    )
    print("前 5 个肿瘤预测概率：", probability[:5])