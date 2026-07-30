"""在预注册的困难任务上比较磷酸化与多类 PTM 的 XGBoost 表现。"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from project_config import CONFIG, configured_template_path
from run_floor import make_xgboost_pipeline


def load_hard_task_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """读取主任务的肿瘤样本矩阵、二分类标签和病人分组。"""

    primary = CONFIG["hard_task"]["primary"]
    cohort_name = primary["cohort"]
    X = pd.read_pickle(
        configured_template_path(
            "residual_matrix_template",
            cohort=cohort_name.lower(),
        )
    )
    X.index = X.index.astype(str)

    labels = pd.read_csv(
        configured_template_path(
            "hard_task_labels_template",
            task=primary["task_name"],
        ),
        index_col="sample_id",
    )
    labels.index = labels.index.astype(str)
    if labels.index.has_duplicates:
        raise ValueError("困难任务标签中存在重复样本 ID。")
    if not labels.index.isin(X.index).all():
        raise ValueError("困难任务标签包含残差矩阵中不存在的样本。")

    X = X.loc[labels.index]
    y = labels["target"].astype(int)
    groups = labels["patient_id"].astype(str)
    if groups.nunique() != len(groups):
        raise ValueError("困难任务中每位病人应只有一个肿瘤样本。")
    return X, y, groups


def select_feature_set(X: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """按配置选择一个或多个 PTM 修饰类型，保留原有特征列顺序。"""

    if not isinstance(X.columns, pd.MultiIndex):
        raise TypeError("困难任务残差矩阵必须保留 Modification 多层列索引。")
    allowed_modifications = CONFIG["hard_task"]["feature_sets"][feature_set_name][
        "modifications"
    ]
    modifications = X.columns.get_level_values("Modification")
    selected = X.loc[:, modifications.isin(allowed_modifications)]
    if selected.shape[1] == 0:
        raise ValueError(f"特征集 {feature_set_name} 未匹配到任何 PTM 特征。")
    return selected


def fixed_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """一次生成分层病人分组划分，供所有特征集公平复用。"""

    cv = StratifiedGroupKFold(
        n_splits=CONFIG["model"]["cv_splits"],
        shuffle=True,
        random_state=CONFIG["model"]["xgboost"]["random_state"],
    )
    return list(cv.split(X, y, groups))


def score_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[object, object]],
    feature_set_name: str,
) -> pd.DataFrame:
    """在固定划分上评估一个特征集，并返回逐折评分。"""

    selected = select_feature_set(X, feature_set_name)
    scoring = CONFIG["model"]["scoring"]
    scores = cross_validate(
        estimator=make_xgboost_pipeline(),
        X=selected,
        y=y,
        groups=groups,
        cv=splits,
        scoring=scoring,
        n_jobs=CONFIG["hard_task"]["cv_parallel_jobs"],
        return_train_score=False,
        error_score="raise",
    )
    results = pd.DataFrame(
        {metric: scores[f"test_{metric}"] for metric in scoring}
    )
    results.index.name = "fold"
    results.insert(0, "feature_set", feature_set_name)
    results.insert(1, "n_features", selected.shape[1])
    return results


def summarize_scores(results_by_set: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """汇总各特征集逐折分数，保留均值和折间标准差。"""

    rows: list[dict[str, object]] = []
    for feature_set_name, scores in results_by_set.items():
        row: dict[str, object] = {
            "feature_set": feature_set_name,
            "n_features": scores["n_features"].iloc[0],
            "n_folds": len(scores),
        }
        for metric in CONFIG["model"]["scoring"]:
            row[f"{metric}_mean"] = scores[metric].mean()
            row[f"{metric}_std"] = scores[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """运行全部配置的修饰类型消融，并保存逐折与汇总结果。"""

    primary = CONFIG["hard_task"]["primary"]
    X, y, groups = load_hard_task_data()
    splits = fixed_splits(X, y, groups)
    print("困难任务：", primary["task_name"])
    print("样本数：", len(y))
    print("类别计数：")
    print(y.value_counts().sort_index().to_string())

    results_by_set: dict[str, pd.DataFrame] = {}
    for feature_set_name in CONFIG["hard_task"]["feature_sets"]:
        results = score_feature_set(X, y, groups, splits, feature_set_name)
        output_path = configured_template_path(
            "hard_task_scores_template",
            task=primary["task_name"],
            feature_set=feature_set_name,
        )
        results.to_csv(output_path)
        results_by_set[feature_set_name] = results
        print(f"\n{feature_set_name} 每折结果：")
        print(results.to_string())
        print("已保存每折结果：", output_path)

    summary = summarize_scores(results_by_set)
    summary_path = configured_template_path(
        "hard_task_summary_template",
        task=primary["task_name"],
    )
    summary.to_csv(summary_path, index=False)
    print("\n汇总结果：")
    print(summary.to_string(index=False))
    print("已保存汇总结果：", summary_path)
    return results_by_set, summary


def main() -> None:
    """执行配置的困难任务 PTM 修饰类型消融。"""

    run_ablation()


if __name__ == "__main__":
    main()
