"""用置换 null、BH 校正和学习曲线完成 Phase 0 的统计结案。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    cross_validate,
    learning_curve,
)

from project_config import CONFIG, configured_path, configured_template_path
from run_floor import load_phase0_data, make_xgboost_pipeline


def fixed_group_splits(X, y, groups):
    """生成与 floor 实验一致、按病人分组的固定五折划分。"""

    cv = GroupKFold(n_splits=CONFIG["model"]["cv_splits"])
    return list(cv.split(X, y, groups))


def score_xgboost(X, y, splits):
    """在给定划分上计算主指标的每折分数。"""

    primary_metric = CONFIG["model"]["primary_metric"]
    scores = cross_validate(
        estimator=make_xgboost_pipeline(),
        X=X,
        y=y,
        cv=splits,
        scoring=primary_metric,
        return_train_score=False,
        error_score="raise",
    )
    return scores["test_score"]


def permute_labels(y, rng):
    """在样本层面置换标签，同时保留原始类别数量与分组划分。"""

    return pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)


def benjamini_hochberg(p_values):
    """返回与输入顺序一致的 Benjamini-Hochberg 校正 q 值。"""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def run_null_for_cohort(cohort_name):
    """运行一个队列的观察分数与标签置换 null。"""

    phase0_config = CONFIG["phase0"]
    X, y, groups = load_phase0_data(cohort_name)
    splits = fixed_group_splits(X, y, groups)

    observed_scores = score_xgboost(X, y, splits)
    observed_mean = observed_scores.mean()
    print(f"{cohort_name} 观察到的 {CONFIG['model']['primary_metric']}：{observed_mean:.4f}")

    rng = np.random.default_rng(phase0_config["null_random_seed"])
    null_rows = []
    n_permutations = phase0_config["null_permutations"]

    for permutation in range(n_permutations):
        null_scores = score_xgboost(X, permute_labels(y, rng), splits)
        null_rows.append(
            {
                "permutation": permutation,
                "average_precision": null_scores.mean(),
            }
        )

        if (permutation + 1) % phase0_config["null_progress_every"] == 0:
            print(f"{cohort_name} 已完成 {permutation + 1} / {n_permutations} 次置换")

    null_scores = pd.DataFrame(null_rows)
    null_path = configured_template_path(
        "phase0_null_scores_template",
        cohort=cohort_name.lower(),
    )
    null_scores.to_csv(null_path, index=False)

    empirical_p = (
        (null_scores["average_precision"] >= observed_mean).sum() + 1
    ) / (n_permutations + 1)

    print(f"{cohort_name} 经验 p 值：{empirical_p:.4f}")
    print(f"已保存置换 null：{null_path}")

    return {
        "cohort": cohort_name,
        "n_samples": len(y),
        "n_tumor": int(y.sum()),
        "n_normal": int((y == 0).sum()),
        "n_permutations": n_permutations,
        "observed_average_precision": observed_mean,
        "null_average_precision_mean": null_scores["average_precision"].mean(),
        "null_average_precision_std": null_scores["average_precision"].std(),
        "empirical_p": empirical_p,
    }


def run_learning_curve(cohort_name):
    """保存队列内 XGBoost 学习曲线，供功效/饱和度诊断使用。"""

    X, y, groups = load_phase0_data(cohort_name)
    cv = GroupKFold(n_splits=CONFIG["model"]["cv_splits"])
    train_sizes, _, test_scores = learning_curve(
        estimator=make_xgboost_pipeline(),
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        scoring=CONFIG["model"]["primary_metric"],
        train_sizes=CONFIG["phase0"]["learning_curve_train_sizes"],
        shuffle=True,
        random_state=CONFIG["phase0"]["null_random_seed"],
        error_score="raise",
    )

    result = pd.DataFrame(
        {
            "train_size": train_sizes,
            "average_precision_mean": test_scores.mean(axis=1),
            "average_precision_std": test_scores.std(axis=1, ddof=1),
            "average_precision_lower_95": np.quantile(test_scores, 0.025, axis=1),
            "average_precision_upper_95": np.quantile(test_scores, 0.975, axis=1),
        }
    )
    output_path = configured_template_path(
        "phase0_learning_curve_template",
        cohort=cohort_name.lower(),
    )
    result.to_csv(output_path, index=False)
    print(f"已保存 {cohort_name} 学习曲线：{output_path}")
    return result


def run_ucec_repeated_cv():
    """用重复的分层病人分组 CV 记录 UCEC 的折间稳定性。"""

    cohort_name = "UCEC"
    X, y, groups = load_phase0_data(cohort_name)
    rows = []

    for repeat in range(CONFIG["phase0"]["ucec_cv_repeats"]):
        cv = StratifiedGroupKFold(
            n_splits=CONFIG["model"]["cv_splits"],
            shuffle=True,
            random_state=CONFIG["phase0"]["null_random_seed"] + repeat,
        )
        splits = list(cv.split(X, y, groups))
        scores = score_xgboost(X, y, splits)
        rows.extend(
            {"repeat": repeat, "fold": fold, "average_precision": score}
            for fold, score in enumerate(scores)
        )

    return pd.DataFrame(rows)


def main():
    output_dir = configured_path("output_dir")
    output_dir.mkdir(exist_ok=True)

    summary_rows = []
    for cohort_name in CONFIG["phase0"]["analysis_cohorts"]:
        summary_rows.append(run_null_for_cohort(cohort_name))
        run_learning_curve(cohort_name)

    summary = pd.DataFrame(summary_rows)
    summary["bh_q"] = benjamini_hochberg(summary["empirical_p"])

    ucec_repeated = run_ucec_repeated_cv()
    ucec_summary = ucec_repeated["average_precision"].agg(["mean", "std"])
    summary.loc[summary["cohort"] == "UCEC", "repeated_cv_mean"] = ucec_summary["mean"]
    summary.loc[summary["cohort"] == "UCEC", "repeated_cv_std"] = ucec_summary["std"]
    summary.loc[summary["cohort"] == "UCEC", "repeated_cv_lower_95"] = (
        ucec_repeated["average_precision"].quantile(0.025)
    )
    summary.loc[summary["cohort"] == "UCEC", "repeated_cv_upper_95"] = (
        ucec_repeated["average_precision"].quantile(0.975)
    )

    summary_path = configured_path("phase0_null_summary")
    summary.to_csv(summary_path, index=False)
    print("\nPhase 0 置换检验汇总：")
    print(summary.to_string(index=False))
    print(f"已保存汇总：{summary_path}")


if __name__ == "__main__":
    main()
