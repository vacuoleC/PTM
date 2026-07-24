"""用重复 CV 与乙酰化特征块置换检验多类 PTM 的增量。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from project_config import CONFIG, configured_template_path
from run_floor import make_xgboost_pipeline
from run_hard_task_ablation import load_hard_task_data, select_feature_set


def make_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series, repeat: int):
    """为一个重复编号生成可复现、分层且按病人分组的折。"""

    cv = StratifiedGroupKFold(
        n_splits=CONFIG["model"]["cv_splits"],
        shuffle=True,
        random_state=CONFIG["hard_task"]["random_seed"] + repeat,
    )
    return list(cv.split(X, y, groups))


def score_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_set_name: str,
) -> pd.DataFrame:
    """在指定折上评估一个特征集，返回逐折所有预注册指标。"""

    scores = cross_validate(
        estimator=make_xgboost_pipeline(),
        X=X,
        y=y,
        groups=groups,
        cv=splits,
        scoring=CONFIG["model"]["scoring"],
        n_jobs=CONFIG["hard_task"]["cv_parallel_jobs"],
        return_train_score=False,
        error_score="raise",
    )
    result = pd.DataFrame(
        {metric: scores[f"test_{metric}"] for metric in CONFIG["model"]["scoring"]}
    )
    result.index.name = "fold"
    result.insert(0, "feature_set", feature_set_name)
    result.insert(1, "n_features", X.shape[1])
    result.insert(2, "n_train", [len(train) for train, _ in splits])
    result.insert(3, "n_test", [len(test) for _, test in splits])
    return result.reset_index()


def repeated_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对磷酸化和多类 PTM 使用完全相同的 repeated-CV 折。"""

    X, y, groups = load_hard_task_data()
    feature_sets = {
        feature_set_name: select_feature_set(X, feature_set_name)
        for feature_set_name in CONFIG["hard_task"]["feature_sets"]
    }
    all_scores: list[pd.DataFrame] = []
    for repeat in range(CONFIG["hard_task"]["repeated_cv_repeats"]):
        splits = make_splits(X, y, groups, repeat)
        for feature_set_name, features in feature_sets.items():
            scores = score_feature_set(features, y, groups, splits, feature_set_name)
            scores.insert(0, "repeat", repeat)
            all_scores.append(scores)

    result = pd.concat(all_scores, ignore_index=True)
    primary = CONFIG["hard_task"]["primary"]
    output_path = configured_template_path(
        "hard_task_repeated_scores_template",
        task=primary["task_name"],
    )
    result.to_csv(output_path, index=False)

    phospho_name, multi_name = CONFIG["hard_task"]["feature_sets"].keys()
    phospho = result.loc[result["feature_set"] == phospho_name].set_index(["repeat", "fold"])
    multi = result.loc[result["feature_set"] == multi_name].set_index(["repeat", "fold"])
    paired = pd.DataFrame(
        {
            "phosphoproteome_auprc": phospho["average_precision"],
            "multi_ptm_auprc": multi["average_precision"],
            "n_train": phospho["n_train"],
            "n_test": phospho["n_test"],
        }
    )
    paired["delta_auprc"] = paired["multi_ptm_auprc"] - paired["phosphoproteome_auprc"]
    return result, paired.reset_index(), feature_sets[multi_name]


def nadeau_bengio_summary(paired: pd.DataFrame) -> dict[str, float]:
    """以 corrected resampled t 描述 repeated-CV 的配对 AUPRC 差。"""

    deltas = paired["delta_auprc"].to_numpy()
    correction = 1 / len(deltas) + (paired["n_test"] / paired["n_train"]).mean()
    standard_error = np.sqrt(deltas.var(ddof=1) * correction)
    t_statistic = deltas.mean() / standard_error if standard_error else np.nan
    p_one_sided = student_t.sf(t_statistic, df=len(deltas) - 1)
    return {
        "repeated_delta_mean": deltas.mean(),
        "repeated_delta_std": deltas.std(ddof=1),
        "nadeau_bengio_correction": correction,
        "nadeau_bengio_t": t_statistic,
        "nadeau_bengio_p_one_sided": p_one_sided,
    }


def acetylation_block(X_multi: pd.DataFrame) -> pd.DataFrame:
    """从多类矩阵中选择相对磷酸化新增的修饰类型块。"""

    phospho_modifications = set(
        CONFIG["hard_task"]["feature_sets"]["phosphoproteome"]["modifications"]
    )
    modifications = X_multi.columns.get_level_values("Modification")
    return X_multi.loc[:, ~modifications.isin(phospho_modifications)]


def block_permutation_null() -> tuple[pd.DataFrame, float, float]:
    """打乱乙酰化样本块，构造“无乙酰化增量”经验 null。"""

    X, y, groups = load_hard_task_data()
    phospho = select_feature_set(X, "phosphoproteome")
    multi = select_feature_set(X, "multi_ptm")
    acetylation = acetylation_block(multi)
    splits = make_splits(X, y, groups, repeat=0)

    baseline = score_feature_set(phospho, y, groups, splits, "phosphoproteome")
    baseline_auprc = baseline["average_precision"].mean()
    observed = score_feature_set(multi, y, groups, splits, "multi_ptm")
    observed_delta = observed["average_precision"].mean() - baseline_auprc

    rng = np.random.default_rng(CONFIG["hard_task"]["random_seed"])
    null_rows: list[dict[str, float]] = []
    for permutation in range(CONFIG["hard_task"]["block_permutations"]):
        shuffled = acetylation.iloc[rng.permutation(len(acetylation))].copy()
        shuffled.index = phospho.index
        null_multi = pd.concat([phospho, shuffled], axis=1)
        null_scores = score_feature_set(null_multi, y, groups, splits, "permuted_acetylation")
        null_rows.append(
            {
                "permutation": permutation,
                "phosphoproteome_auprc": baseline_auprc,
                "permuted_multi_ptm_auprc": null_scores["average_precision"].mean(),
                "delta_auprc": null_scores["average_precision"].mean() - baseline_auprc,
            }
        )
        if (permutation + 1) % CONFIG["phase0"]["null_progress_every"] == 0:
            print(f"已完成 {permutation + 1} / {CONFIG['hard_task']['block_permutations']} 次乙酰化块置换")

    null = pd.DataFrame(null_rows)
    primary = CONFIG["hard_task"]["primary"]
    output_path = configured_template_path(
        "hard_task_increment_null_template",
        task=primary["task_name"],
    )
    null.to_csv(output_path, index=False)
    empirical_p = (np.sum(null["delta_auprc"] >= observed_delta) + 1) / (len(null) + 1)
    return null, observed_delta, empirical_p


def main() -> None:
    """保存 repeated-CV、块置换 null 与统计汇总。"""

    repeated, paired, _ = repeated_scores()
    corrected = nadeau_bengio_summary(paired)
    null, observed_delta, empirical_p = block_permutation_null()
    primary = CONFIG["hard_task"]["primary"]
    summary = pd.DataFrame(
        [
            {
                "task": primary["task_name"],
                "repeated_cv_repeats": CONFIG["hard_task"]["repeated_cv_repeats"],
                "block_permutations": CONFIG["hard_task"]["block_permutations"],
                "fixed_split_observed_delta_auprc": observed_delta,
                "block_permutation_empirical_p_one_sided": empirical_p,
                **corrected,
            }
        ]
    )
    summary_path = configured_template_path(
        "hard_task_increment_summary_template",
        task=primary["task_name"],
    )
    summary.to_csv(summary_path, index=False)
    print("重复 CV 逐折结果：", configured_template_path("hard_task_repeated_scores_template", task=primary["task_name"]))
    print("乙酰化块置换 null：", configured_template_path("hard_task_increment_null_template", task=primary["task_name"]))
    print("增量统计汇总：", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
