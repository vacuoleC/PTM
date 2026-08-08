"""Leakage-safe nested CV core for the frozen pca_elastic_net primary model.

Mirrors nested_raw_elasticnet but adds training-fold PCA dimensionality
reduction before the low-dimensional saga logistic fit. PCA is fitted on
the training fold only (no test leakage), exactly as the frozen
pca_elastic_net pipeline specifies.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from evaluate import fit_score_fold_pca


def parameter_grid(thresholds, components, cs, l1_ratios):
    """Return the predeclared pca-model candidate tuples (threshold, comp, C, l1)."""
    return list(product(thresholds, components, cs, l1_ratios))


def select_parameters(X, y, candidates, inner_splits, random_state):
    """Select one candidate using only stratified inner training/validation folds."""
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    scores = []
    for threshold, n_comp, c_value, l1_ratio in candidates:
        fold_scores = []
        for train_idx, valid_idx in cv.split(X, y):
            probability = fit_score_fold_pca(
                X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx],
                threshold, n_comp, c_value, l1_ratio,
            )
            fold_scores.append(average_precision_score(y.iloc[valid_idx], probability))
        scores.append(float(np.mean(fold_scores)))
    best_index = int(np.argmax(scores))
    return candidates[best_index], scores[best_index]


def nested_oof(X, y, assignments, candidates, inner_splits, random_state):
    """Produce outer-fold OOF scores with inner selection for the pca model."""
    records, selected = [], []
    for fold in sorted(assignments.fold.unique()):
        train_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        test_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        params, inner_ap = select_parameters(
            X.loc[train_ids], y.loc[train_ids], candidates, inner_splits, random_state + int(fold)
        )
        scores = fit_score_fold_pca(X.loc[train_ids], y.loc[train_ids], X.loc[test_ids], *params)
        records.extend(
            {"fold": fold, "patient_id": patient, "target": int(y.loc[patient]), "score": score}
            for patient, score in zip(test_ids, scores)
        )
        selected.append(
            {
                "fold": fold,
                "threshold": params[0], "n_components": params[1],
                "C": params[2], "l1_ratio": params[3], "inner_ap": inner_ap,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(selected)
