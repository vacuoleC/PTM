"""Leakage-safe nested CV core for the frozen pca_elastic_net primary model.

Mirrors nested_raw_elasticnet but adds training-fold PCA dimensionality
reduction before the low-dimensional saga logistic fit. PCA is fitted on
the training fold only (no test leakage), exactly as the frozen
pca_elastic_net pipeline specifies.

The 50 outer folds are fully independent (each uses only its own train/test
patient split), so nested_oof can parallelize across folds via a worker pool.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from evaluate import fit_score_fold_pca

_WORKER_STATE: tuple | None = None


def _init_worker(X, y, assignments, candidates, inner_splits, random_seed):
    global _WORKER_STATE
    _WORKER_STATE = (X, y, assignments, candidates, inner_splits, random_seed)


def parameter_grid(thresholds, components, cs, l1_ratios):
    """Return the predeclared pca-model candidate tuples (threshold, comp, C, l1)."""
    return list(product(thresholds, components, cs, l1_ratios))


def _run_fold(fold: int) -> tuple[list[dict], dict]:
    """Run one outer fold: inner selection then outer OOF prediction."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        X, y, assignments, candidates, inner_splits, random_seed = _WORKER_STATE
        train_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        test_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        X_tr, y_tr = X.loc[train_ids], y.loc[train_ids]
        X_te = X.loc[test_ids]

        # Inner selection on the training fold
        cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed + int(fold))
        best, best_score = None, -1.0
        for threshold, n_comp, C_val, l1 in candidates:
            fold_scores = []
            for ti, vi in cv.split(X_tr, y_tr):
                p = fit_score_fold_pca(
                    X_tr.iloc[ti], y_tr.iloc[ti], X_tr.iloc[vi], threshold, n_comp, C_val, l1,
                )
                fold_scores.append(average_precision_score(y_tr.iloc[vi], p))
            sc = float(np.mean(fold_scores))
            if sc > best_score:
                best_score, best = sc, (threshold, n_comp, C_val, l1)

        scores = fit_score_fold_pca(X_tr, y_tr, X_te, *best)
        records = [
            {"fold": fold, "patient_id": patient, "target": int(y.loc[patient]), "score": score}
            for patient, score in zip(test_ids, scores)
        ]
        selected = {
            "fold": fold,
            "threshold": best[0], "n_components": best[1],
            "C": best[2], "l1_ratio": best[3], "inner_ap": best_score,
        }
        return records, selected


def nested_oof(X, y, assignments, candidates, inner_splits, random_seed, n_jobs: int = 1):
    """Produce outer-fold OOF scores with inner selection, fold-parallel when n_jobs>1."""
    folds = sorted(assignments.fold.unique())
    payload = (X, y, assignments, candidates, inner_splits, random_seed)

    if n_jobs <= 1 or len(folds) == 1:
        _init_worker(*payload)
        results = [_run_fold(f) for f in folds]
    else:
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=payload) as pool:
            results = list(pool.imap_unordered(_run_fold, folds, chunksize=1))

    all_records, all_selected = [], []
    for records, selected in results:
        all_records.extend(records)
        all_selected.append(selected)
    return pd.DataFrame(all_records), pd.DataFrame(all_selected)
