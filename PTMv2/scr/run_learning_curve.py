"""E3.2 learning curve runner for the frozen pca_elastic_net primary model.

For each training_fraction in [0.5, 0.7, 1.0] × 10 repeats:
  - Subsample patients in each training fold at the fraction (patient-level,
    stratified by class), with a repeat-specific random seed.
  - Run the nested pipeline (inner candidate selection + outer OOF prediction)
    on the subsampled training fold.
  - Record pooled OOF AUPRC per (fraction, repeat).

Each fraction retains stratified inner validation, as frozen in study_design.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from nested_pca_elasticnet import parameter_grid
from evaluate import fit_score_fold_pca

_WORKER_STATE: tuple | None = None


def _init_worker(X, y, assignments, candidates, inner_splits, random_seed):
    global _WORKER_STATE
    _WORKER_STATE = (X, y, assignments, candidates, inner_splits, random_seed)


def _select_parameters(X, y, candidates, inner_splits, random_state):
    """Inner selection on a training subset, returns best candidate.

    PCA components are clipped to min(component, n_samples - 1) because
    PCA cannot exceed the number of samples — a mathematical constraint,
    not a design choice. This keeps the frozen component grid [10, 20, 40]
    intact for fractions where it is feasible.
    """
    n_samples = X.shape[0]
    max_comp = max(1, n_samples - 1)
    feasible = [
        (threshold, min(n_comp, max_comp), C_val, l1)
        for threshold, n_comp, C_val, l1 in candidates
    ]
    feasible = list(dict.fromkeys(feasible))
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    best, best_score = None, -1.0
    for threshold, n_comp, C_val, l1 in feasible:
        fold_scores = []
        for ti, vi in cv.split(X, y):
            p = fit_score_fold_pca(
                X.iloc[ti], y.iloc[ti], X.iloc[vi], threshold, n_comp, C_val, l1,
            )
            fold_scores.append(average_precision_score(y.iloc[vi], p))
        sc = float(np.mean(fold_scores))
        if sc > best_score:
            best_score, best = sc, (threshold, n_comp, C_val, l1)
    return best, best_score


def _run_cell(args):
    """One (fraction, repeat) cell across all 50 outer folds."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        fraction, repeat = args
        X, y, assignments, candidates, inner_splits, random_seed = _WORKER_STATE
        rng = np.random.default_rng(random_seed + repeat * 131 + int(fraction * 100))
        records = []
        for fold in sorted(assignments.fold.unique()):
            train_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
            test_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
            # Patient-level stratified subsample of the training fold
            y_train = y.loc[train_ids]
            pos = y_train[y_train == 1].index.to_numpy()
            neg = y_train[y_train == 0].index.to_numpy()
            n_pos = max(1, int(len(pos) * fraction))
            n_neg = max(1, int(len(neg) * fraction))
            keep = np.concatenate([
                rng.choice(pos, size=n_pos, replace=False),
                rng.choice(neg, size=n_neg, replace=False),
            ])
            X_frac = X.loc[keep]
            y_frac = y.loc[keep]
            best, _ = _select_parameters(X_frac, y_frac, candidates, inner_splits, random_seed + int(fold))
            p = fit_score_fold_pca(X_frac, y_frac, X.loc[test_ids], *best)
            records.extend(
                {"fold": fold, "patient_id": pid, "target": int(y.loc[pid]), "score": s}
                for pid, s in zip(test_ids, p)
            )
        oof = pd.DataFrame(records)
        auprc = float(average_precision_score(oof.target, oof.score))
        return {"fraction": fraction, "repeat": repeat, "n_train_patients": len(keep), "oof_auprc": round(auprc, 6)}


def main(config_path: Path, n_jobs: int) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    pca_model = next(model for model in study["models"]["primary"] if model["id"] == "pca_elastic_net")
    candidates = parameter_grid(
        study["preprocessing"]["detection_threshold_candidates"],
        pca_model["pca_components"],
        pca_model["classifier"]["C"], pca_model["classifier"]["l1_ratio"],
    )
    fractions = study["evaluation"]["learning_curve"]["training_fractions"]
    repeats = study["evaluation"]["learning_curve"]["repeats_per_fraction"]
    inner_splits = study["evaluation"]["inner_cv"]["splits"]
    random_seed = study["evaluation"]["random_seed"]

    cells = [(f, r) for f in fractions for r in range(repeats)]
    print(f"learning curve: {len(cells)} cells, n_jobs={n_jobs}", flush=True)

    payload = (X, labels, assignments, candidates, inner_splits, random_seed)
    results = []
    start = time.monotonic()
    if n_jobs <= 1:
        _init_worker(*payload)
        for cell in cells:
            results.append(_run_cell(cell))
            print(f"cell {cell} done ({time.monotonic() - start:.0f}s)", flush=True)
    else:
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=payload) as pool:
            for r in pool.imap_unordered(_run_cell, cells, chunksize=1):
                results.append(r)
                print(f"cell {r['fraction']}/{r['repeat']} done ({time.monotonic() - start:.0f}s)", flush=True)

    df = pd.DataFrame(results).sort_values(["fraction", "repeat"])
    out = root / "outputs/tables/primary_model_learning_curve.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("fraction")["oof_auprc"].agg(["mean", "std", "min", "max"]).round(6)
    print(f"wrote {out}", flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.n_jobs)
