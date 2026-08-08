"""E4 model comparison runner for frozen primary + exploratory models.

Runs full nested CV (inner selection + outer OOF) for each model and reports
pooled OOF AUPRC per repeat, enabling paired comparison per frozen design.
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

from evaluate import fit_score_fold, fit_score_fold_pca

_WORKER_STATE: tuple | None = None


def _init_worker(X, y, assignments, model_id, inner_splits, random_seed):
    global _WORKER_STATE
    _WORKER_STATE = (X, y, assignments, model_id, inner_splits, random_seed)


def _fit_fold(model_id, Xtr, ytr, Xte, params):
    """Fit one candidate for a model id, return test probabilities."""
    if model_id == "raw_elastic_net":
        threshold, C_val, l1 = params
        return fit_score_fold(Xtr, ytr, Xte, threshold, C_val, l1)
    if model_id == "pca_elastic_net":
        threshold, n_comp, C_val, l1 = params
        return fit_score_fold_pca(Xtr, ytr, Xte, threshold, n_comp, C_val, l1)
    raise ValueError(f"unsupported model {model_id}")


def _select(model_id, Xtr, ytr, candidates, inner_splits, seed):
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    best, best_score = None, -1.0
    for cand in candidates:
        fs = []
        for ti, vi in cv.split(Xtr, ytr):
            p = _fit_fold(model_id, Xtr.iloc[ti], ytr.iloc[ti], Xtr.iloc[vi], cand)
            fs.append(average_precision_score(ytr.iloc[vi], p))
        sc = float(np.mean(fs))
        if sc > best_score:
            best_score, best = sc, cand
    return best


def _run_model(args):
    model_id = args
    X, y, assignments, _, inner_splits, random_seed = _WORKER_STATE
    records = []
    for fold in sorted(assignments.fold.unique()):
        train_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        test_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        Xtr, ytr = X.loc[train_ids], y.loc[train_ids]
        Xte = X.loc[test_ids]
        # candidate grid per model
        if model_id == "raw_elastic_net":
            from nested_raw_elasticnet import parameter_grid as raw_grid
            candidates = raw_grid([0.1, 0.3, 0.5], [0.01, 0.1, 1.0], [0.1, 0.5, 0.9])
        else:
            from nested_pca_elasticnet import parameter_grid as pca_grid
            candidates = pca_grid([0.1, 0.3, 0.5], [10, 20, 40], [0.01, 0.1, 1.0], [0.1, 0.5, 0.9])
        best = _select(model_id, Xtr, ytr, candidates, inner_splits, random_seed + int(fold))
        p = _fit_fold(model_id, Xtr, ytr, Xte, best)
        records.extend(
            {"fold": fold, "patient_id": pid, "target": int(y.loc[pid]), "score": s}
            for pid, s in zip(test_ids, p)
        )
    oof = pd.DataFrame(records)
    auprc = float(average_precision_score(oof.target, oof.score))
    return {"model_id": model_id, "oof_auprc": round(auprc, 6), "n_oof": len(oof)}


def main(config_path: Path, models: list[str], n_jobs: int) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    inner_splits = study["evaluation"]["inner_cv"]["splits"]
    random_seed = study["evaluation"]["random_seed"]

    valid = {"raw_elastic_net", "pca_elastic_net"}
    selected = [m for m in models if m in valid]
    if not selected:
        raise SystemExit(f"no valid model in {models}; valid: {valid}")

    print(f"model comparison: {selected} n_jobs={n_jobs}", flush=True)
    payload = (X, labels, assignments, None, inner_splits, random_seed)
    results = []
    start = time.monotonic()
    if n_jobs <= 1:
        _init_worker(*payload)
        for m in selected:
            results.append(_run_model(m))
            print(f"{m} done ({time.monotonic() - start:.0f}s)", flush=True)
    else:
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=payload) as pool:
            for r in pool.imap_unordered(_run_model, selected, chunksize=1):
                results.append(r)
                print(f"{r['model_id']} done ({time.monotonic() - start:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    out = root / "outputs/tables/primary_model_oof_scores.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    parser.add_argument("--models", nargs="+", default=["raw_elastic_net", "pca_elastic_net"])
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.models, args.n_jobs)
