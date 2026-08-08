"""Cached-preprocessing full nested fold: 3 thresholds' stats computed once per
inner fold, then 27 candidates reuse them. Cache key = (row indices tuple,
threshold), NOT id() (id reuse is a real aliasing bug).
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/data/PTM/PTMv2/scr")
from preprocessing import make_preprocessing_pipeline

ROOT = "/data/PTM/PTMv2"

CACHE = {}  # (tuple(row_indices), threshold) -> ndarray


def fit_cuml_cached(Xtr_df, ytr, Xte_df, threshold, C, l1r):
    from cuml.linear_model import LogisticRegression

    key_tr = (tuple(Xtr_df.index), threshold)
    t0 = time.monotonic()
    if key_tr not in CACHE:
        prep = make_preprocessing_pipeline(threshold)
        CACHE[key_tr] = prep.fit_transform(Xtr_df)
    Xt = CACHE[key_tr].astype(np.float32)
    key_te = (tuple(Xte_df.index), threshold)
    if key_te not in CACHE:
        prep = make_preprocessing_pipeline(threshold)
        prep.fit(Xtr_df)
        CACHE[key_te] = prep.transform(Xte_df)
    Xv = CACHE[key_te].astype(np.float32)
    t_prep = time.monotonic() - t0

    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    t_fit = time.monotonic() - t0
    return m.predict_proba(Xv)[:, 1], t_prep, t_fit


def main(fold: int) -> None:
    X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
    labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
    assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
    X = X.loc[labels.index]

    tr_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
    te_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
    Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
    Xte, yte = X.loc[te_ids], labels.loc[te_ids]

    thresholds = [0.1, 0.3, 0.5]
    Cs = [0.01, 0.1, 1.0]
    l1rs = [0.1, 0.5, 0.9]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0 + int(fold))

    t_start = time.monotonic()
    prep_total, fit_total = 0.0, 0.0
    results = []
    for thr in thresholds:
        for C in Cs:
            for l1r in l1rs:
                fs = []
                for ti, vi in cv.split(Xtr, ytr):
                    p, tp, tf = fit_cuml_cached(Xtr.iloc[ti], ytr.iloc[ti], Xtr.iloc[vi], thr, C, l1r)
                    prep_total += tp
                    fit_total += tf
                    fs.append(average_precision_score(ytr.iloc[vi], p))
                results.append(((thr, C, l1r), float(np.mean(fs))))
                print(f"[inner] thr={thr} C={C} l1r={l1r} mean_ap={results[-1][1]:.4f} "
                      f"({time.monotonic() - t_start:.0f}s)", flush=True)
    print(f"inner selection: {time.monotonic() - t_start:.1f}s (prep={prep_total:.1f}s fit={fit_total:.1f}s)", flush=True)

    best = max(results, key=lambda r: r[1])
    print(f"selected={best[0]} inner_ap={best[1]:.4f}", flush=True)
    p, tp, tf = fit_cuml_cached(Xtr, ytr, Xte, *best[0])
    oof_ap = average_precision_score(yte, p)
    print(f"outer fold {fold} oof_ap={oof_ap:.4f} prep={tp:.1f}s fit={tf:.1f}s", flush=True)
    print(f"TOTAL={time.monotonic() - t_start:.1f}s fold={fold} solver=cuml-qn-cached", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    main(args.fold)
