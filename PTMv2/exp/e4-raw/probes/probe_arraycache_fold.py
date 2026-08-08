"""Optimal fast path: cache final Xt/Xv arrays (3 thresholds x 3 inner folds),
then run 81 pure cuml fits. Expected ~200s/fold vs 520s naive-fast.
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

ROOT = "/data/PTM/PTMv2"

ARRAYS = {}


def prep_stats(arr, threshold):
    frac = np.mean(~np.isnan(arr), axis=0)
    keep = frac >= threshold
    a = arr[:, keep]
    s = np.sort(a, axis=0)
    nn = np.sum(~np.isnan(a), axis=0)
    med = (s[nn // 2, np.arange(a.shape[1])] + s[(nn - 1) // 2, np.arange(a.shape[1])]) / 2.0
    imp = np.where(np.isnan(a), med, a)
    mu = imp.mean(axis=0)
    sd = imp.std(axis=0)
    return keep, med, mu, sd


def prep_apply(arr, stats):
    keep, med, mu, sd = stats
    imp = np.where(np.isnan(arr[:, keep]), med, arr[:, keep])
    return (imp - mu) / sd


def get_arrays(arr_tr, arr_va, threshold):
    """Cached Xt/Xv for (train rows, threshold). Keyed by array id + threshold."""
    key_tr = (arr_tr.ctypes.data, threshold)
    if key_tr not in ARRAYS:
        st = prep_stats(arr_tr, threshold)
        ARRAYS[key_tr] = prep_apply(arr_tr, st)
        key_va = (arr_va.ctypes.data, threshold)
        ARRAYS[key_va] = prep_apply(arr_va, st)
    return ARRAYS[key_tr], ARRAYS.get((arr_va.ctypes.data, threshold))


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
    splits = list(cv.split(Xtr, ytr))

    from cuml.linear_model import LogisticRegression

    t_start = time.monotonic()
    prep_total, fit_total = 0.0, 0.0
    results = []
    # pre-convert all inner subsets to f32 arrays once
    inner_arrays = []
    for ti, vi in splits:
        a_tr = np.asarray(Xtr.iloc[ti], dtype=np.float32)
        a_va = np.asarray(Xtr.iloc[vi], dtype=np.float32)
        inner_arrays.append((a_tr, a_va, np.asarray(ytr.iloc[ti]).astype(np.int32), np.asarray(ytr.iloc[vi]).astype(np.int32)))
    for thr in thresholds:
        # prepare once per threshold: all 3 inner folds' Xt/Xv
        for a_tr, a_va, yt, yv in inner_arrays:
            t0 = time.monotonic()
            st = prep_stats(a_tr, thr)
            ARRAYS[(a_tr.ctypes.data, thr)] = prep_apply(a_tr, st)
            ARRAYS[(a_va.ctypes.data, thr)] = prep_apply(a_va, st)
            prep_total += time.monotonic() - t0
        for C in Cs:
            for l1r in l1rs:
                fs = []
                for a_tr, a_va, yt, yv in inner_arrays:
                    Xt = ARRAYS[(a_tr.ctypes.data, thr)]
                    Xv = ARRAYS[(a_va.ctypes.data, thr)]
                    t0 = time.monotonic()
                    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                                           max_iter=10000, tol=1e-4, verbose=False)
                    m.fit(Xt, yt)
                    fit_total += time.monotonic() - t0
                    fs.append(average_precision_score(yv, m.predict_proba(Xv)[:, 1]))
                results.append(((thr, C, l1r), float(np.mean(fs))))
                print(f"[inner] thr={thr} C={C} l1r={l1r} mean_ap={results[-1][1]:.4f} "
                      f"({time.monotonic() - t_start:.0f}s)", flush=True)
    print(f"inner selection: {time.monotonic() - t_start:.1f}s (prep={prep_total:.1f}s fit={fit_total:.1f}s)", flush=True)

    best = max(results, key=lambda r: r[1])
    print(f"selected={best[0]} inner_ap={best[1]:.4f}", flush=True)
    arr_tr = np.asarray(Xtr, dtype=np.float32)
    arr_te = np.asarray(Xte, dtype=np.float32)
    t0 = time.monotonic()
    st = prep_stats(arr_tr, best[0][0])
    Xt = prep_apply(arr_tr, st)
    Xv = prep_apply(arr_te, st)
    t_prep = time.monotonic() - t0
    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=best[0][1], l1_ratio=best[0][2],
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    t_fit = time.monotonic() - t0
    oof_ap = average_precision_score(yte, m.predict_proba(Xv)[:, 1])
    print(f"outer fold {fold} oof_ap={oof_ap:.4f} prep={t_prep:.1f}s fit={t_fit:.1f}s", flush=True)
    print(f"TOTAL={time.monotonic() - t_start:.1f}s fold={fold} solver=cuml-qn-arraycache", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    main(args.fold)
