"""E4.1 raw exploration: full 50-fold nested CV runner (GPU, array-cache).

Mirrors the frozen raw pipeline (detection filter -> median impute -> standard
scale -> elastic-net logistic) but executes the logistic fit with cuml-qn on
GPU and the preprocessing with a numerically-equivalent f32 fast path.
Writes per-fold results incrementally (checkpoint CSV) for resumability.

Usage: python e4raw_run_full.py [--start-fold N] [--n-parallel 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/data/PTM/PTMv2"
OUT = "/data/PTM/PTMv2/exp/e4-raw/outputs"

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


def run_fold(fold, X, labels, assignments, out_csv):
    """Run one outer fold; append records to out_csv. Returns summary line."""
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
    inner_arrays = []
    for ti, vi in splits:
        a_tr = np.asarray(Xtr.iloc[ti], dtype=np.float32)
        a_va = np.asarray(Xtr.iloc[vi], dtype=np.float32)
        inner_arrays.append((a_tr, a_va, np.asarray(ytr.iloc[ti]).astype(np.int32),
                             np.asarray(ytr.iloc[vi]).astype(np.int32)))

    results = []
    for thr in thresholds:
        for a_tr, a_va, yt, yv in inner_arrays:
            st = prep_stats(a_tr, thr)
            ARRAYS[(a_tr.ctypes.data, thr)] = prep_apply(a_tr, st)
            ARRAYS[(a_va.ctypes.data, thr)] = prep_apply(a_va, st)
        for C in Cs:
            for l1r in l1rs:
                fs = []
                for a_tr, a_va, yt, yv in inner_arrays:
                    Xt = ARRAYS[(a_tr.ctypes.data, thr)]
                    Xv = ARRAYS[(a_va.ctypes.data, thr)]
                    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                                           max_iter=10000, tol=1e-4, verbose=False)
                    m.fit(Xt, yt)
                    fs.append(average_precision_score(yv, m.predict_proba(Xv)[:, 1]))
                results.append(((thr, C, l1r), float(np.mean(fs))))
    best = max(results, key=lambda r: r[1])
    inner_ap = best[1]
    best_thr, best_C, best_l1r = best[0]

    arr_tr = np.asarray(Xtr, dtype=np.float32)
    arr_te = np.asarray(Xte, dtype=np.float32)
    st = prep_stats(arr_tr, best_thr)
    Xt = prep_apply(arr_tr, st)
    Xv = prep_apply(arr_te, st)
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=best_C, l1_ratio=best_l1r,
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    p = m.predict_proba(Xv)[:, 1]
    oof_ap = float(average_precision_score(yte, p))

    records = pd.DataFrame({
        "fold": fold,
        "patient_id": te_ids,
        "target": yte.to_numpy(),
        "score": p,
    })
    hdr = not os.path.exists(out_csv)
    records.to_csv(out_csv, mode="a", header=hdr, index=False)

    dt = time.monotonic() - t_start
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] fold {fold}: "
          f"oof_ap={oof_ap:.4f} sel=({best_thr},{best_C},{best_l1r}) inner_ap={inner_ap:.4f} "
          f"took={dt:.0f}s", flush=True)
    return {"fold": fold, "oof_ap": oof_ap, "threshold": best_thr, "C": best_C,
            "l1_ratio": best_l1r, "inner_ap": inner_ap, "seconds": round(dt, 1)}


def main(start_fold: int, n_parallel: int, max_folds: int) -> None:
    X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
    labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
    assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
    X = X.loc[labels.index]

    os.makedirs(OUT, exist_ok=True)
    oof_csv = f"{OUT}/e4raw_full_oof_scores.csv"
    sel_csv = f"{OUT}/e4raw_full_selected_params.csv"

    done = set()
    if os.path.exists(oof_csv):
        existing = pd.read_csv(oof_csv)
        done = set(existing.fold.unique())
        print(f"resume: {len(done)} folds already done", flush=True)

    all_folds = sorted(assignments.fold.unique())
    todo = [f for f in all_folds if f < start_fold + max_folds and f not in done]
    print(f"folds to run: {len(todo)} (start={start_fold} max={max_folds} parallel={n_parallel})", flush=True)

    summaries = []
    # run in parallel batches using multiprocessing with fork-free imports
    if n_parallel > 1 and len(todo) > 1:
        import multiprocessing as mp
        from functools import partial

        def _run(f):
            return run_fold(f, X, labels, assignments, oof_csv)

        with mp.Pool(processes=n_parallel, maxtasksperchild=1) as pool:
            for s in pool.imap_unordered(_run, todo):
                summaries.append(s)
                sel_hdr = not os.path.exists(sel_csv)
                pd.DataFrame([s]).to_csv(sel_csv, mode="a", header=sel_hdr, index=False)
    else:
        for f in todo:
            s = run_fold(f, X, labels, assignments, oof_csv)
            summaries.append(s)
            sel_hdr = not os.path.exists(sel_csv)
            pd.DataFrame([s]).to_csv(sel_csv, mode="a", header=sel_hdr, index=False)

    if summaries:
        df = pd.DataFrame(summaries)
        print("=== SUMMARY ===", flush=True)
        print(df.to_string(index=False), flush=True)
        print(f"pooled mean oof_ap={df.oof_ap.mean():.4f} total_seconds={df.seconds.sum():.0f} "
              f"wall={time.monotonic() - 0:.0f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--n-parallel", type=int, default=2)
    ap.add_argument("--max-folds", type=int, default=50)
    args = ap.parse_args()
    main(args.start_fold, args.n_parallel, args.max_folds)
