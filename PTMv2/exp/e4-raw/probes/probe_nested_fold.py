"""E4.1 exploration: full nested fold with cuml qn (GPU) — single fold speed test.

Runs the complete inner selection (27 candidates x 3 inner folds) + outer fit
for ONE outer fold using cuml qn on GPU, mirroring the frozen raw pipeline
(detection filter -> median impute -> standard scale) but with the solver
swapped from saga to cuml-qn. Reports wall time and the selected parameters.
"""
from __future__ import annotations

import argparse
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
from evaluate import fit_score_fold  # sklearn saga reference

ROOT = "/data/PTM/PTMv2"


def fit_cuml(Xtr, ytr, Xte, threshold, C, l1r):
    from cuml.linear_model import LogisticRegression
    prep = make_preprocessing_pipeline(threshold)
    Xt = prep.fit_transform(Xtr).astype(np.float32)
    Xv = prep.transform(Xte).astype(np.float32)
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    return m.predict_proba(Xv)[:, 1]


def main(fold: int, use_sklearn: bool) -> None:
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
    inner = 3
    seed = 0 + int(fold)

    t_start = time.monotonic()
    cv = StratifiedKFold(n_splits=inner, shuffle=True, random_state=seed)
    results = []
    for thr in thresholds:
        for C in Cs:
            for l1r in l1rs:
                fs = []
                for ti, vi in cv.split(Xtr, ytr):
                    t0 = time.monotonic()
                    if use_sklearn:
                        p = fit_score_fold(Xtr.iloc[ti], ytr.iloc[ti], Xtr.iloc[vi], thr, C, l1r)
                    else:
                        p = fit_cuml(Xtr.iloc[ti], ytr.iloc[ti], Xtr.iloc[vi], thr, C, l1r)
                    fs.append(average_precision_score(ytr.iloc[vi], p))
                    if (thr, C, l1r) == (0.1, 0.1, 0.5) and len(fs) == 1:
                        print(f"[timing] first inner fit {time.monotonic() - t0:.1f}s", flush=True)
                results.append(((thr, C, l1r), float(np.mean(fs))))
                print(f"[inner] thr={thr} C={C} l1r={l1r} mean_ap={results[-1][1]:.4f} "
                      f"({time.monotonic() - t_start:.0f}s elapsed)", flush=True)

    best = max(results, key=lambda r: r[1])
    print(f"selected={best[0]} inner_ap={best[1]:.4f}", flush=True)

    t0 = time.monotonic()
    if use_sklearn:
        p = fit_score_fold(Xtr, ytr, Xte, *best[0])
    else:
        p = fit_cuml(Xtr, ytr, Xte, *best[0])
    oof_ap = average_precision_score(yte, p)
    print(f"outer fold {fold} oof_ap={oof_ap:.4f} outer_fit={time.monotonic() - t0:.1f}s", flush=True)
    print(f"TOTAL={time.monotonic() - t_start:.1f}s fold={fold} solver={'sklearn-saga' if use_sklearn else 'cuml-qn'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sklearn", action="store_true", help="use sklearn saga instead of cuml qn")
    args = ap.parse_args()
    main(args.fold, args.sklearn)
