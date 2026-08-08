"""Diagnostic: per-component timing for folds 0/1/2 + GPU utilization check.

Times preprocessing and cuml fit separately inside the full nested loop for
three outer folds, to localize the 24x difference between fold 0 and fold 1/2.
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
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
X = X.loc[labels.index]

from cuml.linear_model import LogisticRegression

def fit_cuml_timed(Xtr, ytr, Xte, threshold, C, l1r):
    t0 = time.monotonic()
    prep = make_preprocessing_pipeline(threshold)
    Xt = prep.fit_transform(Xtr).astype(np.float32)
    Xv = prep.transform(Xte).astype(np.float32)
    t_prep = time.monotonic() - t0
    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    t_fit = time.monotonic() - t0
    return m.predict_proba(Xv)[:, 1], t_prep, t_fit

for fold in [0, 1, 2]:
    tr_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
    te_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
    Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
    Xte = X.loc[te_ids]
    print(f"=== fold {fold} train={len(Xtr)} test={len(Xte)} ===", flush=True)
    t_start = time.monotonic()
    total_prep, total_fit = 0.0, 0.0
    for thr in [0.1, 0.3, 0.5]:
        for C in [0.01, 0.1, 1.0]:
            for l1r in [0.1, 0.5, 0.9]:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0 + fold)
                for ti, vi in cv.split(Xtr, ytr):
                    _, tp, tf = fit_cuml_timed(Xtr.iloc[ti], ytr.iloc[ti], Xtr.iloc[vi], thr, C, l1r)
                    total_prep += tp
                    total_fit += tf
    print(f"fold {fold}: TOTAL={time.monotonic()-t_start:.1f}s prep_sum={total_prep:.1f}s fit_sum={total_fit:.1f}s", flush=True)

print("done", flush=True)
