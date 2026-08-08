"""E4.1 exploration probe: measure raw saga fit cost per detection threshold.

Reads frozen inputs, builds the frozen preprocessing pipeline, and times
single LogisticRegression(saga, elasticnet) fits on the detection-filtered
matrices for thresholds 0.1/0.3/0.5. Also reports saga iteration counts to
understand what dominates the 133s wall time.
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/PTM/PTMv2/scr")
from preprocessing import make_preprocessing_pipeline

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
X = X.loc[labels.index]
y = labels.to_numpy()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, valid_idx = next(iter(cv.split(X, y)))

print(f"host_nproc={os.cpu_count()}", flush=True)
for thr in [0.1, 0.3, 0.5]:
    prep = make_preprocessing_pipeline(thr)
    t0 = time.monotonic()
    Xt = prep.fit_transform(X.iloc[train_idx])
    Xv = prep.transform(X.iloc[valid_idx])
    tprep = time.monotonic() - t0
    print(f"threshold={thr} filtered_train={Xt.shape} filtered_valid={Xv.shape} prep={tprep:.2f}s", flush=True)

    for C, l1r in [(0.1, 0.5), (1.0, 0.9), (0.01, 0.1), (1.0, 0.1), (0.1, 0.9)]:
        t0 = time.monotonic()
        model = LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1r,
                                   max_iter=10000, random_state=0, tol=1e-4)
        model.fit(Xt, y[train_idx])
        dt = time.monotonic() - t0
        print(f"  thr={thr} C={C} l1r={l1r} fit={dt:.1f}s iters={model.n_iter_[0]}", flush=True)

print("done", flush=True)
