"""Detailed per-fit timing for cuml qn: warmup vs steady-state, param dependence."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/PTM/PTMv2/scr")
from preprocessing import make_preprocessing_pipeline

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
X = X.loc[labels.index]
y = labels.to_numpy()

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_idx, valid_idx = next(iter(cv.split(X, y)))

prep = make_preprocessing_pipeline(0.1)
Xt = prep.fit_transform(X.iloc[train_idx]).astype(np.float32)
Xv = prep.transform(X.iloc[valid_idx]).astype(np.float32)
print(f"X shape {Xt.shape}", flush=True)

from cuml.linear_model import LogisticRegression

# warmup
m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
m.fit(Xt, y[train_idx].astype(np.int32))
print("warmup done", flush=True)

for i in range(5):
    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, y[train_idx].astype(np.int32))
    print(f"steady C=0.1 l1r=0.5 fit#{i}: {time.monotonic()-t0:.2f}s", flush=True)

for C, l1r in [(0.01, 0.1), (1.0, 0.1), (1.0, 0.9), (0.1, 0.9)]:
    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, y[train_idx].astype(np.int32))
    print(f"param C={C} l1r={l1r}: {time.monotonic()-t0:.2f}s iters={m.n_iter_ if hasattr(m,'n_iter_') else '?'}", flush=True)
