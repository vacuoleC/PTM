"""Probe cuml elasticnet support and GPU fit speed on the real matrix."""
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
print(f"train={Xt.shape} dtype={Xt.dtype}", flush=True)

from cuml.linear_model import LogisticRegression as CumlLR

for C, l1r in [(0.1, 0.5), (1.0, 0.9), (0.01, 0.1), (1.0, 0.1)]:
    try:
        t0 = time.monotonic()
        m = CumlLR(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r, max_iter=10000, tol=1e-4, verbose=False)
        m.fit(Xt, y[train_idx])
        dt = time.monotonic() - t0
        p = m.predict_proba(Xv)[:, 1]
        print(f"cuml qn C={C} l1r={l1r} fit={dt:.2f}s proba[{p[0]:.4f}...] nz={np.count_nonzero(m.coef_)}", flush=True)
    except Exception as e:
        print(f"cuml qn C={C} l1r={l1r} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

print("done", flush=True)
