"""Control experiment: preprocessing vs cuml fit vs full fit_cuml, repeated."""
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
Xtr, ytr = X.iloc[train_idx], y[train_idx]

# 1. preprocessing alone, repeated
for i in range(3):
    t0 = time.monotonic()
    prep = make_preprocessing_pipeline(0.1)
    Xt = prep.fit_transform(Xtr)
    dt = time.monotonic() - t0
    print(f"prep-only #{i}: {dt:.2f}s -> {Xt.shape}", flush=True)

# 2. pandas iloc subset cost
t0 = time.monotonic()
sub = Xtr.iloc[:56]
print(f"iloc subset 56 rows: {time.monotonic()-t0:.3f}s", flush=True)

# 3. cuml fit on prepared matrix, repeated (warmup first)
from cuml.linear_model import LogisticRegression
Xt32 = Xt.astype(np.float32)
m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
m.fit(Xt32, ytr[:84].astype(np.int32))
for i in range(3):
    t0 = time.monotonic()
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt32, ytr[:84].astype(np.int32))
    print(f"cuml-fit-only #{i}: {time.monotonic()-t0:.2f}s", flush=True)

# 4. full fit_cuml-equivalent (prep + iloc + fit), repeated
def full(thr, C, l1r):
    prep = make_preprocessing_pipeline(thr)
    Xa = prep.fit_transform(Xtr)
    Xa = Xa.astype(np.float32)
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xa, ytr.astype(np.int32))

for i in range(3):
    t0 = time.monotonic()
    full(0.1, 0.1, 0.5)
    print(f"full-flow #{i}: {time.monotonic()-t0:.2f}s", flush=True)

print("done", flush=True)
