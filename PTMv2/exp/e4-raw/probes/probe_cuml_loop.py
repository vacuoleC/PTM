"""Measure per-fit cuml cost inside a real nested loop: constructor overhead,
first fit, repeated fits, warm-start reuse."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/PTM/PTMv2/scr")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
X = X.loc[labels.index]

tr_ids = assignments.loc[(assignments.fold == 1) & (assignments.role == "train"), "patient_id"]
Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
ti, vi = next(iter(cv.split(Xtr, ytr)))
arr = np.asarray(Xtr.iloc[ti], dtype=np.float32)
frac = np.mean(~np.isnan(arr), axis=0)
arr_f = arr[:, frac >= 0.1][:, :10000]
Xa = arr_f
ytr_a = np.asarray(ytr.iloc[ti]).astype(np.int32)
print(f"subset {Xa.shape} nan={np.isnan(Xa).sum()}", flush=True)

from cuml.linear_model import LogisticRegression

# 1. constructor cost
t0 = time.monotonic()
for _ in range(20):
    LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
print(f"constructor x20: {time.monotonic()-t0:.3f}s ({ (time.monotonic()-t0)/20*1000:.1f}ms each)", flush=True)

# 2. fresh fit loop (like our inner loop)
t0 = time.monotonic()
for i in range(10):
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xa, ytr_a)
    t1 = time.monotonic()
    print(f"  fresh fit #{i}: {t1-t0:.3f}s (cumulative)", flush=True)
    t0 = t1

# 3. reuse same object (refit)
m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
t0 = time.monotonic()
for i in range(10):
    m.fit(Xa, ytr_a)
    print(f"  refit same obj #{i}: {time.monotonic()-t0:.3f}s", flush=True)
    t0 = time.monotonic()
print("done", flush=True)
