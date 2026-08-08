"""Decompose the fast-path inner loop: stats compute vs apply vs cuml fit."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/PTM/PTMv2/scr")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
X = X.loc[labels.index]

tr_ids = assignments.loc[(assignments.fold == 1) & (assignments.role == "train"), "patient_id"]
Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
splits = list(cv.split(Xtr, ytr))

STATS = {}


def prep_stats(df, threshold):
    arr = np.asarray(df, dtype=np.float32)
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


def prep_apply(df, stats):
    keep, med, mu, sd = stats
    arr = np.asarray(df, dtype=np.float32)
    imp = np.where(np.isnan(arr[:, keep]), med, arr[:, keep])
    return (imp - mu) / sd


def prep_apply_direct(arr, stats):
    keep, med, mu, sd = stats
    imp = np.where(np.isnan(arr[:, keep]), med, arr[:, keep])
    return (imp - mu) / sd


# Pre-convert all inner subsets to f32 arrays once
subs = []
for ti, vi in splits:
    arr_tr = np.asarray(Xtr.iloc[ti], dtype=np.float32)
    arr_va = np.asarray(Xtr.iloc[vi], dtype=np.float32)
    subs.append((arr_tr, arr_va, np.asarray(ytr.iloc[ti]).astype(np.int32), np.asarray(ytr.iloc[vi]).astype(np.int32)))

# 1. stats computation for 3 thresholds x 3 inner folds
t0 = time.monotonic()
for ti, vi in splits:
    arr = np.asarray(Xtr.iloc[ti], dtype=np.float32)
    for thr in [0.1, 0.3, 0.5]:
        s = prep_stats(arr, thr)
print(f"9x prep_stats (incl asarray): {time.monotonic()-t0:.2f}s", flush=True)

# 2. apply on pre-converted arrays, repeated (like inner loop)
t0 = time.monotonic()
for thr in [0.1, 0.3, 0.5]:
    for _ in range(27):
        for arr_tr, arr_va, _, _ in subs:
            st = prep_stats(arr_tr, thr)
            Xt = prep_apply_direct(arr_tr, st)
            Xv = prep_apply_direct(arr_va, st)
print(f"81x prep_apply_direct (with 9x stats): {time.monotonic()-t0:.2f}s", flush=True)

# 3. cuml fit on prepped arrays (true GPU cost)
from cuml.linear_model import LogisticRegression
arr_tr = subs[0][0]
st = prep_stats(arr_tr, 0.1)
Xt = prep_apply_direct(arr_tr, st)
m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
m.fit(Xt, subs[0][2])
t0 = time.monotonic()
for i in range(10):
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, subs[0][2])
print(f"10x cuml fit on prepped: {time.monotonic()-t0:.2f}s ({ (time.monotonic()-t0)/10:.2f}s each)", flush=True)

# 4. full loop as probe_fast does it (with asarray each call) for 3 candidates
t0 = time.monotonic()
for thr in [0.1, 0.3]:
    for _ in range(5):
        for idx, (arr_tr, arr_va, yt, yv) in enumerate(subs):
            key = (id(arr_tr), thr)
            if key not in STATS:
                STATS[key] = prep_stats(arr_tr, thr)
            Xt = prep_apply_direct(arr_tr, STATS[key])
            Xv = prep_apply_direct(arr_va, STATS[key])
            m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5, max_iter=10000, tol=1e-4, verbose=False)
            m.fit(Xt, yt)
print(f"30x cached loop iterations: {time.monotonic()-t0:.2f}s", flush=True)
print("done", flush=True)
