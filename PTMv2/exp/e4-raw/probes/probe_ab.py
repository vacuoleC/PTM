"""Minimal A/B: preprocessing with actual e2_2 fold-0 indices vs StratifiedKFold-derived.

Hypothesis: fold 0's 24.8s came from a warm/cached state (cuml fit fast after warmup).
Measures per-component times inside the real fold-0 nested loop for the first
several candidates to see where seconds go.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
sys.path.insert(0, "/data/PTM/PTMv2/scr")
from preprocessing import make_preprocessing_pipeline

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
assignments = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv")
X = X.loc[labels.index]

tr_ids = assignments.loc[(assignments.fold == 0) & (assignments.role == "train"), "patient_id"]
Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
print(f"fold0 train rows: {len(Xtr)} cols: {Xtr.shape[1]}", flush=True)

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
splits = list(cv.split(Xtr, ytr))
print(f"inner split sizes: {[(len(t), len(v)) for t, v in splits]}", flush=True)

for ti, vi in splits[:2]:
    t0 = time.monotonic()
    prep = make_preprocessing_pipeline(0.1)
    Xt = prep.fit_transform(Xtr.iloc[ti])
    t1 = time.monotonic()
    Xv = prep.transform(Xtr.iloc[vi])
    t2 = time.monotonic()
    print(f"inner-fit: {t1-t0:.2f}s inner-transform: {t2-t1:.2f}s -> {Xt.shape}/{Xv.shape}", flush=True)

# nunique + NaN stats on the real matrix (cost of detection filter)
t0 = time.monotonic()
arr = np.asarray(Xtr, dtype=float)
frac = np.mean(~np.isnan(arr), axis=0)
t1 = time.monotonic()
print(f"nan-frac computation on 84x91692: {t1-t0:.2f}s", flush=True)
for thr in [0.1, 0.3, 0.5]:
    print(f"  threshold {thr}: keep {int((frac >= thr).sum())} features", flush=True)
print("done", flush=True)
