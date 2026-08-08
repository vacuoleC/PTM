"""Decompose preprocessing cost: detection / median impute / scaler, float64 vs float32."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
X = X.loc[labels.index]

arr64 = np.asarray(X.iloc[:84], dtype=float)
arr32 = arr64.astype(np.float32)
print(f"arr64 {arr64.shape} {arr64.dtype}, arr32 {arr32.shape} {arr32.dtype}", flush=True)

# 1. detection rate
for name, a in [("f64", arr64), ("f32", arr32)]:
    t0 = time.monotonic()
    mask = np.mean(~np.isnan(a), axis=0) >= 0.1
    print(f"detection [{name}]: {time.monotonic()-t0:.3f}s keep={mask.sum()}", flush=True)

# 2. median impute
for name, a in [("f64", arr64), ("f32", arr32)]:
    t0 = time.monotonic()
    med = np.nanmedian(a, axis=0)
    print(f"nanmedian [{name}]: {time.monotonic()-t0:.3f}s", flush=True)
    t0 = time.monotonic()
    filled = np.where(np.isnan(a), med, a)
    print(f"fillna [{name}]: {time.monotonic()-t0:.3f}s", flush=True)

# 3. scaler (mean/std)
for name, a in [("f64", arr64), ("f32", arr32)]:
    t0 = time.monotonic()
    mu = np.mean(a, axis=0); sd = np.std(a, axis=0)
    print(f"mean/std [{name}]: {time.monotonic()-t0:.3f}s", flush=True)
    t0 = time.monotonic()
    _ = (a - mu) / sd
    print(f"apply scale [{name}]: {time.monotonic()-t0:.3f}s", flush=True)

# 4. sklearn pipeline on f32
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
for name, a in [("f64", arr64), ("f32", arr32)]:
    t0 = time.monotonic()
    pipe = Pipeline([("median", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    out = pipe.fit_transform(a)
    print(f"sklearn pipe [{name}]: {time.monotonic()-t0:.3f}s -> {out.shape}", flush=True)

# 5. SimpleImputer alone
for name, a in [("f64", arr64), ("f32", arr32)]:
    t0 = time.monotonic()
    imp = SimpleImputer(strategy="median").fit(a)
    out = imp.transform(a)
    print(f"SimpleImputer fit+transform [{name}]: {time.monotonic()-t0:.3f}s", flush=True)
print("done", flush=True)
