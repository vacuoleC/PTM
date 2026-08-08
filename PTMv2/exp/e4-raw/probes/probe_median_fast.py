"""Test fast median replacements: sort-trick vs nanmedian vs cudf."""
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
n, p = arr64.shape
print(f"shape {arr64.shape}", flush=True)

# baseline
t0 = time.monotonic()
med_ref = np.nanmedian(arr64, axis=0)
print(f"nanmedian f64: {time.monotonic()-t0:.3f}s", flush=True)

# sort trick f64
t0 = time.monotonic()
s = np.sort(arr64, axis=0)  # NaN sorts to the end
nonnan = np.sum(~np.isnan(arr64), axis=0)
idx_hi = (nonnan - 1) // 2
idx_lo = nonnan // 2
med_sort = (s[idx_lo, np.arange(p)] + s[idx_hi, np.arange(p)]) / 2.0
print(f"sort-trick f64: {time.monotonic()-t0:.3f}s maxdiff={np.max(np.abs(med_sort - med_ref)):.2e}", flush=True)

# sort trick f32
t0 = time.monotonic()
s32 = np.sort(arr32, axis=0)
nonnan32 = np.sum(~np.isnan(arr32), axis=0)
med32 = (s32[(nonnan32 - 1) // 2, np.arange(p)] + s32[nonnan32 // 2, np.arange(p)]) / 2.0
print(f"sort-trick f32: {time.monotonic()-t0:.3f}s", flush=True)

# argpartition trick (potentially faster than full sort)
t0 = time.monotonic()
nonnan = np.sum(~np.isnan(arr64), axis=0)
idx_hi = (nonnan - 1) // 2
idx_lo = nonnan // 2
s1 = np.partition(arr64, np.stack([idx_lo, idx_hi]).T[:, 0], axis=0)[idx_lo, np.arange(p)]
s2 = np.partition(arr64, np.stack([idx_lo, idx_hi]).T[:, 1], axis=0)[idx_hi, np.arange(p)]
med_part = (s1 + s2) / 2.0
print(f"partition-trick f64: {time.monotonic()-t0:.3f}s maxdiff={np.max(np.abs(med_part - med_ref)):.2e}", flush=True)

# cudf availability
try:
    import cudf
    gdf = cudf.DataFrame.from_pandas(X.iloc[:84])
    t0 = time.monotonic()
    med_cudf = gdf.median()
    print(f"cudf median (incl conversion {time.monotonic()-t0:.3f}s): {med_cudf.values[:3]}", flush=True)
except Exception as e:
    print(f"cudf FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
print("done", flush=True)
