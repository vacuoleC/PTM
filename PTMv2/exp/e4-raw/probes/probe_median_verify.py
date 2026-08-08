"""Verify sort-trick median correctness and validate imputed+scaled output
against sklearn SimpleImputer+StandardScaler on the same data.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/PTM/PTMv2/scr")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = "/data/PTM/PTMv2"
X = pd.read_pickle(f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz")
labels = pd.read_csv(f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv").set_index("patient_id")["target"]
X = X.loc[labels.index]
arr = np.asarray(X.iloc[:84], dtype=np.float32)
n, p = arr.shape

# reference: sklearn pipeline (after detection filter, matching frozen order)
from preprocessing import make_preprocessing_pipeline
prep = make_preprocessing_pipeline(0.1)
prep.fit(X.iloc[:84])
mask = prep.named_steps["detection_filter"].support_mask_
ref_imp = prep.named_steps["median_imputer"].transform(np.asarray(X.iloc[:84], dtype=float)[:, mask])
print(f"sklearn imputed (post-filter) shape {ref_imp.shape} median[0:3]={ref_imp[0, :3]}", flush=True)

arrf = arr[:, mask]
# sort-trick on filtered f32
t0 = time.monotonic()
s = np.sort(arrf, axis=0)
nonnan = np.sum(~np.isnan(arrf), axis=0)
idx_lo = nonnan // 2
idx_hi = (nonnan - 1) // 2
med = (s[idx_lo, np.arange(arrf.shape[1])] + s[idx_hi, np.arange(arrf.shape[1])]) / 2.0
print(f"sort-trick median (filtered): {time.monotonic()-t0:.3f}s", flush=True)

# compare medians directly (all filtered cols have >= 10% non-nan)
ref_med = prep.named_steps["median_imputer"].statistics_
diff = np.max(np.abs(med - ref_med))
print(f"median maxdiff vs sklearn: {diff:.2e}", flush=True)
if diff > 1e-5:
    bad = np.where(np.abs(med - ref_med) > 1e-5)[0][:10]
    print(f"  mismatch cols: {bad}", flush=True)
    for c in bad[:3]:
        print(f"  col {c}: mine={med[c]:.6f} sklearn={ref_med[c]:.6f} nonnan={nonnan[c]}", flush=True)

# impute + scale equivalence on filtered cols
t0 = time.monotonic()
imp_mine = np.where(np.isnan(arrf), med, arrf)
print(f"fillna: {time.monotonic()-t0:.3f}s", flush=True)
maxdiff_imp = np.max(np.abs(imp_mine - ref_imp))
print(f"imputed maxdiff vs sklearn: {maxdiff_imp:.2e}", flush=True)

scaler = StandardScaler().fit(ref_imp)
ref_scaled = scaler.transform(ref_imp)
mu = ref_imp.mean(axis=0); sd = ref_imp.std(axis=0)
mine_scaled = (ref_imp - mu) / sd
print(f"scaled maxdiff: {np.max(np.abs(mine_scaled - ref_scaled)):.2e}", flush=True)

# timing full custom pipeline: detection + median + fill + scale (f32)
t0 = time.monotonic()
frac = np.mean(~np.isnan(arr), axis=0)
keep = frac >= 0.1
a2 = arr[:, keep]
s2 = np.sort(a2, axis=0)
nn2 = np.sum(~np.isnan(a2), axis=0)
med2 = (s2[nn2 // 2, np.arange(a2.shape[1])] + s2[(nn2 - 1) // 2, np.arange(a2.shape[1])]) / 2.0
imp2 = np.where(np.isnan(a2), med2, a2)
mu2 = imp2.mean(axis=0); sd2 = imp2.std(axis=0)
out = (imp2 - mu2) / sd2
print(f"full custom prep (f32): {time.monotonic()-t0:.3f}s -> {out.shape}", flush=True)

# compare vs sklearn full pipeline
ref_out = prep.transform(X.iloc[:84])
print(f"sklearn full prep: shape {ref_out.shape}", flush=True)
print(f"mask match: {np.array_equal(mask, prep.named_steps['detection_filter'].support_mask_)}", flush=True)
print(f"final output maxdiff: {np.max(np.abs(out - ref_out.astype(np.float32))) if out.shape == ref_out.shape else 'shape mismatch'}", flush=True)
print("done", flush=True)
