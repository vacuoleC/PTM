"""Numerical consistency: cuml-qn vs sklearn-saga on the same fold/params.

Uses e2_2 smoke fixed parameters (threshold 0.1, C=0.1, l1_ratio=0.5) on
outer fold 0 and compares test predictions side by side with the recorded
sklearn-saga scores.
"""
from __future__ import annotations

import os
import sys

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
te_ids = assignments.loc[(assignments.fold == 0) & (assignments.role == "test"), "patient_id"]
Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
Xte = X.loc[te_ids]

prep = make_preprocessing_pipeline(0.1)
Xt = prep.fit_transform(Xtr).astype(np.float32)
Xv = prep.transform(Xte).astype(np.float32)

from cuml.linear_model import LogisticRegression
m = LogisticRegression(penalty="elasticnet", solver="qn", C=0.1, l1_ratio=0.5,
                       max_iter=10000, tol=1e-4, verbose=False)
m.fit(Xt, np.asarray(ytr).astype(np.int32))
p_cuml = m.predict_proba(Xv)[:, 1]

smoke = pd.read_csv(f"{ROOT}/outputs/tables/e2_2_fixed_parameter_oof_smoke.csv")
smoke0 = smoke[smoke.fold == 0].set_index("patient_id").loc[te_ids]
p_saga = smoke0["score"].to_numpy()
yte_v = labels.loc[te_ids].to_numpy()
df = pd.DataFrame({"patient_id": te_ids, "target": yte_v,
                   "sklearn_saga": p_saga, "cuml_qn": p_cuml})
df["abs_diff"] = (df.sklearn_saga - df.cuml_qn).abs()
print(df.to_string(index=False, float_format="%.4f"))
print(f"\nmax_abs_diff={df.abs_diff.max():.4f} mean_abs_diff={df.abs_diff.mean():.4f} "
      f"corr={np.corrcoef(df.sklearn_saga, df.cuml_qn)[0, 1]:.6f}")
