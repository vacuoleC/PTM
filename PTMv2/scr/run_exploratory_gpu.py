"""E4 exploratory model comparison (GPU cuml-qn, fixed-params).

Runs the two frozen exploratory models on the same 50 outer folds:
  - svd_calibrated_linear_svm: TruncatedSVD → LinearSVM + calibration
  - umap_elastic_net: UMAP → cuml-qn elastic net

Fixed params (from observation-selected strategy) to stay within 24h.
Exploratory results do not alter the primary conclusion (frozen rule).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def _prep_sklearn(X_train, X_test, threshold=0.1):
    """Detection filter + impute + scale (sklearn pipeline, NaN-safe)."""
    from preprocessing import make_preprocessing_pipeline

    prep = make_preprocessing_pipeline(threshold)
    return prep.fit_transform(X_train).astype(np.float32), prep.transform(X_test).astype(np.float32)


def _svm_svd(Xtr, ytr, Xte, n_comp=20, C=0.1):
    """TruncatedSVD → LinearSVM (cuml) + sigmoid calibration via AUPRC-friendly."""
    from cuml.svm import LinearSVC
    from cuml.decomposition import TruncatedSVD
    from cuml.linear_model import LogisticRegression  # calibration proxy

    svd = TruncatedSVD(n_components=n_comp)
    Xtr_l = svd.fit_transform(Xtr).astype(np.float32)
    Xte_l = svd.transform(Xte).astype(np.float32)
    # Calibrated SVM: fit SVM, then map decision values via logistic (calibration_cv=3 proxy)
    svm = LinearSVC(C=C)
    svm.fit(Xtr_l, ytr)
    dec = svm.decision_function(Xte_l).astype(np.float32).reshape(-1, 1)
    # Calibrate on training decision values
    dec_tr = svm.decision_function(Xtr_l).astype(np.float32).reshape(-1, 1)
    cal = LogisticRegression(penalty="l2", solver="qn", C=1.0)
    cal.fit(dec_tr, ytr)
    return cal.predict_proba(dec)[:, 1]


def _umap_en(Xtr, ytr, Xte, n_comp=10, n_neighbors=10, C=0.1, l1r=0.5):
    """UMAP → cuml-qn elastic net."""
    from cuml.manifold import UMAP
    from cuml.linear_model import LogisticRegression

    umap = UMAP(n_components=n_comp, n_neighbors=n_neighbors, min_dist=0.0, random_state=0)
    Xtr_l = umap.fit_transform(Xtr).astype(np.float32)
    Xte_l = umap.transform(Xte).astype(np.float32)
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r, max_iter=10000)
    m.fit(Xtr_l, ytr.astype(np.int32))
    return m.predict_proba(Xte_l)[:, 1]


def main(config_path: Path, model: str) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])

    print(f"E4 exploratory: {model}", flush=True)
    rows = []
    t_start = time.monotonic()
    for fold in sorted(assignments.fold.unique()):
        tr_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        te_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        Xtr, ytr = X.loc[tr_ids], labels.loc[tr_ids]
        Xte, yte = X.loc[te_ids], labels.loc[te_ids]
        Xtr_p, Xte_p = _prep_sklearn(Xtr, Xte)
        if model == "svd_calibrated_linear_svm":
            p = _svm_svd(Xtr_p, ytr.to_numpy(), Xte_p, n_comp=20, C=0.1)
        elif model == "umap_elastic_net":
            p = _umap_en(Xtr_p, ytr.to_numpy(), Xte_p, n_comp=10, n_neighbors=10, C=0.1, l1r=0.5)
        else:
            raise ValueError(f"unknown model {model}")
        ap = float(average_precision_score(yte, p))
        rows.append({"fold": fold, "model": model, "oof_ap": round(ap, 6)})
        print(f"fold {fold}: ap={ap:.4f} ({time.monotonic()-t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = root / "outputs/tables/exploratory_model_scores.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    print(f"mean oof_ap={df.oof_ap.mean():.4f} (baseline 0.4528)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    parser.add_argument("--model", choices=["svd_calibrated_linear_svm", "umap_elastic_net"], required=True)
    main(parser.parse_args().config, parser.parse_args().model)
