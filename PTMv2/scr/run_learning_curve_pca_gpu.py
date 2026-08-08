"""E3.2 learning curve (pca_elastic_net arm) — GPU cuml-qn, fast path.

Dual-track with run_learning_curve_gpu.py (raw arm). Uses the frozen
pca_elastic_net primary model: PCA(20) on training fold → low-dim
cuml-qn logistic. Per-fold params fixed from observation-selected
(e4raw_full_selected_params.csv is raw-arm; here we use PCA20 fixed
settings from the E3.1 pca exploration).

For each training_fraction [0.5, 0.7, 1.0] × 10 repeats:
  - Stratified patient subsample of each training fold at the fraction.
  - PCA(20) fit on subsample → transform test fold (no leakage).
  - cuml-qn fit on PCA space → OOF AUPRC.
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

from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def _fit_cuml(Xt, yt, Xv, C, l1r):
    import cudf
    from cuml.linear_model import LogisticRegression

    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4)
    m.fit(cudf.DataFrame(Xt), cudf.Series(yt.astype(np.int32)))
    return m.predict_proba(cudf.DataFrame(Xv)).to_numpy()[:, 1]


def _prep_pca(X_train, X_test, n_comp=20, seed=0):
    """sklearn pipeline (handles NaN correctly) → PCA(n_comp) fit on train."""
    from preprocessing import make_preprocessing_pipeline

    prep = make_preprocessing_pipeline(0.1)
    a_tr = prep.fit_transform(X_train).astype(np.float32)
    a_te = prep.transform(X_test).astype(np.float32)
    pca = PCA(n_components=min(n_comp, a_tr.shape[0] - 1, a_tr.shape[1]), random_state=seed)
    return pca.fit_transform(a_tr).astype(np.float32), pca.transform(a_te).astype(np.float32)


def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])

    fractions = study["evaluation"]["learning_curve"]["training_fractions"]
    repeats = study["evaluation"]["learning_curve"]["repeats_per_fraction"]
    random_seed = study["evaluation"]["random_seed"]
    # Fixed params for PCA arm: from E3.1 pca nested observed selection (median combo)
    C_fixed, l1r_fixed = 0.1, 0.5

    print(f"learning curve (pca arm): fractions={fractions} repeats={repeats}", flush=True)
    rows = []
    t_start = time.monotonic()
    for fraction in fractions:
        for repeat in range(repeats):
            rng = np.random.default_rng(random_seed + repeat * 131 + int(fraction * 100))
            records = []
            for fold in sorted(assignments.fold.unique()):
                tr_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
                te_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
                y_tr = labels.loc[tr_ids]
                pos = y_tr[y_tr == 1].index.to_numpy()
                neg = y_tr[y_tr == 0].index.to_numpy()
                n_pos = max(1, int(len(pos) * fraction))
                n_neg = max(1, int(len(neg) * fraction))
                keep = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                                       rng.choice(neg, size=n_neg, replace=False)])
                X_frac = X.loc[keep]
                y_frac = labels.loc[keep]
                Xt, Xv = _prep_pca(X_frac, X.loc[te_ids])
                p = _fit_cuml(Xt, y_frac.to_numpy(), Xv, C_fixed, l1r_fixed)
                records.extend({"fold": fold, "patient_id": pid, "target": int(labels.loc[pid]), "score": s}
                               for pid, s in zip(te_ids, p))
            oof = pd.DataFrame(records)
            auprc = float(average_precision_score(oof.target, oof.score))
            rows.append({"fraction": fraction, "repeat": repeat, "n_train": len(keep),
                         "oof_auprc": round(auprc, 6), "arm": "pca"})
            print(f"pca frac={fraction} rep={repeat}: auprc={auprc:.4f} ({time.monotonic()-t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["fraction", "repeat"])
    out = root / "outputs/tables/primary_model_learning_curve_pca.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    print(df.groupby("fraction")["oof_auprc"].agg(["mean", "std"]).round(6).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    main(parser.parse_args().config)
