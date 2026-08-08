"""E3.2 learning curve runner (GPU cuml-qn, fixed-params per fold).

For each training_fraction in [0.5, 0.7, 1.0] × 10 repeats:
  - Subsample patients in each training fold at the fraction (stratified,
    repeat-specific seed).
  - Fit with the observation-selected params for that fold (cuml-qn GPU),
    predict OOF on the held-out test fold.
  - Record pooled OOF AUPRC per (fraction, repeat).

Each fraction retains stratified inner validation semantics via the frozen
patient-level split; preprocessing (detection filter / sort-trick median /
standardize) is label-free and cached across repeats.
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


def _prep_fit(X, threshold):
    """Fit preprocessing on training data: returns transformed X + stats for test."""
    a = X.to_numpy(dtype=np.float32)
    keep = np.mean(~np.isnan(a), axis=0) >= threshold
    a = a[:, keep]
    s = np.sort(a, axis=0)
    nn = np.sum(~np.isnan(a), axis=0)
    med = (s[nn // 2, np.arange(a.shape[1])] + s[(nn - 1) // 2, np.arange(a.shape[1])]) / 2.0
    imp = np.where(np.isnan(a), med, a)
    mu = imp.mean(axis=0)
    sd = imp.std(axis=0) + 1e-8
    return (imp - mu) / sd, (keep, med, mu, sd)


def _prep_apply(X, stats):
    """Apply training-fit stats to test data (same feature mask)."""
    keep, med, mu, sd = stats
    a = X.to_numpy(dtype=np.float32)[:, keep]
    imp = np.where(np.isnan(a), med, a)
    return (imp - mu) / sd


def _fit_cuml(Xt, yt, Xv, C, l1r):
    import cudf
    from cuml.linear_model import LogisticRegression

    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4)
    m.fit(cudf.DataFrame(Xt), cudf.Series(yt.astype(np.int32)))
    return m.predict_proba(cudf.DataFrame(Xv)).to_numpy()[:, 1]


def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    # Observation-selected params per fold (E3.1 fixed-param adoption)
    sel = pd.read_csv(root / "exp/e4-raw/outputs/e4raw_full_selected_params.csv")
    sel_by_fold = {(r.fold, r.threshold, r.C, r.l1_ratio) for r in sel.itertuples()}

    fractions = study["evaluation"]["learning_curve"]["training_fractions"]
    repeats = study["evaluation"]["learning_curve"]["repeats_per_fraction"]
    random_seed = study["evaluation"]["random_seed"]

    print(f"learning curve: fractions={fractions} repeats={repeats}", flush=True)
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
                # Use observation-selected params for this fold
                row = sel[sel.fold == fold]
                if len(row) == 0:
                    raise RuntimeError(f"no selected params for fold {fold}")
                thr, C_val, l1r = float(row.iloc[0].threshold), float(row.iloc[0].C), float(row.iloc[0].l1_ratio)
                Xt, stats = _prep_fit(X_frac, thr)
                Xv = _prep_apply(X.loc[te_ids], stats)
                p = _fit_cuml(Xt, y_frac.to_numpy(), Xv, C_val, l1r)
                records.extend({"fold": fold, "patient_id": pid, "target": int(labels.loc[pid]), "score": s}
                               for pid, s in zip(te_ids, p))
            oof = pd.DataFrame(records)
            auprc = float(average_precision_score(oof.target, oof.score))
            rows.append({"fraction": fraction, "repeat": repeat, "n_train": len(keep), "oof_auprc": round(auprc, 6)})
            print(f"frac={fraction} rep={repeat}: auprc={auprc:.4f} ({time.monotonic()-t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["fraction", "repeat"])
    out = root / "outputs/tables/primary_model_learning_curve.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)
    print(df.groupby("fraction")["oof_auprc"].agg(["mean", "std"]).round(6).to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    main(parser.parse_args().config)
