"""CPU (sklearn saga) vs GPU (PyTorch Adam) consistency check for E3.1.

Runs the fixed-parameter raw Elastic Net on the same outer folds with both
implementations and compares per-fold AUPRC and predicted-probability
Spearman correlation. Passing thresholds: |AUPRC diff| <= 0.01 and
Spearman >= 0.95.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scr"))
from evaluate import fit_score_fold
from preprocessing import make_preprocessing_pipeline


def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    threshold, C, l1_ratio = 0.1, 0.1, 0.5  # frozen fixed parameters

    rows = []
    for fold in sorted(assignments.fold.unique())[:5]:  # 5 folds for the check
        train_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        test_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        X_tr, y_tr = X.loc[train_ids], labels.loc[train_ids]
        X_te, y_te = X.loc[test_ids], labels.loc[test_ids]

        prep = make_preprocessing_pipeline(threshold)
        X_tr_p = prep.fit_transform(X_tr).astype(np.float32)
        X_te_p = prep.transform(X_te).astype(np.float32)

        p_cpu = fit_score_fold(X_tr, y_tr, X_te, threshold, C, l1_ratio)
        p_gpu = fit_score_fold_cuml(X_tr_p, y_tr.to_numpy(), X_te_p, threshold, C, l1_ratio)

        auprc_cpu = average_precision_score(y_te, p_cpu)
        auprc_gpu = average_precision_score(y_te, p_gpu)
        rho, _ = spearmanr(p_cpu, p_gpu)
        rows.append({
            "fold": fold, "n_test": len(y_te),
            "auprc_cpu": round(float(auprc_cpu), 6), "auprc_gpu": round(float(auprc_gpu), 6),
            "auprc_diff": round(float(abs(auprc_cpu - auprc_gpu)), 6),
            "spearman": round(float(rho), 6),
        })
        print(
            f"fold {fold}: cpu={auprc_cpu:.4f} gpu={auprc_gpu:.4f} "
            f"diff={abs(auprc_cpu-auprc_gpu):.4f} spearman={rho:.4f}", flush=True,
        )

    df = pd.DataFrame(rows)
    out_dir = root / "outputs/tables"
    df.to_csv(out_dir / "e3_1_consistency_check.csv", index=False)
    max_diff = df.auprc_diff.max()
    min_rho = df.spearman.min()
    passed = max_diff <= 0.01 and min_rho >= 0.95
    print(f"max_auprc_diff={max_diff:.6f} min_spearman={min_rho:.6f} PASSED={passed}", flush=True)
    (root / "outputs/reports" / "e3_1_consistency_check.md").write_text(
        f"# E3.1 CPU/GPU Consistency Check\n\n"
        f"- max AUPRC diff: {max_diff:.6f} (threshold 0.01)\n"
        f"- min Spearman: {min_rho:.6f} (threshold 0.95)\n"
        f"- result: {'PASS' if passed else 'FAIL'}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    main(parser.parse_args().config)
