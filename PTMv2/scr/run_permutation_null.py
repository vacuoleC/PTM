"""Permutation null runner for the frozen raw Elastic Net primary model.

Each permutation rebuilds the labels by shuffling target values across patients,
then reruns the complete nested pipeline: training-fold detection filter,
imputation, scaling, inner candidate selection, and outer-fold OOF prediction.
No learned state leaks between permutations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from nested_raw_elasticnet import nested_oof, parameter_grid


def run_permutation_null(
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    candidates: list[tuple],
    inner_splits: int,
    random_seed: int,
    n_permutations: int,
    observed_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Return a table of null AUPRC per permutation plus the observed AUPRC."""
    rng = np.random.default_rng(random_seed)
    null_scores = []
    for perm_index in range(n_permutations):
        permuted = y.copy()
        permuted.iloc[:] = rng.permutation(permuted.to_numpy())
        permuted.index = y.index
        oof, _ = nested_oof(X, permuted, assignments, candidates, inner_splits, random_seed + perm_index)
        null_scores.append(
            {"permutation": perm_index, "auprc": average_precision_score(oof.target, oof.score)}
        )
        print(f"permutation {perm_index + 1}/{n_permutations} done", flush=True)
    null_df = pd.DataFrame(null_scores)
    observed_auprc = average_precision_score(observed_oof.target, observed_oof.score)
    p_value = (1.0 + int((null_df.auprc >= observed_auprc).sum())) / (1.0 + n_permutations)
    return null_df, observed_auprc, p_value


def main(config_path: Path, n_permutations: int) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    raw = next(model for model in study["models"]["primary"] if model["id"] == "raw_elastic_net")
    candidates = parameter_grid(
        study["preprocessing"]["detection_threshold_candidates"],
        raw["classifier"]["C"], raw["classifier"]["l1_ratio"],
    )
    observed_oof, _ = nested_oof(
        X, labels, assignments, candidates,
        study["evaluation"]["inner_cv"]["splits"], study["evaluation"]["random_seed"],
    )
    null_df, observed_auprc, p_value = run_permutation_null(
        X, labels, assignments, candidates,
        study["evaluation"]["inner_cv"]["splits"], study["evaluation"]["random_seed"],
        n_permutations, observed_oof,
    )
    out_dir = root / "outputs/tables"
    null_df.to_csv(out_dir / "e3_1_permutation_null_smoke.csv", index=False)
    print(f"observed_auprc={observed_auprc:.6f} p_value={p_value:.6f}", flush=True)
    print(f"wrote {out_dir / 'e3_1_permutation_null_smoke.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    parser.add_argument("--n-permutations", type=int, default=500)
    main(parser.parse_args().config, parser.parse_args().n_permutations)
