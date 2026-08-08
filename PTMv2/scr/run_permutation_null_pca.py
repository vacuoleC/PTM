"""Permutation null runner for the frozen pca_elastic_net primary model.

Each permutation rebuilds the labels by shuffling target values across patients,
then reruns the complete nested pipeline: training-fold detection filter,
imputation, scaling, training-fold PCA, inner candidate selection, and
outer-fold OOF prediction. No learned state leaks between permutations.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from nested_pca_elasticnet import nested_oof, parameter_grid

_WORKER_STATE: tuple | None = None


def _init_worker(X, y, assignments, candidates, inner_splits, random_seed):
    global _WORKER_STATE
    _WORKER_STATE = (X, y, assignments, candidates, inner_splits, random_seed)


def _run_one(perm_index: int) -> float:
    # multiprocessing fork inherits the parent's initialized OpenBLAS thread
    # pool (128 threads per worker), causing oversubscription. Pin each worker
    # to one BLAS thread at runtime — threadpoolctl takes effect after fork,
    # unlike the pre-fork OPENBLAS_NUM_THREADS env var.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        X, y, assignments, candidates, inner_splits, random_seed = _WORKER_STATE
        permuted = y.copy()
        rng = np.random.default_rng(random_seed + perm_index)
        permuted.iloc[:] = rng.permutation(permuted.to_numpy())
        permuted.index = y.index
        oof, _ = nested_oof(X, permuted, assignments, candidates, inner_splits, random_seed + perm_index)
        return float(average_precision_score(oof.target, oof.score))


def _load_checkpoint(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if "permutation" not in header:
            return done
        for line in fh:
            parts = line.strip().split(",")
            if parts and parts[0].isdigit():
                done.add(int(parts[0]))
    return done


def _append_checkpoint(path: Path, perm_index: int, auprc: float) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{perm_index},{auprc:.9f}\n")


def run_permutation_null(
    X, y, assignments, candidates, inner_splits, random_seed,
    n_permutations, observed_oof, *, n_jobs=1, checkpoint=None, max_hours=None,
):
    if checkpoint is not None and checkpoint.exists():
        with checkpoint.open("r", encoding="utf-8") as fh:
            first_line = fh.readline()
        if "permutation" not in first_line:
            checkpoint.write_text("permutation,auprc\n", encoding="utf-8")
    elif checkpoint is not None:
        checkpoint.write_text("permutation,auprc\n", encoding="utf-8")

    done = _load_checkpoint(checkpoint)
    start = time.monotonic()
    observed_auprc = average_precision_score(observed_oof.target, observed_oof.score)
    remaining = [i for i in range(n_permutations) if i not in done]
    print(
        f"permutations: {len(done)} done, {len(remaining)} remaining, "
        f"n_jobs={n_jobs}, max_hours={max_hours}, observed_auprc={observed_auprc:.6f}",
        flush=True,
    )
    if not remaining:
        print("all permutations already present in checkpoint; nothing to do", flush=True)
        null_df = pd.read_csv(checkpoint)
        p_value = (1.0 + int((null_df.auprc >= observed_auprc).sum())) / (1.0 + n_permutations)
        return null_df, observed_auprc, p_value

    payload = (X, y, assignments, candidates, inner_splits, random_seed)
    _init_worker(*payload)
    null_scores: list[dict] = []
    if n_jobs <= 1 or len(remaining) == 1:
        for perm_index in remaining:
            auprc = _run_one(perm_index)
            null_scores.append({"permutation": perm_index, "auprc": auprc})
            if checkpoint is not None:
                _append_checkpoint(checkpoint, perm_index, auprc)
            print(f"permutation {perm_index + 1}/{n_permutations} done ({time.monotonic() - start:.0f}s)", flush=True)
            if max_hours is not None and (time.monotonic() - start) / 3600 >= max_hours:
                print("max-hours reached; stopping gracefully", flush=True)
                break
    else:
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=payload) as pool:
            for perm_index, auprc in zip(remaining, pool.imap_unordered(_run_one, remaining, chunksize=1)):
                null_scores.append({"permutation": perm_index, "auprc": auprc})
                if checkpoint is not None:
                    _append_checkpoint(checkpoint, perm_index, auprc)
                print(f"permutation {perm_index + 1}/{n_permutations} done ({time.monotonic() - start:.0f}s)", flush=True)
                if max_hours is not None and (time.monotonic() - start) / 3600 >= max_hours:
                    print("max-hours reached; stopping gracefully", flush=True)
                    pool.terminate()
                    pool.join()
                    break

    completed_df = pd.read_csv(checkpoint) if checkpoint is not None and checkpoint.exists() else pd.DataFrame(null_scores)
    p_value = (1.0 + int((completed_df.auprc >= observed_auprc).sum())) / (1.0 + n_permutations)
    return completed_df, observed_auprc, p_value


def main(config_path: Path, n_permutations: int, n_jobs: int, checkpoint: Path | None, max_hours: float | None) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = yaml.safe_load((root / config["paths"]["study_design"]).read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    assignments = pd.read_csv(root / config["e2_2_smoke"]["outer_assignments"])
    pca_model = next(model for model in study["models"]["primary"] if model["id"] == "pca_elastic_net")
    candidates = parameter_grid(
        study["preprocessing"]["detection_threshold_candidates"],
        pca_model["pca_components"],
        pca_model["classifier"]["C"], pca_model["classifier"]["l1_ratio"],
    )
    print(f"running observed nested OOF (serial, pca_elastic_net)...", flush=True)
    observed_oof, _ = nested_oof(
        X, labels, assignments, candidates,
        study["evaluation"]["inner_cv"]["splits"], study["evaluation"]["random_seed"],
    )
    print("observed done; starting permutations", flush=True)
    null_df, observed_auprc, p_value = run_permutation_null(
        X, labels, assignments, candidates,
        study["evaluation"]["inner_cv"]["splits"], study["evaluation"]["random_seed"],
        n_permutations, observed_oof,
        n_jobs=n_jobs, checkpoint=checkpoint, max_hours=max_hours,
    )
    print(f"completed {len(null_df)}/{n_permutations} permutations", flush=True)
    print(f"observed_auprc={observed_auprc:.6f} p_value={p_value:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()
    main(args.config, args.n_permutations, args.n_jobs, args.checkpoint, args.max_hours)
