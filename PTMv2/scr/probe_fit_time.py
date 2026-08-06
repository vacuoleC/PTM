"""Remote timing probe: measure one saga fit on the real high-dim matrix.

Prints wall-clock time for a single LogisticRegression(saga, elasticnet) fit
on the frozen detection-filtered training folds, which is the unit of work
inside the nested pipeline. Used to calibrate the E3.1 permutation null
runtime before launching the full 500-permutation job.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scr"))
from preprocessing import make_preprocessing_pipeline


def main(config_path: Path) -> None:
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    X = pd.read_pickle(root / config["paths"]["source_matrix"])
    labels = pd.read_csv(root / config["paths"]["source_labels"]).set_index("patient_id")["target"]
    # Frozen raw-model candidates: threshold 0.1, C=0.1, l1_ratio=0.5 (median candidate)
    prep = make_preprocessing_pipeline(0.1)
    Xt = prep.fit_transform(X.loc[labels.index])
    y = labels.to_numpy()
    print(f"preprocessed shape: {Xt.shape}", flush=True)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=10000, random_state=0)
    start = time.monotonic()
    model.fit(Xt, y)
    fit_seconds = time.monotonic() - start
    print(f"single saga fit: {fit_seconds:.1f}s on shape {Xt.shape}", flush=True)
    print(f"n_jobs={__import__('os').cpu_count()}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    main(parser.parse_args().config)
