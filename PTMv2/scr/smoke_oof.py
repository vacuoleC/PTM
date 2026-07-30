"""Run the configuration-controlled fixed-parameter E2.2 OOF smoke check."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import fit_score_fold
from oof import initialise_oof, record_fold_scores


def timestamp() -> str:
    """Return an ISO timestamp for long-task log monitoring."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(config_path: Path) -> None:
    """Score all frozen outer test folds without fitting on their test patients."""
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    settings = config["e2_2_smoke"]
    interval = config["monitoring"]["long_task_progress_interval"]

    X = pd.read_pickle(root / paths["source_matrix"])
    y = pd.read_csv(root / paths["source_labels"]).set_index("patient_id")
    assignments = pd.read_csv(root / settings["outer_assignments"])
    oof = initialise_oof(assignments)
    folds = sorted(oof.fold.unique())
    print(f"{timestamp()} [E2.2] start: {len(folds)} frozen outer folds", flush=True)

    for ordinal, fold in enumerate(folds, start=1):
        train_ids = assignments.loc[
            (assignments.fold == fold) & (assignments.role == "train"), "patient_id"
        ]
        test_ids = assignments.loc[
            (assignments.fold == fold) & (assignments.role == "test"), "patient_id"
        ]
        scores = fit_score_fold(
            X.loc[train_ids],
            y.loc[train_ids, "target"],
            X.loc[test_ids],
            settings["detection_threshold"],
            settings["classifier_C"],
            settings["classifier_l1_ratio"],
        )
        oof = record_fold_scores(oof, fold, test_ids, scores)
        if ordinal % interval == 0 or ordinal == len(folds):
            print(f"{timestamp()} [E2.2] completed fold {ordinal}/{len(folds)}", flush=True)

    output = root / settings["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output, index=False)
    print(f"{timestamp()} [E2.2] complete: wrote {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml"
    )
    main(parser.parse_args().config)
