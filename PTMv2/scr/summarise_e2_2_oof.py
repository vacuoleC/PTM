"""Summarise validated fixed-parameter E2.2 OOF scores by frozen outer fold."""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from validate_e2_2_oof import validate_oof


def fold_summary(oof: pd.DataFrame, outer_splits: int) -> pd.DataFrame:
    """Return one AUPRC record per frozen outer fold, grouped into CV repeats."""
    records = []
    for fold, frame in oof.groupby("fold", sort=True):
        records.append(
            {
                "fold": fold,
                "repeat": fold // outer_splits,
                "n_test": len(frame),
                "positive_test": int(frame["target"].sum()),
                "average_precision": average_precision_score(frame["target"], frame["score"]),
            }
        )
    return pd.DataFrame(records)


def repeat_summary(oof: pd.DataFrame, outer_splits: int) -> pd.DataFrame:
    """Compute one AUPRC per repeat after aggregating its five test folds."""
    records = []
    for repeat, frame in oof.assign(repeat=oof["fold"] // outer_splits).groupby("repeat", sort=True):
        records.append(
            {
                "repeat": repeat,
                "n_patients": len(frame),
                "average_precision": average_precision_score(frame["target"], frame["score"]),
            }
        )
    return pd.DataFrame(records)


def outer_splits_from_study(root: Path, project_config: dict) -> int:
    """Read frozen outer-fold count from the study design, not runtime paths."""
    study_path = root / project_config["paths"]["study_design"]
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    return study["evaluation"]["outer_cv"]["splits"]


def main(config_path: Path) -> None:
    """Validate the configured OOF artifact then write its fold-level summary."""
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["e2_2_smoke"]
    assignments = pd.read_csv(root / settings["outer_assignments"])
    oof = pd.read_csv(root / settings["output"])
    validate_oof(oof, assignments)
    outer_splits = outer_splits_from_study(root, config)
    fold_output = root / settings["fold_summary"]
    repeat_output = root / settings["repeat_summary"]
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    fold_summary(oof, outer_splits).to_csv(fold_output, index=False)
    repeat_summary(oof, outer_splits).to_csv(repeat_output, index=False)
    print(f"E2.2 fold summary written: {fold_output}")
    print(f"E2.2 repeat summary written: {repeat_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml"
    )
    main(parser.parse_args().config)
