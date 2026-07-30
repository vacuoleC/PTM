"""Validate the fixed-parameter E2.2 out-of-fold prediction artifact."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


KEYS = ["fold", "patient_id", "target"]


def validate_oof(oof: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, int]:
    """Verify that OOF predictions exactly cover the predeclared test assignments."""
    required = set(KEYS + ["score"])
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {sorted(missing)}")

    expected = assignments.loc[assignments["role"].eq("test"), KEYS].copy()
    observed = oof[KEYS].copy()
    if expected.duplicated().any() or observed.duplicated().any():
        raise ValueError("Each fold/patient/target OOF record must appear exactly once.")
    if not expected.sort_values(KEYS).reset_index(drop=True).equals(
        observed.sort_values(KEYS).reset_index(drop=True)
    ):
        raise ValueError("OOF fold, patient, or target coverage differs from frozen test assignments.")

    scores = pd.to_numeric(oof["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("OOF scores must all be finite and non-missing.")
    if not scores.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("OOF scores must lie in [0, 1].")

    return {
        "records": len(oof),
        "folds": oof["fold"].nunique(),
        "unique_patients": oof["patient_id"].nunique(),
        "missing_scores": int(scores.isna().sum()),
    }


def main(config_path: Path) -> None:
    """Load configuration-controlled artifacts and print their validation summary."""
    config_path = config_path.resolve()
    root = config_path.parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["e2_2_smoke"]
    assignments = pd.read_csv(root / settings["outer_assignments"])
    oof = pd.read_csv(root / settings["output"])
    summary = validate_oof(oof, assignments)
    print("E2.2 OOF validation passed")
    for name, value in summary.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml"
    )
    main(parser.parse_args().config)
