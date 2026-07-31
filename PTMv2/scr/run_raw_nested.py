"""Run one complete raw Elastic Net nested-CV evaluation from frozen PTMv2 inputs."""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from nested_raw_elasticnet import nested_oof, parameter_grid


def main(config_path: Path) -> None:
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
    oof, selected = nested_oof(
        X, labels, assignments, candidates, study["evaluation"]["inner_cv"]["splits"], study["evaluation"]["random_seed"]
    )
    output = root / "outputs/tables/e3_1_raw_nested_observed_oof.csv"
    params = root / "outputs/tables/e3_1_raw_nested_selected_parameters.csv"
    oof.to_csv(output, index=False)
    selected.to_csv(params, index=False)
    print(f"wrote {output}", flush=True)
    print(f"wrote {params}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "config" / "project.yaml")
    main(parser.parse_args().config)
