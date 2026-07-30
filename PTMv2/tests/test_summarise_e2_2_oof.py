import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from summarise_e2_2_oof import fold_summary, outer_splits_from_study, repeat_summary


class FoldSummaryTests(unittest.TestCase):
    def test_reports_one_auprc_per_fold_and_repeat(self):
        oof = pd.DataFrame(
            {
                "fold": [0, 0, 1, 1],
                "patient_id": ["A", "B", "C", "D"],
                "target": [0, 1, 0, 1],
                "score": [0.1, 0.9, 0.8, 0.2],
            }
        )
        result = fold_summary(oof, outer_splits=5)
        self.assertEqual(result["repeat"].tolist(), [0, 0])
        self.assertEqual(result["n_test"].tolist(), [2, 2])
        self.assertEqual(result["positive_test"].tolist(), [1, 1])
        self.assertEqual(result["average_precision"].tolist(), [1.0, 0.5])

    def test_reads_outer_splits_from_frozen_study_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "study_design.yaml").write_text(
                yaml.safe_dump({"evaluation": {"outer_cv": {"splits": 5}}}), encoding="utf-8"
            )
            self.assertEqual(
                outer_splits_from_study(root, {"paths": {"study_design": "study_design.yaml"}}), 5
            )

    def test_aggregates_one_auprc_per_repeat(self):
        oof = pd.DataFrame(
            {"fold": [0, 0, 5, 5], "target": [0, 1, 0, 1], "score": [0.1, 0.9, 0.8, 0.2]}
        )
        result = repeat_summary(oof, outer_splits=5)
        self.assertEqual(result["repeat"].tolist(), [0, 1])
        self.assertEqual(result["average_precision"].tolist(), [1.0, 0.5])
