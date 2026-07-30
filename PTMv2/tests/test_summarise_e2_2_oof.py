import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from summarise_e2_2_oof import fold_summary


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
