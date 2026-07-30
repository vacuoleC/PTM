import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from validate_e2_2_oof import validate_oof


def frozen_assignments():
    return pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "role": ["train", "train", "test", "test"],
            "patient_id": ["A", "B", "C", "D"],
            "target": [0, 1, 0, 1],
        }
    )


class ValidateOofTests(unittest.TestCase):
    def test_accepts_exact_finite_probabilities(self):
        oof = pd.DataFrame(
            {"fold": [0, 0], "patient_id": ["C", "D"], "target": [0, 1], "score": [0.2, 0.8]}
        )
        self.assertEqual(
            validate_oof(oof, frozen_assignments()),
            {"records": 2, "folds": 1, "unique_patients": 2, "missing_scores": 0},
        )

    def test_rejects_incomplete_coverage(self):
        oof = pd.DataFrame({"fold": [0], "patient_id": ["C"], "target": [0], "score": [0.2]})
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_oof(oof, frozen_assignments())

    def test_rejects_non_probability_score(self):
        oof = pd.DataFrame(
            {"fold": [0, 0], "patient_id": ["C", "D"], "target": [0, 1], "score": [0.2, 1.1]}
        )
        with self.assertRaisesRegex(ValueError, "\\[0, 1\\]"):
            validate_oof(oof, frozen_assignments())


if __name__ == "__main__":
    unittest.main()
