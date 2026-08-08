import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from nested_pca_elasticnet import nested_oof, parameter_grid


def synthetic_data(n_patients=12, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"P{i}" for i in range(n_patients)]
    X = pd.DataFrame(rng.normal(size=(n_patients, n_features)), index=ids)
    y = pd.Series([0, 1] * (n_patients // 2), index=ids)
    rows = []
    for fold, test_ids in enumerate((ids[:4], ids[4:8], ids[8:])):
        rows.extend({"fold": fold, "role": "test", "patient_id": p} for p in test_ids)
        rows.extend({"fold": fold, "role": "train", "patient_id": p} for p in ids if p not in test_ids)
    return X, y, pd.DataFrame(rows)


class PcaNestedTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.assignments = synthetic_data()
        self.candidates = parameter_grid([0.0], [2], [1.0], [0.5])

    def test_parameter_grid_shape(self):
        grid = parameter_grid([0.1, 0.3], [10, 20], [0.1, 1.0], [0.5])
        self.assertEqual(len(grid), 2 * 2 * 2 * 1)

    def test_nested_oof_covers_all_test_patients(self):
        oof, selected = nested_oof(self.X, self.y, self.assignments, self.candidates, 2, 0)
        self.assertEqual(set(oof.patient_id), set(self.X.index))
        self.assertEqual(len(oof), 12)
        self.assertEqual(len(selected), 3)
        self.assertIn("n_components", selected.columns)


if __name__ == "__main__":
    unittest.main()
