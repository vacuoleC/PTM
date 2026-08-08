import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from run_learning_curve import _select_parameters, _run_cell
from nested_pca_elasticnet import parameter_grid


def synthetic_data(n_patients=24, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"P{i}" for i in range(n_patients)]
    X = pd.DataFrame(rng.normal(size=(n_patients, n_features)), index=ids)
    y = pd.Series([0, 1] * (n_patients // 2), index=ids)
    rows = []
    for fold, test_ids in enumerate((ids[:6], ids[6:12], ids[12:18], ids[18:])):
        rows.extend({"fold": fold, "role": "test", "patient_id": p} for p in test_ids)
        rows.extend({"fold": fold, "role": "train", "patient_id": p} for p in ids if p not in test_ids)
    return X, y, pd.DataFrame(rows)


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.assignments = synthetic_data()
        self.candidates = parameter_grid([0.0], [3], [1.0], [0.5])

    def test_select_parameters_returns_candidate(self):
        best, score = _select_parameters(self.X, self.y, self.candidates, 2, 0)
        self.assertIn("n_components", str(best))
        self.assertGreaterEqual(score, 0.0)

    def test_cell_produces_auprc(self):
        from run_learning_curve import _WORKER_STATE
        global_state_holder = (self.X, self.y, self.assignments, self.candidates, 2, 0)
        # Simulate worker state by direct call with a small wrapper
        from run_learning_curve import _init_worker
        _init_worker(*global_state_holder)
        result = _run_cell((1.0, 0))
        self.assertIn("oof_auprc", result)
        self.assertEqual(result["fraction"], 1.0)
        self.assertGreaterEqual(result["oof_auprc"], 0.0)
        self.assertLessEqual(result["oof_auprc"], 1.0)


if __name__ == "__main__":
    unittest.main()
