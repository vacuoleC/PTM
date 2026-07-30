import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from nested_raw_elasticnet import nested_oof, parameter_grid


class NestedRawTests(unittest.TestCase):
    def test_nested_oof_returns_only_predeclared_test_patients(self):
        rng = np.random.default_rng(0)
        ids = [f"P{i}" for i in range(12)]
        X = pd.DataFrame(rng.normal(size=(12, 4)), index=ids)
        y = pd.Series([0, 1] * 6, index=ids)
        rows = []
        for fold, test_ids in enumerate((ids[:4], ids[4:8], ids[8:])):
            rows.extend({"fold": fold, "role": "test", "patient_id": patient} for patient in test_ids)
            rows.extend({"fold": fold, "role": "train", "patient_id": patient} for patient in ids if patient not in test_ids)
        oof, selected = nested_oof(X, y, pd.DataFrame(rows), parameter_grid([0.0], [1.0], [0.5]), 2, 0)
        self.assertEqual(set(oof.patient_id), set(ids))
        self.assertEqual(len(oof), 12)
        self.assertEqual(len(selected), 3)
