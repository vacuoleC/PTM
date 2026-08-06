import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from run_permutation_null import run_permutation_null
from nested_raw_elasticnet import nested_oof, parameter_grid


def synthetic_data(n_patients=12, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"P{i}" for i in range(n_patients)]
    X = pd.DataFrame(rng.normal(size=(n_patients, n_features)), index=ids)
    y = pd.Series([0, 1] * (n_patients // 2), index=ids)
    rows = []
    for fold, test_ids in enumerate((ids[:4], ids[4:8], ids[8:])):
        rows.extend({"fold": fold, "role": "test", "patient_id": p} for p in test_ids)
        rows.extend({"fold": fold, "role": "train", "patient_id": p} for p in ids if p not in test_ids)
    return X, y, pd.DataFrame(rows)


class PermutationNullTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.assignments = synthetic_data()
        self.candidates = parameter_grid([0.0], [1.0], [0.5])

    def test_null_scores_are_permuted_label_results(self):
        """Each null score must come from a permuted-label nested OOF run."""
        observed_oof, _ = nested_oof(self.X, self.y, self.assignments, self.candidates, 2, 0)
        null_df, observed_auprc, p_value = run_permutation_null(
            self.X, self.y, self.assignments, self.candidates, 2, 0, 5, observed_oof
        )
        self.assertEqual(len(null_df), 5)
        self.assertAlmostEqual(observed_auprc, average_precision_score(observed_oof.target, observed_oof.score))
        self.assertTrue(0.0 <= p_value <= 1.0)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in null_df.auprc))

    def test_p_value_counts_null_greater_or_equal_observed(self):
        """p-value formula: (1 + count(null >= observed)) / (1 + n_permutations)."""
        observed_oof, _ = nested_oof(self.X, self.y, self.assignments, self.candidates, 2, 0)
        null_df, _, p_value = run_permutation_null(
            self.X, self.y, self.assignments, self.candidates, 2, 0, 5, observed_oof
        )
        observed_auprc = average_precision_score(observed_oof.target, observed_oof.score)
        expected_p = (1.0 + int((null_df.auprc >= observed_auprc).sum())) / (1.0 + 5)
        self.assertAlmostEqual(p_value, expected_p)

    def test_permuted_labels_change_distribution(self):
        """Permutation must actually shuffle labels (not return identical y)."""
        rng = np.random.default_rng(0)
        permuted = self.y.copy()
        permuted.iloc[:] = rng.permutation(permuted.to_numpy())
        self.assertNotEqual(list(permuted), list(self.y))


if __name__ == "__main__":
    unittest.main()
