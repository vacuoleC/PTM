import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scr"))
from evaluate_gpu import fit_score_fold_gpu


class GpuEvaluateTests(unittest.TestCase):
    def test_shape_and_range(self):
        rng = np.random.default_rng(0)
        X_tr = rng.normal(size=(40, 20)).astype(np.float32)
        y_tr = np.array([0, 1] * 20, dtype=np.float32)
        X_te = rng.normal(size=(10, 20)).astype(np.float32)
        p = fit_score_fold_gpu(X_tr, y_tr, X_te, 0.1, 0.1, 0.5, steps=50, device="cpu")
        self.assertEqual(p.shape, (10,))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))

    def test_separates_classes(self):
        """A separable synthetic problem should produce higher proba for class 1."""
        rng = np.random.default_rng(1)
        X_pos = rng.normal(loc=2.0, size=(30, 8)).astype(np.float32)
        X_neg = rng.normal(loc=-2.0, size=(30, 8)).astype(np.float32)
        X_tr = np.vstack([X_pos, X_neg])
        y_tr = np.array([1] * 30 + [0] * 30, dtype=np.float32)
        X_te = np.vstack([rng.normal(loc=2.0, size=(5, 8)), rng.normal(loc=-2.0, size=(5, 8))]).astype(np.float32)
        p = fit_score_fold_gpu(X_tr, y_tr, X_te, 0.1, 0.1, 0.5, steps=300, device="cpu")
        self.assertGreater(float(np.mean(p[:5])), float(np.mean(p[5:])))

    def test_deterministic_seed(self):
        rng = np.random.default_rng(2)
        X_tr = rng.normal(size=(30, 10)).astype(np.float32)
        y_tr = np.array([0, 1] * 15, dtype=np.float32)
        X_te = rng.normal(size=(8, 10)).astype(np.float32)
        p1 = fit_score_fold_gpu(X_tr, y_tr, X_te, 0.1, 0.1, 0.5, steps=100, seed=42, device="cpu")
        p2 = fit_score_fold_gpu(X_tr, y_tr, X_te, 0.1, 0.1, 0.5, steps=100, seed=42, device="cpu")
        np.testing.assert_allclose(p1, p2, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
