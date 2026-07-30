"""Leakage-safe preprocessing components for PTMv2 sklearn pipelines."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DetectionRateFilter(BaseEstimator, TransformerMixin):
    """Keep features whose nonmissing fraction in the training fold meets threshold."""
    def __init__(self, threshold: float = 0.1): self.threshold = threshold
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.support_mask_ = np.mean(~np.isnan(X), axis=0) >= self.threshold
        if not self.support_mask_.any(): raise ValueError("Detection threshold removes every feature.")
        return self
    def transform(self, X):
        if not hasattr(self, "support_mask_"): raise RuntimeError("Fit DetectionRateFilter before transform.")
        return np.asarray(X, dtype=float)[:, self.support_mask_]
