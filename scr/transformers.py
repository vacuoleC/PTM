import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DetectionFilter(BaseEstimator, TransformerMixin):
    """只保留训练集中检测率达到阈值的特征"""

    def __init__(self, min_detection=0.5):
        self.min_detection = min_detection

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.keep_ = (
            frame.notna().mean(axis=0) >= self.min_detection
        ).to_numpy()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        return frame.loc[:, self.keep_].to_numpy()


class MedianImputer(BaseEstimator, TransformerMixin):
    """用训练集每列的中位数填补缺失值。"""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(values, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        missing_rows, missing_cols = np.where(np.isnan(values))
        values[missing_rows, missing_cols] = self.median_[missing_cols]
        return values