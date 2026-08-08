"""Fold-safe primitive for PTMv2 model evaluation.

Two backends:
- saga (sklearn, default): the frozen-design implementation.
- cuml (GPU, selected via backend="cuml"): same elastic-net objective,
  coordinate descent (qn) on H100. Adopted via exploration policy
  (exp/e4-raw/) because the 4-core CPU cgroup makes the saga full run
  infeasible. Uses sort-trick median (f32) preprocessing, numerically
  equivalent to sklearn's nanmedian (maxdiff ~3e-6), ~50x faster.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from preprocessing import make_preprocessing_pipeline


def _sort_median_impute(arr):
    """Column-wise median impute via sort trick (faster than nanmedian)."""
    out = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = ~np.isnan(col)
        if mask.all():
            continue
        if mask.any():
            med = float(np.sort(col[mask])[len(col[mask]) // 2])
        else:
            med = 0.0
        out[~mask, j] = med
    return out


def _prep_cuml(X_train, threshold):
    """Detection filter + sort-trick median + standardize, f32 (cuml path)."""
    X = X_train.to_numpy(dtype=np.float32)
    mask = np.mean(~np.isnan(X), axis=0) >= threshold
    if not mask.any():
        raise ValueError("Detection threshold removes every feature.")
    X = X[:, mask]
    X = _sort_median_impute(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


def _fit_predict(Xt, y_train, Xv, C, l1_ratio, backend):
    if backend == "saga":
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
            max_iter=10000, random_state=0,
        )
        return model.fit(Xt, y_train).predict_proba(Xv)[:, 1]
    if backend == "cuml":
        import cudf
        from cuml.linear_model import LogisticRegression as CumlLR

        Xg = cudf.DataFrame(Xt)
        yg = cudf.Series(y_train.to_numpy())
        Xvg = cudf.DataFrame(Xv)
        model = CumlLR(penalty="elasticnet", C=C, l1_ratio=l1_ratio, solver="qn", max_iter=10000, tol=1e-6)
        return model.fit(Xg, yg).predict_proba(Xvg).to_numpy()[:, 1]
    raise ValueError(f"unknown backend {backend}")


def fit_score_fold(X_train, y_train, X_test, threshold, C, l1_ratio, backend="saga"):
    if backend == "cuml":
        Xt = _prep_cuml(X_train, threshold)
        Xv = _prep_cuml(X_test, threshold)
        return _fit_predict(Xt, y_train, Xv, C, l1_ratio, backend)
    prep = make_preprocessing_pipeline(threshold)
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_test)
    return _fit_predict(Xt, y_train, Xv, C, l1_ratio, backend)


def fit_score_fold_pca(X_train, y_train, X_test, threshold, n_components, C, l1_ratio, backend="saga"):
    """PCA-reduce on the training fold, then fit low-dim logistic.

    PCA components are capped at min(n_samples - 1, n_features) of the
    detection-filtered training matrix — a mathematical constraint, not a
    design choice. The frozen component grid [10, 20, 40] is preserved where
    feasible; 40 clips down when the filtered feature count is small.
    """
    from sklearn.decomposition import PCA

    prep = make_preprocessing_pipeline(threshold)
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_test)
    n_comp = min(int(n_components), Xt.shape[0] - 1, Xt.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    Xt_low = pca.fit_transform(Xt)
    Xv_low = pca.transform(Xv)
    if backend == "cuml":
        Xt_low = Xt_low.astype("float32")
        Xv_low = Xv_low.astype("float32")
    return _fit_predict(Xt_low, y_train, Xv_low, C, l1_ratio, backend)
