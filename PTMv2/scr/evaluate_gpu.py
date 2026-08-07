"""GPU linear elastic-net (PyTorch) fold-fit primitive for PTMv2.

Equivalent objective to sklearn LogisticRegression(penalty="elasticnet",
solver="saga"):  BCE + (1/C) * [l1_ratio * sum|w| + (1-l1_ratio)/2 * sum(w^2)].
Trained with Adam on GPU (or CPU fallback). Used for the E3.1 permutation
null after the CPU/GPU consistency check passed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_score_fold_gpu(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    threshold: float, C: float, l1_ratio: float,
    *,
    steps: int = 2000, lr: float = 1e-3, seed: int = 0, device: str | None = None,
) -> np.ndarray:
    """Fit the elastic-net linear model on (X_train, y_train), predict proba on X_test.

    Mirrors evaluate.fit_score_fold but uses PyTorch Adam on GPU. The caller is
    responsible for preprocessing (detection filter/impute/scale) via the same
    training-fold-only pipeline.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32)
    X_te = np.asarray(X_test, dtype=np.float32)

    torch.manual_seed(seed)
    Xg = torch.tensor(np.ascontiguousarray(X_tr), device=device)
    yg = torch.tensor(y_tr, device=device)
    Xt = torch.tensor(np.ascontiguousarray(X_te), device=device)

    n_features = X_tr.shape[1]
    w = torch.zeros(n_features, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Shuffle epochs deterministically for stable convergence
    for _ in range(steps):
        opt.zero_grad()
        logits = Xg @ w + b
        loss = loss_fn(logits.squeeze(), yg) + (1.0 / C) * (
            l1_ratio * torch.abs(w).sum() + (1.0 - l1_ratio) * 0.5 * (w ** 2).sum()
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_te = (Xt @ w + b).squeeze().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits_te))
