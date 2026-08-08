"""Paired comparison of fixed-param vs nested permutation nulls (E3.1 exploration).

Reads the two checkpoint arms (same perm indices, same fold subset 0-9) and
reports:
  - null distribution moments/shape for both arms
  - paired differences (Wilcoxon signed-rank, mean diff, correlation)
  - KS test on pooled samples
  - p-values vs the observed pooled AP on the shared fold subset
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

OBS_FOLD_AP = None  # filled from e4raw oof

def load_null(path):
    df = pd.read_csv(path)
    return df.set_index("permutation")["auprc"].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--nested", required=True)
    ap.add_argument("--observed-ap", type=float, default=None)
    ap.add_argument("--folds", default="0-9")
    args = ap.parse_args()

    fixed = load_null(args.fixed)
    nested = load_null(args.nested)
    common = fixed.index.intersection(nested.index)
    f = fixed.loc[common].to_numpy()
    n = nested.loc[common].to_numpy()

    print(f"paired perms: {len(common)}  folds: {args.folds}")
    print(f"fixed : n={len(f)} mean={f.mean():.6f} std={f.std(ddof=1):.6f} "
          f"min={f.min():.6f} max={f.max():.6f} median={np.median(f):.6f}")
    print(f"nested: n={len(n)} mean={n.mean():.6f} std={n.std(ddof=1):.6f} "
          f"min={n.min():.6f} max={n.max():.6f} median={np.median(n):.6f}")
    d = n - f
    print(f"nested - fixed: mean diff={d.mean():.6f} median diff={np.median(d):.6f} "
          f"corr={np.corrcoef(f, n)[0, 1]:.4f}")
    try:
        w = stats.wilcoxon(d)
        print(f"wilcoxon signed-rank: stat={w.statistic:.1f} p={w.pvalue:.4f}")
    except ValueError as e:
        print(f"wilcoxon failed: {e}")
    ks = stats.ks_2samp(f, n)
    print(f"ks_2samp: stat={ks.statistic:.4f} p={ks.pvalue:.4f}")

    if args.observed_ap is not None:
        obs = args.observed_ap
        p_f = (1 + int((f >= obs).sum())) / (1 + len(f))
        p_n = (1 + int((n >= obs).sum())) / (1 + len(n))
        print(f"observed pooled AP = {obs:.6f}")
        print(f"p(fixed)  = {p_f:.4f}  ({int((f >= obs).sum())}/{len(f)} nulls >= obs)")
        print(f"p(nested) = {p_n:.4f}  ({int((n >= obs).sum())}/{len(n)} nulls >= obs)")
        # quantile of observed within each null
        q_f = (f < obs).mean()
        q_n = (n < obs).mean()
        print(f"observed quantile: fixed={q_f:.4f} nested={q_n:.4f}")
    # permutation-level table
    out = pd.DataFrame({"fixed": f, "nested": n, "diff": d}, index=common)
    print(out.to_string())


if __name__ == "__main__":
    main()
