"""E3.1 fixed-param permutation exploration runner (GPU cuml-qn).

Modes:
  fixed   : per-fold params from the observed run (e4raw_full_selected_params.csv),
            NO inner selection. Statistic = pooled OOF AP over the fold subset,
            matching the frozen E3.1 statistic definition.
  nested  : full nested per fold (27 candidates x 3 inner folds + outer fit),
            same permutation indices as frozen E3.1
            (complete_primary_pipeline_including_inner_selection).
  observed: real labels, fixed params only; per-fold + pooled AP, validated
            against e4raw_full_oof_scores.csv (must reproduce fold 0 = 0.5450).

Permutation scheme (identical to scr/run_permutation_null.py):
  rng = np.random.default_rng(0 + perm_index); permuted.iloc[:] = rng.permutation(y)
  (shuffles all 106 labels; per-fold labels gathered by patient position).

Usage:
  python probe_fixed_perm.py --mode observed --folds 0-49
  python probe_fixed_perm.py --mode fixed --n-permutations 50 --folds 0-49 \
      --checkpoint out_fixed.csv --detail out_detail.csv
  python probe_fixed_perm.py --mode nested --n-permutations 15 --folds 0-9 \
      --checkpoint out_nested.csv --detail out_detail.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = "/data/PTM/PTMv2"
X_PATH = f"{ROOT}/../PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz"
LABELS_PATH = f"{ROOT}/../PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv"
ASSIGN_PATH = f"{ROOT}/outputs/tables/e2_2_outer_split_assignments.csv"
SEL_PATH = f"{ROOT}/exp/e4-raw/outputs/e4raw_full_selected_params.csv"
OBS_OOF_PATH = f"{ROOT}/exp/e4-raw/outputs/e4raw_full_oof_scores.csv"

THRESHOLDS = [0.1, 0.3, 0.5]
CS = [0.01, 0.1, 1.0]
L1RS = [0.1, 0.5, 0.9]


def prep_stats(arr, threshold):
    """Detection filter + sort-trick median stats (identical to e4raw_run_full.py)."""
    frac = np.mean(~np.isnan(arr), axis=0)
    keep = frac >= threshold
    a = arr[:, keep]
    s = np.sort(a, axis=0)
    nn = np.sum(~np.isnan(a), axis=0)
    med = (s[nn // 2, np.arange(a.shape[1])] + s[(nn - 1) // 2, np.arange(a.shape[1])]) / 2.0
    imp = np.where(np.isnan(a), med, a)
    mu = imp.mean(axis=0)
    sd = imp.std(axis=0)
    return keep, med, mu, sd


def prep_apply(arr, stats):
    keep, med, mu, sd = stats
    imp = np.where(np.isnan(arr[:, keep]), med, arr[:, keep])
    return (imp - mu) / sd


def fit_predict(Xt, ytr, Xv, C, l1r):
    from cuml.linear_model import LogisticRegression
    m = LogisticRegression(penalty="elasticnet", solver="qn", C=C, l1_ratio=l1r,
                           max_iter=10000, tol=1e-4, verbose=False)
    m.fit(Xt, np.asarray(ytr).astype(np.int32))
    return m.predict_proba(Xv)[:, 1]


def inner_arrays_fold(Xtr_np, ytr_np, fold):
    """Stratified 3-fold inner splits; returns list of (train_idx, valid_idx)."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0 + int(fold))
    return list(cv.split(Xtr_np, ytr_np))


def load_data():
    X = pd.read_pickle(X_PATH)
    labels = pd.read_csv(LABELS_PATH).set_index("patient_id")["target"]
    X = X.loc[labels.index]
    assignments = pd.read_csv(ASSIGN_PATH)
    X_np = np.asarray(X, dtype=np.float32)
    y_np = labels.to_numpy().astype(np.int32)
    pos = {pid: i for i, pid in enumerate(labels.index)}
    fold_meta = {}
    for fold in sorted(assignments.fold.unique()):
        tr_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), "patient_id"]
        te_ids = assignments.loc[(assignments.fold == fold) & (assignments.role == "test"), "patient_id"]
        fold_meta[fold] = ([pos[p] for p in tr_ids], [pos[p] for p in te_ids])
    return X_np, y_np, fold_meta


def run_fixed_fold(X_np, y_perm, fold_meta, fold, fixed_params, array_cache):
    """Fixed-param outer fold. Returns (score array for test patients, seconds)."""
    tr_idx, te_idx = fold_meta[fold]
    thr, C, l1r = fixed_params[fold]
    key = ("outer", fold, thr)
    if key not in array_cache:
        arr_tr = X_np[tr_idx]
        st = prep_stats(arr_tr, thr)
        Xt = prep_apply(arr_tr, st)
        Xv = prep_apply(X_np[te_idx], st)
        array_cache[key] = (Xt, Xv)
    Xt, Xv = array_cache[key]
    t0 = time.monotonic()
    p = fit_predict(Xt, y_perm[tr_idx], Xv, C, l1r)
    return p, time.monotonic() - t0


def run_nested_fold(X_np, y_perm, fold_meta, fold, array_cache, inner_arrays, perm_index):
    """Full nested fold (frozen semantics). Returns (score array, seconds, selected)."""
    tr_idx, te_idx = fold_meta[fold]
    ytr = y_perm[tr_idx]
    # inner selection: 27 candidates x 3 inner folds; inner cv seeded 0+perm_index+fold
    # (cache keyed by (perm_index, fold) — the splits depend on both!)
    splits = inner_arrays.get((perm_index, fold)) if inner_arrays is not None else None
    if splits is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0 + perm_index + int(fold))
        splits = list(cv.split(X_np[tr_idx], ytr))
        if inner_arrays is not None:
            inner_arrays[(perm_index, fold)] = splits
    results = []
    for thr in THRESHOLDS:
        for C in CS:
            for l1r in L1RS:
                fs = []
                for ti, vi in splits:
                    key = ("inner", fold, thr, tuple(ti), tuple(vi))
                    if key not in array_cache:
                        arr_tr = X_np[tr_idx][ti]
                        st = prep_stats(arr_tr, thr)
                        Xt = prep_apply(arr_tr, st)
                        Xv = prep_apply(X_np[tr_idx][vi], st)
                        array_cache[key] = (Xt, Xv)
                    Xt, Xv = array_cache[key]
                    p = fit_predict(Xt, ytr[ti], Xv, C, l1r)
                    fs.append(average_precision_score(ytr[vi], p))
                results.append(((thr, C, l1r), float(np.mean(fs))))
    best = max(results, key=lambda r: r[1])
    thr, C, l1r = best[0]
    arr_tr = X_np[tr_idx]
    st = prep_stats(arr_tr, thr)
    Xt = prep_apply(arr_tr, st)
    Xv = prep_apply(X_np[te_idx], st)
    t0 = time.monotonic()
    p = fit_predict(Xt, ytr, Xv, C, l1r)
    return p, time.monotonic() - t0, {"fold": fold, "threshold": thr, "C": C,
                                       "l1_ratio": l1r, "inner_ap": best[1]}


def pooled_ap_from_scores(all_p, all_y):
    return float(average_precision_score(np.concatenate(all_y), np.concatenate(all_p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "nested", "observed"], default="fixed")
    ap.add_argument("--n-permutations", type=int, default=50)
    ap.add_argument("--start-perm", type=int, default=0)
    ap.add_argument("--folds", type=str, default="0-49", help="e.g. 0-49 or 0-9")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--detail", type=str, default=None)
    ap.add_argument("--sel-log", type=str, default=None,
                    help="append selected params per fold (nested mode)")
    ap.add_argument("--nested", action="store_true",
                    help="observed mode: also run full nested path (perm_index=0 "
                         "=> e4raw semantics) for validation")
    args = ap.parse_args()

    if "-" in args.folds:
        a, b = args.folds.split("-")
        folds = list(range(int(a), int(b) + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]
    folds = sorted(folds)

    X_np, y_np, fold_meta = load_data()
    fixed_params = pd.read_csv(SEL_PATH).set_index("fold")
    fixed_params = {
        int(f): (float(fixed_params.loc[f, "threshold"]),
                 float(fixed_params.loc[f, "C"]),
                 float(fixed_params.loc[f, "l1_ratio"]))
        for f in folds if f in fixed_params.index
    }

    if args.mode == "observed":
        print("== observed validation (fixed params, real labels) ==", flush=True)
        cache = {}
        all_p, all_y, recs = [], [], []
        for fold in folds:
            p, dt = run_fixed_fold(X_np, y_np, fold_meta, fold, fixed_params, cache)
            te_idx = fold_meta[fold][1]
            all_p.append(p); all_y.append(y_np[te_idx])
            recs.append({"fold": fold, "ap": average_precision_score(y_np[te_idx], p), "seconds": round(dt, 2)})
        df = pd.DataFrame(recs)
        print(df.to_string(index=False), flush=True)
        print(f"pooled AP over {len(folds)} folds = {pooled_ap_from_scores(all_p, all_y):.6f}", flush=True)
        # compare against e4raw observed per-fold APs
        obs = pd.read_csv(OBS_OOF_PATH)
        obs_ap_fixed = float(average_precision_score(obs[obs.fold.isin(folds)].target,
                                                     obs[obs.fold.isin(folds)].score))
        for fold in folds:
            sub = obs[obs.fold == fold]
            ref = average_precision_score(sub.target, sub.score)
            got = recs[[r["fold"] for r in recs].index(fold)]["ap"]
            print(f"fold {fold}: rerun={got:.6f} e4raw={ref:.6f} diff={got - ref:+.2e}", flush=True)
        if args.nested:
            print("== nested path validation (perm_index=0, e4raw semantics) ==", flush=True)
            cache2 = {}
            inner_arrays = {}
            sel_recs = []
            n_all_p, n_all_y = [], []
            for fold in folds:
                p, dt, sel = run_nested_fold(X_np, y_np, fold_meta, fold, cache2, inner_arrays, 0)
                te_idx = fold_meta[fold][1]
                n_all_p.append(p); n_all_y.append(y_np[te_idx])
                sel_recs.append(sel)
                print(f"fold {fold}: nested_ap={average_precision_score(y_np[te_idx], p):.6f} "
                      f"sel=({sel['threshold']},{sel['C']},{sel['l1_ratio']}) inner_ap={sel['inner_ap']:.4f} "
                      f"took={dt:.1f}s", flush=True)
            nsel = pd.DataFrame(sel_recs).set_index("fold")
            ref_sel = pd.read_csv(SEL_PATH).set_index("fold")
            nsel = nsel.join(ref_sel, lsuffix="_rerun", rsuffix="_e4raw")
            nsel["thr_match"] = nsel["threshold_rerun"] == nsel["threshold_e4raw"]
            nsel["C_match"] = nsel["C_rerun"] == nsel["C_e4raw"]
            nsel["lr_match"] = nsel["l1_ratio_rerun"] == nsel["l1_ratio_e4raw"]
            nsel["sel_match"] = nsel["thr_match"] & nsel["C_match"] & nsel["lr_match"]
            print(nsel[["threshold_rerun", "C_rerun", "l1_ratio_rerun",
                        "threshold_e4raw", "C_e4raw", "l1_ratio_e4raw", "sel_match"]].to_string(),
                  flush=True)
            print(f"selection match: {int(nsel.sel_match.sum())}/{len(nsel)} folds; "
                  f"nested pooled AP = {pooled_ap_from_scores(n_all_p, n_all_y):.6f} "
                  f"(e4raw {obs_ap_fixed:.6f})", flush=True)
        return

    # permutation modes
    done = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as fh:
            fh.readline()
            for line in fh:
                parts = line.strip().split(",")
                if parts and parts[0].isdigit():
                    done.add(int(parts[0]))
    remaining = [i for i in range(args.start_perm, args.start_perm + args.n_permutations) if i not in done]
    print(f"mode={args.mode} folds={folds} remaining={len(remaining)} perms "
          f"({len(done)} done)", flush=True)

    # observed pooled AP on the same fold subset (for p-value comparison)
    obs = pd.read_csv(OBS_OOF_PATH)
    obs_sub = obs[obs.fold.isin(folds)]
    obs_ap = float(average_precision_score(obs_sub.target, obs_sub.score))
    print(f"observed pooled AP on folds {folds[0]}-{folds[-1]} = {obs_ap:.6f}", flush=True)

    t_start = time.monotonic()
    null_aps = []
    cache = {}
    inner_arrays = {}
    for perm_index in remaining:
        rng = np.random.default_rng(0 + perm_index)
        y_perm = rng.permutation(y_np)
        all_p, all_y = [], []
        t_perm = time.monotonic()
        for fold in folds:
            if args.mode == "fixed":
                p, dt = run_fixed_fold(X_np, y_perm, fold_meta, fold, fixed_params, cache)
            else:
                p, dt, sel = run_nested_fold(X_np, y_perm, fold_meta, fold, cache, inner_arrays, perm_index)
                if args.sel_log:
                    with open(args.sel_log, "a") as fh:
                        if not os.path.exists(args.sel_log) or os.path.getsize(args.sel_log) == 0:
                            fh.write("permutation,fold,threshold,C,l1_ratio,inner_ap\n")
                        fh.write(f"{perm_index},{sel['fold']},{sel['threshold']},{sel['C']},"
                                 f"{sel['l1_ratio']},{sel['inner_ap']:.6f}\n")
            te_idx = fold_meta[fold][1]
            all_p.append(p); all_y.append(y_perm[te_idx])
            if args.detail:
                with open(args.detail, "a") as fh:
                    if not os.path.exists(args.detail) or os.path.getsize(args.detail) == 0:
                        fh.write("permutation,fold,arm,ap,seconds\n")
                    fh.write(f"{perm_index},{fold},{args.mode},{average_precision_score(y_perm[te_idx], p):.6f},{dt:.3f}\n")
        null_ap = pooled_ap_from_scores(all_p, all_y)
        null_aps.append(null_ap)
        if args.checkpoint:
            if not os.path.exists(args.checkpoint) or os.path.getsize(args.checkpoint) == 0:
                with open(args.checkpoint, "w") as fh:
                    fh.write("permutation,auprc\n")
            with open(args.checkpoint, "a") as fh:
                fh.write(f"{perm_index},{null_ap:.9f}\n")
        print(f"[{time.monotonic() - t_start:.0f}s] perm {perm_index}: auprc={null_ap:.6f} "
              f"perm_took={time.monotonic() - t_perm:.1f}s", flush=True)

    null_aps = np.array(null_aps)
    print("=== SUMMARY ===", flush=True)
    print(f"n={len(null_aps)} mean={null_aps.mean():.6f} std={null_aps.std():.6f} "
          f"min={null_aps.min():.6f} max={null_aps.max():.6f}", flush=True)
    p_value = (1.0 + int((null_aps >= obs_ap).sum())) / (1.0 + len(null_aps))
    print(f"observed={obs_ap:.6f} p_value={p_value:.6f} "
          f"({int((null_aps >= obs_ap).sum())}/{len(null_aps)} nulls >= observed)", flush=True)
    print(f"total_seconds={time.monotonic() - t_start:.0f} "
          f"mean_perm_seconds={(time.monotonic() - t_start) / len(null_aps):.1f}", flush=True)


if __name__ == "__main__":
    main()
