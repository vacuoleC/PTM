"""Build the reproducible E3.1 permutation-null remote bundle.

The bundle contains only frozen design, config, predeclared candidates,
outer-fold assignments, and the runner scripts — no raw PTMv1 data.
Output: releases/e3_1_permutation_null_bundle.tar.gz
"""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import yaml

BUNDLE_FILES = [
    "scr/run_permutation_null.py",
    "scr/nested_raw_elasticnet.py",
    "scr/evaluate.py",
    "scr/preprocessing.py",
    "config/project.yaml",
    "study_design.yaml",
    "outputs/tables/e2_2_outer_split_assignments.csv",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "releases"
    out_dir.mkdir(exist_ok=True)
    bundle_path = out_dir / "e3_1_permutation_null_bundle.tar.gz"

    missing = [f for f in BUNDLE_FILES if not (root / f).exists()]
    if missing:
        raise SystemExit(f"missing bundle files: {missing}")

    with tarfile.open(bundle_path, "w:gz") as tar:
        for rel in BUNDLE_FILES:
            tar.add(root / rel, arcname=rel)

    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    size = bundle_path.stat().st_size
    print(f"wrote {bundle_path} ({size} bytes)")
    print(f"sha256: {digest}")
    print("members:")
    for rel in BUNDLE_FILES:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
