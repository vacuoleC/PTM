"""Build the exact, small E2.2 remote execution tarball tracked in Git."""
from pathlib import Path
import hashlib
import tarfile


ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "releases" / "e2_2_remote_oof_bundle.tar.gz"
MEMBERS = [
    "PTMv2/config/project.yaml",
    "PTMv2/data/manifest.tsv",
    "PTMv2/study_design.yaml",
    "PTMv2/remote/E2_2_REMOTE_RUN.md",
    "PTMv2/scr/preprocessing.py",
    "PTMv2/scr/evaluate.py",
    "PTMv2/scr/oof.py",
    "PTMv2/scr/smoke_oof.py",
    "PTMv2/outputs/tables/e2_2_outer_split_assignments.csv",
]


def main() -> None:
    """Create a tarball with exactly the files declared in MEMBERS."""
    repository = ROOT.parent
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE, "w:gz") as archive:
        for relative in MEMBERS:
            source = repository / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.add(source, arcname=relative)
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    print(f"bundle={BUNDLE}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
