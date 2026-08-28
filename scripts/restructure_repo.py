#!/usr/bin/env python3

from pathlib import Path
import shutil
import argparse


ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Files/directories that are clearly dataset/generated artifacts.
# These are NOT deleted by this script.
# They are only reported so that .gitignore can exclude them.
# ----------------------------------------------------------------------

DATA_DIRS = [
    ROOT / "benign",
    ROOT / "data/raw",
    ROOT / "data/processed_assembly",
    ROOT / "data/ngrams",
]


GENERATED_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*.npz",
    "*.npy",
    "*.onnx",
]


# ----------------------------------------------------------------------
# Candidate obsolete files identified from the inventory.
# ----------------------------------------------------------------------

OBSOLETE_FILES = [
    ROOT / "models/onnx/lightgbm_metadata (copy 1).json",
    ROOT / "models/siamese_svm1.pkl",
]


# ----------------------------------------------------------------------
# Files that are reports / snapshots rather than source code.
# ----------------------------------------------------------------------

AUDIT_FILES = [
    ROOT / "codebase_snapshot.txt",
    ROOT / "ml_relevant_codebase.txt",
    ROOT / "repo_audit",
]


# ----------------------------------------------------------------------
# Current source layout → proposed layout.
#
# IMPORTANT:
# This first pass does NOT merge files.
# Merging is dangerous until imports/references have been verified.
# ----------------------------------------------------------------------

MOVE_MAP = {
    "pipelines/tabular": "src/iot_malware/tabular",
    "pipelines/assembly": "src/iot_malware/assembly",
    "pipelines/training": "src/iot_malware/training",
    "pipelines/export": "src/iot_malware/export",

    "fl-orchestration/common": "src/iot_malware/fl/common",
    "fl-orchestration/client": "src/iot_malware/fl/client",
    "fl-orchestration/server": "src/iot_malware/fl/server",

    "fl-orchestration/config": "configs",
}


def exists(path):
    return path.exists()


def show(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def report_data():
    print_header("DATA / GENERATED CONTENT")

    for directory in DATA_DIRS:
        if directory.exists():
            print(f"[DATA] {show(directory)}")

    for pattern in GENERATED_PATTERNS:
        matches = list(ROOT.rglob(pattern))

        if matches:
            print(
                f"[GENERATED] {pattern}: "
                f"{len(matches)} files"
            )


def report_obsolete():
    print_header("CANDIDATE OBSOLETE FILES")

    for path in OBSOLETE_FILES:
        if path.exists():
            print(f"[REMOVE CANDIDATE] {show(path)}")
        else:
            print(f"[MISSING] {show(path)}")


def report_audits():
    print_header("AUDIT / SNAPSHOT FILES")

    for path in AUDIT_FILES:
        if path.exists():
            print(f"[AUDIT] {show(path)}")


def report_moves():
    print_header("PROPOSED SOURCE MOVES")

    for source, destination in MOVE_MAP.items():
        src = ROOT / source
        dst = ROOT / destination

        if not src.exists():
            print(f"[SKIP] {source}")
            continue

        files = [
            p for p in src.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
        ]

        print(
            f"[MOVE] {source}"
            f" -> {destination}"
            f" ({len(files)} files)"
        )


def safe_move(src, dst, apply):
    if not src.exists():
        return

    if dst.exists():
        raise RuntimeError(
            f"Destination already exists: {show(dst)}"
        )

    print(
        f"[MOVE] {show(src)} -> {show(dst)}"
    )

    if apply:
        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.move(
            str(src),
            str(dst),
        )


def remove_empty_dirs(apply):
    print_header("EMPTY DIRECTORIES")

    candidates = []

    for path in sorted(
        ROOT.rglob("*"),
        reverse=True,
    ):
        if not path.is_dir():
            continue

        if ".git" in path.parts:
            continue

        try:
            next(path.iterdir())
            continue
        except StopIteration:
            candidates.append(path)

    for path in candidates:
        print(f"[EMPTY] {show(path)}")

        if apply:
            path.rmdir()


def write_gitignore(apply):
    print_header(".gitignore")

    gitignore = ROOT / ".gitignore"

    content = """# ============================================================
# Python
# ============================================================

__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
env/

# ============================================================
# ELF / binary samples
# ============================================================

*.elf
*.bin
*.exe
*.out
*.so
*.dll
*.dylib
*.o
*.a

# Extracted firmware
benign/
data/raw/
data/processed_assembly/
data/ngrams/

# ============================================================
# ML datasets / generated artifacts
# ============================================================

*.pkl
*.pickle
*.joblib
*.npz
*.npy
*.csv
*.parquet

# ============================================================
# Generated reports
# ============================================================

*.log
results/*.csv
results/*.png
results/*.html

# ============================================================
# Keep lightweight deployment models
# ============================================================

!models/
!models/onnx/
!models/onnx/*.onnx
!models/onnx/*.json

# ============================================================
# Local audit files
# ============================================================

repo_audit/
codebase_snapshot.txt
ml_relevant_codebase.txt

# ============================================================
# OS / editor
# ============================================================

.DS_Store
.vscode/
.idea/
"""

    print(content)

    if apply:
        gitignore.write_text(
            content,
            encoding="utf-8",
        )


def remove_obsolete(apply):
    print_header("REMOVING CONFIRMED OBSOLETE ARTIFACTS")

    for path in OBSOLETE_FILES:
        if not path.exists():
            continue

        print(f"[REMOVE] {show(path)}")

        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform changes",
    )

    args = parser.parse_args()

    print_header(
        "IoT MALWARE REPOSITORY RESTRUCTURE"
    )

    print(
        f"Repository: {ROOT}"
    )

    if not args.apply:
        print(
            "\nDRY RUN — no files will be moved or deleted."
        )

    report_data()
    report_obsolete()
    report_audits()
    report_moves()

    write_gitignore(args.apply)

    if args.apply:
        print_header("MOVING SOURCE CODE")

        for source, destination in MOVE_MAP.items():
            safe_move(
                ROOT / source,
                ROOT / destination,
                True,
            )

        remove_obsolete(True)
        remove_empty_dirs(True)

    else:
        print_header("DRY-RUN COMPLETE")

        print(
            "No filesystem changes were made."
        )

        print(
            "\nReview the proposed moves above."
        )

        print(
            "Run with --apply only after verification."
        )


if __name__ == "__main__":
    main()
