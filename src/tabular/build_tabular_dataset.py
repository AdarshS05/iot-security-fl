"""
Build the final N1 tabular feature dataset using the project's existing LIEF
extractor and the already-created split manifest.

Run from the repository root:
  python3 scripts/build_tabular_dataset.py \
    --manifest data/splits/final_n1/split_manifest.csv \
    --extractor src/tabular/elf_feature_extractor.py \
    --output data/features/tabular_n1/tabular_features.csv \
    --failures data/features/tabular_n1/extraction_failures.csv

The script is deliberately strict: any extraction or SHA-256 verification
failure aborts dataset creation. It does not change or recreate the split.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id", "binary_path", "binary_sha256",
    "label", "source", "split",
}
VALID_SPLITS = {"train", "validation", "test"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_extractor(extractor_path: Path):
    if not extractor_path.is_file():
        raise FileNotFoundError(f"Extractor not found: {extractor_path}")
    spec = importlib.util.spec_from_file_location("project_elf_feature_extractor", extractor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extractor module: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "extract_features"):
        raise AttributeError(
            f"{extractor_path} does not expose extract_features(path, include_opcode=False)"
        )
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Existing split_manifest.csv")
    parser.add_argument("--extractor", default="src/tabular/elf_feature_extractor.py")
    parser.add_argument("--output", default="data/features/tabular_n1/tabular_features.csv")
    parser.add_argument("--failures", default="data/features/tabular_n1/extraction_failures.csv")
    parser.add_argument("--include-opcode", action="store_true",
                        help="Enable optional radare2 features; normally leave disabled.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    extractor_path = Path(args.extractor).resolve()
    output_path = Path(args.output).resolve()
    failures_path = Path(args.failures).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest["binary_sha256"] = manifest["binary_sha256"].astype(str).str.lower().str.strip()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["label"] = manifest["label"].astype(str).str.lower().str.strip()
    manifest["source"] = manifest["source"].astype(str).str.strip()
    manifest["split"] = manifest["split"].astype(str).str.lower().str.strip()

    if manifest["sample_id"].duplicated().any():
        dupes = manifest.loc[manifest["sample_id"].duplicated(False), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values in split manifest: {dupes[:10]}")
    if manifest["binary_sha256"].duplicated().any():
        dupes = manifest.loc[manifest["binary_sha256"].duplicated(False), "binary_sha256"].tolist()
        raise ValueError(f"Duplicate raw binary hashes in split manifest: {dupes[:10]}")
    if not set(manifest["split"]).issubset(VALID_SPLITS):
        raise ValueError(f"Unexpected split values: {sorted(set(manifest['split']) - VALID_SPLITS)}")
    if not set(manifest["label"]).issubset({"benign", "malware"}):
        raise ValueError(f"Unexpected labels: {sorted(set(manifest['label']) - {'benign', 'malware'})}")

    # Verify no raw-binary or N1 hash crosses partitions, if the N1 hash exists.
    for hash_col in ("binary_sha256", "n1_conservative_normalized_sha256"):
        if hash_col in manifest.columns:
            cross = manifest.groupby(hash_col)["split"].nunique()
            if (cross > 1).any():
                raise ValueError(f"Cross-split leakage detected in {hash_col}")

    expected_counts = {"train": 699, "validation": 150, "test": 150}
    actual_counts = manifest["split"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(
            f"Split counts differ from the verified 699/150/150 split. "
            f"Expected {expected_counts}, found {actual_counts}. "
            "Check that --manifest points to data/splits/final_n1/split_manifest.csv."
        )

    extractor = load_extractor(extractor_path)
    if getattr(extractor, "lief", None) is None:
        raise RuntimeError("LIEF is not importable in this Python environment.")

    rows, failures = [], []
    total = len(manifest)

    for idx, row in manifest.iterrows():
        sample_id = row["sample_id"]
        binary_path = Path(str(row["binary_path"])).expanduser()
        if not binary_path.is_absolute():
            binary_path = (manifest_path.parent / binary_path).resolve()

        try:
            if not binary_path.is_file():
                raise FileNotFoundError(f"Binary does not exist: {binary_path}")

            actual_hash = sha256_file(binary_path)
            expected_hash = row["binary_sha256"]
            if actual_hash.lower() != expected_hash.lower():
                raise ValueError(
                    f"Manifest SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
                )

            features = extractor.extract_features(
                str(binary_path), include_opcode=args.include_opcode
            )
            extractor_hash = str(features.get("sha256", "")).lower()
            if extractor_hash and extractor_hash != actual_hash.lower():
                raise ValueError(
                    f"Extractor SHA-256 mismatch: extractor={extractor_hash}, file={actual_hash}"
                )

            record = {
                "sample_id": sample_id,
                "label": row["label"],
                "source": row["source"],
                "split": row["split"],
                "binary_path": str(binary_path),
                "binary_sha256": actual_hash,
            }
            # Keep extractor metadata for audit, but the training script excludes
            # file_name/md5/sha256 and all manifest metadata from model inputs.
            record.update(features)
            rows.append(record)

        except Exception as exc:
            failures.append({
                "sample_id": sample_id,
                "binary_path": str(binary_path),
                "expected_sha256": row["binary_sha256"],
                "error": repr(exc),
            })

        done = idx + 1
        if done % 50 == 0 or done == total:
            print(f"[{done}/{total}] processed; successful={len(rows)}, failures={len(failures)}")

    failures_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=[
        "sample_id", "binary_path", "expected_sha256", "error"
    ]).to_csv(failures_path, index=False)

    if failures:
        print(f"\nExtraction failures written to: {failures_path}", file=sys.stderr)
        raise RuntimeError(
            f"Aborting: {len(failures)} of {total} samples failed. "
            "Fix the failures and rerun; no partial feature dataset was written."
        )

    result = pd.DataFrame(rows)
    if len(result) != len(manifest):
        raise RuntimeError(f"Expected {len(manifest)} rows, generated {len(result)}")

    # Re-check join identity and partition counts after extraction.
    if result["sample_id"].duplicated().any() or result["binary_sha256"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs or raw hashes appeared in extracted dataset.")

    split_counts = result["split"].value_counts().to_dict()
    if split_counts != expected_counts:
        raise RuntimeError(f"Extracted split counts changed unexpectedly: {split_counts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nTABULAR DATASET BUILD COMPLETE")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print(f"Output: {output_path}")
    print(f"Failures report: {failures_path}")
    print("\nSplit counts:")
    print(result.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print("\nSource counts:")
    print(result.groupby(["split", "source"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
