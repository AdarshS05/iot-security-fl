#!/usr/bin/env python3
"""Create one reproducible split manifest shared by assembly and tabular pipelines.

Example:
python3 scripts/make_selected_splits.py \
  --manifest data/reports/n1_selection/selected_1000_manifest.csv \
  --output-dir data/splits/final_n1 \
  --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/splits/final_n1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    if abs(args.train + args.validation + args.test - 1.0) > 1e-8:
        parser.error("--train + --validation + --test must equal 1.0")

    manifest_path = Path(args.manifest).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required = {
        "label",
        "source",
        "binary_sha256",
        "n1_conservative_normalized_sha256",
        "n1_conservative_path",
        "binary_path",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip()

    if not set(df["label"]).issubset({"benign", "malware"}):
        raise ValueError(f"Unexpected labels: {sorted(df['label'].unique())}")

    # Refuse accidental duplicates rather than silently allowing leakage.
    for col in ["binary_sha256", "n1_conservative_normalized_sha256"]:
        duplicated = df[col].duplicated(keep=False)
        if duplicated.any():
            examples = df.loc[duplicated, col].astype(str).head(10).tolist()
            raise ValueError(
                f"Duplicate values found in {col}; resolve before splitting. "
                f"Examples: {examples}"
            )

    # Stratify by class AND corpus source, keeping C/F represented in each split.
    df["_stratum"] = df["label"] + "::" + df["source"]

    trainval_frac = args.train + args.validation
    test_frac_of_all = args.test
    val_frac_of_trainval = args.validation / trainval_frac

    trainval, test = train_test_split(
        df,
        test_size=test_frac_of_all,
        random_state=args.seed,
        stratify=df["_stratum"],
    )

    train, val = train_test_split(
        trainval,
        test_size=val_frac_of_trainval,
        random_state=args.seed,
        stratify=trainval["_stratum"],
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    combined = pd.concat([train, val, test], ignore_index=True)
    combined = combined.drop(columns=["_stratum"])
    combined = combined.sort_values(["split", "label", "source", "sample_id"])
    combined.to_csv(out / "split_manifest.csv", index=False)

    for name, part in [
        ("train", train), ("validation", val), ("test", test)
    ]:
        part.drop(columns=["_stratum"]).to_csv(out / f"{name}.csv", index=False)

    # Explicit cross-split checks.
    for col in ["binary_sha256", "n1_conservative_normalized_sha256"]:
        split_counts = combined.groupby(col)["split"].nunique()
        crossing = int((split_counts > 1).sum())
        if crossing:
            raise RuntimeError(
                f"Leakage check failed: {crossing} {col} groups cross splits."
            )

    print("\nSplit counts:")
    print(pd.crosstab(combined["split"], combined["label"]))
    print("\nSplit by source:")
    print(pd.crosstab(combined["split"], combined["source"]))
    print(f"\nSaved: {out / 'split_manifest.csv'}")
    print("Raw-binary and N1-hash cross-split leakage checks: PASS")


if __name__ == "__main__":
    main()
