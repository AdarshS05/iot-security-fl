"""
Build TF-IDF matrices from an EXISTING split_manifest.csv.
Fits the vocabulary on TRAIN ONLY; validation/test are transform-only.

Run:
python3 scripts/build_tfidf_from_split.py \
  --manifest data/splits/final_n1/split_manifest.csv \
  --output-dir data/features/tfidf_n1 \
  --max-features 30000 \
  --ngram-max 2
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def read_assembly(path_value):
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"Assembly file not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/features/tfidf_n1")
    parser.add_argument("--assembly-col", default="n1_conservative_path")
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--ngram-max", type=int, default=2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"split", "label", "sample_id", args.assembly_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    expected_splits = {"train", "validation", "test"}
    if set(df["split"]) != expected_splits:
        raise ValueError(
            f"Expected exactly {expected_splits}, got {set(df['split'])}"
        )

    if not set(df["label"]).issubset({"benign", "malware"}):
        raise ValueError(f"Unexpected labels: {sorted(df['label'].unique())}")

    # Stable row order within each split.
    parts = {
        name: df[df["split"] == name].copy().reset_index(drop=True)
        for name in ["train", "validation", "test"]
    }

    texts = {}
    for split, part in parts.items():
        texts[split] = [
            read_assembly(p) for p in part[args.assembly_col].tolist()
        ]
        if any(not text.strip() for text in texts[split]):
            raise ValueError(f"Empty assembly text found in {split} split.")

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w.$:+-]+\b",
        lowercase=False,
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    # Fit only on training corpus to prevent vocabulary leakage.
    X_train = vectorizer.fit_transform(texts["train"])
    X_val = vectorizer.transform(texts["validation"])
    X_test = vectorizer.transform(texts["test"])

    matrices = {
        "train": X_train,
        "validation": X_val,
        "test": X_test,
    }

    for split, matrix in matrices.items():
        save_npz(out / f"X_{split}.npz", matrix)
        part = parts[split].copy()
        part["y"] = (part["label"] == "malware").astype(np.int8)
        part.to_csv(out / f"{split}_samples.csv", index=False)

    joblib.dump(vectorizer, out / "tfidf_vectorizer.joblib")
    pd.Series(
        vectorizer.get_feature_names_out(),
        name="feature"
    ).to_csv(out / "feature_names.csv", index=False)

    summary = []
    for split, matrix in matrices.items():
        summary.append({
            "split": split,
            "samples": matrix.shape[0],
            "features": matrix.shape[1],
            "nonzero_values": int(matrix.nnz),
            "sparsity": 1.0 - matrix.nnz / (
                matrix.shape[0] * matrix.shape[1]
            ),
        })

    pd.DataFrame(summary).to_csv(out / "tfidf_summary.csv", index=False)

    print("\nTF-IDF generation complete")
    print(f"Output directory: {out}")
    print(f"Vocabulary size: {X_train.shape[1]}")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nVocabulary was fitted on train only.")
    print("Validation and test were transformed using the train vocabulary.")


if __name__ == "__main__":
    main()
