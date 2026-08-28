#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import joblib

ROOT = Path(__file__).resolve().parents[2]

FL_ROOT = ROOT / "fl-orchestration"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FL_ROOT))

from common.onnx_models import ONNXDetector


FEATURES = [
    "global_entropy",
    "overlay_size_bytes",
    "has_overlay",
    "entropy_high_ratio",
    "num_strings",
    "avg_string_len",
    "max_string_len",
    "num_ip_strings",
    "num_url_strings",
    "num_suspicious_strings",
    "num_high_entropy_strings",
]


def load_extractor():
    """
    Import the project's existing LIEF extractor without duplicating
    feature-extraction logic.
    """
    import importlib.util

    path = ROOT / "pipelines" / "tabular" / "extract_lief.py"

    spec = importlib.util.spec_from_file_location(
        "project_extract_lief",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.extract_features


def find_elfs(root):
    if root.is_file():
        return [root]

    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not any(
            part in {
                "__pycache__",
                ".git",
                ".venv",
                "venv",
            }
            for part in p.parts
        )
        and p.read_bytes()[:4] == b"\x7fELF"
    )


def build_tabular_vector(features):
    missing = [f for f in FEATURES if f not in features]

    if missing:
        raise ValueError(
            "Missing LightGBM features: "
            + ", ".join(missing)
        )

    values = []

    for name in FEATURES:
        value = features[name]

        if isinstance(value, bool):
            value = int(value)

        values.append(float(value))

    return np.asarray(
        values,
        dtype=np.float32,
    )


def find_assembly_for_elf(elf_path):
    """
    Current assembly pipeline names its output:

        <original filename>.asm

    Search the project's generated assembly directories.
    """
    candidates = [
        ROOT / "data" / "ngrams" / "benign_mips",
        ROOT / "data" / "ngrams" / "malware_mips",
        ROOT / "data" / "normalized_assembly" / "benign_mips",
        ROOT / "data" / "normalized_assembly" / "malware_mips",
        ROOT / "data" / "processed_assembly" / "benign_mips",
        ROOT / "data" / "processed_assembly" / "malware_mips",
    ]

    wanted = [
        elf_path.name + ".asm",
        elf_path.stem + ".asm",
    ]

    for directory in candidates:
        for name in wanted:
            path = directory / name
            if path.exists():
                return path

    return None


def tfidf_from_assembly(asm_path, vectorizer):
    with open(
        asm_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        text = " ".join(
            line.strip()
            for line in f
            if line.strip()
        )

    return vectorizer.transform([text])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "target",
        help="ELF file or directory containing ELF files",
    )

    parser.add_argument(
        "--reference",
        type=int,
        default=500,
        help="TF-IDF reference row index",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of ELF files to test",
    )

    args = parser.parse_args()

    target = Path(args.target).resolve()

    if not target.exists():
        raise SystemExit(
            f"Target does not exist: {target}"
        )

    print("=" * 80)
    print("REAL ELF -> ONNX INFERENCE TEST")
    print("=" * 80)

    detector = ONNXDetector(ROOT)
    extract_features = load_extractor()

    # ------------------------------------------------------------
    # TF-IDF reference
    # ------------------------------------------------------------

    vectorizer_path = (
        ROOT
        / "data"
        / "tfidf"
        / "tfidf_vectorizer.pkl"
    )

    matrix_path = (
        ROOT
        / "data"
        / "tfidf"
        / "tfidf_matrix.npz"
    )

    filenames_path = (
        ROOT
        / "data"
        / "tfidf"
        / "filenames.pkl"
    )

    labels_path = (
        ROOT
        / "data"
        / "tfidf"
        / "labels.pkl"
    )

    vectorizer = joblib.load(vectorizer_path)
    X_reference = sparse.load_npz(matrix_path)

    filenames = joblib.load(filenames_path)
    labels = np.asarray(joblib.load(labels_path))

    ref = args.reference

    if not 0 <= ref < X_reference.shape[0]:
        raise SystemExit(
            f"Reference index must be in "
            f"[0, {X_reference.shape[0] - 1}]"
        )

    reference_vector = (
        X_reference[ref]
        .toarray()
        .astype(np.float32)
    )

    reference_label = int(labels[ref])

    print()
    print(f"Reference index : {ref}")
    print(f"Reference label : {reference_label}")
    print(f"Reference file  : {filenames[ref]}")

    # ------------------------------------------------------------
    # ELF files
    # ------------------------------------------------------------

    elfs = find_elfs(target)

    if not elfs:
        raise SystemExit(
            "No ELF files found."
        )

    elfs = elfs[:args.limit]

    print()
    print(f"ELF files tested: {len(elfs)}")
    print()

    rows = []

    for index, elf in enumerate(elfs, 1):

        print("-" * 80)
        print(
            f"[{index}/{len(elfs)}] {elf.name}"
        )

        # --------------------------------------------------------
        # TABULAR
        # --------------------------------------------------------

        try:
            start = time.perf_counter()

            features = extract_features(
                str(elf),
                include_opcode=False,
            )

            X_tabular = build_tabular_vector(
                features
            ).reshape(1, -1)

            result = detector.predict_tabular(
                X_tabular
            )

            tabular_time = (
                time.perf_counter() - start
            )

            tabular_label = int(
                result["labels"][0]
            )

            malware_probability = float(
                result["malware_probability"][0]
            )

            print(
                "TABULAR:"
                f" P(malware)="
                f"{malware_probability:.6f}"
                f"  -> "
                f"{'MALWARE' if tabular_label else 'BENIGN'}"
            )

        except Exception as exc:
            print(
                f"TABULAR ERROR: {exc}"
            )

            rows.append({
                "file": str(elf),
                "tabular": "ERROR",
                "tabular_probability": np.nan,
                "siamese": "ERROR",
            })

            continue

        # --------------------------------------------------------
        # ASSEMBLY
        # --------------------------------------------------------

        asm_path = find_assembly_for_elf(
            elf
        )

        if asm_path is None:
            print(
                "SIAMESE: assembly file not found"
            )

            rows.append({
                "file": str(elf),
                "tabular": (
                    "MALWARE"
                    if tabular_label
                    else "BENIGN"
                ),
                "tabular_probability":
                    malware_probability,
                "siamese": "NO_ASM",
            })

            continue

        try:
            start = time.perf_counter()

            X_tfidf = tfidf_from_assembly(
                asm_path,
                vectorizer,
            )

            pair_difference = np.abs(
                X_tfidf.toarray().astype(
                    np.float32
                )
                - reference_vector
            )

            siamese = detector.predict_siamese(
                pair_difference
            )

            siamese_time = (
                time.perf_counter() - start
            )

            pair_label = int(
                siamese["labels"][0]
            )

            raw_score = float(
                siamese["raw_output"][0][-1]
            )

            relationship = (
                "SAME CLASS"
                if pair_label == 1
                else "DIFFERENT CLASS"
            )

            print(
                f"SIAMESE:"
                f" {relationship}"
                f"  score={raw_score:+.6f}"
            )

            print(
                f"TF-IDF:"
                f" {X_tfidf.shape}"
            )

            print(
                f"Reference:"
                f" {reference_label}"
            )

            print(
                f"Latency:"
                f" tabular={tabular_time*1000:.2f} ms"
                f", siamese={siamese_time*1000:.2f} ms"
            )

            rows.append({
                "file": str(elf),
                "tabular": (
                    "MALWARE"
                    if tabular_label
                    else "BENIGN"
                ),
                "tabular_probability":
                    malware_probability,
                "siamese": relationship,
                "siamese_score": raw_score,
                "reference_label":
                    reference_label,
            })

        except Exception as exc:
            print(
                f"SIAMESE ERROR: {exc}"
            )

            rows.append({
                "file": str(elf),
                "tabular": (
                    "MALWARE"
                    if tabular_label
                    else "BENIGN"
                ),
                "tabular_probability":
                    malware_probability,
                "siamese": "ERROR",
            })

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    df = pd.DataFrame(rows)

    output = (
        ROOT
        / "results"
        / "real_onnx_inference.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Files processed : {len(df)}"
    )

    if "tabular" in df:
        print(
            "Tabular predictions:"
        )
        print(
            df["tabular"].value_counts()
        )

    if "siamese" in df:
        print(
            "Siamese results:"
        )
        print(
            df["siamese"].value_counts()
        )

    print()
    print(
        f"Results saved to: {output}"
    )


if __name__ == "__main__":
    main()
