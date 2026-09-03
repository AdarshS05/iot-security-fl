from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = MODEL_DIR / "onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIGHTGBM_MODEL = MODEL_DIR / "lightgbm_model.pkl"
SIAMESE_MODEL = MODEL_DIR / "siamese_svm1.pkl"

TABULAR_METADATA = OUTPUT_DIR / "lightgbm_metadata.json"


def export_lightgbm():
    model_path = LIGHTGBM_MODEL

    print(f"Loading LightGBM model: {model_path}")

    loaded = joblib.load(model_path)

    if isinstance(loaded, dict):
        model = loaded.get("model", loaded.get("lightgbm_model"))
        threshold = loaded.get("best_threshold", 0.5)
        label_encoder = loaded.get("label_encoder")
    else:
        model = loaded
        threshold = 0.5
        label_encoder = None

    if model is None:
        raise RuntimeError("Could not find LightGBM model inside artifact.")

    if not hasattr(model, "n_features_in_"):
        raise RuntimeError(
            "Loaded LightGBM object does not expose n_features_in_."
        )

    n_features = int(model.n_features_in_)

    print(f"Features: {n_features}")
    print(f"Threshold: {threshold}")

    initial_types = [
        ("input", FloatTensorType([None, n_features]))
    ]

    print("Converting LightGBM -> ONNX...")

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_types,
        target_opset=15,
        zipmap=False,
    )

    output_path = OUTPUT_DIR / "lightgbm.onnx"

    onnx.save_model(onnx_model, output_path)

    metadata = {
        "model_type": "LightGBM",
        "source_model": str(model_path.relative_to(ROOT)),
        "onnx_model": str(output_path.relative_to(ROOT)),
        "n_features": n_features,
        "input_name": "input",
        "threshold": float(threshold),
        "classes": (
            label_encoder.classes_.tolist()
            if label_encoder is not None
            else [0, 1]
        ),
        "feature_names": (
            list(model.feature_name_)
            if hasattr(model, "feature_name_")
            else []
        ),
    }

    with TABULAR_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {output_path}")
    print(f"Saved: {TABULAR_METADATA}")

    return model, output_path


def export_siamese_svm(model_dir= MODEL_DIR):
    model_path = model_dir / "siamese_svm1.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Siamese SVM model not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    model = artifact["model"]
    scaler = artifact["scaler"]

    n_features = int(model.n_features_in_)

    print(f"Siamese SVM features: {n_features}")
    print(f"SVM type: {type(model).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")

    initial_types = [
        (
            "pair_difference",
            SklearnFloatTensorType([None, n_features]),
        )
    ]

    # Convert the complete sklearn pipeline.
    #
    # IMPORTANT:
    # The existing model expects:
    #
    # abs(tfidf_a - tfidf_b)
    #       ↓
    # StandardScaler
    #       ↓
    # LinearSVC
    #
    # The ONNX input is therefore the already-computed
    # absolute TF-IDF difference vector.

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("scaler", scaler),
            ("model", model),
        ]
    )

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=15,
        options={
            id(model): {
                "raw_scores": True
            }
        },
    )

    output_path = model_dir / "siamese_svm.onnx"

    onnx.save_model(
        onnx_model,
        output_path,
    )

    metadata = {
        "model_type": "SiameseSVM",
        "source_model": str(model_path),
        "onnx_model": str(output_path),
        "input_name": "pair_difference",
        "n_features": n_features,
        "pair_operation": "absolute_difference",
        "scaler": type(scaler).__name__,
        "classifier": type(model).__name__,
        "classes": model.classes_.tolist(),
    }

    metadata_path = model_dir / "siamese_svm_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Siamese SVM ONNX: {output_path}")
    print(f"Metadata:         {metadata_path}")


def main():
    print("=" * 70)
    print("IoT MALWARE MODEL -> ONNX EXPORT")
    print("=" * 70)

    export_lightgbm()
    export_siamese_svm()

    print("\nExport completed.")


if __name__ == "__main__":
    main()
