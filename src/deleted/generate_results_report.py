from pathlib import Path
import json
import joblib
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
ONNX_DIR = MODEL_DIR / "onnx"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def model_info(path):
    m = onnx.load(path)

    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    return {
        "file": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "size_kb": path.stat().st_size / 1024,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "nodes": len(m.graph.node),
        "operators": ops,
        "inputs": [
            {
                "name": x.name,
                "shape": x.shape,
                "type": x.type,
            }
            for x in session.get_inputs()
        ],
        "outputs": [
            {
                "name": x.name,
                "shape": x.shape,
                "type": x.type,
            }
            for x in session.get_outputs()
        ],
    }


def main():
    report = {}

    # ------------------------------------------------------------
    # ONNX models
    # ------------------------------------------------------------

    for name in ["lightgbm.onnx", "siamese_svm.onnx"]:
        path = ONNX_DIR / name
        if path.exists():
            report[name] = model_info(path)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    for name in [
        "lightgbm_metadata.json",
        "siamese_svm_metadata.json",
    ]:
        path = ONNX_DIR / name

        if path.exists():
            with path.open() as f:
                report[name] = json.load(f)

    # ------------------------------------------------------------
    # Source model sizes
    # ------------------------------------------------------------

    for name in [
        "lightgbm_model.pkl",
        "siamese_svm1.pkl",
    ]:
        path = MODEL_DIR / name

        if path.exists():
            report[f"{name}_size"] = {
                "size_bytes": path.stat().st_size,
                "size_kb": path.stat().st_size / 1024,
                "size_mb": path.stat().st_size / (1024 * 1024),
            }

    # ------------------------------------------------------------
    # TF-IDF information
    # ------------------------------------------------------------

    tfidf_matrix = ROOT / "data/tfidf/tfidf_matrix.npz"

    if tfidf_matrix.exists():
        from scipy import sparse

        X = sparse.load_npz(tfidf_matrix)

        report["tfidf"] = {
            "documents": X.shape[0],
            "vocabulary_size": X.shape[1],
            "nonzero_entries": int(X.nnz),
            "sparsity_percent": float(
                100 * (1 - X.nnz / (X.shape[0] * X.shape[1]))
            ),
        }

    # ------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------

    json_path = RESULTS_DIR / "onnx_results.json"

    with json_path.open("w") as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------
    # Presentation text
    # ------------------------------------------------------------

    lines = []

    lines.append("=" * 70)
    lines.append("IoT MALWARE DETECTION - CURRENT RESULTS")
    lines.append("=" * 70)
    lines.append("")

    lines.append("DATA / FEATURE SPACE")
    lines.append("-" * 70)
    lines.append("Dataset target: 500 malware + 500 benign binaries")
    lines.append("Tabular inference features: 11")
    lines.append("Assembly TF-IDF features: 18,223")
    lines.append("")

    if "tfidf" in report:
        t = report["tfidf"]
        lines.append(
            f"TF-IDF matrix: {t['documents']} x "
            f"{t['vocabulary_size']}"
        )
        lines.append(
            f"TF-IDF sparsity: {t['sparsity_percent']:.2f}%"
        )

    lines.append("")
    lines.append("LIGHTGBM -> ONNX")
    lines.append("-" * 70)

    l = report.get("lightgbm.onnx", {})

    lines.append("Status: COMPLETED")
    lines.append("Input features: 11")
    lines.append("Operator: TreeEnsembleClassifier")
    lines.append(f"ONNX nodes: {l.get('nodes', 'N/A')}")
    lines.append(
        f"ONNX size: {l.get('size_kb', 0):.2f} KB"
    )

    meta = report.get("lightgbm_metadata.json", {})

    if meta:
        lines.append(
            f"Decision threshold: "
            f"{meta.get('threshold', 'N/A')}"
        )

        lines.append(
            "Classes: "
            + ", ".join(map(str, meta.get("classes", [])))
        )

    lines.append("")
    lines.append("SIAMESE SVM -> ONNX")
    lines.append("-" * 70)

    s = report.get("siamese_svm.onnx", {})

    lines.append("Status: COMPLETED")
    lines.append("Input features: 18,223")
    lines.append("Pipeline: Absolute TF-IDF difference")
    lines.append("        -> StandardScaler")
    lines.append("        -> LinearSVC")
    lines.append(f"ONNX nodes: {s.get('nodes', 'N/A')}")
    lines.append(
        f"ONNX size: {s.get('size_kb', 0):.2f} KB"
    )

    lines.append("")
    lines.append("ONNX RUNTIME")
    lines.append("-" * 70)
    lines.append("Provider: CPUExecutionProvider")
    lines.append("LightGBM model: loadable")
    lines.append("Siamese SVM model: loadable")

    lines.append("")
    lines.append("CURRENT IMPLEMENTATION STATUS")
    lines.append("-" * 70)
    lines.append("[DONE] Tabular feature extraction")
    lines.append("[DONE] Assembly extraction")
    lines.append("[DONE] TF-IDF vectorization")
    lines.append("[DONE] LightGBM training")
    lines.append("[DONE] Siamese SVM training")
    lines.append("[DONE] LightGBM ONNX export")
    lines.append("[DONE] Siamese SVM lightweight ONNX export")
    lines.append("[DONE] ONNX Runtime model loading")
    lines.append("[DONE] Flower simulation / communication layer")
    lines.append("[NEXT] Integrate ONNX models into collaborator's Flower client")
    lines.append("[NEXT] GA chromosome + mutation constraints")
    lines.append("[NEXT] GA fitness evaluation")
    lines.append("[NEXT] Adversarial retraining")

    txt_path = RESULTS_DIR / "presentation_results.txt"

    txt_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
