import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FL_ROOT = PROJECT_ROOT / "fl-orchestration"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FL_ROOT))

from common.onnx_models import ONNXDetector
from scipy import sparse
import joblib
import numpy as np

def main():
    detector = ONNXDetector(PROJECT_ROOT)

    X = sparse.load_npz(
        PROJECT_ROOT / "data" / "tfidf" / "tfidf_matrix.npz"
    )

    y = np.asarray(
        joblib.load(PROJECT_ROOT / "data" / "tfidf" / "labels.pkl")
    )

    filenames = joblib.load(
        PROJECT_ROOT / "data" / "tfidf" / "filenames.pkl"
    )

    print("=" * 70)
    print("SIAMESE SVM — DETERMINISTIC PAIR TEST")
    print("=" * 70)

    print(f"Dataset shape : {X.shape}")
    print(f"Label counts  : {np.bincount(y)}")

    pairs = [
        ("BENIGN → BENIGN", 0, 1),
        ("MALWARE → MALWARE", 500, 501),
        ("BENIGN → MALWARE", 0, 500),
        ("MALWARE → BENIGN", 500, 0),
    ]

    print()

    results = []

    for name, sample_idx, reference_idx in pairs:

        sample = X[sample_idx].toarray().astype(np.float32)
        reference = X[reference_idx].toarray().astype(np.float32)

        difference = detector.make_pair_difference(
            sample,
            reference,
        )

        result = detector.predict_siamese(
            difference
        )

        prediction = int(result["labels"][0])
        raw = result["raw_output"][0]

        # LinearSVC ONNX raw output is represented as
        # opposing class scores.
        decision_score = float(raw[-1])

        expected_same_class = (
            y[sample_idx] == y[reference_idx]
        )

        results.append(
            (
                name,
                sample_idx,
                reference_idx,
                int(y[sample_idx]),
                int(y[reference_idx]),
                prediction,
                decision_score,
                expected_same_class,
            )
        )

    for (
        name,
        sample_idx,
        reference_idx,
        sample_label,
        reference_label,
        prediction,
        score,
        same_class,
    ) in results:

        print("-" * 70)
        print(name)
        print(
            f"Sample    : {sample_idx} "
            f"({filenames[sample_idx]})"
        )
        print(
            f"Reference : {reference_idx} "
            f"({filenames[reference_idx]})"
        )

        print(
            f"Labels    : "
            f"{sample_label} → {reference_label}"
        )

        print(
            f"Expected  : "
            f"{'SAME CLASS' if same_class else 'CROSS CLASS'}"
        )

        #print(
            #f"Prediction: "
            #f"{'MALWARE' if prediction else 'BENIGN'}"
        #)

        print(
            f"Raw score : {score:+.6f}"
        )

    print()
    print("=" * 70)
    print("NOTE")
    print("=" * 70)
    print(
        "The Siamese SVM predicts the trained pairwise class label."
    )
    print(
        "It should NOT be interpreted as a direct "
        "malware-probability score."
    )


if __name__ == "__main__":
    main()
