from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
import joblib


class ONNXDetector:
    def __init__(self, project_root=None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.root = Path(project_root)
        self.model_dir = self.root / "models" / "onnx"

        self.lightgbm_path = self.model_dir / "lightgbm.onnx"
        self.siamese_path = self.model_dir / "siamese_svm.onnx"

        self.tfidf_dir = self.root / "data" / "tfidf"

        self.tfidf_vectorizer_path = (
            self.tfidf_dir / "tfidf_vectorizer.pkl"
        )

        self.tfidf_matrix_path = (
            self.tfidf_dir / "tfidf_matrix.npz"
        )

        self.filenames_path = (
            self.tfidf_dir / "filenames.pkl"
        )

        self.labels_path = (
            self.tfidf_dir / "labels.pkl"
        )

        # ---------------------------------------------------------
        # Load ONNX sessions
        # ---------------------------------------------------------

        if not self.lightgbm_path.exists():
            raise FileNotFoundError(self.lightgbm_path)

        if not self.siamese_path.exists():
            raise FileNotFoundError(self.siamese_path)

        self.lightgbm = ort.InferenceSession(
            str(self.lightgbm_path),
            providers=["CPUExecutionProvider"],
        )

        self.siamese = ort.InferenceSession(
            str(self.siamese_path),
            providers=["CPUExecutionProvider"],
        )

        # ---------------------------------------------------------
        # Load metadata
        # ---------------------------------------------------------

        with open(
            self.model_dir / "lightgbm_metadata.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.lightgbm_metadata = json.load(f)

        with open(
            self.model_dir / "siamese_svm_metadata.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.siamese_metadata = json.load(f)

        # ---------------------------------------------------------
        # TF-IDF vectorizer
        # ---------------------------------------------------------

        self.vectorizer = None

        if self.tfidf_vectorizer_path.exists():
            self.vectorizer = joblib.load(
                self.tfidf_vectorizer_path
            )

        print("[ONNX] Models loaded")
        print(
            f"[ONNX] LightGBM input: "
            f"{self.lightgbm.get_inputs()[0].shape}"
        )
        print(
            f"[ONNX] Siamese input: "
            f"{self.siamese.get_inputs()[0].shape}"
        )

    # =============================================================
    # TABULAR
    # =============================================================

    def predict_tabular(self, X):
        """
        Run LightGBM ONNX inference.

        X:
            shape (N, 11)
        """

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 11:
            raise ValueError(
                f"Expected 11 tabular features, "
                f"got {X.shape[1]}"
            )

        input_name = self.lightgbm.get_inputs()[0].name

        outputs = self.lightgbm.run(
            None,
            {
                input_name: X,
            },
        )

        labels = np.asarray(outputs[0]).reshape(-1)
        probabilities = np.asarray(outputs[1])

        # Malware probability
        malware_probability = probabilities[:, 1]

        threshold = float(
            self.lightgbm_metadata.get(
                "threshold",
                0.5,
            )
        )

        predictions = (
            malware_probability >= threshold
        ).astype(np.int64)

        return {
            "labels": predictions,
            "probabilities": probabilities,
            "malware_probability": malware_probability,
        }

    # =============================================================
    # SIAMESE
    # =============================================================

    def predict_siamese(self, pair_difference):
        """
        Run Siamese SVM ONNX inference.

        Input:
            absolute TF-IDF difference

        Shape:
            (N, 18223)
        """

        X = np.asarray(
            pair_difference,
            dtype=np.float32,
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        expected = int(
            self.siamese_metadata["n_features"]
        )

        if X.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} Siamese features, "
                f"got {X.shape[1]}"
            )

        input_name = self.siamese.get_inputs()[0].name

        outputs = self.siamese.run(
            None,
            {
                input_name: X,
            },
        )

        labels = np.asarray(outputs[0]).reshape(-1)

        raw_output = np.asarray(outputs[1])
        same_class = labels.astype(np.int64)
        return {
            "labels": same_class,
            "same_class": same_class,
            "different_class": 1 - same_class,
            "raw_output": raw_output,
        }

    # =============================================================
    # SIAMESE PAIR
    # =============================================================

    def make_pair_difference(self, X, reference):
        X = np.asarray(X, dtype=np.float32)
        reference = np.asarray(
            reference,
            dtype=np.float32,
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if reference.ndim == 1:
            reference = reference.reshape(1, -1)

        if reference.shape[0] == 1:
            reference = np.repeat(
                reference,
                X.shape[0],
                axis=0,
            )

        if X.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch: X={X.shape}, "
                f"reference={reference.shape}"
            )

        return np.abs(X - reference)

    # =============================================================
    # COMBINED
    # =============================================================

    def predict_both(
        self,
        tabular_features,
        pair_difference,
    ):
        tabular = self.predict_tabular(
            tabular_features
        )

        siamese = self.predict_siamese(
            pair_difference
        )

        tabular_score = float(
            np.mean(tabular["malware_probability"])
        )

        siamese_labels = siamese["labels"]

        siamese_score = float(
            np.mean(siamese_labels)
        )

        # Initial equal-weight consensus.
        # Calibration can replace this later.
        consensus_score = (
            0.5 * tabular_score
            + 0.5 * siamese_score
        )

        consensus_label = int(
            consensus_score >= 0.5
        )

        return {
            "tabular": tabular,
            "siamese": siamese,
            "tabular_score": tabular_score,
            "siamese_score": siamese_score,
            "consensus_score": consensus_score,
            "consensus_label": consensus_label,
        }
