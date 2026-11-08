from pathlib import Path
import json
import joblib
import numpy as np
import lightgbm as lgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LGBM_DIR = (
    PROJECT_ROOT / "models/final_n1/entropy_strings_final"
)

SIAMESE_DIR = (
    PROJECT_ROOT / "models/siamese_svm_n1"
)


class ModelInterface:

    def __init__(self):

        # --------------------------------------------------
        # Load LightGBM
        # --------------------------------------------------

        with open(LGBM_DIR / "metadata.json") as f:
            self.lgbm_metadata = json.load(f)

        self.lgbm = lgb.Booster(
            model_file=str(
                LGBM_DIR / self.lgbm_metadata["model_file"]
            )
        )

        self.lgbm_features = self.lgbm_metadata[
            "feature_columns"
        ]

        self.lgbm_threshold = float(
            self.lgbm_metadata["threshold"]
        )

        # --------------------------------------------------
        # Load Siamese SVM
        # --------------------------------------------------

        with open(SIAMESE_DIR / "metadata.json") as f:
            self.siamese_metadata = json.load(f)

        self.siamese_bundle = joblib.load(
            SIAMESE_DIR / "siamese_svm_fewshot.joblib"
        )

        self.pair_classifier = (
            self.siamese_bundle["pair_classifier"]
        )

        self.vectorizer = (
            self.siamese_bundle["vectorizer"]
        )

        self.selected_k = int(
            self.siamese_metadata["selected_k"]
        )

        self.class_mapping = (
            self.siamese_metadata["class_mapping"]
        )

        # --------------------------------------------------
        # Validate artifact compatibility
        # --------------------------------------------------

        assert len(self.lgbm_features) == 12

        assert self.vectorizer.max_features == 30000

        assert self.pair_classifier.n_features_in_ == 30000

        assert self.selected_k == 3

        assert self.siamese_metadata[
            "feature_representation"
        ] == "absolute_tfidf_difference"

        print("Model interface initialized successfully.")

        print("LightGBM threshold:", self.lgbm_threshold)
        print("LightGBM features:", len(self.lgbm_features))
        print("Siamese TF-IDF features:", len(
            self.vectorizer.get_feature_names_out()
        ))
        print("Siamese selected K:", self.selected_k)

    # ------------------------------------------------------
    # LightGBM inference
    # ------------------------------------------------------

    def score_lightgbm(self, features):

        """
        features: dictionary containing the 12 LightGBM
                  feature names and their numeric values.

        Returns malware probability and threshold decision.
        """

        missing = set(self.lgbm_features) - set(features)

        if missing:
            raise ValueError(
                f"Missing LightGBM features: {sorted(missing)}"
            )

        X = np.asarray(
            [[features[name] for name in self.lgbm_features]],
            dtype=np.float64
        )

        probability = float(self.lgbm.predict(X)[0])

        prediction = int(
            probability >= self.lgbm_threshold
        )

        return {
            "malware_probability": probability,
            "threshold": self.lgbm_threshold,
            "prediction": prediction
        }

    # ------------------------------------------------------
    # Siamese pairwise inference
    # ------------------------------------------------------

    def score_siamese(self, query_vector, support_vectors):
        """
        query_vector:
            One TF-IDF vector, shape (1, 30000).

        support_vectors:
            {
                0: benign support vectors, shape (3, 30000),
                1: malware support vectors, shape (3, 30000)
            }

        Positive pair score means same class.

        The class with the higher mean pair score wins.
        """

        query_vector = query_vector.reshape(1, -1)

        class_scores = {}

        for class_id in (0, 1):

            supports = support_vectors[class_id]

            if supports.shape[0] != self.selected_k:
                raise ValueError(
                    f"Expected {self.selected_k} supports "
                    f"for class {class_id}; got {supports.shape[0]}"
                )

            differences = abs(
                supports - query_vector
            )

            pair_scores = self.pair_classifier.decision_function(
                differences
            )

            class_scores[class_id] = float(
                np.mean(pair_scores)
            )

        benign_score = class_scores[0]
        malware_score = class_scores[1]

        prediction = int(
            malware_score > benign_score
        )

        return {
            "benign_same_class_score": benign_score,
            "malware_same_class_score": malware_score,
            "malware_score_margin": (
                malware_score - benign_score
            ),
            "prediction": prediction
        }
