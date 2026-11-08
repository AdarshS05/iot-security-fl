
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.sparse import vstack, load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data/features/tfidf_n1"
SPLIT_DIR = ROOT / "data/splits/final_n1"
OUTPUT_DIR = ROOT / "models/siamese_svm_n1"

LABEL_MAP = {
    "benign": 0,
    "malware": 1
}

K_VALUES = [1, 3, 5, 10]

RANDOM_STATE = 42
EPISODES = 100
QUERIES_PER_CLASS = 10

SVM_PARAMETERS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 20000,
    "random_state": RANDOM_STATE
}


# =========================================================
# VALIDATION
# =========================================================

def validate_arguments(args, parser):

    if args.pairs_per_relation <= 0:
        parser.error("--pairs-per-relation must be positive")

    if args.pairs_per_relation % 2 != 0:
        parser.error(
            "--pairs-per-relation must be even because "
            "same-class pairs are generated equally per class"
        )

    if args.episodes <= 1:
        parser.error("--episodes must be greater than 1")

    if args.queries_per_class <= 0:
        parser.error("--queries-per-class must be positive")


def validate_partition_disjointness(partitions):

    print("\nChecking partition disjointness...")

    for field in ["sample_id", "binary_sha256"]:

        if not all(field in df.columns for df in partitions.values()):
            raise ValueError(
                f"Required leakage-check column missing: {field}"
            )

        sets = {
            name: set(df[field].dropna())
            for name, df in partitions.items()
        }

        names = list(sets.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a = names[i]
                b = names[j]

                overlap = sets[a] & sets[b]

                if overlap:
                    raise ValueError(
                        f"Leakage detected: {field} overlaps between "
                        f"{a} and {b}: {len(overlap)} values"
                    )

        print(f"{field}: no cross-partition overlap")


# =========================================================
# LOAD AND VALIDATE TF-IDF MATRICES
# =========================================================

def load_partition(name):

    matrix_path = FEATURE_DIR / f"X_{name}.npz"
    samples_path = FEATURE_DIR / f"{name}_samples.csv"
    split_path = SPLIT_DIR / f"{name}.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    if not split_path.exists():
        raise FileNotFoundError(split_path)

    X = load_npz(matrix_path).tocsr()

    samples = pd.read_csv(samples_path)
    split = pd.read_csv(split_path)

    if X.shape[0] != len(samples):
        raise ValueError(
            f"{name}: matrix rows {X.shape[0]} "
            f"!= sample rows {len(samples)}"
        )

    if X.shape[1] == 0:
        raise ValueError(f"{name}: empty feature matrix")

    required_columns = [
        "sample_id",
        "label",
        "binary_sha256"
    ]

    for column in required_columns:

        if column not in samples.columns:
            raise ValueError(
                f"{name}: missing {column} in sample metadata"
            )

        if column not in split.columns:
            raise ValueError(
                f"{name}: missing {column} in split manifest"
            )

    # Verify exact sample order.
    if samples["sample_id"].tolist() != split["sample_id"].tolist():

        raise ValueError(
            f"{name}: TF-IDF sample order differs from split CSV. "
            "Correct alignment before training."
        )

    # Verify labels also agree with the split manifest.
    sample_labels = samples["label"].str.lower().tolist()
    split_labels = split["label"].str.lower().tolist()

    if sample_labels != split_labels:
        raise ValueError(
            f"{name}: labels differ between TF-IDF metadata "
            "and split manifest"
        )

    y = samples["label"].str.lower().map(LABEL_MAP)

    if y.isna().any():
        raise ValueError(f"{name}: unknown labels detected")

    y = y.to_numpy(dtype=np.int32)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"{name}: both benign and malware classes are required"
        )

    print(
        f"{name}: X={X.shape}, "
        f"benign={int(np.sum(y == 0))}, "
        f"malware={int(np.sum(y == 1))}"
    )

    return X, y, samples


# =========================================================
# GENERATE TRAINING PAIRS
# =========================================================

def generate_pair_indices(y, pairs_per_relation, rng):

    if pairs_per_relation % 2 != 0:
        raise ValueError("pairs_per_relation must be even")

    class_0 = np.flatnonzero(y == 0)
    class_1 = np.flatnonzero(y == 1)

    if min(len(class_0), len(class_1)) < 2:
        raise ValueError(
            "At least two samples per class are required"
        )

    # Same-class pairs.
    # Half benign-benign and half malware-malware.

    pos_a = []
    pos_b = []

    for _ in range(pairs_per_relation // 2):

        for indices in (class_0, class_1):

            a = rng.choice(indices)
            b = rng.choice(indices)

            while a == b:
                b = rng.choice(indices)

            pos_a.append(a)
            pos_b.append(b)

    # Different-class pairs.
    neg_a = rng.choice(
        class_0,
        size=pairs_per_relation,
        replace=True
    )

    neg_b = rng.choice(
        class_1,
        size=pairs_per_relation,
        replace=True
    )

    a = np.concatenate([
        np.asarray(pos_a),
        neg_a
    ])

    b = np.concatenate([
        np.asarray(pos_b),
        neg_b
    ])

    # 1 = same class
    # 0 = different class

    pair_y = np.concatenate([
        np.ones(len(pos_a), dtype=np.int32),
        np.zeros(len(neg_a), dtype=np.int32)
    ])

    order = rng.permutation(len(pair_y))

    return a[order], b[order], pair_y[order]


# =========================================================
# ABSOLUTE TF-IDF DIFFERENCE REPRESENTATION
# =========================================================

def make_difference_matrix(X, indices_a, indices_b):

    if len(indices_a) != len(indices_b):
        raise ValueError("Pair index arrays have different lengths")

    rows = []

    for a, b in zip(indices_a, indices_b):

        diff = X[a] - X[b]

        # Absolute TF-IDF difference.
        diff.data = np.abs(diff.data)
        diff.eliminate_zeros()

        rows.append(diff)

    return vstack(rows, format="csr")


# =========================================================
# TRAIN PAIRWISE LINEAR SVM
# =========================================================

def train_pair_model(X_train, y_train, pairs_per_relation):

    rng = np.random.default_rng(RANDOM_STATE)

    print("\nGenerating training pairs...")

    a, b, pair_y = generate_pair_indices(
        y_train,
        pairs_per_relation,
        rng
    )

    same_count = int(np.sum(pair_y == 1))
    different_count = int(np.sum(pair_y == 0))

    print("Same-class pairs:", same_count)
    print("Different-class pairs:", different_count)

    if same_count != pairs_per_relation:
        raise RuntimeError("Unexpected same-class pair count")

    if different_count != pairs_per_relation:
        raise RuntimeError("Unexpected different-class pair count")

    X_pairs = make_difference_matrix(
        X_train,
        a,
        b
    )

    print("Pair matrix:", X_pairs.shape)
    print("Training Linear SVM...")

    svm = LinearSVC(**SVM_PARAMETERS)

    svm.fit(X_pairs, pair_y)

    print("SVM classes:", svm.classes_)
    print("SVM feature count:", svm.n_features_in_)

    if not np.array_equal(svm.classes_, np.array([0, 1])):
        raise RuntimeError(
            "Unexpected pair classifier class ordering"
        )

    return svm, pair_y


# =========================================================
# FEW-SHOT EPISODIC EVALUATION
# =========================================================

def evaluate_episodes(
    model,
    X,
    y,
    k,
    episodes,
    queries_per_class,
    seed
):

    rng = np.random.default_rng(seed)

    class_0 = np.flatnonzero(y == 0)
    class_1 = np.flatnonzero(y == 1)

    required_samples = k + queries_per_class

    if len(class_0) < required_samples:
        raise ValueError(
            f"Insufficient benign samples for {k}-shot evaluation"
        )

    if len(class_1) < required_samples:
        raise ValueError(
            f"Insufficient malware samples for {k}-shot evaluation"
        )

    metrics = []

    for episode in range(episodes):

        # ---------------------------------------------
        # Support selection
        # ---------------------------------------------

        support_0 = rng.choice(
            class_0,
            size=k,
            replace=False
        )

        support_1 = rng.choice(
            class_1,
            size=k,
            replace=False
        )

        # ---------------------------------------------
        # Query selection
        # ---------------------------------------------

        remaining_0 = np.setdiff1d(
            class_0,
            support_0
        )

        remaining_1 = np.setdiff1d(
            class_1,
            support_1
        )

        q0 = rng.choice(
            remaining_0,
            size=queries_per_class,
            replace=False
        )

        q1 = rng.choice(
            remaining_1,
            size=queries_per_class,
            replace=False
        )

        query_indices = np.concatenate([
            q0,
            q1
        ])

        query_y = y[query_indices]

        # ---------------------------------------------
        # Support ordering:
        # First benign, then malware.
        # ---------------------------------------------

        support_indices = np.concatenate([
            support_0,
            support_1
        ])

        # ---------------------------------------------
        # Construct query-support pairs
        # ---------------------------------------------

        pair_a = np.repeat(
            query_indices,
            2 * k
        )

        pair_b = np.tile(
            support_indices,
            len(query_indices)
        )

        X_pairs = make_difference_matrix(
            X,
            pair_a,
            pair_b
        )

        # Positive score = same class.
        similarities = model.decision_function(X_pairs)

        similarities = similarities.reshape(
            len(query_indices),
            2,
            k
        )

        # Mean score across support examples.
        class_scores = similarities.mean(axis=2)

        predictions = np.argmax(
            class_scores,
            axis=1
        )

        metrics.append({

            "episode": episode + 1,

            "accuracy": accuracy_score(
                query_y,
                predictions
            ),

            "precision": precision_score(
                query_y,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                query_y,
                predictions,
                zero_division=0
            ),

            "f1": f1_score(
                query_y,
                predictions,
                zero_division=0
            )
        })

    episode_df = pd.DataFrame(metrics)

    summary = {}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:

        summary[f"{metric}_mean"] = float(
            episode_df[metric].mean()
        )

        summary[f"{metric}_std"] = float(
            episode_df[metric].std(ddof=1)
        )

    return summary, episode_df


# =========================================================
# JSON-SAFE METADATA
# =========================================================

def build_metadata(
    args,
    X_train,
    y_train,
    pair_y,
    vectorizer,
    best_k,
    validation_results,
    test_summary
):

    metadata = {

        "model_type": "Siamese-style pairwise Linear SVM",

        "feature_representation": (
            "absolute_tfidf_difference"
        ),

        "feature_count": int(X_train.shape[1]),

        "class_mapping": LABEL_MAP,

        "pair_label_mapping": {
            "0": "different_class",
            "1": "same_class"
        },

        "selected_k": int(best_k),

        "n_way": 2,

        "aggregation": (
            "mean_pairwise_same_class_decision_score"
        ),

        "pair_classifier": {
            "type": "LinearSVC",
            "parameters": SVM_PARAMETERS,
            "classes": [int(x) for x in [0, 1]],
            "decision_score_interpretation": (
                "positive = same class; negative = different class"
            )
        },

        "pair_training": {

            "pairs_per_relation": int(args.pairs_per_relation),

            "same_class_pairs": int(
                np.sum(pair_y == 1)
            ),

            "different_class_pairs": int(
                np.sum(pair_y == 0)
            ),

            "total_pairs": int(len(pair_y)),

            "training_samples": int(len(y_train))
        },

        "tfidf": {

            "vocabulary_size": int(
                len(vectorizer.get_feature_names_out())
            ),

            "parameters": {
                "max_features": vectorizer.max_features,
                "ngram_range": list(vectorizer.ngram_range),
                "lowercase": vectorizer.lowercase,
                "sublinear_tf": vectorizer.sublinear_tf,
                "use_idf": vectorizer.use_idf,
                "norm": vectorizer.norm,
                "dtype": str(vectorizer.dtype),
                "token_pattern": vectorizer.token_pattern
            },

            "fit_partition": (
                "NOT_VERIFIED: confirm vectorizer was fitted "
                "on training data only"
            )
        },

        "evaluation": {

            "protocol": "binary_few_shot_episodic",

            "episodes": int(args.episodes),

            "queries_per_class": int(
                args.queries_per_class
            ),

            "k_values": K_VALUES,

            "validation_seeds": {
                str(k): int(RANDOM_STATE + k)
                for k in K_VALUES
            },

            "test_seed": int(
                RANDOM_STATE + 1000 + best_k
            ),

            "support_query_disjoint_within_episode": True,

            "test_samples_reused_across_episodes": True,

            "test_metrics_interpretation": (
                "Mean and standard deviation across episodic "
                "evaluations, not one fixed prediction per "
                "test sample."
            )
        },

        "validation_results": {
            str(k): value
            for k, value in validation_results.items()
        },

        "test_results": test_summary,

        "random_state": RANDOM_STATE,

        "software_versions": {

            "python": platform.python_version(),

            "numpy": np.__version__,

            "pandas": pd.__version__,

            "scipy": scipy.__version__,

            "scikit_learn": sklearn.__version__,

            "joblib": joblib.__version__
        }
    }

    return metadata


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Binary few-shot Siamese-style SVM"
    )

    parser.add_argument(
        "--pairs-per-relation",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES
    )

    parser.add_argument(
        "--queries-per-class",
        type=int,
        default=QUERIES_PER_CLASS
    )

    args = parser.parse_args()

    validate_arguments(args, parser)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("BINARY FEW-SHOT SIAMESE SVM")
    print("=" * 65)

    # -----------------------------------------------------
    # Load fixed partitions
    # -----------------------------------------------------

    X_train, y_train, train_samples = load_partition("train")

    X_val, y_val, val_samples = load_partition("validation")

    X_test, y_test, test_samples = load_partition("test")

    print("\nPartition shapes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # -----------------------------------------------------
    # Feature dimension checks
    # -----------------------------------------------------

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train/validation feature dimensions differ")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions differ")

    # -----------------------------------------------------
    # Cross-partition leakage checks
    # -----------------------------------------------------

    validate_partition_disjointness({

        "train": train_samples,

        "validation": val_samples,

        "test": test_samples
    })

    # -----------------------------------------------------
    # Train using training partition only
    # -----------------------------------------------------

    pair_model, pair_y = train_pair_model(
        X_train,
        y_train,
        args.pairs_per_relation
    )

    # -----------------------------------------------------
    # Validation: select K
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION EPISODES")
    print("=" * 65)

    validation_results = {}

    for k in K_VALUES:

        summary, episodes_df = evaluate_episodes(
            pair_model,
            X_val,
            y_val,
            k,
            args.episodes,
            args.queries_per_class,
            RANDOM_STATE + k
        )

        validation_results[k] = summary

        episodes_df.to_csv(
            OUTPUT_DIR / f"validation_episodes_{k}shot.csv",
            index=False
        )

        print(f"\n{k}-shot:")
        print(json.dumps(summary, indent=2))

    # Highest mean F1; accuracy breaks ties.
    best_k = max(
        K_VALUES,
        key=lambda k: (
            validation_results[k]["f1_mean"],
            validation_results[k]["accuracy_mean"]
        )
    )

    print("\nSelected K:", best_k)

    # -----------------------------------------------------
    # Test: evaluate only selected K
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("HELD-OUT TEST EPISODES")
    print("=" * 65)

    test_summary, test_episodes = evaluate_episodes(
        pair_model,
        X_test,
        y_test,
        best_k,
        args.episodes,
        args.queries_per_class,
        RANDOM_STATE + 1000 + best_k
    )

    print(json.dumps(test_summary, indent=2))

    test_episodes.to_csv(
        OUTPUT_DIR / "test_episodes.csv",
        index=False
    )

    # -----------------------------------------------------
    # Load and validate existing vectorizer
    # -----------------------------------------------------

    vectorizer_path = FEATURE_DIR / "tfidf_vectorizer.joblib"

    if not vectorizer_path.exists():
        raise FileNotFoundError(vectorizer_path)

    vectorizer = joblib.load(vectorizer_path)

    vocabulary_size = len(
        vectorizer.get_feature_names_out()
    )

    if vocabulary_size != X_train.shape[1]:
        raise ValueError(
            "Vectorizer vocabulary does not match TF-IDF matrices"
        )

    if not hasattr(vectorizer, "idf_"):
        raise ValueError(
            "Loaded vectorizer does not contain fitted IDF weights"
        )

    # -----------------------------------------------------
    # Save model bundle
    # -----------------------------------------------------

    bundle = {

        "pair_classifier": pair_model,

        "vectorizer": vectorizer,

        "feature_count": X_train.shape[1],

        "feature_representation": (
            "absolute_tfidf_difference"
        ),

        "class_mapping": LABEL_MAP,

        "pair_label_mapping": {
            0: "different_class",
            1: "same_class"
        },

        "selected_k": best_k,

        "n_way": 2,

        "aggregation": (
            "mean_pairwise_same_class_decision_score"
        ),

        "pair_training": {
            "pairs_per_relation": args.pairs_per_relation,
            "same_class_pairs": int(np.sum(pair_y == 1)),
            "different_class_pairs": int(np.sum(pair_y == 0)),
            "training_samples": len(y_train)
        },

        "validation_results": validation_results,

        "test_results": test_summary,

        "random_state": RANDOM_STATE
    }

    model_path = OUTPUT_DIR / "siamese_svm_fewshot.joblib"

    joblib.dump(
        bundle,
        model_path
    )

    # -----------------------------------------------------
    # Save JSON-safe metadata
    # -----------------------------------------------------

    metadata = build_metadata(
        args,
        X_train,
        y_train,
        pair_y,
        vectorizer,
        best_k,
        validation_results,
        test_summary
    )

    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # Save validation K comparison
    # -----------------------------------------------------

    comparison = []

    for k in K_VALUES:

        comparison.append({

            "k_shot": k,

            **validation_results[k]
        })

    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / "validation_k_comparison.csv",
        index=False
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL RESULT")
    print("=" * 65)

    print("Selected K:", best_k)

    print(
        "Test metrics:",
        json.dumps(test_summary, indent=2)
    )

    print("Model saved:", model_path)

    print("Metadata saved:", metadata_path)

    print("\nSoftware versions:")
    print("Python:", platform.python_version())
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("Joblib:", joblib.__version__)


if __name__ == "__main__":
    main()
