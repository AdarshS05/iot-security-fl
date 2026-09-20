from pathlib import Path
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
TFIDF_DIR = PROJECT_ROOT / "data" / "tfidf"

OUTPUT_DIR = PROJECT_ROOT / "results" / "model_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LIGHTGBM_MODEL = MODEL_DIR / "lightgbm_model.pkl"
LIGHTGBM_METADATA = MODEL_DIR / "lightgbm_metadata.json"

SIAMESE_MODEL = MODEL_DIR / "siamese_svm1.pkl"

TFIDF_MATRIX = TFIDF_DIR / "tfidf_matrix.npz"
TFIDF_LABELS = TFIDF_DIR / "labels.pkl"


# ============================================================
# Utility
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# LIGHTGBM
# ============================================================

def load_lightgbm_data():
    artifact = joblib.load(LIGHTGBM_MODEL)

    model = artifact["model"]
    threshold = artifact["best_threshold"]
    feature_names = artifact["feature_names"]

    metadata = None

    if LIGHTGBM_METADATA.exists():
        with open(LIGHTGBM_METADATA, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, threshold, feature_names, metadata


def prepare_lightgbm_test_data(feature_names):
    csv_path = PROJECT_ROOT / "data" / "processed" / "benign_dataset.csv"

    df = pd.read_csv(csv_path)

    # Same exclusions used during training
    identifier_columns = [
        "filename", "file_name", "file_path",
        "md5", "sha256", "sha1", "top_opcodes"
    ]

    constant_columns = [
        "elf_type", "os_abi", "endianness",
        "load_segments", "has_debug_info",
        "has_packer_signature"
    ]

    shortcut_columns = [
        "is_statically_linked",
        "program_headers",
        "has_interp",
        "has_dynamic",
        "num_libraries",
        "num_imported_functions",
        "num_dynamic_symbols",
        "import_export_ratio",
        "has_relro",
        "num_uncommon_libraries",
        "num_suspicious_imports",
        "num_sections",
        "section_headers_present",
        "num_nonstandard_sections",
        "num_executable_sections",
        "text_size_ratio",
        "max_section_entropy",
        "mean_section_entropy",
        "num_static_symbols",
        "is_stripped",
        "rwx_segments",
        "has_gnu_stack",
        "num_exported_functions",
    ]

    columns_to_drop = (
        identifier_columns
        + constant_columns
        + shortcut_columns
        + ["label"]
    )

    columns_to_drop = [
        c for c in columns_to_drop
        if c in df.columns
    ]

    X = df.drop(columns=columns_to_drop)

    for feature in [
        "entropy_variance",
        "entry_point",
        "file_size_bytes",
    ]:
        if feature in X.columns:
            X = X.drop(columns=[feature])

    y = df["label"]

    # Ensure exact feature order expected by the model
    X = X[feature_names]

    categorical_cols = X.select_dtypes(
        include=["object", "bool"]
    ).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype("category")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_test, np.asarray(y_test)


def plot_lightgbm_confusion(model, threshold, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malware"],
    )

    disp.plot()
    plt.title("LightGBM Confusion Matrix")
    save_plot("lightgbm_confusion_matrix.png")


def plot_lightgbm_roc(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        label=f"LightGBM (AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LightGBM ROC Curve")
    plt.legend()

    save_plot("lightgbm_roc_curve.png")


def plot_lightgbm_pr(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(
        recall,
        precision,
        label=f"LightGBM (AUC = {pr_auc:.4f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("LightGBM Precision-Recall Curve")
    plt.legend()

    save_plot("lightgbm_precision_recall.png")


def plot_lightgbm_probabilities(model, X_test, y_test, threshold):
    probabilities = model.predict_proba(X_test)[:, 1]

    benign = probabilities[y_test == 0]
    malware = probabilities[y_test == 1]

    plt.figure()

    plt.hist(
        benign,
        bins=30,
        alpha=0.6,
        label="Benign",
    )

    plt.hist(
        malware,
        bins=30,
        alpha=0.6,
        label="Malware",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.4f}",
    )

    plt.xlabel("Predicted P(Malware)")
    plt.ylabel("Number of Samples")
    plt.title("LightGBM Prediction Probability Distribution")
    plt.legend()

    save_plot("lightgbm_probability_distribution.png")


def plot_lightgbm_feature_importance(model):
    importance = model.booster_.feature_importance(
        importance_type="gain"
    )

    names = model.feature_name_

    df = pd.DataFrame({
        "feature": names,
        "importance": importance,
    })

    df = df.sort_values(
        "importance",
        ascending=False,
    ).head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        df["feature"][::-1],
        df["importance"][::-1],
    )

    plt.xlabel("Gain")
    plt.ylabel("Feature")
    plt.title("LightGBM Top 20 Features by Gain")

    save_plot("lightgbm_feature_importance.png")


# ============================================================
# SIAMESE SVM
# ============================================================

def generate_pairs(X, y, pairs_per_sample=2, random_state=42):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    pair_features = []
    pair_labels = []

    seen_pairs = set()

    class0 = np.where(y == 0)[0]
    class1 = np.where(y == 1)[0]

    def add_pair(i, j, label):
        key = (
            min(int(i), int(j)),
            max(int(i), int(j)),
        )

        if key in seen_pairs:
            return False

        seen_pairs.add(key)

        diff = np.abs(X[i] - X[j])

        pair_features.append(diff)
        pair_labels.append(label)

        return True

    for idx in range(len(y)):

        if y[idx] == 0:
            positive_pool = class0[class0 != idx]
            negative_pool = class1
        else:
            positive_pool = class1[class1 != idx]
            negative_pool = class0

        if len(positive_pool) > 0:
            n = min(
                pairs_per_sample,
                len(positive_pool),
            )

            for j in rng.choice(
                positive_pool,
                size=n,
                replace=False,
            ):
                add_pair(idx, j, 1)

        if len(negative_pool) > 0:
            n = min(
                pairs_per_sample,
                len(negative_pool),
            )

            for j in rng.choice(
                negative_pool,
                size=n,
                replace=False,
            ):
                add_pair(idx, j, 0)

    return (
        np.asarray(pair_features),
        np.asarray(pair_labels),
    )


def load_siamese_test_data():
    X = sparse.load_npz(TFIDF_MATRIX)

    with open(TFIDF_LABELS, "rb") as f:
        y = pickle.load(f)

    X = X.toarray()
    y = np.asarray(y)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pair_X_test, pair_y_test = generate_pairs(
        X_test,
        y_test,
        pairs_per_sample=2,
        random_state=42,
    )

    return pair_X_test, pair_y_test


def plot_siamese_confusion(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)

    predictions = model.predict(X_scaled)

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Different Class",
            "Same Class",
        ],
    )

    disp.plot()

    plt.title("Siamese SVM Confusion Matrix")

    save_plot("siamese_confusion_matrix.png")


def plot_siamese_roc(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)

    scores = model.decision_function(X_scaled)

    fpr, tpr, _ = roc_curve(
        y_test,
        scores,
    )

    roc_auc = roc_auc_score(
        y_test,
        scores,
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"Siamese SVM (AUC = {roc_auc:.4f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Siamese SVM ROC Curve")
    plt.legend()

    save_plot("siamese_roc_curve.png")


def plot_siamese_pr(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)

    scores = model.decision_function(X_scaled)

    precision, recall, _ = precision_recall_curve(
        y_test,
        scores,
    )

    pr_auc = auc(
        recall,
        precision,
    )

    plt.figure()

    plt.plot(
        recall,
        precision,
        label=f"Siamese SVM (AUC = {pr_auc:.4f})",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Siamese SVM Precision-Recall Curve")
    plt.legend()

    save_plot("siamese_precision_recall.png")


def plot_siamese_scores(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)

    scores = model.decision_function(X_scaled)

    same = scores[y_test == 1]
    different = scores[y_test == 0]

    plt.figure()

    plt.hist(
        different,
        bins=30,
        alpha=0.6,
        label="Different Class",
    )

    plt.hist(
        same,
        bins=30,
        alpha=0.6,
        label="Same Class",
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Decision boundary",
    )

    plt.xlabel("SVM Decision Score")
    plt.ylabel("Number of Pairs")
    plt.title("Siamese SVM Decision Score Distribution")
    plt.legend()

    save_plot("siamese_decision_scores.png")


def plot_siamese_pair_balance(y_test):
    same = int(np.sum(y_test == 1))
    different = int(np.sum(y_test == 0))

    plt.figure()

    plt.bar(
        ["Same Class", "Different Class"],
        [same, different],
    )

    plt.ylabel("Number of Pairs")
    plt.title("Siamese Test Pair Distribution")

    save_plot("siamese_pair_distribution.png")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING MODEL EVALUATION PLOTS")
    print("=" * 60)

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    print("\n[LightGBM]")

    model, threshold, feature_names, metadata = (
        load_lightgbm_data()
    )
    
    

    X_test, y_test = prepare_lightgbm_test_data(
        feature_names
    )
    
    y_test = np.array([
        1 if str(label).lower() == "malware" else 0
        for label in y_test
    ])

    plot_lightgbm_confusion(
        model,
        threshold,
        X_test,
        y_test,
    )

    plot_lightgbm_roc(
        model,
        X_test,
        y_test,
    )

    plot_lightgbm_pr(
        model,
        X_test,
        y_test,
    )

    plot_lightgbm_probabilities(
        model,
        X_test,
        y_test,
        threshold,
    )

    plot_lightgbm_feature_importance(
        model,
    )

    # --------------------------------------------------------
    # Siamese SVM
    # --------------------------------------------------------

    print("\n[Siamese SVM]")

    siamese_artifact = joblib.load(
        SIAMESE_MODEL
    )

    siamese_model = siamese_artifact["model"]
    siamese_scaler = siamese_artifact["scaler"]

    pair_X_test, pair_y_test = (
        load_siamese_test_data()
    )

    plot_siamese_confusion(
        siamese_model,
        siamese_scaler,
        pair_X_test,
        pair_y_test,
    )

    plot_siamese_roc(
        siamese_model,
        siamese_scaler,
        pair_X_test,
        pair_y_test,
    )

    plot_siamese_pr(
        siamese_model,
        siamese_scaler,
        pair_X_test,
        pair_y_test,
    )

    plot_siamese_scores(
        siamese_model,
        siamese_scaler,
        pair_X_test,
        pair_y_test,
    )

    plot_siamese_pair_balance(
        pair_y_test,
    )

    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
