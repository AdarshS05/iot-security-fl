from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from load_tfidf_dataset import load_tfidf_dataset
from pair_generator import generate_pairs

MATRIX = "../../data/tfidf/tfidf_matrix.npz"
LABELS = "../../data/tfidf/labels.pkl"

MODEL_DIR = Path("../../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "siamese_svm1.pkl"

print("Loading TF-IDF dataset...")
X,y,filename= load_tfidf_dataset(MATRIX, LABELS)
X= X.toarray()
y= np.asarray(y)

print("Generating training pairs...")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pair_X_train, pair_y_train = generate_pairs(X_train_raw,y_train_raw,pairs_per_sample=2)
pair_X_test, pair_y_test = generate_pairs(X_test_raw,y_test_raw,pairs_per_sample=2)

print(f"\nTrain pairs: {len(pair_y_train)} "
f"(same-class={sum(pair_y_train == 1)}, diff-class={sum(pair_y_train == 0)})")
print(f"Test pairs : {len(pair_y_test)} "
          f"(same-class={sum(pair_y_test == 1)}, diff-class={sum(pair_y_test == 0)})")


scaler = StandardScaler()

pair_X_train = scaler.fit_transform(pair_X_train)
pair_X_test = scaler.transform(pair_X_test)

from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LinearSVC(random_state=42, max_iter=20000), param_grid, cv=5, scoring="f1")
grid.fit(pair_X_train, pair_y_train)
model = grid.best_estimator_
print("Best C:", grid.best_params_)

print("Training Siamese SVM...")

model = LinearSVC(random_state=42, max_iter=5000)
model.fit(pair_X_train,pair_y_train)

pred= model.predict(pair_X_test)

print("\nEvaluation")

print(f"Accuracy : {accuracy_score(pair_y_test, pred):.4f}")
print(f"Precision: {precision_score(pair_y_test, pred):.4f}")
print(f"Recall   : {recall_score(pair_y_test, pred):.4f}")
print(f"F1 Score : {f1_score(pair_y_test, pred):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(pair_y_test, pred))

print("\nClassification Report")
print(classification_report(pair_y_test, pred))

joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
print(f"\nModel + scaler saved to {MODEL_PATH}")
