from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from scipy import sparse

def generate_pairs(X,y, pairs_per_sample=2, random_state=42):
	rng= np.random.default_rng(random_state)
	
	X = np.asarray(X)
	y = np.asarray(y)
	
	pair_features = []
	pair_labels = []
	
	seen_pairs = set()
	
	class0= np.where(y==0)[0]
	class1= np.where(y==1)[0]
	
	
	def add_pair(i,j,label):
		key = (min(int(i), int(j)), max(int(i), int(j)))
		if key in seen_pairs:
			return False
		seen_pairs.add(key)
		
		diff = np.abs(X[i]-X[j])
		pair_features.append(diff)
		pair_labels.append(label)
		return True
		

	for idx in range(len(y)):	
		if y[idx]==0:
			positive_pool = class0[class0 != idx]
			negative_pool = class1
		else:
			positive_pool = class1[class1 != idx]
			negative_pool = class0
		
		if len(positive_pool) > 0:
			n = min(pairs_per_sample, len(positive_pool))
			for j in rng.choice(positive_pool, size=n, replace=False):
				add_pair(idx,j,1)
		
		if len(negative_pool) > 0:
			n = min(pairs_per_sample, len(negative_pool))
			for j in rng.choice(negative_pool, size=n, replace=False):
				add_pair(idx,j,0)
	return (np.asarray(pair_features),np.asarray(pair_labels))

def load_tfidf_dataset(matrix_path, labels_path):
	X = sparse.load_npz(matrix_path)
	with open(labels_path,"rb")  as f:
		metadata= pickle.load(f)
		
	if isinstance(metadata, dict):
		labels= metadata["labels"]
		filenames= metadata.get("filenames",None)
	elif isinstance(metadata, tuple):
		filenames,labels=metadata
	elif(isinstance(metadata,list)):
		labels=metadata
		filenames= None
	
	if len(labels) != X.shape[0]:
		raise ValueError(f"Mismatch: TF-IDF matrix has {X.shape[0]} rows but label.csv has {len(labels)} entries")

	print("TF-IDF DATASET LOADED")
	print(f"Samples          : {X.shape[0]}")
	print(f"Vocabulary Size  : {X.shape[1]}")
	print(f"Benign Samples   : {sum(l == 0 for l in labels)}")
	print(f"Malware Samples  : {sum(l == 1 for l in labels)}")
	
	return X, labels, filenames

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATRIX = PROJECT_ROOT / "data" / "tfidf" / "tfidf_matrix.npz"
LABELS = PROJECT_ROOT / "data" / "tfidf" / "labels.pkl"

MODEL_DIR = PROJECT_ROOT / "models"
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

param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LinearSVC(random_state=42, max_iter=1000), param_grid, cv=5, scoring="f1")
grid.fit(pair_X_train, pair_y_train)
model = grid.best_estimator_
print("Best C:", grid.best_params_)

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
