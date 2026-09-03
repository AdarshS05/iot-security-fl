from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import pickle
from scipy import sparse

def generate_pairs(X,y, pairs_per_sample=2, random_state=42):
	rng= np.random.default_rng(random_state)
	
	pair_features = []
	pair_labels = []
	
	
	y=np.asarray(y)
	class0= np.where(y==0)[0]
	class1= np.where(y==1)[0]
	
	seen_pairs = set()
	
	def add_pair(i,j,label):
		key = (min(i,j),max(i,j))
		if key in seen_pairs:
			return False
		seen_pairs.add(key)
		
		diff = np.abs(X[i]-X[j])
		pair_features.append(diff)
		pair_labels.append(label)
		return True
		

	for idx in range(len(y)):
		current = X[idx]
		
		if y[idx]==0:
			positive_pool = class0[class0 != idx]
			negative_pool = class1
		else:
			positive_pool = class1[class1 != idx]
			negative_pool = class0
	
		def sample_unique_partners(pool,k):
			pool= rng.permutation(pool)
			chosen = []
			
			for candidate in pool:
				key = (min(idx,candidate), max(idx,candidate))
				if key not in seen_pairs:
					chosen.append(candidate)
					
					
		pos_choices = rng.choice(positive_pool, size=pairs_per_sample, replace=False if len(positive_pool) >= pairs_per_sample else True)
		
		for j in pos_choices:
			diff = np.abs(current-X[j])
			pair_features.append(diff)
			pair_labels.append(1)
		
		neg_choices = rng.choice(negative_pool, size=pairs_per_sample, replace=False if len(negative_pool) >= pairs_per_sample else True)
		
		for j in neg_choices:
			diff= np.abs(current-X[j])
			pair_features.append(diff)
			pair_labels.append(0)
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
		raise ValueError(f"Mismatch: TF-IDF matrix has {X.shape[0]} rows but label.csv has {len(df)} entries")

	print("TF-IDF DATASET LOADED")
	print(f"Samples          : {X.shape[0]}")
	print(f"Vocabulary Size  : {X.shape[1]}")
	print(f"Benign Samples   : {sum(l == 0 for l in labels)}")
	print(f"Malware Samples  : {sum(l == 1 for l in labels)}")
	
	return X, labels, filenames


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
