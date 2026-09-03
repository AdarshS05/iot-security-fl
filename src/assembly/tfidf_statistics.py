import json
import joblib
import numpy as np
from scipy import sparse
from collections import Counter

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

TFIDF_DIR = "data/tfidf"

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

X = sparse.load_npz(f"{TFIDF_DIR}/tfidf_matrix.npz")

labels = joblib.load(f"{TFIDF_DIR}/labels.pkl")
filenames = joblib.load(f"{TFIDF_DIR}/filenames.pkl")
vectorizer = joblib.load(f"{TFIDF_DIR}/tfidf_vectorizer.pkl")

with open(f"{TFIDF_DIR}/feature_names.json", "r") as f:
    feature_names = json.load(f)

# -------------------------------------------------------
# Basic Dataset Statistics
# -------------------------------------------------------

num_docs = X.shape[0]
vocab_size = X.shape[1]

benign = labels.count(0)
malware = labels.count(1)

# -------------------------------------------------------
# Document Length Statistics
# -------------------------------------------------------

# Number of unique n-grams present in each document
doc_lengths = np.diff(X.indptr)

avg_doc_length = np.mean(doc_lengths)
min_doc_length = np.min(doc_lengths)
max_doc_length = np.max(doc_lengths)

# -------------------------------------------------------
# Matrix Sparsity
# -------------------------------------------------------

nonzero = X.nnz
total = X.shape[0] * X.shape[1]

sparsity = 100 * (1 - nonzero / total)

# -------------------------------------------------------
# Top N-grams
# -------------------------------------------------------

term_frequencies = np.asarray(X.sum(axis=0)).ravel()

top_indices = np.argsort(term_frequencies)[::-1][:20]

print("=" * 65)
print("TF-IDF DATASET REPORT")
print("=" * 65)

print(f"Benign Files                : {benign}")
print(f"Malware Files               : {malware}")
print(f"Total Documents             : {num_docs}")

print()

print(f"Vocabulary Size             : {vocab_size}")

print(f"Average N-grams / File      : {avg_doc_length:.2f}")
print(f"Maximum Document Length     : {max_doc_length}")
print(f"Minimum Document Length     : {min_doc_length}")

print()

print(f"TF-IDF Matrix Shape         : {X.shape}")

print(f"Non-zero Entries            : {nonzero:,}")

print(f"Matrix Sparsity             : {sparsity:.2f}%")

print()

print("=" * 65)
print("TOP 20 MOST IMPORTANT N-GRAMS")
print("=" * 65)

for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:2d}. {feature_names[idx]:40s} {term_frequencies[idx]:.3f}")

print("=" * 65)
