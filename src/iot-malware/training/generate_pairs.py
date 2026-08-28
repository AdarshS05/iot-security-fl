import random
import numpy as np

random.seed(42)

def generate_pairs(X,y):
	y=np.asarray(y)
	class_indices={}
	
	for idx,label in enumerate(y):
		class_indices.setdefault(label,[]).append(idx)
	
	positive_pairs = []
	negative_pairs = []
	
	labels = list(class_indices.keys())
	
	for label in labels:
		indices = class_indices[label]
		for i in range(len(indices) -1):
			positive_pairs.append((indices[i], indices[i+1],1))
			
	positives=len(positive_pairs)
	while len(negative_pairs) < positives:
		label1,label2 = random.sample(labels,2)
		idx1 = random.choice(class_indices[label1])
		idx2 = random.choice(class_indices[label2])
		negative_pairs.append((idx1,idx2,0))
	
	pairs= positive_pairs + negative_pairs
	random.shuffle(pairs)
	
	left = [p[0] for p in pairs]
	right = [p[1] for p in pairs]
	labels = np.array([p[2] for p in pairs])
	return left, right, labels
	
from load_tfidf_dataset import load_tfidf_dataset

MATRIX = "../../data/tfidf/tfidf_matrix.npz"
LABELS = "../../data/tfidf/labels.pkl"

X, y, filenames = load_tfidf_dataset(MATRIX, LABELS)

left, right, pair_labels = generate_pairs(X, y)

print("Number of pairs :", len(pair_labels))
print("Positive :", pair_labels.sum())
print("Negative :", len(pair_labels)-pair_labels.sum())
