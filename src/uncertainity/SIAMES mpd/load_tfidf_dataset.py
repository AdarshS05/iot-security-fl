import pandas as pd
import pickle
from scipy import sparse

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
	
if __name__=="__main__":	

	MATRIX_PATH= "../../data/tfidf/tfidf_matrix.npz"
	LABELS_PATH= "../../data/tfidf/labels.pkl"
	X,y, filenames= load_tfidf_dataset(MATRIX_PATH,LABELS_PATH)
