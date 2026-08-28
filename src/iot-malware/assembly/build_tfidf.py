import json
from pathlib import Path
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_ROOT = Path("data/ngrams")
OUTPUT_ROOT= Path("data/tfidf")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

documents= []
labels = []
filenames= []

datasets = [("benign_mips",0),("malware_mips",1)]

for folder, label in datasets:
	directory = INPUT_ROOT / folder
	for file in sorted(directory.glob("*.asm")):
		with open(file, "r", encoding="utf-8") as f:
			text = " ".join(line.strip() for line in f if line.strip())
			
			documents.append(text)
			labels.append(label)
			filenames.append(file.name)

print(f"Documents Loaded : {len(documents)}")

vectorizer = TfidfVectorizer(
	tokenizer= str.split,
	preprocessor = None,
	token_pattern= None,
	lowercase= False
)

X= vectorizer.fit_transform(documents)

print(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")
print(f"TF-IDF Shape: {X.shape}")


sparse.save_npz(OUTPUT_ROOT / "tfidf_matrix.npz", X)
joblib.dump(labels, OUTPUT_ROOT / "labels.pkl")

joblib.dump(filenames, OUTPUT_ROOT/"filenames.pkl")

with open(OUTPUT_ROOT / "feature_names.json","w") as f:
	json.dump(vectorizer.get_feature_names_out().tolist(),f,indent=2)
	
joblib.dump(vectorizer, OUTPUT_ROOT / "tfidf_vectorizer.pkl")
print("\nTF-IDF Dataset Saved Successfully.")
