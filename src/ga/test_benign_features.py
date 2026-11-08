import sys
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import SiameseSVMScorer

model_path = '/data/iot-malware-fl/iot-malware-abhi/models/siamese_svm_n1/siamese_svm_fewshot.joblib'
matrix_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/X_train.npz'
labels_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/train_samples.csv'

scorer = SiameseSVMScorer(
    model_path=model_path,
    matrix_path=matrix_path,
    labels_path=labels_path,
    random_state=42
)

# Benign support samples:
b_vecs = scorer.support_vectors[0].toarray() # (3, 30000)
# Malware support samples:
m_vecs = scorer.support_vectors[1].toarray() # (3, 30000)

w = scorer.model.coef_[0] # (30000,)
feature_names = np.array(scorer.vectorizer.get_feature_names_out())

# Average benign prototype vs malware prototype
b_mean = b_vecs.mean(axis=0)
m_mean = m_vecs.mean(axis=0)

# What features are present in benign prototypes but NOT in malware prototypes?
diff = b_mean - m_mean
top_benign_features = np.argsort(diff)[-20:]
print("Top 20 features strongest in Benign support prototypes vs Malware:")
for idx in reversed(top_benign_features):
    print(f"  feat {idx:5d}: diff={diff[idx]:+.4f} (b={b_mean[idx]:.4f}, m={m_mean[idx]:.4f}) | {feature_names[idx]}")

# When query has feature j, |q_j - s_j| for benign vs malware:
# If q_j matches b_mean_j, |q_j - b_mean_j| is small.
