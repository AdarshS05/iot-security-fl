import sys
from pathlib import Path

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

# Test 00db9215cdb75735c1d6
path = '/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/00db9215cdb75735c1d6365b75b9cf41d8e11a30f3c5d6e0956276d10db842b1'
if not Path(path).exists():
    matches = list(Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips').glob('00db9215*'))
    if matches:
        path = str(matches[0])

print("Testing path:", path)
det = scorer.predict_details(path)
print("Details:", det)
