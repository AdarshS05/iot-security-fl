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

raw_mips = list(Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips').glob('*'))
print(f"Total MIPS binaries: {len(raw_mips)}")

results = []
for p in raw_mips:
    if p.is_file():
        try:
            det = scorer.predict_details(str(p))
            results.append((p.name, det['score'], det['prediction']))
            print(f"{p.name[:16]}: margin={det['score']:+.4f} pred={det['prediction']}")
        except Exception as e:
            print(f"{p.name[:16]}: err={e}")

results.sort(key=lambda x: x[1])
print("\nSorted by margin (closest to boundary / most vulnerable to flip):")
for name, margin, pred in results:
    print(f"  {name}: margin={margin:+.4f} pred={pred}")
