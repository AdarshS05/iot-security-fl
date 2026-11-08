import sys
from pathlib import Path
import shutil

sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/ga')
sys.path.insert(0, '/data/iot-malware-fl/iot-malware-abhi/src/assembly2')

from genetic_algorithm_mutator import (
    SiameseSVMScorer,
    find_code_cave,
    find_trampoline_insertion_points
)

model_path = '/data/iot-malware-fl/iot-malware-abhi/models/siamese_svm_n1/siamese_svm_fewshot.joblib'
matrix_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/X_train.npz'
labels_path = '/data/iot-malware-fl/iot-malware-abhi/data/features/tfidf_n1/train_samples.csv'

scorer = SiameseSVMScorer(
    model_path=model_path,
    matrix_path=matrix_path,
    labels_path=labels_path,
    random_state=42
)

candidates = [
    Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/006b8e5bcf14a22fba1155e87003ebf9ec003d9d386cdd4a7a680b96efc8f4ca'),
    Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/0068070679dc07f54aecc4073bfde92ef9c48438b86eba7c1df979ffb5abca49'),
    Path('/data/iot-malware-fl/iot-malware-abhi/data/raw/raw_mips/1a19777cc215e563f04f96d2bd8095fed8a3f04167426beaa61968ce682580eb'),
]

for p in candidates:
    print(f"\n--- Testing {p.name[:20]} ---")
    det = scorer.predict_details(str(p))
    print(f"Baseline SVM margin: {det['score']:+.4f} prediction: {det['prediction']}")
    cave = find_code_cave(p, min_size=16)
    if cave:
        aligned_offset = (cave['file_offset'] + 3) & ~3
        print(f"Cave found: offset={hex(cave['file_offset'])} aligned={hex(aligned_offset)} usable={cave['usable']}")
    else:
        print("No cave found")
    try:
        pairs = find_trampoline_insertion_points(p)
        print(f"Trampoline pairs found: {len(pairs)}")
    except Exception as e:
        print(f"Trampoline error: {e}")
