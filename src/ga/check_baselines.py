import json
import glob

files = sorted(glob.glob('/data/iot-malware-fl/iot-malware-abhi/reports/ga/batch_instr*/**/ga_results.json', recursive=True))
print(f"Found {len(files)} ga_results.json files in batch_instr*")
for path in files:
    try:
        d = json.load(open(path))
        b = d.get('baseline', {})
        sample_id = path.split('/')[-2]
        print(f"{sample_id[:20]}: lgbm={b.get('lightgbm_score'):.4f} (pred={b.get('lightgbm_prediction')}) | svm={b.get('siamese_svm_score'):+.4f} (pred={b.get('siamese_svm_prediction')})")
    except Exception as e:
        print(f"Error {path}: {e}")
