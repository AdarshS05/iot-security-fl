import sys
import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GA_DIRS = [
    ROOT / "reports/ga/full_dataset_run3",
    ROOT / "reports/ga/full_dataset_run3_adv_model",
]

spec = importlib.util.spec_from_file_location(
    "elf_extractor", ROOT / "src/tabular/elf_feature_extractor.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rows = []
seen = set()

for ga_dir in GA_DIRS:
    if not ga_dir.exists():
        print("Missing:", ga_dir)
        continue

    for binary in ga_dir.rglob("*.elf"):
        if binary in seen:
            continue
        seen.add(binary)

        try:
            features = module.extract_features(str(binary))
            features["variant_path"] = str(binary)
            features["ga_run"] = ga_dir.name
            rows.append(features)
        except Exception as exc:
            print(f"[SKIP] {binary}: {exc}")

if not rows:
    sys.exit("No features extracted.")

out = ROOT / "results/ga_variant_features.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)

print(f"Extracted {len(rows)} variants")
print("Saved:", out)
