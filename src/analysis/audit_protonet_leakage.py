import json
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/experiments/f1_protonet"
PREPARED = DATA / "prepared"

with open(PREPARED / "split_manifest.json") as f:
    manifest = json.load(f)

splits = manifest["splits"]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_duplicates(kind):
    print(f"\n===== {kind.upper()} EXACT DUPLICATE AUDIT =====")

    locations = defaultdict(list)

    for split, records in splits.items():
        for record in records:
            path = DATA / kind / record["class"] / record["filename"]

            if not path.exists():
                print("MISSING:", path)
                continue

            locations[digest(path)].append(
                (split, record["class"], record["filename"])
            )

    cross_split = []
    within_split = []

    for entries in locations.values():
        if len(entries) < 2:
            continue

        split_names = set(x[0] for x in entries)

        if len(split_names) > 1:
            cross_split.append(entries)
        else:
            within_split.append(entries)

    print("Unique contents:", len(locations))
    print("Cross-split duplicate groups:", len(cross_split))
    print("Within-split duplicate groups:", len(within_split))

    for group in cross_split[:10]:
        print("\nDUPLICATE GROUP:")
        for item in group:
            print(" ", item)

    return len(cross_split)


ngram_duplicates = check_duplicates("ngrams")
normalized_duplicates = check_duplicates("normalized")

print("\n===== SUMMARY =====")
print("N-gram cross-split duplicates:", ngram_duplicates)
print("Normalized cross-split duplicates:", normalized_duplicates)
