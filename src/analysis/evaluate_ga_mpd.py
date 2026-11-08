from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path.cwd()

FEATURES = [
    "global_entropy",
    "max_section_entropy",
    "mean_section_entropy",
    "entropy_variance",
    "entropy_high_ratio",
    "num_strings",
    "avg_string_len",
    "max_string_len",
    "num_ip_strings",
    "num_url_strings",
    "num_suspicious_strings",
    "num_high_entropy_strings",
]

VARIANT_CSV = ROOT / "results/ga_variant_features.csv"
VALIDATION_CSV = ROOT / (
    "results/uncertainty_adversarial_lgbm/"
    "lgbm_clean_validation_mpd.csv"
)

REPLICA_DIR = ROOT / (
    "results/uncertainty_adversarial_lgbm/lgbm/replicas"
)

CANONICAL_MODEL = ROOT / (
    "models/final_n1_adv/lightgbm_adversarial.txt"
)

OUTPUT = ROOT / "results/ga_variant_mpd.csv"

# Load GA variants
variants = pd.read_csv(VARIANT_CSV)

X = variants[FEATURES].astype(float).values

# Load the 25 existing bootstrap replicas
replica_paths = sorted(REPLICA_DIR.glob("model_*.txt"))

if len(replica_paths) != 25:
    raise RuntimeError(
        f"Expected 25 replicas, found {len(replica_paths)}"
    )

replicas = [
    lgb.Booster(model_file=str(path))
    for path in replica_paths
]

# Generate replica probabilities: N samples x 25 replicas
q = np.column_stack([
    model.predict(X) for model in replicas
])

# Correct MPD calculation
u0 = np.sqrt(np.mean(q ** 2, axis=1))
u1 = np.sqrt(np.mean((q - 1) ** 2, axis=1))
mpd = np.minimum(u0, u1)

# Load canonical updated model
canonical = lgb.Booster(
    model_file=str(CANONICAL_MODEL)
)

canonical_prob = canonical.predict(X)

# Operational prediction threshold
canonical_pred = (canonical_prob >= 0.5).astype(int)

# Obtain clean-validation T95 threshold
validation = pd.read_csv(VALIDATION_CSV)

if "mpd" not in validation.columns:
    raise ValueError("Validation CSV has no MPD column")

t95 = validation["mpd"].quantile(0.95)
t99 = validation["mpd"].quantile(0.99)

# Save results
variants["canonical_probability"] = canonical_prob
variants["canonical_prediction"] = canonical_pred
variants["u0"] = u0
variants["u1"] = u1
variants["mpd"] = mpd
variants["mpd_t95"] = t95
variants["mpd_t99"] = t99
variants["flag_t95"] = mpd >= t95
variants["flag_t99"] = mpd >= t99

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
variants.to_csv(OUTPUT, index=False)

print("\nGA MPD Evaluation")
print("-----------------")
print("Variants:", len(variants))
print("Replicas:", len(replicas))
print("Clean validation T95:", t95)
print("Clean validation T99:", t99)
print("Mean MPD:", np.mean(mpd))
print("Median MPD:", np.median(mpd))
print("MPD standard deviation:", np.std(mpd))
print("T95 flagged:", np.sum(mpd >= t95))
print("T99 flagged:", np.sum(mpd >= t99))
print("Maximum MPD:", np.max(mpd))
print("Saved:", OUTPUT)
