import os
import math
import matplotlib.pyplot as plt

# -----------------------------
# Shannon Entropy
# -----------------------------
def calculate_entropy(data):
    if not data:
        return 0

    counts = [0] * 256

    for b in data:
        counts[b] += 1

    entropy = 0
    total = len(data)

    for c in counts:
        if c:
            p = c / total
            entropy -= p * math.log2(p)

    return entropy


# -----------------------------
# Compute entropy of one file
# -----------------------------
def file_entropy(path):
    with open(path, "rb") as f:
        return calculate_entropy(f.read())


# -----------------------------
# Scan a directory
# -----------------------------
def collect_entropies(directory):
    entropies = []

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)

            try:
                e = file_entropy(path)
                entropies.append(e)
            except Exception:
                continue

    return entropies


# -----------------------------
# Directories
# -----------------------------
benign_dir = "/home/kali/iot-security-fl/data/raw/firmware"
malware_dir = "/home/kali/iot-security-fl/data/raw/malware_dataset"

benign_entropy = collect_entropies(benign_dir)
malware_entropy = collect_entropies(malware_dir)

print(f"Benign samples : {len(benign_entropy)}")
print(f"Malware samples: {len(malware_entropy)}")

print(f"Average Benign Entropy : {sum(benign_entropy)/len(benign_entropy):.2f}")
print(f"Average Malware Entropy: {sum(malware_entropy)/len(malware_entropy):.2f}")

# -----------------------------
# Histogram
# -----------------------------
plt.figure(figsize=(8,5))

plt.hist(
    benign_entropy,
    bins=20,
    alpha=0.6,
    label="Benign"
)

plt.hist(
    malware_entropy,
    bins=20,
    alpha=0.6,
    label="Malware"
)

plt.xlabel("Global Shannon Entropy")
plt.ylabel("Number of Binaries")
plt.title("Distribution of Global Entropy")
plt.legend()

plt.tight_layout()
plt.savefig("entropy_histogram.png", dpi=300)

# -----------------------------
# Boxplot
# -----------------------------
plt.figure(figsize=(5,5))

plt.boxplot(
    [benign_entropy, malware_entropy],
    tick_labels=["Benign", "Malware"]
)

plt.ylabel("Global Shannon Entropy")
plt.title("Entropy Comparison")

plt.tight_layout()
plt.savefig("entropy_boxplot.png", dpi=300)

plt.show()
