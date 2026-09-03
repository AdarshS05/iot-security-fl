import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# Load the dataset
# ==============================
df = pd.read_csv("data/processed/output.csv")

# ==============================
# Separate features and labels
# Replace 'Binary Class' with your label column name
# ==============================
label_column = "label"

X = df.drop(columns=[label_column])
y = df[label_column]

# If there are non-numeric columns besides the label, remove them
X = X.select_dtypes(include=["number"])

# ==============================
# Standardize features
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Perform PCA
# ==============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ==============================
# Create DataFrame for plotting
# ==============================
pca_df = pd.DataFrame({
    "Principal Component 1": X_pca[:, 0],
    "Principal Component 2": X_pca[:, 1],
    "Binary Class": y.values
})

# ==============================
# Plot
# ==============================
plt.figure(figsize=(8,6))

colors = {
    "Benign": "#4C72B0",
    "Malware": "#DD4A48"
}

markers = {
    "Benign": "o",
    "Malware": "X"
}

for cls in pca_df["Binary Class"].unique():
    subset = pca_df[pca_df["Binary Class"] == cls]

    plt.scatter(
        subset["Principal Component 1"],
        subset["Principal Component 2"],
        c=colors.get(cls, "gray"),
        marker=markers.get(cls, "o"),
        s=40,
        alpha=0.8,
        label=cls
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.legend(title="Binary Class")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================
# Explained variance
# ==============================
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
