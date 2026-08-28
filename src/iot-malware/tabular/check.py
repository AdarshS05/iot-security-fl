import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
 
# Columns that should never be treated as candidate features.
IDENTIFIER_COLUMNS = [
    "filename", "file_name", "file_path", "md5", "sha256", "sha1", "top_opcodes"
]
 
# Already-known shortcuts (kept here so the report shows them as
# "known", not flagged as new findings).
KNOWN_SHORTCUT_COLUMNS = [
    "is_statically_linked", "program_headers", "has_interp", "has_dynamic",
    "num_libraries", "num_imported_functions", "num_dynamic_symbols",
    "import_export_ratio", "has_relro", "num_uncommon_libraries",
    "num_suspicious_imports", "num_sections", "section_headers_present",
    "num_nonstandard_sections", "num_executable_sections", "text_size_ratio",
    "max_section_entropy", "mean_section_entropy", "num_static_symbols",
    "is_stripped", "rwx_segments", "has_gnu_stack", "num_exported_functions",
]
 
AUC_FLAG_THRESHOLD = 0.90   # single-feature AUC at/above this = suspicious
KS_FLAG_THRESHOLD = 0.80    # KS stat at/above this = suspicious
CONST_FLAG_RATIO = 0.98     # if >=98% of a class shares one value = suspicious
 
 
def scan_features(csv_path, label_col="label"):
    df = pd.read_csv(csv_path)
 
    if "sha256" in df.columns:
        df = df.drop_duplicates(subset=["sha256"])
    elif "md5" in df.columns:
        df = df.drop_duplicates(subset=["md5"])
    else:
        df = df.drop_duplicates()
 
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in {csv_path}")
 
    if pd.api.types.is_numeric_dtype(df[label_col]) and not pd.api.types.is_bool_dtype(df[label_col]):
        y = df[label_col].astype(int)
    else:
        y = (df[label_col].astype(str).str.lower() == "malware").astype(int)
 
    candidate_cols = [
        c for c in df.columns
        if c != label_col and c not in IDENTIFIER_COLUMNS
    ]
 
    rows = []
    for col in candidate_cols:
        series = df[col]
 
        # Coerce booleans/categoricals to numeric for AUC/KS purposes.
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(int)
        elif not pd.api.types.is_numeric_dtype(series):
            # Skip free-text / non-ordinal columns -- AUC isn't meaningful.
            continue
 
        if series.isna().any():
            series = series.fillna(series.median())
 
        class0 = series[y == 0]
        class1 = series[y == 1]
 
        # Single-feature AUC: how well does this feature alone rank-order
        # the classes? Works regardless of scale.
        try:
            auc = roc_auc_score(y, series)
            auc = max(auc, 1 - auc)  # direction-agnostic
        except ValueError:
            auc = np.nan
 
        # KS statistic: max gap between the two classes' empirical CDFs.
        try:
            ks_stat = ks_2samp(class0, class1).statistic
        except ValueError:
            ks_stat = np.nan
 
        # Near-constant-within-a-class check (e.g. rwx_segments == 0 for
        # 100% of benign): dominant-value ratio per class.
        def dominant_ratio(s):
            if len(s) == 0:
                return np.nan
            return s.value_counts(normalize=True).iloc[0]
 
        dom0 = dominant_ratio(class0)
        dom1 = dominant_ratio(class1)
 
        flagged = (
            (not np.isnan(auc) and auc >= AUC_FLAG_THRESHOLD) or
            (not np.isnan(ks_stat) and ks_stat >= KS_FLAG_THRESHOLD) or
            (not np.isnan(dom0) and dom0 >= CONST_FLAG_RATIO) or
            (not np.isnan(dom1) and dom1 >= CONST_FLAG_RATIO)
        )
 
        rows.append({
            "feature": col,
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "ks_stat": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
            "dominant_val_ratio_class0": round(dom0, 4) if not np.isnan(dom0) else None,
            "dominant_val_ratio_class1": round(dom1, 4) if not np.isnan(dom1) else None,
            "known_shortcut": col in KNOWN_SHORTCUT_COLUMNS,
            "flagged": flagged,
        })
 
    report = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    return report
 
 
def main():
    parser = argparse.ArgumentParser(description="Scan a dataset for single-feature leakage.")
    parser.add_argument("--csv_path", default="data/processed/benign_dataset.csv")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()
 
    report = scan_features(args.csv_path, args.label_col)
 
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
 
    print(f"\nFeature leakage scan: {args.csv_path}\n" + "=" * 60)
    print(report.to_string(index=False))
 
    new_flags = report[report["flagged"] & ~report["known_shortcut"]]
    print("\n" + "=" * 60)
    if len(new_flags):
        print(f"NEW suspicious features not already in SHORTCUT_COLUMNS ({len(new_flags)}):")
        for _, row in new_flags.iterrows():
            print(f"  - {row['feature']}  (AUC={row['auc']}, KS={row['ks_stat']})")
    else:
        print("No new suspicious features found beyond the known shortcut columns.")
 
 
if __name__ == "__main__":
    main()
