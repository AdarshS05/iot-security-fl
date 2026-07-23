import os
import math
import lief
import pandas as pd


def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    entropy = 0.0
    total = len(data)

    counts = [0] * 256
    for b in data:
        counts[b] += 1

    for c in counts:
        if c:
            p = c / total
            entropy -= p * math.log2(p)

    return entropy


def extract_features(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    binary = lief.parse(file_path)

    if binary is None or not isinstance(binary, lief.ELF.Binary):
        return {"error": "Invalid ELF"}

    with open(file_path, "rb") as f:
        raw = f.read()

    # --------------------------------------------------
    # Global Entropy
    # --------------------------------------------------
    global_entropy = calculate_entropy(raw)

    # --------------------------------------------------
    # Segment Statistics
    # --------------------------------------------------
    load_segments= 0
    rwx_segments = 0
    largest_segment=0

    has_interp = 0
    has_dynamic = 0
    has_relro = 0
    has_gnu_stack = 0

    for seg in binary.segments:
        largest_segment = max(largest_segment, seg.physical_size)

        if seg.type == lief.ELF.Segment.TYPE.LOAD:
            load_segments += 1

        if seg.type == lief.ELF.Segment.TYPE.INTERP:
            has_interp = 1

        if seg.type == lief.ELF.Segment.TYPE.DYNAMIC:
            has_dynamic = 1

        if seg.type == lief.ELF.Segment.TYPE.GNU_RELRO:
            has_relro = 1

        if seg.type == lief.ELF.Segment.TYPE.GNU_STACK:
            has_gnu_stack = 1

        r = seg.has(lief.ELF.Segment.FLAGS.R)
        w = seg.has(lief.ELF.Segment.FLAGS.W)
        x = seg.has(lief.ELF.Segment.FLAGS.X)


        if r and w and x:
            rwx_segments += 1

    # --------------------------------------------------
    # Dynamic Information
    # --------------------------------------------------
    libraries = list(binary.libraries)

    imported_symbols = [
        s.name for s in binary.imported_symbols if s.imported
    ]

    dynamic_entry_count = len(binary.dynamic_entries)

    # --------------------------------------------------
    # Binary Security Features
    # --------------------------------------------------
    has_textrel = any(
        entry.tag == lief.ELF.DynamicEntry.TAG.TEXTREL
        for entry in binary.dynamic_entries
    )

    # --------------------------------------------------
    # Feature Dictionary
    # --------------------------------------------------
    features = {
        # Label
        "filename": os.path.basename(file_path),

        # File
        "file_size_bytes": len(raw),
        "global_entropy": round(global_entropy, 3),

        # ELF Header
        "architecture": binary.header.machine_type.name,
        "elf_type": binary.header.file_type.name,
        "os_abi": binary.header.identity_os_abi.name,

        # Program Header
        "program_headers": len(binary.segments),
        "load_segments": load_segments,
        "rwx_segments": rwx_segments,

        # Dynamic Loader
        "has_interp": has_interp,
        "has_dynamic": has_dynamic,
        "has_relro": has_relro,
        "has_gnu_stack": has_gnu_stack,

        "num_libraries": len(libraries),
        "num_imported_functions": len(imported_symbols),

        # Symbols
        "num_dynamic_symbols": len(binary.dynamic_symbols),
    }

    return features


def collect_features(directory):
    dataset = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                features = extract_features(path)
                # Ensure we only append successful parses to avoid breaking pandas
                if "error" not in features:
                    dataset.append(features)
            except Exception:
                continue
    return dataset


def summarise(dataset):
    df = pd.DataFrame(dataset)
    numeric = df.select_dtypes(include="number")
    
    summary = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std": numeric.std(),
    })
    
    return summary


if __name__ == "__main__":
    benign_dir = "/home/kali/iot-security-fl/data/raw/benign_mips/executables/"
    malware_dir = "/home/kali/malware_baazar/"
    features = [
        # Label
        "filename",

        # File
        "file_size_bytes",
        "global_entropy",

        # ELF Header
        "architecture",
        "elf_type",
        "os_abi",

        # Program Header
        "program_headers",
        "load_segments",
        "rwx_segments",

        # Dynamic Loader
        "has_interp",
        "has_dynamic",
        "has_relro",
        "has_gnu_stack",

        "num_libraries",
        "num_imported_functions",

        # Symbols
        "num_dynamic_symbols",
    ]
    benign = collect_features(benign_dir)
    malware = collect_features(malware_dir)
    df= pd.DataFrame(benign+malware)
    df.to_csv('output.csv',index=False)
    benign_summary = summarise(benign)
    malware_summary = summarise(malware)
    
    print("=== Benign Summary ===")
    print(benign_summary)
    print("\n=== Malware Summary ===")
    print(malware_summary)
