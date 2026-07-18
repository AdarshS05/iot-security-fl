import json
import math
import os
import sys
import lief


def calculate_entropy(data: bytes) -> float:
    """Calculates the Shannon entropy of a byte array."""
    if not data:
        return 0.0

    entropy = 0.0
    total_len = len(data)
    counts = [0] * 256

    for byte in data:
        counts[byte] += 1

    for count in counts:
        if count == 0:
            continue
        p = count / total_len
        entropy -= p * math.log2(p)

    return entropy


def extract_rodata_strings(binary: lief.ELF.Binary) -> list:
    """Extracts printable ASCII strings (length > 4) from the .rodata section."""
    if not binary.has_section(".rodata"):
        return []

    rodata = binary.get_section(".rodata")
    raw_bytes = bytes(rodata.content)

    # Convert bytes to characters, replacing non-printables with newlines
    decoded = "".join(
        [chr(b) if 32 <= b < 127 else "\n" for b in raw_bytes]
    )
    strings = [s.strip() for s in decoded.split("\n") if len(s.strip()) > 4]
    return strings


def extract_features(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"error": f"File '{file_path}' not found."}

    binary = lief.parse(file_path)
    if not binary or not isinstance(binary, lief.ELF.Binary):
        return {"error": "Target file is not a valid ELF binary."}

    # Calculate global file entropy
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    global_entropy = calculate_entropy(file_bytes)

    # Extract dynamic imports
    imported_libs = [lib for lib in binary.libraries]
    imported_funcs = [
        sym.name for sym in binary.imported_symbols if sym.imported
    ]

    # Process Sections
    sections_data = []
    for sec in binary.sections:
        sections_data.append(
            {
                "name": sec.name,
                "size": sec.size,
                "entropy": round(calculate_entropy(bytes(sec.content)), 2),
            }
        )

    # Process Segments & check for RWE (Read-Write-Execute) risk
    has_rwe_segment = 0
    segments_data = []
    for seg in binary.segments:
        is_rwe = (
            seg.has(lief.ELF.Segment.FLAGS.R)
            and seg.has(seg.FLAGS.W)
            and seg.has(seg.FLAGS.X)
        )
        if is_rwe:
            has_rwe_segment = 1

        segments_data.append(
            {"type": seg.type.name, "flags": str(seg.flags), "is_rwe": is_rwe}
        )

    # Check for text relocation security bypasses (DT_TEXTREL)
    has_textrel = 0
    for entry in binary.dynamic_entries:
        if entry.tag == lief.ELF.DynamicEntry.TAG.TEXTREL:
            has_textrel = 1
            break

    # Build the flattened ML-ready feature dictionary
    features = {
        "filename": os.path.basename(file_path),
        # Basic Metadata
        "file_size_bytes": os.path.getsize(file_path),
        "global_entropy": round(global_entropy, 2),
        "entry_point": hex(binary.entrypoint),
        "program_header_offset": binary.header.program_header_offset,
        # Environment & Architecture Fingerprint
        "architecture": binary.header.machine_type.name,
        "os_abi": binary.header.identity_os_abi.name,
        # Evasion / Packing Indicators
        "has_rwe_segment": has_rwe_segment,
        "has_textrel": has_textrel,
        # Structural Counts
        "num_sections": len(binary.sections),
        "num_segments": len(binary.segments),
        "num_dynamic_symbols": len(binary.dynamic_symbols),
        # Detailed Collections (Good for NLP tokenization or list matching)
        "imported_libraries": imported_libs,
        "imported_functions_count": len(imported_funcs),
        "imported_functions_sample": imported_funcs[:15],
        "sections": sections_data,
        "strings_extracted_count": len(extract_rodata_strings(binary)),
    }

    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_elf.py <path_to_elf_binary>")
        sys.exit(1)

    elf_features = extract_features(sys.argv[1])

    # Pretty print the JSON payload
    print(json.dumps(elf_features, indent=2))
