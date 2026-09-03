#!/usr/bin/env python3
"""
elf_feature_extractor.py

Extracts static features from ELF binaries for malware dataset construction.
Covers: file/header metadata, sections, segments, symbols, imports, strings,
entropy, packing indicators, and basic code-structure features.

Dependencies:
    pip install lief pyelftools --break-system-packages

Optional (for opcode/CFG features):
    - radare2 installed on system (https://github.com/radareorg/radare2)
    - pip install r2pipe --break-system-packages

Usage:
    python3 elf_feature_extractor.py <path_to_elf_or_directory> [-o output.csv] [--opcode]

    Single file  -> prints feature dict as JSON
    Directory    -> extracts features for every file in the directory and
                    writes a CSV dataset (one row per binary)
"""

import os
import sys
import json
import math
import re
import string
import argparse
import hashlib
from collections import Counter

try:
    import lief
except ImportError:
    lief = None

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
except ImportError:
    ELFFile = None

try:
    import r2pipe
    HAS_R2 = True
except ImportError:
    HAS_R2 = False

def is_elf(file_path: str) -> bool:
    """
    Check if a file is an ELF binary by reading its magic bytes (\x7fELF).
    """
    ELF_MAGIC = b"\x7fELF"
    
    try:
        if not os.path.isfile(file_path):
            return False
        with open(file_path, "rb") as f:
            return f.read(4) == ELF_MAGIC
    except (OSError, PermissionError):
        return False

# ---------------------------------------------------------------------------
# Suspicious API / string indicators commonly seen in IoT / Linux malware
# ---------------------------------------------------------------------------
SUSPICIOUS_IMPORTS = [
    "ptrace", "mprotect", "mmap", "dlopen", "dlsym", "dlerror",
    "fork", "vfork", "execve", "execv", "execl", "system", "popen",
    "socket", "connect", "bind", "listen", "accept", "recv", "send",
    "chmod", "chown", "setuid", "setgid", "seteuid", "setegid",
    "getenv", "setenv", "unlink", "remove", "kill", "signal",
    "prctl", "syscall", "inet_addr", "gethostbyname",
]

SUSPICIOUS_STRINGS = [
    "/bin/sh", "/bin/bash", "busybox", "wget", "tftp", "curl",
    "chmod 777", "chmod +x", "iptables", "/dev/watchdog",
    "LD_PRELOAD", "proc/self", "/tmp/", "reboot", "rm -rf",
    ".onion", "http://", "https://",
]

KNOWN_PACKER_SECTION_NAMES = [
    "UPX0", "UPX1", "UPX2", ".upx", "packed", ".packed",
]

STANDARD_ELF_SECTIONS = {
    ".text", ".data", ".bss", ".rodata", ".init", ".fini",
    ".dynamic", ".dynsym", ".dynstr", ".symtab", ".strtab",
    ".interp", ".comment", ".shstrtab", ".plt", ".got", ".got.plt",
    ".init_array", ".fini_array", ".eh_frame", ".eh_frame_hdr",
    ".note.ABI-tag", ".note.gnu.build-id", ".gnu.hash", ".hash",
    ".gnu.version", ".gnu.version_r", ".rela.dyn", ".rela.plt",
}


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------
def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy (0-8) of a byte sequence."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def entropy_histogram(data: bytes, chunk_size: int = 256, bins=(2, 4, 6, 7)) -> dict:
    """
    Slide over the file in chunks and bucket chunk-entropy values.
    Returns fraction of chunks falling into low/med/high entropy bands
    plus variance -- useful to distinguish 'uniformly packed' from
    'one embedded compressed blob'.
    """
    if not data:
        return {"entropy_variance": 0.0, "entropy_high_ratio": 0.0}

    chunk_entropies = [
        shannon_entropy(data[i:i + chunk_size])
        for i in range(0, len(data), chunk_size)
    ]
    if not chunk_entropies:
        return {"entropy_variance": 0.0, "entropy_high_ratio": 0.0}

    mean_e = sum(chunk_entropies) / len(chunk_entropies)
    variance = sum((e - mean_e) ** 2 for e in chunk_entropies) / len(chunk_entropies)
    high_ratio = sum(1 for e in chunk_entropies if e >= 7.0) / len(chunk_entropies)

    return {"entropy_variance": round(variance, 4), "entropy_high_ratio": round(high_ratio, 4)}


# ---------------------------------------------------------------------------
# String extraction (equivalent to `strings` command)
# ---------------------------------------------------------------------------
def extract_strings(data: bytes, min_len: int = 4):
    pattern = re.compile(
        b"[" + re.escape(string.printable[:-6].encode()) + b"]{%d,}" % min_len
    )
    return [m.decode(errors="ignore") for m in pattern.findall(data)]


IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def string_features(data: bytes) -> dict:
    strs = extract_strings(data)
    n = len(strs)
    lengths = [len(s) for s in strs] or [0]
    joined_sample = " ".join(strs)

    ip_hits = len(IP_RE.findall(joined_sample))
    url_hits = len(URL_RE.findall(joined_sample))
    suspicious_hits = sum(1 for s in strs if any(sig in s for sig in SUSPICIOUS_STRINGS))

    high_entropy_strings = sum(1 for s in strs if len(s) >= 8 and shannon_entropy(s.encode()) >= 4.5)

    return {
        "num_strings": n,
        "avg_string_len": round(sum(lengths) / n, 2) if n else 0.0,
        "max_string_len": max(lengths) if lengths else 0,
        "num_ip_strings": ip_hits,
        "num_url_strings": url_hits,
        "num_suspicious_strings": suspicious_hits,
        "num_high_entropy_strings": high_entropy_strings,
    }


# ---------------------------------------------------------------------------
# Core ELF metadata / header / segment features (your existing feature set)
# ---------------------------------------------------------------------------
def basic_and_header_features(path: str, binary, data: bytes) -> dict:
    feats = {}

    # --- file-level ---
    feats["file_size_bytes"] = len(data)
    feats["global_entropy"] = round(shannon_entropy(data), 4)
    feats["md5"] = hashlib.md5(data).hexdigest()
    feats["sha256"] = hashlib.sha256(data).hexdigest()

    # --- ELF header ---
    header = binary.header
    feats["architecture"] = str(header.machine_type).split(".")[-1]
    feats["elf_type"] = str(header.file_type).split(".")[-1]
    feats["os_abi"] = str(header.identity_os_abi).split(".")[-1]
    feats["abi_version"] = int(header.identity_abi_version)
    feats["endianness"] = str(header.identity_data).split(".")[-1]
    feats["entry_point"] = header.entrypoint

    # --- program headers / segments ---
    segments = binary.segments
    feats["program_headers"] = len(segments)
    load_segments = [s for s in segments if str(s.type).split(".")[-1] == "LOAD"]
    feats["load_segments"] = len(load_segments)

    rwx_count = 0
    for s in segments:
        flags = str(s.flags)
        # LIEF segment flags: R, W, X bits
        is_r = s.has(lief.ELF.Segment.FLAGS.R)
        is_w = s.has(lief.ELF.Segment.FLAGS.W)
        is_x = s.has(lief.ELF.Segment.FLAGS.X)
        if is_r and is_w and is_x:
            rwx_count += 1
    feats["rwx_segments"] = rwx_count

    feats["has_interp"] = binary.has_interpreter
    feats["has_dynamic"] = any(str(s.type).split(".")[-1] == "DYNAMIC" for s in segments)
    feats["has_gnu_stack"] = any(str(s.type).split(".")[-1] == "GNU_STACK" for s in segments)
    feats["has_relro"] = any(
        str(s.type).split(".")[-1] in ("GNU_RELRO",) for s in segments
    )

    return feats


# ---------------------------------------------------------------------------
# Dynamic linking / import / symbol features
# ---------------------------------------------------------------------------
def linking_and_symbol_features(binary) -> dict:
    feats = {}

    libs = list(binary.libraries)
    feats["num_libraries"] = len(libs)
    feats["is_statically_linked"] = int(len(libs) == 0 and not binary.has_interpreter)

    # Uncommon / non-standard library names (heuristic: not matching libc/libm/ld family)
    common_lib_prefixes = ("libc.", "libm.", "ld-linux", "libpthread", "libdl", "librt", "libgcc")
    uncommon_libs = [l for l in libs if not any(l.startswith(p) for p in common_lib_prefixes)]
    feats["num_uncommon_libraries"] = len(uncommon_libs)

    # Imported functions (dynamic symbols that are undefined -> imported)
    imported_funcs = []
    try:
        imported_funcs = [f.name for f in binary.imported_functions]
    except Exception:
        pass
    feats["num_imported_functions"] = len(imported_funcs)

    exported_funcs = []
    try:
        exported_funcs = [f.name for f in binary.exported_functions]
    except Exception:
        pass
    feats["num_exported_functions"] = len(exported_funcs)

    ratio = (len(imported_funcs) / len(exported_funcs)) if exported_funcs else float(len(imported_funcs))
    feats["import_export_ratio"] = round(ratio, 4)

    # Symbol information is optional in ELF files.
    # Sectionless ELF binaries may not have a usable static symbol table,
    # so every symbol-table access must be guarded.
    try:
        dynamic_symbols = list(binary.dynamic_symbols)
    except Exception:
        dynamic_symbols = []

    try:
        static_symbols = list(binary.symtab_symbols)
    except Exception:
        static_symbols = []

    feats["num_dynamic_symbols"] = len(dynamic_symbols)
    feats["num_static_symbols"] = len(static_symbols)
    feats["is_stripped"] = int(len(static_symbols) == 0)

    # Suspicious imported API calls
    suspicious_hits = sum(
        1
        for f in imported_funcs
        if any(sig == f or sig in f for sig in SUSPICIOUS_IMPORTS)
    )
    feats["num_suspicious_imports"] = suspicious_hits

    # Debug information requires section headers.
    # Sectionless ELF has no section metadata from which to identify
    # .debug* sections, so report it as absent.
    try:
        sections = list(binary.sections)
    except Exception:
        sections = []

    feats["has_debug_info"] = int(
        any(".debug" in s.name for s in sections)
    )

    return feats


# ---------------------------------------------------------------------------
# Section-level features
# ---------------------------------------------------------------------------
def section_features(binary, data: bytes) -> dict:
    feats = {}

    # Section headers are optional in ELF.
    # Some stripped/packed binaries have no section-header table at all.
    # LIEF may emit a warning when sections are requested, so handle this
    # explicitly and continue with segment/file-level features.
    try:
        sections = list(binary.sections)
    except Exception:
        sections = []

    section_headers_present = len(sections) > 0

    feats["num_sections"] = len(sections)
    feats["section_headers_present"] = int(section_headers_present)

    names = [s.name for s in sections]
    nonstandard = [n for n in names if n and n not in STANDARD_ELF_SECTIONS]
    feats["num_nonstandard_sections"] = len(nonstandard)

    # executable sections
    exec_sections = [s for s in sections if lief.ELF.Section.FLAGS.EXECINSTR in s.flags_list]
    feats["num_executable_sections"] = len(exec_sections)

    # text section size ratio
    text_size = 0
    for s in sections:
        if s.name == ".text":
            text_size = s.size
            break
    feats["text_size_ratio"] = round(text_size / len(data), 6) if data else 0.0

    # per-section entropy (max and mean across sections, plus flag for suspiciously packed section)
    section_entropies = []
    for s in sections:
        try:
            content = bytes(s.content)
        except Exception:
            content = b""
        if content:
            section_entropies.append(shannon_entropy(content))

    if section_entropies:
        feats["max_section_entropy"] = round(max(section_entropies), 4)
        feats["mean_section_entropy"] = round(sum(section_entropies) / len(section_entropies), 4)
    else:
        feats["max_section_entropy"] = 0.0
        feats["mean_section_entropy"] = 0.0

    # packer signature detection
    feats["has_packer_signature"] = int(
        any(any(pk.lower() in n.lower() for pk in KNOWN_PACKER_SECTION_NAMES) for n in names)
    )

    # overlay data: bytes beyond the last section/segment end
    max_end = 0
    for s in sections:
        end = s.offset + s.size
        if end > max_end:
            max_end = end
    for seg in binary.segments:
        end = seg.file_offset + seg.physical_size
        if end > max_end:
            max_end = end
    overlay_size = max(0, len(data) - max_end)
    feats["overlay_size_bytes"] = overlay_size
    feats["has_overlay"] = int(overlay_size > 64)  # small tail padding is common/benign

    return feats


# ---------------------------------------------------------------------------
# Optional: opcode / control-flow features via radare2 (r2pipe)
# ---------------------------------------------------------------------------
def opcode_and_cfg_features(path: str) -> dict:
    """
    Requires radare2 + r2pipe installed. Returns opcode frequency (top-N),
    function count, and basic CFG complexity metrics.
    Falls back to zeros/empty if r2 is unavailable or analysis fails.
    """
    feats = {
        "num_functions": 0,
        "avg_basic_blocks_per_func": 0.0,
        "avg_cyclomatic_complexity": 0.0,
        "jump_to_instruction_ratio": 0.0,
    }
    top_opcodes = {}

    if not HAS_R2:
        return {**feats, "top_opcodes": top_opcodes}

    try:
        r2 = r2pipe.open(path, flags=["-2"])  # -2: suppress stderr
        r2.cmd("aaa")  # analyze all

        funcs = json.loads(r2.cmd("aflj") or "[]")
        feats["num_functions"] = len(funcs)

        if funcs:
            total_bb = sum(f.get("nbbs", 0) for f in funcs)
            total_cc = sum(f.get("cc", 0) for f in funcs)
            feats["avg_basic_blocks_per_func"] = round(total_bb / len(funcs), 3)
            feats["avg_cyclomatic_complexity"] = round(total_cc / len(funcs), 3)

        # opcode frequency across whole .text via disassembly
        insns = json.loads(r2.cmd("pdrj @@ fcn.*") or "[]") if funcs else []
        opcode_counter = Counter()
        jump_like = 0
        total_insns = 0
        # pdrj per function returns nested lists; flatten safely
        def flatten(x):
            if isinstance(x, list):
                for i in x:
                    yield from flatten(i)
            elif isinstance(x, dict):
                yield x

        for insn in flatten(insns):
            mnem = insn.get("opcode", "").split()[0] if insn.get("opcode") else None
            if mnem:
                opcode_counter[mnem] += 1
                total_insns += 1
                if mnem.startswith("j") or mnem.startswith("b") or mnem in ("call", "bl", "blx"):
                    jump_like += 1

        if total_insns:
            feats["jump_to_instruction_ratio"] = round(jump_like / total_insns, 4)
            top_opcodes = dict(opcode_counter.most_common(15))

        r2.quit()
    except Exception:
        pass

    return {**feats, "top_opcodes": top_opcodes}


# ---------------------------------------------------------------------------
# Main extraction orchestrator
# ---------------------------------------------------------------------------
def extract_features(path: str, include_opcode: bool = False) -> dict:
    with open(path, "rb") as f:
        data = f.read()

    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF failed to parse {path} as ELF")

    feats = {"file_name": os.path.basename(path)}

    # Always-available ELF/header/segment features.
    feats.update(basic_and_header_features(path, binary, data))

    # Dynamic/import information may still be available even when
    # section headers are absent.
    feats.update(linking_and_symbol_features(binary))

    # Section-derived features explicitly tolerate sectionless ELF.
    feats.update(section_features(binary, data))

    # These operate directly on raw file bytes and therefore work
    # regardless of whether section headers exist.
    feats.update(entropy_histogram(data))
    feats.update(string_features(data))

    if include_opcode:
        opcode_feats = opcode_and_cfg_features(path)
        feats["top_opcodes"] = json.dumps(opcode_feats.pop("top_opcodes"))
        feats.update(opcode_feats)

    return feats


def main():
    parser = argparse.ArgumentParser(description="Extract static ELF features for malware dataset building.")
    parser.add_argument("target", help="Path to a single ELF file or a directory of ELF files")
    parser.add_argument("-o", "--output", default="elf_features.csv", help="Output CSV path (directory mode)")
    parser.add_argument("--opcode", action="store_true", help="Also extract opcode/CFG features (requires radare2 + r2pipe)")
    args = parser.parse_args()

    if lief is None:
        sys.exit("ERROR: 'lief' is required. Install with: pip install lief --break-system-packages")

    if os.path.isdir(args.target):
        rows = []
        for fname in sorted(os.listdir(args.target)):
            fpath = os.path.join(args.target, fname)
            if not os.path.isfile(fpath) or not is_elf(fpath):
                continue
            try:
                rows.append(extract_features(fpath, include_opcode=args.opcode))
            except Exception as e:
                print(f"[WARN] Skipping {fname}: {e}", file=sys.stderr)

        if not rows:
            sys.exit("No valid ELF files found in directory.")

        # Union of all keys, preserving stable column order
        all_keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"Wrote {len(rows)} rows x {len(all_keys)} features -> {args.output}")

    else:
        if not is_elf(args.target):
            sys.exit(f"{args.target} does not look like an ELF file.")
        feats = extract_features(args.target, include_opcode=args.opcode)
        print(json.dumps(feats, indent=2))


if __name__ == "__main__":
    main()

