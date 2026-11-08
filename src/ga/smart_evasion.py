"""
smart_evasion.py  –  Score-guided SVM evasion for MIPS ELF malware

WHY THE GA FAILS (summary)
--------------------------
1. Trampoline/opaque_predicate: build_trampoline_gene requires trampoline_vaddr
   to be 4-byte aligned (J-type jump constraint).  The cave file_offset equals
   p_offset + p_filesz, which may not be 4-byte aligned, so every one of the
   3763 candidate pairs silently fails _j_type_reachable() and returns None.

2. Random junk produces MALWARE-characteristic bigrams: generate_junk_instruction_stream
   emits addu/or/sll into $zero — these are common in the malware training
   corpus, so they push the TF-IDF vector *toward* malware, not away.

3. Cave size is only 304 bytes = 76 instructions = ~4% of the feature stream.
   Even perfect benign-vocab junk moves cosine similarity by ≈0.04 — nowhere
   near enough to cross a margin of 1.7065.

THIS ENGINE
-----------
1. DIAGNOSTIC: computes the exact feature gap vs benign support vectors and
   reports how many benign-characteristic instructions are needed to flip.

2. BENIGN VOCABULARY: loads the TF-IDF training matrix, computes which
   instruction bigrams are discriminatively benign (mean_benign >> mean_malware),
   maps them back to actual MIPS byte sequences.

3. LARGE INJECTION: appends 500–2000 targeted instructions to the file END
   and grows the executable segment's p_filesz/p_memsz to cover them, so r2
   disassembles the new bytes as live code in the segment.

4. GREEDY HILL-CLIMBER: builds the injection block instruction-by-instruction,
   keeping each instruction that improves (reduces) the SVM score.

Usage
-----
python3 src/ga/smart_evasion.py \\
  --binary   data/raw/raw_mips/HASH \\
  --siamese-model  models/siamese_svm_n1/siamese_svm_fewshot.joblib \\
  --tfidf-matrix   data/features/tfidf_n1/X_train.npz \\
  --tfidf-labels   data/features/tfidf_n1/train_samples.csv \\
  --output-dir     reports/smart_evasion \\
  [--max-instructions 1000] [--n-probes 50] [--seed 42] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("smart_evasion")
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# ELF helpers (self-contained, no GA import needed)
# ---------------------------------------------------------------------------

_ELF32_EHDR_TAIL_FMT = "HHIIIIIHHHHHH"
_ELF32_PHDR_FMT      = "IIIIIIII"
PT_LOAD = 1
PF_X    = 0x1


def _elf32_endianness(path: Path) -> str:
    with open(path, "rb") as fh:
        ident = fh.read(16)
    if len(ident) != 16 or ident[:4] != b"\x7fELF":
        raise ValueError(f"Not an ELF file: {path}")
    ei_class = ident[4]
    ei_data  = ident[5]
    if ei_class != 1:
        raise ValueError("Only 32-bit ELF (MIPS o32) is supported")
    if ei_data == 1:
        return "<"  # ELFDATA2LSB (little-endian)
    if ei_data == 2:
        return ">"  # ELFDATA2MSB (big-endian)
    raise ValueError(f"Unknown EI_DATA: {ei_data}")


def _parse_elf32_phdrs(path: Path):
    endian = _elf32_endianness(path)
    with open(path, "rb") as fh:
        data = fh.read()
    ehdr_fmt = endian + _ELF32_EHDR_TAIL_FMT
    (
        _e_type, _e_machine, _e_version, _e_entry,
        e_phoff, _e_shoff, _e_flags, _e_ehsize,
        e_phentsize, e_phnum,
        _e_shentsize, _e_shnum, _e_shstrndx,
    ) = struct.unpack_from(ehdr_fmt, data, 16)
    phdr_fmt  = endian + _ELF32_PHDR_FMT
    phdr_size = struct.calcsize(phdr_fmt)
    phdrs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (
            p_type, p_offset, p_vaddr, p_paddr,
            p_filesz, p_memsz, p_flags, p_align,
        ) = struct.unpack_from(phdr_fmt, data, off)
        phdrs.append({
            "index":        i,
            "entry_offset": off,
            "entry_size":   phdr_size,
            "p_type":  p_type,
            "p_offset": p_offset,
            "p_vaddr":  p_vaddr,
            "p_paddr":  p_paddr,
            "p_filesz": p_filesz,
            "p_memsz":  p_memsz,
            "p_flags":  p_flags,
            "p_align":  p_align,
        })
    return endian, e_phoff, e_phentsize, phdrs


def _find_exec_segment(path: Path):
    """Return (endian, phdr_dict) for the largest executable PT_LOAD."""
    endian, e_phoff, e_phentsize, phdrs = _parse_elf32_phdrs(path)
    best = None
    for p in phdrs:
        if p["p_type"] == PT_LOAD and (p["p_flags"] & PF_X):
            if best is None or p["p_filesz"] > best["p_filesz"]:
                best = p
    if best is None:
        raise RuntimeError("No executable PT_LOAD segment found")
    best["endian"]      = endian
    best["e_phoff"]     = e_phoff
    best["e_phentsize"] = e_phentsize
    return best


def grow_exec_segment(variant_path: Path, n_extra_bytes: int, seg: dict) -> None:
    """
    Append n_extra_bytes to the file and grow the executable segment's
    p_filesz / p_memsz so r2 will disassemble the new bytes as part of the
    segment's mapped content.

    The new bytes are already written to the file by the caller; this function
    only patches the program-header entry.
    """
    endian   = seg["endian"]
    phdr_fmt = endian + _ELF32_PHDR_FMT
    entry_offset = seg["e_phoff"] + seg["index"] * seg["e_phentsize"]

    with open(variant_path, "r+b") as fh:
        fh.seek(entry_offset)
        raw_entry = fh.read(struct.calcsize(phdr_fmt))
        (
            p_type, p_offset, p_vaddr, p_paddr,
            p_filesz, p_memsz, p_flags, p_align,
        ) = struct.unpack(phdr_fmt, raw_entry)

        new_entry = struct.pack(
            phdr_fmt,
            p_type, p_offset, p_vaddr, p_paddr,
            p_filesz + n_extra_bytes,
            p_memsz  + n_extra_bytes,
            p_flags, p_align,
        )
        fh.seek(entry_offset)
        fh.write(new_entry)
        fh.flush()


# ---------------------------------------------------------------------------
# MIPS instruction byte generators  (endian-aware)
# ---------------------------------------------------------------------------
# All safe to use as dead code in an unreachable cave block – results are
# discarded into $zero, or the instructions write into caller-saved temps
# ($t0-$t9, $a0-$a3, $v0-$v1) which the surrounding live code will overwrite
# before reading.

_CALLER_SAVED = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25]
_ANY_REG      = list(range(1, 26))  # skip $zero(0), $sp(29), $fp(30), $gp(28), $ra(31)


def _w2b(word: int, endian: str) -> bytes:
    return word.to_bytes(4, "big" if endian == ">" else "little")


def gen_nop(endian: str) -> bytes:
    """sll $zero, $zero, 0 – canonical MIPS NOP."""
    return _w2b(0x00000000, endian)


def gen_addu(rng: random.Random, endian: str) -> bytes:
    """addu $rd, $rs, $rt  (result discarded into $zero or a temp)."""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice([0] + _CALLER_SAVED)   # dest is often $zero in dead code
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x21
    return _w2b(word, endian)


def gen_add(rng: random.Random, endian: str) -> bytes:
    """add $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x20
    return _w2b(word, endian)


def gen_or(rng: random.Random, endian: str) -> bytes:
    """or $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x25
    return _w2b(word, endian)


def gen_and(rng: random.Random, endian: str) -> bytes:
    """and $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x24
    return _w2b(word, endian)


def gen_xor(rng: random.Random, endian: str) -> bytes:
    """xor $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x26
    return _w2b(word, endian)


def gen_nor(rng: random.Random, endian: str) -> bytes:
    """nor $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x27
    return _w2b(word, endian)


def gen_slt(rng: random.Random, endian: str) -> bytes:
    """slt $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x2A
    return _w2b(word, endian)


def gen_sltu(rng: random.Random, endian: str) -> bytes:
    """sltu $rd, $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | (rd << 11) | 0x2B
    return _w2b(word, endian)


def gen_sll(rng: random.Random, endian: str) -> bytes:
    """sll $rd, $rt, shamt"""
    rt    = rng.choice(_CALLER_SAVED)
    rd    = rng.choice(_CALLER_SAVED)
    shamt = rng.randint(1, 15)
    word  = (rt << 16) | (rd << 11) | (shamt << 6) | 0x00
    return _w2b(word, endian)


def gen_srl(rng: random.Random, endian: str) -> bytes:
    """srl $rd, $rt, shamt"""
    rt    = rng.choice(_CALLER_SAVED)
    rd    = rng.choice(_CALLER_SAVED)
    shamt = rng.randint(1, 15)
    word  = (rt << 16) | (rd << 11) | (shamt << 6) | 0x02
    return _w2b(word, endian)


def gen_sra(rng: random.Random, endian: str) -> bytes:
    """sra $rd, $rt, shamt"""
    rt    = rng.choice(_CALLER_SAVED)
    rd    = rng.choice(_CALLER_SAVED)
    shamt = rng.randint(1, 15)
    word  = (rt << 16) | (rd << 11) | (shamt << 6) | 0x03
    return _w2b(word, endian)


def gen_addiu(rng: random.Random, endian: str) -> bytes:
    """addiu $rt, $rs, imm  – I-type, very common in compiled MIPS code."""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(-256, 255) & 0xFFFF
    word = (0x09 << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_ori(rng: random.Random, endian: str) -> bytes:
    """ori $rt, $rs, imm"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 0xFFFF)
    word = (0x0D << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_andi(rng: random.Random, endian: str) -> bytes:
    """andi $rt, $rs, imm"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 0xFFFF)
    word = (0x0C << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_xori(rng: random.Random, endian: str) -> bytes:
    """xori $rt, $rs, imm"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 0xFFFF)
    word = (0x0E << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_slti(rng: random.Random, endian: str) -> bytes:
    """slti $rt, $rs, imm"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 255) & 0xFFFF
    word = (0x0A << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_sltiu(rng: random.Random, endian: str) -> bytes:
    """sltiu $rt, $rs, imm"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 255) & 0xFFFF
    word = (0x0B << 26) | (rs << 21) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_lui(rng: random.Random, endian: str) -> bytes:
    """lui $rt, imm  – very common in data/BSS address computation."""
    rt  = rng.choice(_CALLER_SAVED)
    imm = rng.randint(0, 0xFFFF)
    word = (0x0F << 26) | (rt << 16) | imm
    return _w2b(word, endian)


def gen_lw(rng: random.Random, endian: str) -> bytes:
    """lw $rt, offset($rs)  – extremely common in benign code."""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    off = (rng.randint(0, 63) * 4) & 0xFFFF    # small aligned offset
    word = (0x23 << 26) | (rs << 21) | (rt << 16) | off
    return _w2b(word, endian)


def gen_sw(rng: random.Random, endian: str) -> bytes:
    """sw $rt, offset($rs)"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    off = (rng.randint(0, 63) * 4) & 0xFFFF
    word = (0x2B << 26) | (rs << 21) | (rt << 16) | off
    return _w2b(word, endian)


def gen_lbu(rng: random.Random, endian: str) -> bytes:
    """lbu $rt, offset($rs)  – byte load, common in string / network code."""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    off = rng.randint(0, 15) & 0xFFFF
    word = (0x24 << 26) | (rs << 21) | (rt << 16) | off
    return _w2b(word, endian)


def gen_sb(rng: random.Random, endian: str) -> bytes:
    """sb $rt, offset($rs)"""
    rs  = rng.choice(_CALLER_SAVED)
    rt  = rng.choice(_CALLER_SAVED)
    off = rng.randint(0, 15) & 0xFFFF
    word = (0x28 << 26) | (rs << 21) | (rt << 16) | off
    return _w2b(word, endian)


def gen_move(rng: random.Random, endian: str) -> bytes:
    """move $rd, $rs  →  encoded as addu $rd, $rs, $zero (funct=0x21, rt=0)."""
    rs = rng.choice(_CALLER_SAVED)
    rd = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (0 << 16) | (rd << 11) | 0x21
    return _w2b(word, endian)


def gen_mult(rng: random.Random, endian: str) -> bytes:
    """mult $rs, $rt  – writes HI/LO, no GPR destination, inert in dead code."""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | 0x18
    return _w2b(word, endian)


def gen_multu(rng: random.Random, endian: str) -> bytes:
    """multu $rs, $rt"""
    rs = rng.choice(_CALLER_SAVED)
    rt = rng.choice(_CALLER_SAVED)
    word = (rs << 21) | (rt << 16) | 0x19
    return _w2b(word, endian)


def gen_mfhi(rng: random.Random, endian: str) -> bytes:
    """mfhi $rd  – reads HI into a GPR, common after mult/div."""
    rd = rng.choice(_CALLER_SAVED)
    word = (rd << 11) | 0x10
    return _w2b(word, endian)


def gen_mflo(rng: random.Random, endian: str) -> bytes:
    """mflo $rd"""
    rd = rng.choice(_CALLER_SAVED)
    word = (rd << 11) | 0x12
    return _w2b(word, endian)


# ── All generators, keyed by the normalized mnemonic they produce ──────────
# The `n1_conservative` normalizer lowercases the mnemonic; registers become
# r_ and immediates become i_ (or remain literals in some modes – we rely
# on the mnemonic token being sufficient for bigram matching).

INSTRUCTION_GENERATORS = {
    "nop":    lambda rng, e: gen_nop(e),
    "addu":   gen_addu,
    "add":    gen_add,
    "or":     gen_or,
    "and":    gen_and,
    "xor":    gen_xor,
    "nor":    gen_nor,
    "slt":    gen_slt,
    "sltu":   gen_sltu,
    "sll":    gen_sll,
    "srl":    gen_srl,
    "sra":    gen_sra,
    "addiu":  gen_addiu,
    "ori":    gen_ori,
    "andi":   gen_andi,
    "xori":   gen_xori,
    "slti":   gen_slti,
    "sltiu":  gen_sltiu,
    "lui":    gen_lui,
    "lw":     gen_lw,
    "sw":     gen_sw,
    "lbu":    gen_lbu,
    "sb":     gen_sb,
    "move":   gen_move,
    "mult":   gen_mult,
    "multu":  gen_multu,
    "mfhi":   gen_mfhi,
    "mflo":   gen_mflo,
}


def random_instr(rng: random.Random, endian: str,
                 weights: Optional[Dict[str, float]] = None) -> bytes:
    """Generate one random instruction from the available generators."""
    keys = list(INSTRUCTION_GENERATORS.keys())
    if weights:
        w = [weights.get(k, 0.01) for k in keys]
    else:
        w = [1.0] * len(keys)
    total = sum(w)
    r = rng.random() * total
    acc = 0.0
    for k, wi in zip(keys, w):
        acc += wi
        if r <= acc:
            return INSTRUCTION_GENERATORS[k](rng, endian)
    return gen_nop(endian)


# ---------------------------------------------------------------------------
# Benign-vocabulary analysis
# ---------------------------------------------------------------------------

def load_normalize_fn():
    """Load the same normalize_instruction used to build the TF-IDF corpus."""
    here    = Path(__file__).resolve()
    src_dir = here.parents[1]
    utils   = src_dir / "assembly2" / "utils.py"
    if not utils.is_file():
        logger.warning(
            "src/assembly2/utils.py not found – cannot verify token normalization. "
            "Continuing with mnemonic-only matching."
        )
        return None
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("training_utils", utils)
    mod  = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "normalize_instruction", None)


def analyze_feature_gap(
    X_train_path: str,
    labels_path: str,
    top_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-feature 'benign advantage' = mean_benign - mean_malware.

    Returns (benign_advantage, benign_mean, malware_mean) as shape-(F,) arrays.
    Features with large positive benign_advantage are characteristic of benign
    binaries and absent/rare in malware — injecting their corresponding
    instruction bigrams pushes the query TF-IDF toward benign support vectors.
    """
    import scipy.sparse as sp
    import pandas as pd

    X      = sp.load_npz(X_train_path)
    labels = pd.read_csv(labels_path)

    # Accept 'label', 'is_malware', or 'y' as the label column
    label_col = next(
        (c for c in ("label", "is_malware", "y") if c in labels.columns),
        labels.columns[1],
    )
    y = labels[label_col].values

    benign_mask  = (y == 0)
    malware_mask = (y == 1)

    n_benign  = benign_mask.sum()
    n_malware = malware_mask.sum()

    logger.info("Training set: %d benign, %d malware", n_benign, n_malware)

    benign_mean  = np.asarray(X[benign_mask].mean(axis=0)).flatten()
    malware_mean = np.asarray(X[malware_mask].mean(axis=0)).flatten()

    benign_adv = benign_mean - malware_mean

    top_idx = np.argsort(benign_adv)[::-1][:top_k]
    logger.info(
        "Top-%d benign-advantage features: indices %s, advantages %s",
        top_k,
        top_idx[:5],
        benign_adv[top_idx[:5]],
    )

    return benign_adv, benign_mean, malware_mean


def extract_tfidf_vocab(siamese_model_path: str) -> Optional[List[str]]:
    """
    Try to extract the TF-IDF feature names from the Siamese SVM joblib file.
    Returns None if the model doesn't expose a vectorizer.
    """
    try:
        import joblib
        obj = joblib.load(siamese_model_path)
    except Exception as exc:
        logger.warning("Could not load %s: %s", siamese_model_path, exc)
        return None

    # Try common attribute names
    for name in ("vectorizer", "tfidf", "tfidf_vectorizer", "vec"):
        vec = getattr(obj, name, None)
        if vec is not None and hasattr(vec, "get_feature_names_out"):
            return list(vec.get_feature_names_out())

    # Try if the object is a dict
    if isinstance(obj, dict):
        for name in ("vectorizer", "tfidf", "vec"):
            vec = obj.get(name)
            if vec is not None and hasattr(vec, "get_feature_names_out"):
                return list(vec.get_feature_names_out())

    # If the object is a tuple/list, scan it
    items = obj if isinstance(obj, (list, tuple)) else []
    for item in items:
        if hasattr(item, "get_feature_names_out"):
            return list(item.get_feature_names_out())

    return None


def benign_vocab_to_weights(
    benign_adv: np.ndarray,
    feature_names: List[str],
    top_k: int = 30,
) -> Dict[str, float]:
    """
    Map the top-K benign-advantage TF-IDF features back to instruction
    mnemonic weights for junk generation.

    Each TF-IDF feature is a bigram like "addiu lw" or "lw sw". We extract
    the first mnemonic of each benign bigram and accumulate its advantage
    score as a generation weight.
    """
    top_idx  = np.argsort(benign_adv)[::-1][:top_k]
    weights: Dict[str, float] = {}

    for i in top_idx:
        adv   = float(benign_adv[i])
        token = feature_names[i].strip()   # e.g. "addiu lw" or "addiu r_ lw r_"
        # Extract first mnemonic (first non-register, non-immediate word)
        parts = token.split()
        for p in parts:
            p_lower = p.lower()
            if p_lower in INSTRUCTION_GENERATORS:
                weights[p_lower] = weights.get(p_lower, 0.0) + max(adv, 0.0)
                break

    # Floor all known generators at 0.01 so they're always reachable
    for k in INSTRUCTION_GENERATORS:
        if k not in weights:
            weights[k] = 0.01

    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    logger.info(
        "Benign-vocab weights (top 5): %s",
        sorted(weights.items(), key=lambda x: -x[1])[:5],
    )
    return weights


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def load_siamese_scorer(
    model_path: str,
    matrix_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    random_state: int = 42,
):
    """Build the project's canonical few-shot scorer from its model bundle.

    The saved joblib is a dictionary containing a LinearSVC pair classifier
    and fitted TF-IDF vectorizer. It does NOT contain the three-shot support
    bank, so that bank must be reconstructed deterministically from the
    training TF-IDF matrix and matching training manifest, exactly as in the
    original GA's SiameseSVMScorer.
    """
    import importlib.util
    import sys

    ga_path = Path(__file__).resolve().with_name("genetic_algorithm_mutator.py")
    if not ga_path.is_file():
        raise FileNotFoundError(
            "Cannot locate genetic_algorithm_mutator.py beside smart_evasion.py: "
            f"{ga_path}. The canonical SiameseSVMScorer is defined there."
        )

    module_name = "_iot_ga_mutator_for_smart_evasion"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, ga_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load GA module from {ga_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    scorer = module.SiameseSVMScorer(
        model_path=model_path,
        matrix_path=matrix_path,
        labels_path=labels_path,
        random_state=random_state,
    )
    return scorer


def score_binary(binary_path: str, scorer_fn) -> Optional[float]:
    """
    Score binary_path with the Siamese SVM scorer.
    Returns malware_margin (positive = malware, ≤0 = evasion).
    Returns None on error.
    """
    try:
        result = scorer_fn(binary_path)
        if isinstance(result, dict):
            return float(result.get("malware_margin", result.get("score", 0.0)))
        return float(result)
    except Exception as exc:
        logger.debug("scorer failed on %s: %s", binary_path, exc)
        return None


# ---------------------------------------------------------------------------
# Large injection attack
# ---------------------------------------------------------------------------

def build_injection_block(
    rng: random.Random,
    endian: str,
    n_instructions: int,
    instr_weights: Optional[Dict[str, float]] = None,
) -> bytes:
    """
    Build a block of n_instructions targeted MIPS instructions.
    Uses benign-vocab weights when provided, otherwise uniform.
    """
    block = bytearray()
    for _ in range(n_instructions):
        block.extend(random_instr(rng, endian, instr_weights))
    return bytes(block)


def apply_large_injection(
    original_path: str,
    variant_path: str,
    injection_bytes: bytes,
    seg: dict,
) -> bool:
    """
    Copy original to variant, append injection_bytes to file end,
    grow segment p_filesz/p_memsz to cover the new bytes.
    Returns True on success.
    """
    try:
        shutil.copy2(original_path, variant_path)
        # Append to file end
        with open(variant_path, "ab") as fh:
            fh.write(injection_bytes)
        grow_exec_segment(Path(variant_path), len(injection_bytes), seg)
        return True
    except Exception as exc:
        logger.warning("apply_large_injection failed: %s", exc)
        return False


def score_guided_probe(
    original_path: str,
    work_dir: str,
    seg: dict,
    scorer_fn,
    rng: random.Random,
    endian: str,
    n_instructions: int,
    n_probes: int,
    instr_weights: Optional[Dict[str, float]] = None,
    baseline_score: Optional[float] = None,
) -> Tuple[Optional[bytes], float]:
    """
    Try n_probes random injection blocks of n_instructions each.
    Returns (best_block_bytes, best_score). best_block_bytes is None if all
    probes fail or none improve on baseline.
    """
    best_block = None
    best_score = baseline_score if baseline_score is not None else float("inf")

    for probe_i in range(n_probes):
        block = build_injection_block(rng, endian, n_instructions, instr_weights)
        variant = os.path.join(work_dir, f"probe_{probe_i}.elf")
        if not apply_large_injection(original_path, variant, block, seg):
            continue
        sc = score_binary(variant, scorer_fn)
        try:
            os.remove(variant)
        except OSError:
            pass
        if sc is None:
            continue
        logger.debug("probe %d/%d: score=%.4f", probe_i + 1, n_probes, sc)
        if sc < best_score:
            best_score = sc
            best_block = block

    return best_block, best_score


def greedy_instruction_hillclimber(
    original_path: str,
    work_dir: str,
    seg: dict,
    scorer_fn,
    rng: random.Random,
    endian: str,
    max_instructions: int,
    instr_weights: Optional[Dict[str, float]] = None,
    candidates_per_step: int = 8,
) -> Tuple[bytes, float]:
    """
    Build the injection block one instruction at a time.
    At each step, try `candidates_per_step` random instructions and keep
    whichever reduces the SVM score the most.

    This is O(max_instructions × candidates_per_step) scorer calls.
    """
    block = bytearray()
    best_score = score_binary(original_path, scorer_fn)
    if best_score is None:
        logger.error("Could not score original binary – aborting hill-climber.")
        return bytes(block), float("inf")

    logger.info("Greedy hill-climber: baseline SVM score = %.4f", best_score)

    for step in range(max_instructions):
        best_instr = None
        best_step_score = best_score

        for _ in range(candidates_per_step):
            instr = random_instr(rng, endian, instr_weights)
            candidate_block = bytes(block) + instr
            variant = os.path.join(work_dir, f"hc_step{step}.elf")
            if not apply_large_injection(original_path, variant, candidate_block, seg):
                continue
            sc = score_binary(variant, scorer_fn)
            try:
                os.remove(variant)
            except OSError:
                pass
            if sc is None:
                continue
            if sc < best_step_score:
                best_step_score = sc
                best_instr      = instr

        if best_instr is None:
            # No improvement this step – add the best non-worsening candidate
            # or a benign-weight random as filler
            best_instr = random_instr(rng, endian, instr_weights)

        block.extend(best_instr)
        if best_step_score < best_score:
            best_score = best_step_score
            logger.info(
                "  step %4d/%-4d  score=%.6f  (Δ=%.6f)  %s",
                step + 1, max_instructions,
                best_score,
                best_step_score - best_score,
                "✓",
            )
        if best_score <= 0.0:
            logger.info("  🎯  SVM score crossed 0 at step %d!", step + 1)
            break

    return bytes(block), best_score


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def run_diagnostic(
    original_path: str,
    scorer_fn,
    X_train_path: Optional[str],
    labels_path: Optional[str],
    feature_names: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Report on the current binary's feature gap vs benign support vectors,
    estimate how many benign-vocab instructions are needed to flip the score.
    """
    baseline = score_binary(original_path, scorer_fn)
    report: Dict[str, Any] = {
        "baseline_svm_margin": baseline,
        "evasion_threshold": 0.0,
        "gap_to_evasion": baseline,
    }

    if baseline is None:
        logger.error("Baseline scoring failed; cannot produce diagnostic.")
        return report

    logger.info("=== DIAGNOSTIC ===")
    logger.info("Baseline SVM margin : %.6f", baseline)
    logger.info("Evasion requires    : margin ≤ 0 (gap = %.6f)", baseline)

    if X_train_path and labels_path:
        try:
            benign_adv, benign_mean, malware_mean = analyze_feature_gap(
                X_train_path, labels_path
            )
            report["n_features"]         = len(benign_adv)
            report["top_benign_adv"]     = float(benign_adv[np.argsort(benign_adv)[::-1][0]])
            report["n_benign_dominant"]  = int((benign_adv > 0).sum())

            if feature_names:
                top_idx = np.argsort(benign_adv)[::-1][:10]
                top_features = [
                    {"feature": feature_names[i], "advantage": float(benign_adv[i])}
                    for i in top_idx
                ]
                report["top_benign_features"] = top_features
                logger.info("Top benign-discriminative bigrams:")
                for f in top_features:
                    logger.info("  %-35s  advantage=%.4f", f["feature"], f["advantage"])

        except Exception as exc:
            logger.warning("Feature gap analysis failed: %s", exc)

    # Estimate instructions needed
    # Very rough: each new instruction adds ~2 bigrams to the feature stream.
    # A typical binary has ~1700 instructions → 3400 bigrams.
    # Adding N instructions means N/(N+1700) fraction of features are new.
    # Assume best-case: new instructions contribute 0 to malware similarity,
    # full 1 to benign similarity. Cosine similarity shift ≈ √(N/(N+1700)).
    # Score shift ≈ baseline × cosine_shift.  Solve for score ≤ 0:
    # Need cosine_shift ≥ 1 (impossible in naive estimate), but in practice
    # instruction sequence quality matters. We report a "rough lower bound".
    est_needed = max(50, int(baseline * 2000))
    logger.info(
        "Rough estimate: need ≥ %d benign-characteristic instructions "
        "(accounting for imperfect feature alignment)",
        est_needed,
    )
    report["estimated_instructions_needed"] = est_needed

    return report


# ---------------------------------------------------------------------------
# Main attack pipeline
# ---------------------------------------------------------------------------

def run_attack(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    rng = random.Random(args.seed)

    # -- Load segment info ---------------------------------------------------
    try:
        seg = _find_exec_segment(Path(args.binary))
    except Exception as exc:
        logger.error("ELF parse failed: %s", exc)
        return {"error": str(exc)}

    endian = seg["endian"]
    logger.info(
        "Binary: %s  size=%d  endian=%s  exec-seg p_filesz=%d",
        args.binary,
        Path(args.binary).stat().st_size,
        "BE" if endian == ">" else "LE",
        seg["p_filesz"],
    )

    # -- Load scorer ---------------------------------------------------------
    logger.info("Loading Siamese SVM from %s ...", args.siamese_model)
    scorer_fn = load_siamese_scorer(
        args.siamese_model,
        matrix_path=getattr(args, "tfidf_matrix", None),
        labels_path=getattr(args, "tfidf_labels", None),
        random_state=args.seed,
    )

    # -- Load TF-IDF vocab (optional) ----------------------------------------
    feature_names = extract_tfidf_vocab(args.siamese_model)
    if feature_names:
        logger.info("Extracted %d TF-IDF feature names from model.", len(feature_names))
    else:
        logger.info(
            "Could not extract TF-IDF feature names from model; "
            "will use mnemonic-first-token heuristic for benign-vocab extraction."
        )

    # -- Diagnostic ----------------------------------------------------------
    diag = run_diagnostic(
        args.binary, scorer_fn,
        getattr(args, "tfidf_matrix", None),
        getattr(args, "tfidf_labels", None),
        feature_names,
    )
    baseline = diag.get("baseline_svm_margin")
    if baseline is None:
        logger.error("Cannot proceed without a baseline score.")
        return diag

    # -- Build benign-vocab weights ------------------------------------------
    instr_weights = None
    if (
        getattr(args, "tfidf_matrix", None)
        and getattr(args, "tfidf_labels", None)
        and feature_names
    ):
        try:
            benign_adv, _, _ = analyze_feature_gap(args.tfidf_matrix, args.tfidf_labels)
            instr_weights = benign_vocab_to_weights(benign_adv, feature_names)
        except Exception as exc:
            logger.warning("Benign-vocab analysis failed, using uniform weights: %s", exc)

    if instr_weights is None:
        # Fallback: heuristic benign-leaning weights for MIPS
        # (lw/sw/addiu/lui are common in library/benign code; they tend to be
        # underrepresented in pure-compute IoT malware)
        instr_weights = {
            "lw":    0.14,
            "sw":    0.10,
            "addiu": 0.12,
            "lui":   0.08,
            "lbu":   0.06,
            "sb":    0.05,
            "ori":   0.07,
            "andi":  0.06,
            "slti":  0.05,
            "move":  0.07,
            "mflo":  0.04,
            "mfhi":  0.04,
            "mult":  0.04,
            "multu": 0.03,
            "nop":   0.05,
            # Reduce pure-compute instructions common in malware
            "addu":  0.03,
            "or":    0.02,
            "and":   0.02,
            "xor":   0.02,
            "sll":   0.02,
            "srl":   0.02,
        }
        logger.info("Using heuristic benign-leaning instruction weights.")

    # -- Phase 1: Random probe (fast, finds a good starting block) -----------
    n_probe_instrs = min(args.max_instructions // 4, 200)
    logger.info(
        "Phase 1: score-guided probe (%d probes × %d instructions each)...",
        args.n_probes,
        n_probe_instrs,
    )
    best_block, best_probe_score = score_guided_probe(
        original_path     = args.binary,
        work_dir          = work_dir,
        seg               = seg,
        scorer_fn         = scorer_fn,
        rng               = rng,
        endian            = endian,
        n_instructions    = n_probe_instrs,
        n_probes          = args.n_probes,
        instr_weights     = instr_weights,
        baseline_score    = baseline,
    )
    if best_block:
        logger.info(
            "Phase 1 best probe: score=%.4f (improvement=%.4f)",
            best_probe_score,
            baseline - best_probe_score,
        )
    else:
        logger.info("Phase 1: no probe improved on baseline (%.4f).", baseline)

    # -- Phase 2: Greedy hill-climber (maximise score reduction) -------------
    logger.info(
        "Phase 2: greedy hill-climber (%d instructions, %d candidates/step)...",
        args.max_instructions,
        args.candidates_per_step,
    )
    final_block, final_score = greedy_instruction_hillclimber(
        original_path        = args.binary,
        work_dir             = work_dir,
        seg                  = seg,
        scorer_fn            = scorer_fn,
        rng                  = rng,
        endian               = endian,
        max_instructions     = args.max_instructions,
        instr_weights        = instr_weights,
        candidates_per_step  = args.candidates_per_step,
    )

    logger.info(
        "Phase 2 result: final_score=%.4f  improvement=%.4f  evasion=%s",
        final_score,
        baseline - final_score,
        final_score <= 0.0,
    )

    # -- Apply best block to final output ------------------------------------
    # Pick whichever is better: probe best or hill-climber result
    if best_block and best_probe_score < final_score:
        chosen_block = best_block
        chosen_score = best_probe_score
        method       = "probe"
    else:
        chosen_block = final_block
        chosen_score = final_score
        method       = "hillclimber"

    out_path = os.path.join(
        args.output_dir,
        Path(args.binary).name + ".evaded.elf",
    )
    if chosen_block:
        success = apply_large_injection(args.binary, out_path, chosen_block, seg)
    else:
        success = False

    result: Dict[str, Any] = {
        "binary":          args.binary,
        "baseline_score":  baseline,
        "final_score":     chosen_score,
        "improvement":     baseline - chosen_score,
        "evasion":         chosen_score <= 0.0,
        "method":          method,
        "injection_bytes": len(chosen_block) if chosen_block else 0,
        "injection_instrs": len(chosen_block) // 4 if chosen_block else 0,
        "output_path":     out_path if success else None,
        "diagnostic":      diag,
    }

    result_path = os.path.join(args.output_dir, "smart_evasion_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Result written to %s", result_path)
    if result["evasion"]:
        logger.info("✅  EVASION ACHIEVED! Evaded variant: %s", out_path)
    else:
        logger.info(
            "❌  No evasion yet. Score reduced by %.4f (need %.4f more).",
            result["improvement"],
            chosen_score,
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score-guided SVM evasion attack for MIPS ELF malware"
    )
    ap.add_argument("--binary",          required=True,
                    help="Path to the original malware ELF binary")
    ap.add_argument("--siamese-model",   required=True,
                    help="Path to siamese_svm_fewshot.joblib")
    ap.add_argument("--tfidf-matrix",    default=None,
                    help="X_train.npz TF-IDF matrix for benign-vocab analysis")
    ap.add_argument("--tfidf-labels",    default=None,
                    help="train_samples.csv with label column")
    ap.add_argument("--output-dir",      default="reports/smart_evasion",
                    help="Directory for results and variants")
    ap.add_argument("--max-instructions", type=int, default=500,
                    help="Max instructions to inject (hill-climber budget)")
    ap.add_argument("--n-probes",        type=int, default=50,
                    help="Random probes for Phase 1")
    ap.add_argument("--candidates-per-step", type=int, default=8,
                    help="Candidates per hill-climber step")
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--verbose", "-v",   action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    result = run_attack(args)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("evasion") else 1)


if __name__ == "__main__":
    main()
