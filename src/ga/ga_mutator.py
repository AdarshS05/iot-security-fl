"""
genetic_algorithm_mutator.py

GA-based physical mutation framework for ELF malware robustness testing.

Design goals
------------
1. Represent mutations as chromosomes of explicit genes.
2. Generate a real GA population (not a single individual).
3. Keep sectionless ELF binaries byte-for-byte identical in the original
   file region when using overlay mutations.
4. Never invoke LIEF's ELF.Builder for overlay-only mutations.
5. Validate both ELF correctness and preservation invariants.
6. Keep chromosome metadata, materialized bytes, and fitness connected.
7. Re-extract features from every accepted variant before scoring it.
8. Keep instruction-level transforms disabled until explicitly whitelisted
   and validated.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import uuid
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import lief
import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

"""Canonical fixed reference set for the Siamese SVM.
Used identically by the GA (ga_mutator.py) and by ONNX inference
(common/onnx_models.py) so both score against the same math.
"""
import numpy as np
import joblib
from scipy import sparse

REF_PER_CLASS = 5
SEED = 42


def load_fixed_reference(tfidf_dir: str):
    """Returns (ref_vectors: np.ndarray (10,D) float32, ref_labels: np.ndarray (10,), ref_indices: np.ndarray (10,))."""
    X = sparse.load_npz(f"{tfidf_dir}/tfidf_matrix.npz")
    labels = np.asarray(joblib.load(f"{tfidf_dir}/labels.pkl"))

    rng = np.random.default_rng(SEED)
    chosen = []
    for lbl in (0, 1):
        available = np.flatnonzero(labels == lbl)
        n = min(REF_PER_CLASS, len(available))
        chosen.extend(rng.choice(available, size=n, replace=False).tolist())

    idx = np.asarray(sorted(chosen), dtype=int)
    ref_vectors = X[idx].toarray().astype(np.float32)
    ref_labels = labels[idx]
    return ref_vectors, ref_labels, idx


def score_against_reference(diffs_or_margins: np.ndarray, ref_labels: np.ndarray) -> float:
    """Given per-reference margins (shape (10,)), return malware_margin - benign_margin."""
    malware_m = float(diffs_or_margins[ref_labels == 1].mean())
    benign_m = float(diffs_or_margins[ref_labels == 0].mean())
    return malware_m - benign_m


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# All the former ad-hoc `print("[DEBUG] ...")` calls now go through this
# logger instead. They are silent by default and only appear with
# `--verbose` / `-v` (see main()), instead of always printing to stdout
# regardless of whether anyone asked for them.

logger = logging.getLogger("ga_mutator")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _debug(*parts: Any) -> None:
    """Drop-in replacement for the old `print("[DEBUG] ...", ...)` calls.

    Accepts the same multi-argument call style as print() but routes
    through logging.debug, so it is silent unless --verbose is passed.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(" ".join(str(p) for p in parts))


# ---------------------------------------------------------------------------
# Input-file triage (packer / architecture / basic sanity checks)
# ---------------------------------------------------------------------------
# Added because the GA's structural mutators (code-cave/trampoline/opaque
# predicate insertion, instruction-level transforms) all assume the static
# instruction stream in the ELF *is* the real program. A UPX-packed sample
# fails that assumption silently: LIEF still parses it as a valid ELF and
# radare2 still disassembles *something*, but what gets disassembled is
# just the small UPX unpacking stub (tens of instructions), not the actual
# malware logic, which only exists decompressed in memory at runtime. Any
# mutation the GA makes in that scenario is mutating the stub, not the
# payload, and any feature/behavior claims drawn from it are unreliable.
# This was observed directly against a sample in this dataset: `aflj`
# found only a handful of tiny functions (entry0 at 45 instrs, a ~124-instr
# stub, ...) and `izzj` turned up literal "UPX!" / "$Info: This file is
# packed with the UPX executable packer" strings -- a textbook packed
# binary. We now detect and flag this before spending a GA run on it.

UPX_MARKERS = (b"UPX!", b"$Info: This file is packed with the UPX")


class BinaryTriageError(RuntimeError):
    """Raised when a candidate file is not suitable for GA mutation."""


def _is_upx_packed(raw: bytes) -> bool:
    return any(marker in raw for marker in UPX_MARKERS)


def _basic_elf_triage(path: str) -> Tuple["lief.Binary", bytes]:
    """File-exists / ELF-magic / LIEF-parses checks only (no packer check).

    Returns (parsed_binary, raw_bytes). Raises BinaryTriageError on any
    failure, same contract as triage_binary().
    """
    p = Path(path)
    if not p.is_file():
        raise BinaryTriageError(f"not a regular file: {path}")
    if p.stat().st_size == 0:
        raise BinaryTriageError(f"empty file: {path}")

    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise BinaryTriageError(f"could not read file: {exc}") from exc

    if raw[:4] != b"\x7fELF":
        raise BinaryTriageError("not an ELF file (bad magic)")

    binary = lief.parse(str(p))
    if binary is None:
        raise BinaryTriageError("LIEF could not parse this ELF")

    return binary, raw


def attempt_upx_unpack(path: str, work_dir: str, timeout: int = 60) -> Optional[str]:
    """Try `upx -d` on a copy of `path`. Returns the unpacked file's path
    on success (markers gone, still a parsable ELF), or None if unpacking
    isn't possible (upx missing) or didn't succeed (e.g. a Mirai/Gafgyt
    sample with a deliberately corrupted UPX header, a known anti-analysis
    trick -- `upx -d` then exits non-zero and we fall back cleanly).
    """
    if shutil.which("upx") is None:
        logger.warning(
            "upx not found on PATH -- cannot auto-unpack %s "
            "(install UPX, e.g. `apt install upx-ucl`, or pass "
            "--no-auto-unpack to silence this)",
            path,
        )
        return None

    os.makedirs(work_dir, exist_ok=True)
    stem = Path(path).name
    staged_in = os.path.join(work_dir, f"{stem}.packed")
    staged_out = os.path.join(work_dir, f"{stem}.unpacked")
    shutil.copy2(path, staged_in)

    try:
        proc = subprocess.run(
            ["upx", "-d", "-o", staged_out, staged_in],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("upx -d on %s raised %s", path, exc)
        return None

    if proc.returncode != 0 or not os.path.isfile(staged_out):
        logger.warning(
            "upx -d failed on %s (exit %s): %s -- likely a deliberately "
            "corrupted UPX header (common IoT-malware anti-unpacking "
            "trick); leaving this sample packed",
            path, proc.returncode, proc.stderr.strip(),
        )
        return None

    try:
        raw = Path(staged_out).read_bytes()
    except OSError as exc:
        logger.warning("could not read upx output for %s: %s", path, exc)
        return None

    if _is_upx_packed(raw):
        logger.warning("upx -d on %s ran but markers are still present; discarding", path)
        return None

    if lief.parse(staged_out) is None:
        logger.warning("upx -d on %s produced an unparsable ELF; discarding", path)
        return None

    logger.info("%s: auto-unpacked with upx -> %s", path, staged_out)
    return staged_out


def triage_binary(
    path: str,
    expected_machine: Optional["lief.ELF.ARCH"] = None,
    allow_packed: bool = False,
    auto_unpack: bool = True,
    unpack_work_dir: Optional[str] = None,
) -> Tuple["lief.Binary", str]:
    """Validate (and, if needed and possible, unpack) a candidate file.

    Returns (parsed_binary, path_to_use_for_the_ga) on success -- the
    returned path is `path` unchanged, or the auto-unpacked copy if one
    was produced. Raises BinaryTriageError if the file should be skipped
    entirely (bad file, wrong arch, or still packed after an unpack
    attempt with allow_packed=False).
    """
    binary, raw = _basic_elf_triage(path)

    if expected_machine is not None and binary.header.machine_type != expected_machine:
        raise BinaryTriageError(
            f"architecture mismatch: expected {expected_machine}, "
            f"got {binary.header.machine_type}"
        )

    if not _is_upx_packed(raw):
        return binary, path

    # Packed. Try to unpack it for real before deciding to skip/force it.
    if auto_unpack:
        unpacked_path = attempt_upx_unpack(
            path, unpack_work_dir or os.path.join(os.path.dirname(path) or ".", "_unpacked")
        )
        if unpacked_path is not None:
            unpacked_binary = lief.parse(unpacked_path)
            if (
                expected_machine is not None
                and unpacked_binary.header.machine_type != expected_machine
            ):
                raise BinaryTriageError(
                    f"architecture mismatch after unpacking: expected "
                    f"{expected_machine}, got {unpacked_binary.header.machine_type}"
                )
            return unpacked_binary, unpacked_path

    # Still packed (auto_unpack off, upx missing, or upx -d failed).
    if allow_packed:
        logger.warning(
            "%s: still UPX-packed and could not be auto-unpacked; "
            "proceeding anyway because --allow-packed was set -- "
            "mutations will only touch the decompression stub",
            path,
        )
        return binary, path

    raise BinaryTriageError(
        "binary is UPX-packed and could not be auto-unpacked (upx missing "
        "or `upx -d` failed, e.g. a deliberately corrupted header); "
        "install upx and retry, or re-run with --allow-packed to force "
        "mutation of the packed stub"
    )


# ---------------------------------------------------------------------------
# Chromosome representation
# ---------------------------------------------------------------------------

class Operator(str, Enum):
    PADDING = "padding"
    ELF_TRANSFORM = "elf_transform"
    INSTRUCTION_TRANSFORM = "instruction_transform"
    CODE_CAVE_INSERTION = "code_cave_insertion"
    TRAMPOLINE_INSERTION = "trampoline_insertion"
    OPAQUE_PREDICATE_INSERTION = "opaque_predicate_insertion"


@dataclass
class Gene:
    """One mutation gene: operator + location + parameters."""

    operator: Operator
    location: Any = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Gene":
        return Gene(
            operator=self.operator,
            location=copy.deepcopy(self.location),
            parameters=copy.deepcopy(self.parameters),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator.value,
            "location": self.location,
            "parameters": copy.deepcopy(self.parameters),
        }


@dataclass
class Chromosome:
    """An ordered set of genes applied to one source binary."""

    genes: List[Gene] = field(default_factory=list)
    source_binary: str = ""
    fitness: Optional[float] = None
    fitness_result: Optional[Dict[str, Any]] = None
    modification_cost: Optional[float] = None
    variant_path: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    genes_applied: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "Chromosome":
        return Chromosome(
            genes=[gene.copy() for gene in self.genes],
            source_binary=self.source_binary,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_binary": self.source_binary,
            "genes": [gene.to_dict() for gene in self.genes],
            "fitness": self.fitness,
            "fitness_breakdown": self.fitness_result,
            "modification_cost": self.modification_cost,
            "variant_path": self.variant_path,
            "validation": self.validation,
            "genes_applied": copy.deepcopy(self.genes_applied),
        }


# ---------------------------------------------------------------------------
# Binary invariants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinarySnapshot:
    """Properties that must remain stable for overlay-only mutation."""

    path: str
    size: int
    sha256: str
    machine: Any
    file_type: Any
    entrypoint: int
    segment_count: int
    load_segments: Tuple[Tuple[int, int, int, int, int], ...]
    section_count: int


def _sha256_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_segment_signature(binary: lief.Binary) -> Tuple[Tuple[int, int, int, int, int], ...]:
    result = []
    for seg in binary.segments:
        if seg.type != lief.ELF.Segment.TYPE.LOAD:
            continue
        result.append(
            (
                int(seg.virtual_address),
                int(seg.physical_address),
                int(seg.physical_size),
                int(seg.virtual_size),
                int(seg.flags),
            )
        )
    return tuple(result)


def snapshot_binary(path: str) -> BinarySnapshot:
    binary = lief.parse(path)
    if binary is None:
        raise ValueError(f"LIEF could not parse ELF binary: {path}")

    return BinarySnapshot(
        path=path,
        size=os.path.getsize(path),
        sha256=_sha256_file(path),
        machine=binary.header.machine_type,
        file_type=binary.header.file_type,
        entrypoint=int(binary.header.entrypoint),
        segment_count=len(binary.segments),
        load_segments=_load_segment_signature(binary),
        section_count=len(binary.sections),
    )


# ---------------------------------------------------------------------------
# Mutation mask
# ---------------------------------------------------------------------------

class MutationMask:
    """
    Binary-specific mutation constraints.

    For sectionless ELFs, only overlay padding is allowed. This prevents
    accidental LIEF reconstruction of an ELF whose original layout must be
    preserved byte-for-byte.
    """

    def __init__(self, binary: lief.Binary, max_padding_bytes: int = 4096):
        if max_padding_bytes <= 0:
            raise ValueError("max_padding_bytes must be positive")

        self.binary = binary
        self.max_padding_bytes = int(max_padding_bytes)
        self.sectionless = len(binary.sections) == 0
        self._exec_ranges = self._compute_executable_ranges()

    def _compute_executable_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []

        for seg in self.binary.segments:
            if seg.type != lief.ELF.Segment.TYPE.LOAD:
                continue

            try:
                executable = bool(
                    int(seg.flags) & int(lief.ELF.Segment.FLAGS.X)
                )
            except Exception:
                executable = False

            if not executable:
                continue

            start = int(seg.virtual_address)
            end = start + int(seg.virtual_size)
            if end > start:
                ranges.append((start, end))

        return ranges

    def allows_padding(
        self,
        region: str,
        size: int,
        location: Any = None,
    ) -> bool:
        try:
            size = int(size)
        except (TypeError, ValueError):
            return False

        if size <= 0 or size > self.max_padding_bytes:
            return False

        if region == "overlay":
            return True

        return False

    def allows_elf_transform(
        self,
        field_name: str,
        location: Any = None,
        new_value: Any = None,
    ) -> bool:
        # Only the reserved e_ident EI_PAD bytes are whitelisted: they are
        # unread by any conformant loader, so rewriting them can never
        # break execution or the ELF-structure invariants checked by the
        # Validator.
        return field_name == "ei_pad"

    def allows_instruction_transform(self, address, transform_name=None):
        if not InstructionMutator.SAFE_TRANSFORMS:
            return False
        try:
            address = int(address)
        except (TypeError, ValueError):
            return False
        if not any(lo <= address < hi for lo, hi in self._exec_ranges):
            return False
        return transform_name in InstructionMutator.SAFE_TRANSFORMS


    def allows_gene(self, gene: Gene) -> bool:
        if not isinstance(gene, Gene):
            return False

        if gene.operator == Operator.PADDING:
            return self.allows_padding(
                gene.parameters.get("region", "overlay"),
                gene.parameters.get("size", 0),
                gene.location,
            )

        if gene.operator == Operator.ELF_TRANSFORM:
            return self.allows_elf_transform(
                gene.parameters.get("field"),
                gene.location,
                gene.parameters.get("new_value"),
            )

        if gene.operator == Operator.INSTRUCTION_TRANSFORM:
            return self.allows_instruction_transform(
                gene.location,
                gene.parameters.get("transform"),
            )

        if gene.operator == Operator.CODE_CAVE_INSERTION:
            return self.allows_code_cave(gene)

        if gene.operator in (
            Operator.TRAMPOLINE_INSERTION,
            Operator.OPAQUE_PREDICATE_INSERTION,
        ):
            # Both operators share the exact same on-disk shape (redirect
            # word + relocated delay-slot word + a cave-resident body +
            # segment growth) -- they only differ in how the cave body is
            # constructed (flat junk vs. a fake-but-deterministic branch
            # wrapped around dead junk), so one legality check covers both.
            return self.allows_trampoline_style(gene)

        return False

    def allows_code_cave(self, gene: Gene) -> bool:
        try:
            new_bytes = bytes.fromhex(gene.parameters.get("new_bytes", ""))
        except ValueError:
            return False

        if not new_bytes or len(new_bytes) % 4 != 0:
            return False

        for key in (
            "file_offset", "segment_index", "e_phoff", "e_phentsize",
            "original_filesz", "original_memsz", "endian",
        ):
            if gene.parameters.get(key) is None:
                return False

        return True

    def allows_trampoline_style(self, gene: Gene) -> bool:
        try:
            trampoline_bytes = bytes.fromhex(
                gene.parameters.get("trampoline_bytes", "")
            )
            site_a_new = bytes.fromhex(gene.parameters.get("site_a_new", ""))
            site_b_new = bytes.fromhex(gene.parameters.get("site_b_new", ""))
        except ValueError:
            return False

        if len(site_a_new) != 4 or len(site_b_new) != 4:
            return False

        if not trampoline_bytes or len(trampoline_bytes) % 4 != 0:
            return False

        # Trampoline body must be at least [>=1 junk word] + [return jump]
        # + [relocated delay-slot word].
        if len(trampoline_bytes) < 12:
            return False

        for key in (
            "site_a_file_offset", "site_b_file_offset",
            "trampoline_file_offset", "trampoline_vaddr", "return_vaddr",
            "segment_index", "e_phoff", "e_phentsize",
            "original_filesz", "original_memsz", "endian",
        ):
            if gene.parameters.get(key) is None:
                return False

        if (
            gene.parameters["site_b_file_offset"]
            != gene.parameters["site_a_file_offset"] + 4
        ):
            return False

        return True

    def allows_chromosome(self, chromosome: Chromosome, max_genes: Optional[int] = None) -> bool:
        allowed, _reason = self.explain_chromosome(chromosome, max_genes)
        return allowed

    def explain_chromosome(
        self,
        chromosome: Chromosome,
        max_genes: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Same legality check as allows_chromosome(), but returns a specific
        reason string instead of a bare bool, distinguishing:
          - "empty"                     -> structurally empty chromosome
          - "gene_invalid:<n>:<op>"     -> gene #n itself is illegal
          - "duplicate_instruction_offset" -> two genes patch the same word
          - "duplicate_elf_transform_field" -> two genes touch the same field
          - "multiple_segment_growth_genes" -> more than one gene tries to
            grow a PT_LOAD segment (code cave and/or trampoline both
            allocate from the same cave-finding logic against the pristine
            source binary, so at most one may be present per chromosome)
          - "ok"                        -> chromosome is legal

        This lets the GA's logging separate "gene individually invalid"
        from "valid genes, invalid combination" instead of collapsing both
        into one opaque "chromosome rejected" line.
        """
        if not chromosome.genes:
            return False, "empty"

        for index, gene in enumerate(chromosome.genes):
            if not self.allows_gene(gene):
                return False, f"gene_invalid:{index}:{gene.operator.value}"

        # Every gene that patches an exact 4-byte instruction word --
        # INSTRUCTION_TRANSFORM, plus TRAMPOLINE_INSERTION's two redirect
        # sites -- must target disjoint offsets, or the second write in
        # materialization order will find the byte-identity check on
        # "original_bytes" already stale and the whole chromosome will
        # fail at apply time instead of at this cheap up-front check.
        instruction_offsets = []

        for gene in chromosome.genes:
            if gene.operator == Operator.INSTRUCTION_TRANSFORM:
                file_offset = gene.parameters.get("file_offset")

                if file_offset is None:
                    return False, "instruction_gene_missing_offset"

                instruction_offsets.append(file_offset)

            elif gene.operator in (
                Operator.TRAMPOLINE_INSERTION,
                Operator.OPAQUE_PREDICATE_INSERTION,
            ):
                site_a = gene.parameters.get("site_a_file_offset")
                site_b = gene.parameters.get("site_b_file_offset")

                if site_a is None or site_b is None:
                    return False, "trampoline_gene_missing_offset"

                instruction_offsets.append(site_a)
                instruction_offsets.append(site_b)

        if len(instruction_offsets) != len(set(instruction_offsets)):
            return False, "duplicate_instruction_offset"

        elf_transform_fields = [
            gene.parameters.get("field")
            for gene in chromosome.genes
            if gene.operator == Operator.ELF_TRANSFORM
        ]

        if len(elf_transform_fields) != len(set(elf_transform_fields)):
            return False, "duplicate_elf_transform_field"

        segment_growth_genes = sum(
            1
            for gene in chromosome.genes
            if gene.operator in (
                Operator.CODE_CAVE_INSERTION,
                Operator.TRAMPOLINE_INSERTION,
                Operator.OPAQUE_PREDICATE_INSERTION,
            )
        )

        if segment_growth_genes > 1:
            return False, "multiple_segment_growth_genes"

        if max_genes is not None and len(chromosome.genes) > max_genes:
            return False, "too_many_genes"

        return True, "ok"


# ---------------------------------------------------------------------------
# Instruction mutation extension point
# ---------------------------------------------------------------------------

class InstructionMutator:
    """
    Explicit whitelist for validated same-length MIPS32 substitutions.
    """

    SAFE_TRANSFORMS: Dict[
        str, Callable[[lief.Binary, int, Dict[str, Any]], bool]
    ] = {}

    @classmethod
    def apply(
        cls,
        binary: lief.Binary,
        address: int,
        transform_name: str,
        params: Dict[str, Any],
    ) -> bool:
        fn = cls.SAFE_TRANSFORMS.get(transform_name)
        if fn is None:
            return False

        try:
            return bool(fn(binary, address, params))
        except Exception:
            return False


# Generic validator for any exact-4-byte, same-length MIPS instruction swap.
# Reused across every whitelisted single-instruction substitution technique
# below, since all of them share the same "4 bytes in, 4 different bytes
# out" shape. The technique-specific *semantics* live in the mutator classes
# (MIPSConservativeMutator, MIPSSignednessSwapMutator, ...), not here.
def _mips_same_length_instruction_swap(binary, address, params):
    original_hex = params.get("original_bytes")
    new_hex = params.get("new_bytes")

    if not original_hex or not new_hex:
        return False

    try:
        original_bytes = bytes.fromhex(original_hex)
        new_bytes = bytes.fromhex(new_hex)
    except ValueError:
        return False

    if len(original_bytes) != 4 or len(new_bytes) != 4:
        return False

    if original_bytes == new_bytes:
        return False

    return True

# ---------------------------------------------------------------------------
# Technique 3: independent adjacent-instruction reorder (8-byte transform)
# ---------------------------------------------------------------------------
# Swaps two consecutive instructions when neither reads a register the
# other writes and neither writes a register the other writes (no RAW/WAW/
# WAR hazard). Changes every bigram touching the swap boundary, which is a
# much larger perturbation per gene than a single-operand swap.

def _no_dependency(ins_a: "MIPSInstruction", ins_b: "MIPSInstruction") -> bool:
    da = _decode_mips_rtype(ins_a.raw_bytes)
    db = _decode_mips_rtype(ins_b.raw_bytes)
    if da is None or db is None:
        return False

    writes_a = {da["rd"]} if da["rd"] != 0 else set()
    reads_a = {da["rs"], da["rt"]} - {0}
    writes_b = {db["rd"]} if db["rd"] != 0 else set()
    reads_b = {db["rs"], db["rt"]} - {0}

    if writes_a & (reads_b | writes_b):
        return False
    if writes_b & reads_a:
        return False
    return True


def _mips_adjacent_reorder_validator(binary, address, params) -> bool:
    original_hex = params.get("original_bytes")
    new_hex = params.get("new_bytes")
    if not original_hex or not new_hex:
        return False
    try:
        original_bytes = bytes.fromhex(original_hex)
        new_bytes = bytes.fromhex(new_hex)
    except ValueError:
        return False
    if len(original_bytes) != 8 or len(new_bytes) != 8:
        return False
    if original_bytes == new_bytes:
        return False
    # new_bytes must be original_bytes with the two 4-byte halves swapped
    return new_bytes == original_bytes[4:8] + original_bytes[0:4]


InstructionMutator.SAFE_TRANSFORMS["mips_adjacent_reorder"] = _mips_adjacent_reorder_validator


def build_reorder_gene(binary_path, rng, pairs=None):
    """pairs: list of (MIPSInstruction, MIPSInstruction) consecutive pairs,
    reuse find_trampoline_insertion_points() output or build your own."""
    if pairs is None:
        pairs = find_trampoline_insertion_points(binary_path)
    pairs = list(pairs)
    rng.shuffle(pairs)

    for ins_a, ins_b in pairs:
        if ins_b.file_offset != ins_a.file_offset + 4:
            continue
        if not _no_dependency(ins_a, ins_b):
            continue

        original_bytes = ins_a.raw_bytes + ins_b.raw_bytes
        new_bytes = ins_b.raw_bytes + ins_a.raw_bytes

        return Gene(
            operator=Operator.INSTRUCTION_TRANSFORM,
            location=ins_a.address,
            parameters={
                "transform": "mips_adjacent_reorder",
                "file_offset": ins_a.file_offset,
                "size": 8,
                "original_bytes": original_bytes.hex(),
                "new_bytes": new_bytes.hex(),
                "disasm": f"{ins_a.disasm} <-> {ins_b.disasm}",
                "function": ins_a.function,
                "delay_slot": False,
            },
        )
    return None

# Backwards-compatible alias (kept in case other code imports the old name).
_mips_commutative_swap = _mips_same_length_instruction_swap

# Technique 1: commutative operand swap (rs <-> rt) for commutative R-type
# ALU ops (AND/OR/XOR/NOR/ADDU/ADD). This is a *semantic n-gram* evasion
# technique: it changes the literal opcode bytes seen by the TF-IDF/n-gram
# tokenizer while leaving program behavior identical.
InstructionMutator.SAFE_TRANSFORMS[
    "mips_commutative_rs_rt_swap"
] = _mips_same_length_instruction_swap

# Technique 2: signed/unsigned ADD <-> ADDU equivalence substitution.
# Same rs/rt/rd/shamt fields, only the 6-bit funct field differs
# (0x20 ADD vs 0x21 ADDU). For the overwhelming majority of real-world
# code (no reliance on MIPS integer-overflow traps) these are behaviorally
# interchangeable, and toolchains routinely emit one or the other for the
# same source-level `+`. This is a distinct opcode-substitution technique
# separate from the operand-swap above, and further perturbs the
# instruction n-gram stream that the Siamese SVM is trained on.
InstructionMutator.SAFE_TRANSFORMS[
    "mips_add_addu_signedness_swap"
] = _mips_same_length_instruction_swap


# ---------------------------------------------------------------------------
# MIPS32 instruction inventory and conservative mutation support
# ---------------------------------------------------------------------------

@dataclass
class MIPSInstruction:
    """Decoded 32-bit MIPS instruction mapped to its exact ELF file offset."""
    address: int
    file_offset: int
    raw_bytes: bytes
    opcode: str
    operands: str
    disasm: str
    function: Optional[str] = None
    instruction_type: Optional[str] = None
    is_branch: bool = False
    is_jump: bool = False
    is_delay_slot: bool = False
    has_relocation: bool = False
    protected: bool = False
    reason: str = ""


class Radare2InstructionExtractor:
    """Extract executable MIPS instructions without rebuilding the ELF."""

    BRANCHES = {
        "b", "beq", "bne", "beql", "bnel", "bgez", "bgezal",
        "bgezall", "bgezl", "bltz", "bltzal", "bltzall", "bltzl",
        "blez", "blezl", "bgtz", "bgtzl",
    }
    JUMPS = {"j", "jal", "jr", "jalr", "b", "bal"}
    SENSITIVE = {"syscall", "break", "eret", "wait", "jr", "jalr", "j", "jal"}
    PROTECTED_REGS = {"$sp", "$ra", "$gp", "$fp", "$k0", "$k1"}

    def __init__(self, binary_path: Path, timeout: int = 30):
        self.binary_path = Path(binary_path)
        self.timeout = timeout

    def _run(self, command: str) -> str:
        p = subprocess.run(
            ["r2", "-q", "-c", command, str(self.binary_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(f"radare2 failed: {p.stderr.strip()}")
        if p.stderr.strip():
            # r2 prints non-fatal diagnostics (e.g. "esil_neg: unknown reg
            # w27" during aaef emulation on unusual instructions) to stderr
            # even on success. These don't affect the stdout JSON we parse,
            # but surface them at debug level instead of discarding them.
            logger.debug("%s: r2 stderr: %s", self.binary_path, p.stderr.strip())
        return p.stdout

    @staticmethod
    def _integer(value):
        if value is None:
            return None
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _split(opcode):
        parts = str(opcode or "").strip().split(None, 1)
        return (
            parts[0].lower() if parts else "",
            parts[1].strip() if len(parts) > 1 else "",
        )

    def _disassemble_range(self, address: int, size: int, label: str) -> List[dict]:
        """Disassemble a file-backed virtual-address range with radare2.

        `aoj` operates on virtual addresses and does not require ELF section
        headers. That makes it suitable for stripped/sectionless IoT malware
        where executable PT_LOAD segments are present but `.text` is absent.
        """
        data: List[dict] = []
        if address is None or size is None or int(size) <= 0:
            return data

        # MIPS32 instructions are 4 bytes. Keep chunk boundaries aligned so
        # the disassembler never starts a chunk in the middle of an
        # instruction. The final chunk may be shorter.
        chunk_size = 65536
        size = int(size)
        address = int(address)
        chunk_failures = 0

        for offset in range(0, size, chunk_size):
            chunk_addr = address + offset
            remaining = size - offset
            chunk_len = min(chunk_size, remaining)
            chunk_len -= chunk_len % 4
            if chunk_len < 4:
                continue

            inst_count = chunk_len // 4
            try:
                raw = self._run(
                    f"e scr.color=false;aoj {inst_count} @ {hex(chunk_addr)}"
                )
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    data.extend(parsed)
            except Exception as chunk_exc:
                chunk_failures += 1
                logger.debug(
                    "%s: r2 disassembly chunk @ %s (%s) failed: %s",
                    self.binary_path, hex(chunk_addr), label, chunk_exc,
                )

        if chunk_failures:
            logger.warning(
                "%s: %d %s disassembly chunks failed",
                self.binary_path, chunk_failures, label,
            )
        return data

    def _executable_load_segments(self) -> List[Tuple[int, int, int]]:
        """Return (virtual_address, file_offset, file_size) for executable PT_LOADs.

        This deliberately uses LIEF program headers rather than section
        headers. Sectionless ELF files still retain their program headers,
        and the loader executes PT_LOAD segments rather than `.text`.
        """
        try:
            binary = lief.parse(str(self.binary_path))
        except Exception as exc:
            logger.debug("%s: LIEF segment discovery failed: %s", self.binary_path, exc)
            return []

        if binary is None:
            return []

        ranges: List[Tuple[int, int, int]] = []
        for seg in binary.segments:
            try:
                is_load = seg.type == lief.ELF.Segment.TYPE.LOAD
                is_exec = bool(
                    int(seg.flags) & int(lief.ELF.Segment.FLAGS.X)
                )
                vaddr = int(seg.virtual_address)
                file_offset = int(seg.file_offset)
                file_size = int(seg.physical_size)
            except Exception:
                continue

            if is_load and is_exec and file_size > 0:
                ranges.append((vaddr, file_offset, file_size))

        return ranges

    def _extract_sectionless_segments(self) -> List[dict]:
        """Fallback disassembly for stripped/sectionless ELF files."""
        segments = self._executable_load_segments()
        if not segments:
            logger.warning(
                "%s: no executable PT_LOAD segment available for sectionless "
                "instruction extraction", self.binary_path
            )
            return []

        data: List[dict] = []
        for vaddr, file_offset, file_size in segments:
            # Only disassemble bytes that actually exist in the file. Do not
            # walk the zero-filled BSS portion (p_memsz > p_filesz).
            segment_data = self._disassemble_range(
                vaddr, file_size, f"executable PT_LOAD @ file+0x{file_offset:x}"
            )
            data.extend(segment_data)

        # A segment can overlap another executable mapping on unusual ELFs;
        # deduplicate by instruction address before normalisation.
        unique = {}
        for item in data:
            if isinstance(item, dict):
                addr = self._integer(item.get("offset", item.get("addr")))
                if addr is not None:
                    unique[addr] = item
        return [unique[k] for k in sorted(unique)]

    def extract(self) -> List[MIPSInstruction]:
        # Disassemble all executable PT_LOAD segments to ensure injected code caves,
        # trampolines, and junk instruction streams are included in the instruction stream.
        data = self._extract_sectionless_segments()
        if not data:
            try:
                sections = json.loads(self._run("aaa;e scr.color=false;iSj"))
                text_section = next((s for s in sections if str(s.get("name", "")) == ".text"), None) if isinstance(sections, list) else None
                if text_section is not None:
                    addr = self._integer(text_section.get("vaddr"))
                    size = self._integer(text_section.get("size"))
                    if addr is not None and size:
                        data = self._disassemble_range(addr, size, ".text")
            except Exception as exc:
                logger.warning("%s: fallback section analysis failed: %s", self.binary_path, exc)

        # If a named .text exists but radare2 could not decode it, the same
        # segment-based fallback is useful and costs nothing for normal ELFs.
        if not data and not used_section_fallback:
            logger.warning(
                "%s: .text produced 0 instructions; trying executable PT_LOAD fallback",
                self.binary_path,
            )
            data = self._extract_sectionless_segments()

        _debug("[DEBUG] r2 data type:", type(data).__name__)
        _debug("[DEBUG] r2 instruction count:", len(data))
        if data:
            _debug("[DEBUG] first r2 item:", data[0])
        _debug("[DEBUG] r2 data type:", type(data).__name__)
        _debug("[DEBUG] r2 instruction count:", len(data))
        if data:
            _debug("[DEBUG] first r2 item:", data[0])
        result = []
        for item in data:
            if not isinstance(item, dict):
                continue
            address = self._integer(item.get("offset", item.get("addr")))
            size = self._integer(item.get("size"))
            hexbytes = str(item.get("bytes") or "").strip()
            disasm = str(item.get("opcode") or item.get("disasm") or "").strip()

            if address is None or size != 4 or len(hexbytes) != 8:
                continue
            try:
                raw_bytes = bytes.fromhex(hexbytes)
            except ValueError:
                continue

            mnemonic, operands = self._split(disasm)
            is_branch = mnemonic in self.BRANCHES
            is_jump = mnemonic in self.JUMPS
            relocation = bool(item.get("reloc", False))
            protected_reg = any(
                reg in operands.lower() for reg in self.PROTECTED_REGS
            )
            sensitive = mnemonic in self.SENSITIVE
            protected = (
                relocation or protected_reg or sensitive or is_branch or is_jump
            )

            reasons = []
            if relocation:
                reasons.append("relocation")
            if protected_reg:
                reasons.append("protected-register")
            if sensitive:
                reasons.append("control-flow-sensitive")
            if is_branch:
                reasons.append("branch")
            if is_jump:
                reasons.append("jump")

            fcn = self._integer(item.get("fcn_addr"))
            result.append(
                MIPSInstruction(
                    address=address,
                    file_offset=-1,
                    raw_bytes=raw_bytes,
                    opcode=mnemonic,
                    operands=operands,
                    disasm=disasm,
                    function=hex(fcn) if fcn is not None else None,
                    instruction_type=str(item.get("type") or ""),
                    is_branch=is_branch,
                    is_jump=is_jump,
                    has_relocation=relocation,
                    protected=protected,
                    reason=",".join(reasons),
                )
            )

        result.sort(key=lambda x: x.address)

        for i in range(1, len(result)):
            previous = result[i - 1]
            if previous.is_branch or previous.is_jump:
                result[i].is_delay_slot = True
                result[i].protected = True
                result[i].reason = (
                    f"{result[i].reason},delay-slot"
                    if result[i].reason else "delay-slot"
                )

        return result


def map_mips_address_to_file_offset(binary_path: Path, address: int):
    try:
        output = subprocess.run(
            ["readelf", "-lW", str(binary_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except Exception:
        return None

    pattern = re.compile(
        r"^\s*LOAD\s+"
        r"(0x[0-9a-fA-F]+)\s+"
        r"(0x[0-9a-fA-F]+)\s+"
        r"(0x[0-9a-fA-F]+)\s+"
        r"(0x[0-9a-fA-F]+)\s+"
        r"(0x[0-9a-fA-F]+)\s+"
        r"([RWE ]+)\s+"
        r"(0x[0-9a-fA-F]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        offset = int(m.group(1), 16)
        vaddr = int(m.group(2), 16)
        filesz = int(m.group(4), 16)
        if vaddr <= address < vaddr + filesz:
            return offset + address - vaddr
    return None


def attach_mips_file_offsets(binary_path: Path, instructions):
    for ins in instructions:
        off = map_mips_address_to_file_offset(binary_path, ins.address)
        if off is None:
            ins.protected = True
            ins.reason = (
                f"{ins.reason},no-file-offset"
                if ins.reason else "no-file-offset"
            )
        else:
            ins.file_offset = off
    return instructions


def _decode_mips_rtype(raw_bytes: bytes) -> Optional[Dict[str, int]]:
    """
    Decode `raw_bytes` as a MIPS word and return its R-type (SPECIAL,
    opcode == 0) fields, or None if it is not actually an R-type
    instruction.

    This check is the fix for a real bug: funct-field-only matching
    (checking just the low 6 bits of the word) can misfire on I-type/
    J-type instructions whose low 6 bits happen to coincide with an
    R-type funct code by pure coincidence -- e.g. `addiu v1, zero, 0x20`
    has opcode=0x09 but its low byte is 0x20, which collides with the
    ADD funct code. Every R-type mutator must go through this decoder
    instead of reading `word & 0x3F` directly.
    """
    if len(raw_bytes) != 4:
        return None

    word = int.from_bytes(raw_bytes, "little")
    opcode = (word >> 26) & 0x3F

    if opcode != 0:
        return None

    return {
        "word": word,
        "rs": (word >> 21) & 0x1F,
        "rt": (word >> 16) & 0x1F,
        "rd": (word >> 11) & 0x1F,
        "shamt": (word >> 6) & 0x1F,
        "funct": word & 0x3F,
    }


class MIPSConservativeMutator:
    COMMUTATIVE_FUNCTS = {0x21, 0x24, 0x25, 0x26, 0x27}
    PROTECTED_REG_NUMBERS = {28, 29, 30, 31}

    @classmethod
    def mutate(cls, instruction: MIPSInstruction):
        decoded = _decode_mips_rtype(instruction.raw_bytes)
        if decoded is None:
            return None

        rs = decoded["rs"]
        rt = decoded["rt"]
        rd = decoded["rd"]
        shamt = decoded["shamt"]
        funct = decoded["funct"]

        if funct not in cls.COMMUTATIVE_FUNCTS:
            return None
        if rs == rt:
            return None
        if rs in cls.PROTECTED_REG_NUMBERS:
            return None
        if rt in cls.PROTECTED_REG_NUMBERS:
            return None
        if rd in cls.PROTECTED_REG_NUMBERS:
            return None

        new_word = (
            (rt << 21) | (rs << 16) | (rd << 11) |
            (shamt << 6) | funct
        )
        mutated = new_word.to_bytes(4, "little")
        return mutated if mutated != instruction.raw_bytes else None


class MIPSSignednessSwapMutator:
    """
    ADD (funct=0x20) <-> ADDU (funct=0x21) opcode-substitution technique.

    Only the 6-bit funct field is flipped; rs/rt/rd/shamt are untouched, so
    the instruction remains exactly 4 bytes and targets exactly the same
    registers. This is functionally a no-op for code that does not rely on
    the (rarely used, rarely handled) MIPS integer-overflow trap, while
    still changing the raw opcode bytes/mnemonic that an n-gram/TF-IDF
    model tokenizes on.
    """

    SWAP_FUNCTS = {0x20: 0x21, 0x21: 0x20}
    PROTECTED_REG_NUMBERS = {28, 29, 30, 31}

    @classmethod
    def mutate(cls, instruction: MIPSInstruction):
        decoded = _decode_mips_rtype(instruction.raw_bytes)
        if decoded is None:
            return None

        rs = decoded["rs"]
        rt = decoded["rt"]
        rd = decoded["rd"]
        shamt = decoded["shamt"]
        funct = decoded["funct"]

        if funct not in cls.SWAP_FUNCTS:
            return None
        if rs in cls.PROTECTED_REG_NUMBERS:
            return None
        if rt in cls.PROTECTED_REG_NUMBERS:
            return None
        if rd in cls.PROTECTED_REG_NUMBERS:
            return None

        new_funct = cls.SWAP_FUNCTS[funct]
        new_word = (
            (rs << 21) | (rt << 16) | (rd << 11) |
            (shamt << 6) | new_funct
        )
        mutated = new_word.to_bytes(4, "little")
        return mutated if mutated != instruction.raw_bytes else None


# Ordered list of (transform_name, mutator_class) pairs the GA tries, in
# randomized order, for every candidate instruction. Add new same-length
# substitution techniques here (and register a SAFE_TRANSFORMS validator
# above) to extend the instruction-level mutation arsenal without touching
# any of the GA driver logic below.
INSTRUCTION_MUTATION_TECHNIQUES: List[Tuple[str, Any]] = [
    ("mips_commutative_rs_rt_swap", MIPSConservativeMutator),
    ("mips_add_addu_signedness_swap", MIPSSignednessSwapMutator),
]


def find_safe_mips_instruction_candidates(binary_path: Path):
    extractor = Radare2InstructionExtractor(binary_path)

    instructions = extractor.extract()

    _debug(f"[DEBUG] Extracted instructions: {len(instructions)}")

    instructions = attach_mips_file_offsets(binary_path, instructions)

    _debug(
        "[DEBUG] With valid file offsets:",
        sum(1 for ins in instructions if ins.file_offset >= 0),
    )

    _debug(
        "[DEBUG] Protected:",
        sum(1 for ins in instructions if ins.protected),
    )

    _debug(
        "[DEBUG] Delay slots:",
        sum(1 for ins in instructions if ins.is_delay_slot),
    )

    candidates = [
        ins for ins in instructions
        if ins.file_offset >= 0
        and not ins.protected
        and not ins.is_delay_slot
    ]

    _debug(f"[DEBUG] Safe candidates: {len(candidates)}")

    return candidates


def apply_exact_instruction_patch(binary_path: Path, instruction_params: Dict[str, Any]):
    file_offset = instruction_params.get("file_offset")
    new_bytes = bytes.fromhex(instruction_params.get("new_bytes", ""))
    original_bytes = bytes.fromhex(instruction_params.get("original_bytes", ""))
    size = instruction_params.get("size", 4)

    if len(new_bytes) != size or len(original_bytes) != size:
        raise ValueError(f"Instruction patch size mismatch: expected {size} bytes")

    with open(binary_path, "r+b") as fh:
        fh.seek(file_offset)
        current = fh.read(size)
        if current != original_bytes:
            raise ValueError("Source changed since inventory was created; refusing patch")
        fh.seek(file_offset)
        fh.write(new_bytes)
        fh.flush()

def build_mips_instruction_gene(
    binary_path: Path,
    rng: random.Random,
    candidates: Optional[List[MIPSInstruction]] = None,
) -> Optional[Gene]:
    """
    Build one INSTRUCTION_TRANSFORM gene by trying every registered
    same-length substitution technique (see INSTRUCTION_MUTATION_TECHNIQUES)
    against a shuffled list of safe candidate instructions, in a shuffled
    technique order. Returns the first legal (instruction, technique) hit.
    """
    if candidates is None:
        candidates = find_safe_mips_instruction_candidates(binary_path)

    candidates = list(candidates)
    rng.shuffle(candidates)

    techniques = list(INSTRUCTION_MUTATION_TECHNIQUES)
    rng.shuffle(techniques)

    for ins in candidates:
        for transform_name, mutator_cls in techniques:
            new_bytes = mutator_cls.mutate(ins)
            if new_bytes is None:
                continue

            return Gene(
                operator=Operator.INSTRUCTION_TRANSFORM,
                location=ins.address,
                parameters={
                    "transform": transform_name,
                    "file_offset": ins.file_offset,
                    "size": 4,
                    "original_bytes": ins.raw_bytes.hex(),
                    "new_bytes": new_bytes.hex(),
                    "disasm": ins.disasm,
                    "function": ins.function,
                    "delay_slot": ins.is_delay_slot,
                },
            )
    return None


# ---------------------------------------------------------------------------
# ELF header reserved-byte (EI_PAD) mutation
# ---------------------------------------------------------------------------
#
# Bytes 9-15 of e_ident (EI_PAD, 7 bytes) are explicitly reserved and
# unspecified by the ELF spec: conformant loaders (Linux kernel, musl,
# glibc, uClibc used on IoT/MIPS targets) never read them. Rewriting them
# is a zero-risk, length-preserving structural mutation that perturbs
# tabular/metadata features (and any naive "raw header bytes" feature) fed
# to LightGBM, without touching code, relocations, or program behavior.

ELF_EI_PAD_OFFSET = 9
ELF_EI_PAD_LENGTH = 7


def apply_ei_pad_patch(binary_path: Path, params: Dict[str, Any]) -> None:
    new_bytes = bytes.fromhex(params.get("new_bytes", ""))
    original_bytes = bytes.fromhex(params.get("original_bytes", ""))

    if len(new_bytes) != ELF_EI_PAD_LENGTH or len(original_bytes) != ELF_EI_PAD_LENGTH:
        raise ValueError(
            f"EI_PAD patches must be exactly {ELF_EI_PAD_LENGTH} bytes"
        )

    with open(binary_path, "r+b") as fh:
        fh.seek(ELF_EI_PAD_OFFSET)
        current = fh.read(ELF_EI_PAD_LENGTH)
        if current != original_bytes:
            raise ValueError(
                "EI_PAD region changed since inventory was created; "
                "refusing patch"
            )
        fh.seek(ELF_EI_PAD_OFFSET)
        fh.write(new_bytes)
        fh.flush()


def build_ei_pad_gene(binary_path: Path, rng: random.Random) -> Optional[Gene]:
    try:
        with open(binary_path, "rb") as fh:
            fh.seek(ELF_EI_PAD_OFFSET)
            original = fh.read(ELF_EI_PAD_LENGTH)
    except OSError:
        return None

    if len(original) != ELF_EI_PAD_LENGTH:
        return None

    new_bytes = bytes(rng.getrandbits(8) for _ in range(ELF_EI_PAD_LENGTH))
    if new_bytes == original:
        return None

    return Gene(
        operator=Operator.ELF_TRANSFORM,
        location="e_ident.ei_pad",
        parameters={
            "field": "ei_pad",
            "file_offset": ELF_EI_PAD_OFFSET,
            "size": ELF_EI_PAD_LENGTH,
            "original_bytes": original.hex(),
            "new_bytes": new_bytes.hex(),
        },
    )


# ---------------------------------------------------------------------------
# Overlay padding payload styles
# ---------------------------------------------------------------------------
#
# Pure random bytes are the easiest thing for an entropy-based LightGBM
# feature to key on (random overlay data drives file entropy sharply
# upward, which is itself a well-known malware signal). These alternate
# styles let the GA search for padding that changes size/structural
# features without producing a suspiciously random entropy fingerprint.

PADDING_STYLES: Tuple[str, ...] = ("random", "low_entropy", "string_like")

_PRINTABLE_BYTES = bytes(range(0x20, 0x7F))


def generate_padding_payload(
    rng: random.Random,
    size: int,
    style: str = "random",
) -> bytes:
    if size <= 0:
        return b""

    if style == "low_entropy":
        # Repeat a short (1-4 byte) pattern, mimicking the low-entropy
        # zero/alignment padding real compilers/linkers emit.
        pattern_len = rng.randint(1, 4)
        pattern = bytes(rng.getrandbits(8) for _ in range(pattern_len))
        repeats = (size // pattern_len) + 1
        return (pattern * repeats)[:size]

    if style == "string_like":
        # Printable-ASCII filler, mimicking embedded version/string data
        # that's common in benign binaries and pulls string-derived and
        # entropy-derived tabular features toward the benign distribution.
        return bytes(rng.choice(_PRINTABLE_BYTES) for _ in range(size))

    # "random": original behavior, kept as the default/fallback style.
    return bytes(rng.getrandbits(8) for _ in range(size))


# ---------------------------------------------------------------------------
# Relocation-aware rewriting -- Phase 1: ELF "code cave" junk insertion
# ---------------------------------------------------------------------------
#
# This is the first, lowest-risk step toward true length-changing mutation.
# It does NOT require fixing up branches/jumps/relocations, because it never
# moves or resizes anything that already has references pointing at it:
#
#   * Linkers page-align each PT_LOAD segment's p_offset/p_vaddr. That
#     alignment routinely leaves a gap of already-present (usually zero)
#     bytes in the file between one segment's used region
#     (p_offset + p_filesz) and the next segment's p_offset.
#   * Those gap bytes are already part of the file (file size never
#     changes) -- they're just not currently counted as part of any
#     segment's loaded content.
#   * We overwrite that gap with junk MIPS instructions and grow the
#     owning segment's p_filesz/p_memsz to cover it, so the loader now
#     maps those bytes as executable content and any linear-sweep
#     disassembler (radare2, the TF-IDF pipeline) picks them up as real
#     instructions -- perturbing the n-gram stream without moving a
#     single existing byte or address.
#   * Nothing else in the file shifts, so no branch/jump/relocation target
#     ever needs recalculating. That's Phase 2's job (true mid-function
#     insertion with address-map fixups) and Phase 3's (control-flow
#     flattening) -- both real, separate pieces of work on top of this.
#
# The cave bytes are dead code: nothing jumps into them, so their content
# only has to be *safe if ever reached by accident*, not correct. We fill
# them with instructions whose only effect is writing to $zero (discarded)
# or true NOPs, so even accidental execution is inert.

PT_LOAD = 1
PF_X = 0x1
PF_W = 0x2
PF_R = 0x4

# Elf32_Ehdr fields after the 16-byte e_ident: e_type, e_machine (H,H);
# e_version (I); e_entry, e_phoff, e_shoff (I,I,I); e_flags (I);
# e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx (H x6)
_ELF32_EHDR_TAIL_FMT = "HHIIIIIHHHHHH"
# Elf32_Phdr: p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
# p_flags, p_align -- all 4-byte fields, same order on every 32-bit ELF
# target including MIPS o32.
_ELF32_PHDR_FMT = "IIIIIIII"


def _elf32_endianness(path: Path) -> str:
    with open(path, "rb") as fh:
        ident = fh.read(16)

    if len(ident) != 16 or ident[:4] != b"\x7fELF":
        raise ValueError(f"Not an ELF file: {path}")

    ei_class = ident[4]
    ei_data = ident[5]

    if ei_class != 1:
        raise ValueError(
            "Code-cave engine currently supports 32-bit ELF (MIPS o32) "
            "only"
        )

    if ei_data == 1:
        return "<"  # ELFDATA2LSB
    if ei_data == 2:
        return ">"  # ELFDATA2MSB
    raise ValueError(f"Unknown ELF data encoding: {ei_data}")


def _parse_elf32_program_headers(path: Path):
    endian = _elf32_endianness(path)

    with open(path, "rb") as fh:
        data = fh.read()

    ehdr_fmt = endian + _ELF32_EHDR_TAIL_FMT
    (
        _e_type, _e_machine, _e_version, _e_entry, e_phoff, _e_shoff,
        _e_flags, _e_ehsize, e_phentsize, e_phnum, _e_shentsize,
        _e_shnum, _e_shstrndx,
    ) = struct.unpack_from(ehdr_fmt, data, 16)

    phdr_fmt = endian + _ELF32_PHDR_FMT
    phdr_size = struct.calcsize(phdr_fmt)

    phdrs = []
    for i in range(e_phnum):
        entry_offset = e_phoff + i * e_phentsize
        (
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
            p_flags, p_align,
        ) = struct.unpack_from(phdr_fmt, data, entry_offset)
        phdrs.append(
            {
                "index": i,
                "entry_offset": entry_offset,
                "entry_size": phdr_size,
                "p_type": p_type,
                "p_offset": p_offset,
                "p_vaddr": p_vaddr,
                "p_paddr": p_paddr,
                "p_filesz": p_filesz,
                "p_memsz": p_memsz,
                "p_flags": p_flags,
                "p_align": p_align,
            }
        )

    return endian, e_phoff, e_phentsize, e_phnum, phdrs


def find_code_cave(
    binary_path: Path,
    min_size: int = 16,
) -> Optional[Dict[str, Any]]:
    """
    Locate the largest already-present, currently-unclaimed file gap that
    directly follows an executable PT_LOAD segment's used region, with
    enough matching virtual-address headroom before the next segment to
    grow into safely. Returns None if no binary-safe cave exists.
    """
    path = Path(binary_path)
    file_size = path.stat().st_size

    try:
        endian, e_phoff, e_phentsize, e_phnum, phdrs = (
            _parse_elf32_program_headers(path)
        )
    except ValueError:
        return None

    load_segs = [p for p in phdrs if p["p_type"] == PT_LOAD]
    exec_segs = [p for p in load_segs if p["p_flags"] & PF_X]

    if not exec_segs:
        return None

    # Every claimed file interval (any segment type), used to make sure the
    # cave we pick doesn't overlap something else already in the file
    # (e.g. PT_NOTE, PT_DYNAMIC, another PT_LOAD).
    intervals = sorted(
        (p["p_offset"], p["p_offset"] + p["p_filesz"])
        for p in phdrs
        if p["p_filesz"] > 0
    )

    best: Optional[Dict[str, Any]] = None

    for seg in exec_segs:
        seg_file_end = seg["p_offset"] + seg["p_filesz"]

        next_file_start = file_size
        for start, _end in intervals:
            if seg_file_end <= start < next_file_start:
                next_file_start = start

        file_gap = next_file_start - seg_file_end
        if file_gap <= 0:
            continue

        seg_vaddr_end = seg["p_vaddr"] + seg["p_memsz"]
        next_vaddr = None
        for other in load_segs:
            if other is seg:
                continue
            if other["p_vaddr"] >= seg_vaddr_end:
                if next_vaddr is None or other["p_vaddr"] < next_vaddr:
                    next_vaddr = other["p_vaddr"]

        vaddr_gap = (
            (next_vaddr - seg_vaddr_end)
            if next_vaddr is not None
            else file_gap
        )

        usable = min(file_gap, vaddr_gap)
        # Keep the cave 4-byte (instruction) aligned.
        usable -= usable % 4

        if usable < min_size:
            continue

        if best is None or usable > best["usable"]:
            best = {
                "segment": seg,
                "file_offset": seg_file_end,
                "usable": usable,
                "endian": endian,
                "e_phoff": e_phoff,
                "e_phentsize": e_phentsize,
            }

    return best


# Safe "dead" MIPS words: either a true NOP, or an ALU op whose result is
# discarded into $zero, so the encoding is harmless even in the (intended
# to be impossible) event it's ever actually executed.
def generate_junk_instruction_stream(
    rng: random.Random,
    count: int,
) -> bytes:
    safe_operand_regs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    words = bytearray()

    for _ in range(count):
        choice = rng.randint(0, 2)

        if choice == 0:
            # sll $zero, $zero, shamt -- true architectural NOP family.
            shamt = rng.randint(0, 31)
            word = shamt << 6

        elif choice == 1:
            # addu $zero, $rs, $rt -- result discarded, no overflow trap.
            rs = rng.choice(safe_operand_regs)
            rt = rng.choice(safe_operand_regs)
            word = (rs << 21) | (rt << 16) | (0 << 11) | 0x21

        else:
            # or $zero, $rs, $rt -- result discarded, no side effects.
            rs = rng.choice(safe_operand_regs)
            rt = rng.choice(safe_operand_regs)
            word = (rs << 21) | (rt << 16) | (0 << 11) | 0x25

        words.extend(word.to_bytes(4, "little"))

    return bytes(words)


def build_code_cave_gene(
    binary_path: Path,
    rng: random.Random,
    min_junk_instructions: int = 4,
    max_junk_instructions: int = 32,
) -> Optional[Gene]:
    cave = find_code_cave(binary_path, min_size=min_junk_instructions * 4)
    if cave is None:
        return None

    max_words = cave["usable"] // 4
    if max_words < min_junk_instructions:
        return None

    n = rng.randint(
        min_junk_instructions,
        min(max_junk_instructions, max_words),
    )
    junk = generate_junk_instruction_stream(rng, n)
    seg = cave["segment"]

    return Gene(
        operator=Operator.CODE_CAVE_INSERTION,
        location=cave["file_offset"],
        parameters={
            "file_offset": cave["file_offset"],
            "new_bytes": junk.hex(),
            "segment_index": seg["index"],
            "segment_vaddr": seg["p_vaddr"],
            "endian": cave["endian"],
            "e_phoff": cave["e_phoff"],
            "e_phentsize": cave["e_phentsize"],
            "original_filesz": seg["p_filesz"],
            "original_memsz": seg["p_memsz"],
            "instruction_count": n,
        },
    )


def apply_code_cave_patch(
    binary_path: Path,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Write junk instructions into the cave and grow the owning segment's
    p_filesz/p_memsz to cover them. Returns the list of (offset, size,
    original_bytes, new_bytes) modification records -- one for the cave
    bytes, one for the patched program-header entry -- in the same shape
    Validator._validate_instruction_changes already expects.
    """
    file_offset = int(params["file_offset"])
    new_bytes = bytes.fromhex(params["new_bytes"])
    grow = len(new_bytes)

    entry_offset = int(params["e_phoff"]) + (
        int(params["segment_index"]) * int(params["e_phentsize"])
    )
    endian = params["endian"]
    phdr_fmt = endian + _ELF32_PHDR_FMT
    entry_size = struct.calcsize(phdr_fmt)

    with open(binary_path, "r+b") as fh:
        # 1. Verify + write the cave bytes.
        fh.seek(file_offset)
        original_cave_bytes = fh.read(grow)
        fh.seek(file_offset)
        fh.write(new_bytes)

        # 2. Verify + patch the program-header entry (p_filesz, p_memsz).
        fh.seek(entry_offset)
        original_entry = fh.read(entry_size)
        (
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
            p_flags, p_align,
        ) = struct.unpack(phdr_fmt, original_entry)

        if (
            p_filesz != int(params["original_filesz"])
            or p_memsz != int(params["original_memsz"])
        ):
            raise ValueError(
                "Segment size changed since inventory was created; "
                "refusing code-cave patch"
            )

        new_entry = struct.pack(
            phdr_fmt,
            p_type, p_offset, p_vaddr, p_paddr,
            p_filesz + grow, p_memsz + grow,
            p_flags, p_align,
        )
        fh.seek(entry_offset)
        fh.write(new_entry)
        fh.flush()

    return [
        {
            "file_offset": file_offset,
            "size": grow,
            "original_bytes": original_cave_bytes.hex(),
            "new_bytes": new_bytes.hex(),
        },
        {
            "file_offset": entry_offset,
            "size": entry_size,
            "original_bytes": original_entry.hex(),
            "new_bytes": new_entry.hex(),
        },
    ]


# ---------------------------------------------------------------------------
# Phase 2: trampoline (detour) insertion -- mid-function junk insertion
# without any global address relocation
# ---------------------------------------------------------------------------
#
# The naive way to "insert code in the middle of a function" is to shift
# every byte after the insertion point and fix up every branch/jump/
# relocation whose target now falls on the wrong side of the shift. That's
# real, general-purpose relocation -- and for MIPS it also has to keep
# every PT_LOAD's file offset congruent to its virtual address modulo
# p_align, which a plain byte-shift breaks unless the shift itself is
# padded out to a full page. It's substantial, and substantially riskier
# to get exactly right.
#
# There's a standard alternative used by real binary patchers (Microsoft
# Detours, e9patch, and friends): a *trampoline*. Nothing else's address
# ever moves. Instead:
#
#   1. Pick two consecutive instructions A (at address X) and B (at X+4).
#   2. Overwrite A with `j trampoline`.
#   3. Overwrite B with a COPY of A's original bytes.
#   4. The trampoline (placed in a Phase-1 code cave) contains:
#      [junk instructions] + `j (X+8)` + [a copy of B's original bytes as
#      that jump's delay slot].
#
# Why this preserves execution order despite the MIPS branch-delay-slot
# rule (the instruction physically after a jump always executes before
# the jump lands): the delay slot of `j trampoline` at X is now B's
# *address* but A's *original content* -- so A's effect still runs first,
# immediately, exactly where it always did in program order. Control then
# reaches the trampoline, junk runs, and the final `j (X+8)`'s delay slot
# is a copy of B's original content -- so B's effect runs immediately
# after A's, in the same order as before. The instruction physically
# residing at X+8 onward is completely untouched, at its original
# address, so nothing anywhere else in the binary needs to change.
#
# Soundness requires the two instructions being relocated to be "inert" to
# move:
#   - Neither A nor B may itself be a branch/jump/jalr/jr: MIPS forbids
#     (or leaves undefined) a branch/jump inside another branch/jump's
#     delay slot, and `jal`/`bal`/`jalr` compute their return address from
#     their *own* runtime PC -- moving them would silently corrupt the
#     return address.
#   - Neither A nor B may already be flagged `protected` or
#     `is_delay_slot` by the existing radare2-based extractor (the same
#     conservative policy Phase 1 already applies).
#   - B's address must not be a branch/jump target anywhere else in the
#     binary: after the patch, B's address holds A's old content, not
#     B's -- so anything that jumps directly to B would now run the wrong
#     instruction. This check is necessarily best-effort: it only sees
#     statically-decodable direct branches/jumps (computed from the
#     binary's own instruction stream). Indirect control flow through
#     `jr`/`jalr` (returns, function pointers, jump tables) is invisible
#     to static analysis and is NOT covered by this check. Do not treat
#     trampoline-inserted variants as behavior-verified without a runtime
#     check (QEMU-user / Unicorn) on top of this static screening --
#     that's exactly the "behavioural analysis" fallback your own
#     Challenges slide already calls for on aggressive mutations.

_MIPS_BRANCH_OPCODES = {0x01, 0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17}


def _is_mips_control_flow(word: int) -> bool:
    opcode = (word >> 26) & 0x3F

    if opcode == 0:  # SPECIAL
        funct = word & 0x3F
        return funct in (0x08, 0x09)  # JR, JALR

    if opcode in (0x02, 0x03):  # J, JAL
        return True

    if opcode in _MIPS_BRANCH_OPCODES:
        return True

    # Coprocessor branches (BC1F/BC1T/BC0*/BC2*): conservatively treated
    # as control flow whenever the "branch" sub-format bit pattern
    # appears, rather than trying to fully decode every coprocessor
    # variant.
    if opcode == 0x11:  # COP1
        fmt = (word >> 21) & 0x1F
        return fmt == 0x08
    if opcode in (0x10, 0x12):  # COP0 / COP2
        fmt = (word >> 21) & 0x1F
        return fmt == 0x08

    return False


def _mips_branch_target(address: int, word: int) -> Optional[int]:
    opcode = (word >> 26) & 0x3F

    if opcode in (0x02, 0x03):  # J, JAL
        top4 = (address + 4) & 0xF0000000
        instr_index = word & 0x03FFFFFF
        return top4 | (instr_index << 2)

    if opcode in _MIPS_BRANCH_OPCODES:
        imm = word & 0xFFFF
        if imm & 0x8000:
            imm -= 0x10000
        return (address + 4) + (imm << 2)

    return None


def _compute_static_control_flow_targets(instructions) -> set:
    targets = set()
    for ins in instructions:
        if len(ins.raw_bytes) != 4:
            continue
        word = int.from_bytes(ins.raw_bytes, "little")
        target = _mips_branch_target(ins.address, word)
        if target is not None:
            targets.add(target)
    return targets


def _j_type_reachable(from_addr_plus4: int, target: int) -> bool:
    if target % 4 != 0:
        return False
    return (from_addr_plus4 & 0xF0000000) == (target & 0xF0000000)


def find_trampoline_insertion_points(
    binary_path: Path,
) -> List[Tuple[MIPSInstruction, MIPSInstruction]]:
    """
    Find (A, B) consecutive-instruction pairs safe to redirect through a
    trampoline: neither is control flow, neither is protected/a delay
    slot, they're contiguous in the file, and B's address is not a
    (statically visible) branch/jump target anywhere in the binary.
    """
    extractor = Radare2InstructionExtractor(binary_path)
    instructions = extractor.extract()
    instructions = attach_mips_file_offsets(binary_path, instructions)

    by_address = {
        ins.address: ins for ins in instructions if ins.file_offset >= 0
    }
    targets = _compute_static_control_flow_targets(instructions)

    pairs = []

    for ins_a in instructions:
        if ins_a.file_offset < 0 or ins_a.protected or ins_a.is_delay_slot:
            continue
        if len(ins_a.raw_bytes) != 4:
            continue

        word_a = int.from_bytes(ins_a.raw_bytes, "little")
        if _is_mips_control_flow(word_a) or ins_a.is_branch or ins_a.is_jump:
            continue

        ins_b = by_address.get(ins_a.address + 4)
        if ins_b is None:
            continue
        if ins_b.file_offset != ins_a.file_offset + 4:
            continue
        if ins_b.protected or ins_b.is_delay_slot:
            continue
        if len(ins_b.raw_bytes) != 4:
            continue

        word_b = int.from_bytes(ins_b.raw_bytes, "little")
        if _is_mips_control_flow(word_b) or ins_b.is_branch or ins_b.is_jump:
            continue

        if ins_b.address in targets:
            continue

        pairs.append((ins_a, ins_b))

    _debug(f"[DEBUG] Trampoline-safe instruction pairs: {len(pairs)}")
    return pairs


def build_trampoline_gene(
    binary_path: Path,
    rng: random.Random,
    min_junk_instructions: int = 2,
    max_junk_instructions: int = 16,
    pairs: Optional[List[Tuple[MIPSInstruction, MIPSInstruction]]] = None,
) -> Optional[Gene]:
    if pairs is None:
        pairs = find_trampoline_insertion_points(binary_path)
    if not pairs:
        return None

    pairs = list(pairs)
    rng.shuffle(pairs)

    for ins_a, ins_b in pairs:
        # Trampoline body = junk + return-jump + relocated delay slot.
        min_words_needed = min_junk_instructions + 2
        cave = find_code_cave(binary_path, min_size=min_words_needed * 4)
        if cave is None:
            return None

        max_words = cave["usable"] // 4
        max_junk_possible = max_words - 2
        if max_junk_possible < min_junk_instructions:
            continue

        n_junk = rng.randint(
            min_junk_instructions,
            min(max_junk_instructions, max_junk_possible),
        )

        seg = cave["segment"]
        trampoline_vaddr = seg["p_vaddr"] + (
            cave["file_offset"] - seg["p_offset"]
        )

        junk_bytes = generate_junk_instruction_stream(rng, n_junk)
        return_vaddr = ins_b.address + 4  # i.e. the original "A+8"
        return_jump_vaddr = trampoline_vaddr + len(junk_bytes)

        if not _j_type_reachable(return_jump_vaddr + 4, return_vaddr):
            continue
        if not _j_type_reachable(ins_a.address + 4, trampoline_vaddr):
            continue

        return_jump_word = (0x02 << 26) | ((return_vaddr >> 2) & 0x03FFFFFF)
        return_jump_bytes = return_jump_word.to_bytes(4, "little")

        trampoline_bytes = junk_bytes + return_jump_bytes + ins_b.raw_bytes

        redirect_word = (0x02 << 26) | (
            (trampoline_vaddr >> 2) & 0x03FFFFFF
        )
        site_a_new = redirect_word.to_bytes(4, "little")
        site_b_new = ins_a.raw_bytes  # A's content moves into B's slot

        return Gene(
            operator=Operator.TRAMPOLINE_INSERTION,
            location=ins_a.address,
            parameters={
                "site_a_file_offset": ins_a.file_offset,
                "site_a_original": ins_a.raw_bytes.hex(),
                "site_a_new": site_a_new.hex(),
                "site_b_file_offset": ins_b.file_offset,
                "site_b_original": ins_b.raw_bytes.hex(),
                "site_b_new": site_b_new.hex(),
                "trampoline_file_offset": cave["file_offset"],
                "trampoline_vaddr": trampoline_vaddr,
                "trampoline_bytes": trampoline_bytes.hex(),
                "return_vaddr": return_vaddr,
                "junk_instruction_count": n_junk,
                "segment_index": seg["index"],
                "segment_vaddr": seg["p_vaddr"],
                "endian": cave["endian"],
                "e_phoff": cave["e_phoff"],
                "e_phentsize": cave["e_phentsize"],
                "original_filesz": seg["p_filesz"],
                "original_memsz": seg["p_memsz"],
            },
        )

    return None


# ---------------------------------------------------------------------------
# Phase 3: opaque predicates
# ---------------------------------------------------------------------------
#
# Same trampoline mechanism as Phase 2 -- same redirect site pair, same
# code-cave placement, same delay-slot-preserving relocation, same segment
# growth -- with one change to what's inside the cave. Instead of a flat
# run of inert ALU junk, the body opens with a real MIPS conditional
# branch whose condition is a tautology:
#
#     beq $zero, $zero, skip_junk     ; ALWAYS taken -- $zero == $zero
#     <delay slot: true NOP>          ; branch delay slots always execute
#     <N junk instructions>           ; DEAD CODE: skip_junk always jumps
#                                      ; past this, so it is never reached
#                                      ; at runtime
#     skip_junk:
#     j return_vaddr
#     <relocated original delay slot>
#
# This is the standard "opaque predicate" idiom (a branch condition that
# is always true/false by construction, so control flow is 100%
# deterministic even though the byte stream contains a genuine branch
# instruction and a block that looks like a live alternate path). Runtime
# behavior is identical to the flat-junk trampoline -- the dead block is
# never executed -- but the static instruction/n-gram stream a
# disassembler-driven feature extractor sees now contains an actual
# conditional branch feeding into what looks like a real branch target,
# rather than a long homogeneous run of ALU ops. That's a qualitatively
# different signal than Phase 1/2's flat junk, and it's specifically
# aimed at any feature (branch-instruction ratio, branch/jump density,
# n-grams anchored on branch opcodes) that a classifier might use to
# separate "looks like real compiled code" from "looks like inserted
# filler." The same static-target caveat from Phase 2 applies here
# identically (see find_trampoline_insertion_points()'s docstring).

def build_opaque_predicate_gene(
    binary_path: Path,
    rng: random.Random,
    min_junk_instructions: int = 2,
    max_junk_instructions: int = 16,
    pairs: Optional[List[Tuple[MIPSInstruction, MIPSInstruction]]] = None,
) -> Optional[Gene]:
    if pairs is None:
        pairs = find_trampoline_insertion_points(binary_path)
    if not pairs:
        return None

    pairs = list(pairs)
    rng.shuffle(pairs)

    for ins_a, ins_b in pairs:
        # Body = beq + delay-slot filler + junk(N) + return-jump +
        # relocated delay slot = 4 fixed words + N junk words.
        min_words_needed = 4 + min_junk_instructions
        cave = find_code_cave(binary_path, min_size=min_words_needed * 4)
        if cave is None:
            return None

        max_words = cave["usable"] // 4
        max_junk_possible = max_words - 4
        if max_junk_possible < min_junk_instructions:
            continue

        n_junk = rng.randint(
            min_junk_instructions,
            min(max_junk_instructions, max_junk_possible),
        )

        seg = cave["segment"]
        trampoline_vaddr = seg["p_vaddr"] + (
            cave["file_offset"] - seg["p_offset"]
        )

        beq_addr = trampoline_vaddr
        delay_slot_addr = beq_addr + 4
        junk_start_addr = delay_slot_addr + 4
        skip_target_addr = junk_start_addr + 4 * n_junk
        return_jump_addr = skip_target_addr
        return_vaddr = ins_b.address + 4

        offset_words, remainder = divmod(
            skip_target_addr - (beq_addr + 4), 4
        )
        if remainder != 0:
            continue
        if not (-32768 <= offset_words <= 32767):
            continue

        if not _j_type_reachable(return_jump_addr + 4, return_vaddr):
            continue
        if not _j_type_reachable(ins_a.address + 4, trampoline_vaddr):
            continue

        # beq $zero, $zero, offset_words -- unconditionally taken.
        beq_word = (0x04 << 26) | (0 << 21) | (0 << 16) | (
            offset_words & 0xFFFF
        )
        delay_slot_filler_word = 0  # sll $zero, $zero, 0 -- true NOP

        junk_bytes = generate_junk_instruction_stream(rng, n_junk)

        return_jump_word = (0x02 << 26) | ((return_vaddr >> 2) & 0x03FFFFFF)

        trampoline_bytes = (
            beq_word.to_bytes(4, "little")
            + delay_slot_filler_word.to_bytes(4, "little")
            + junk_bytes
            + return_jump_word.to_bytes(4, "little")
            + ins_b.raw_bytes
        )

        redirect_word = (0x02 << 26) | (
            (trampoline_vaddr >> 2) & 0x03FFFFFF
        )
        site_a_new = redirect_word.to_bytes(4, "little")
        site_b_new = ins_a.raw_bytes

        return Gene(
            operator=Operator.OPAQUE_PREDICATE_INSERTION,
            location=ins_a.address,
            parameters={
                "site_a_file_offset": ins_a.file_offset,
                "site_a_original": ins_a.raw_bytes.hex(),
                "site_a_new": site_a_new.hex(),
                "site_b_file_offset": ins_b.file_offset,
                "site_b_original": ins_b.raw_bytes.hex(),
                "site_b_new": site_b_new.hex(),
                "trampoline_file_offset": cave["file_offset"],
                "trampoline_vaddr": trampoline_vaddr,
                "trampoline_bytes": trampoline_bytes.hex(),
                "return_vaddr": return_vaddr,
                "junk_instruction_count": n_junk,
                "predicate_kind": "beq_zero_zero_always_taken",
                "segment_index": seg["index"],
                "segment_vaddr": seg["p_vaddr"],
                "endian": cave["endian"],
                "e_phoff": cave["e_phoff"],
                "e_phentsize": cave["e_phentsize"],
                "original_filesz": seg["p_filesz"],
                "original_memsz": seg["p_memsz"],
            },
        )

    return None


def apply_trampoline_patch(
    binary_path: Path,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Apply a TRAMPOLINE_INSERTION gene: two in-place 4-byte word rewrites
    (the redirect + the relocated delay slot), one cave write (the
    trampoline body), and one program-header growth patch. Returns
    modification records for all four writes.
    """
    site_a_offset = int(params["site_a_file_offset"])
    site_a_original = bytes.fromhex(params["site_a_original"])
    site_a_new = bytes.fromhex(params["site_a_new"])

    site_b_offset = int(params["site_b_file_offset"])
    site_b_original = bytes.fromhex(params["site_b_original"])
    site_b_new = bytes.fromhex(params["site_b_new"])

    trampoline_offset = int(params["trampoline_file_offset"])
    trampoline_bytes = bytes.fromhex(params["trampoline_bytes"])
    grow = len(trampoline_bytes)

    entry_offset = int(params["e_phoff"]) + (
        int(params["segment_index"]) * int(params["e_phentsize"])
    )
    endian = params["endian"]
    phdr_fmt = endian + _ELF32_PHDR_FMT
    entry_size = struct.calcsize(phdr_fmt)

    modifications: List[Dict[str, Any]] = []

    with open(binary_path, "r+b") as fh:
        # 1. Redirect: A becomes `j trampoline`.
        fh.seek(site_a_offset)
        current = fh.read(4)
        if current != site_a_original:
            raise ValueError(
                "Trampoline site A changed since inventory; refusing patch"
            )
        fh.seek(site_a_offset)
        fh.write(site_a_new)
        modifications.append({
            "file_offset": site_a_offset,
            "size": 4,
            "original_bytes": site_a_original.hex(),
            "new_bytes": site_a_new.hex(),
        })

        # 2. Relocated delay slot: B becomes A's original content.
        fh.seek(site_b_offset)
        current = fh.read(4)
        if current != site_b_original:
            raise ValueError(
                "Trampoline site B changed since inventory; refusing patch"
            )
        fh.seek(site_b_offset)
        fh.write(site_b_new)
        modifications.append({
            "file_offset": site_b_offset,
            "size": 4,
            "original_bytes": site_b_original.hex(),
            "new_bytes": site_b_new.hex(),
        })

        # 3. Trampoline body, written into the code cave.
        fh.seek(trampoline_offset)
        original_trampoline_region = fh.read(grow)
        fh.seek(trampoline_offset)
        fh.write(trampoline_bytes)
        modifications.append({
            "file_offset": trampoline_offset,
            "size": grow,
            "original_bytes": original_trampoline_region.hex(),
            "new_bytes": trampoline_bytes.hex(),
        })

        # 4. Grow the owning segment to cover the trampoline bytes.
        fh.seek(entry_offset)
        original_entry = fh.read(entry_size)
        (
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
            p_flags, p_align,
        ) = struct.unpack(phdr_fmt, original_entry)

        if (
            p_filesz != int(params["original_filesz"])
            or p_memsz != int(params["original_memsz"])
        ):
            raise ValueError(
                "Segment size changed since inventory was created; "
                "refusing trampoline patch"
            )

        new_entry = struct.pack(
            phdr_fmt,
            p_type, p_offset, p_vaddr, p_paddr,
            p_filesz + grow, p_memsz + grow,
            p_flags, p_align,
        )
        fh.seek(entry_offset)
        fh.write(new_entry)
        modifications.append({
            "file_offset": entry_offset,
            "size": entry_size,
            "original_bytes": original_entry.hex(),
            "new_bytes": new_entry.hex(),
        })

        fh.flush()

    return modifications


# ---------------------------------------------------------------------------
# Mutation engine
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    path: str
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "path": self.path,
            "checks": dict(self.checks),
            "metrics": copy.deepcopy(self.metrics),
            "reason": self.reason,
        }


class Validator:
    """Validate parseability plus original/variant preservation invariants."""

    def __init__(
        self,
        expected_machine: Any,
        baseline_path: Optional[str] = None,
        use_radare2: bool = False,
        report_path: Optional[str] = None,
        allow_byte_modifications: bool = False,
    ):
        self.expected_machine = expected_machine
        self.baseline_path = baseline_path
        self.use_radare2 = use_radare2 and shutil.which("r2") is not None
        self.report_path = report_path
        self.allow_byte_modifications = allow_byte_modifications
        self.results: List[ValidationResult] = []
        self.baseline: Optional[BinarySnapshot] = (
            snapshot_binary(baseline_path) if baseline_path else None
        )
    def _validate_instruction_changes(
        self,
        original_path: Path,
        variant_path: Path,
        allowed_modifications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        original = original_path.read_bytes()
        variant = variant_path.read_bytes()

        changed_offsets = []

        common_size = min(len(original), len(variant))

        for offset in range(common_size):
            if original[offset] != variant[offset]:
                changed_offsets.append(offset)

        # Instruction mutations must not truncate the binary. Appended overlay
        # padding is permitted, but the original binary range must only differ
        # at the exact expected mutation offsets.
        if len(variant) < len(original):
            return {
                "valid": False,
                "reason": (
                    f"File truncated: "
                    f"{len(original)} -> {len(variant)} bytes"
                ),
                "changed_offsets": changed_offsets,
            }

        expected_offsets = set()

        for modification in allowed_modifications:
            offset = modification["file_offset"]
            size = modification["size"]

            for i in range(size):
                expected_offsets.add(offset + i)

            expected = bytes.fromhex(modification["original_bytes"])
            actual_new = bytes.fromhex(modification["new_bytes"])

            if original[offset:offset + size] != expected:
                return {
                    "valid": False,
                    "reason": f"Original bytes do not match at offset {offset}",
                    "changed_offsets": changed_offsets,
                }

            if variant[offset:offset + size] != actual_new:
                return {
                    "valid": False,
                    "reason": f"Mutation bytes do not match at offset {offset}",
                    "changed_offsets": changed_offsets,
                }

        unexpected_offsets = [
            offset
            for offset in changed_offsets
            if offset not in expected_offsets
        ]

        return {
            "valid": len(unexpected_offsets) == 0,
            "changed_offsets": changed_offsets,
            "expected_offsets": sorted(expected_offsets),
            "unexpected_offsets": unexpected_offsets,
        }
        
    def validate_file(self, path: str) -> bool:
        return self.validate(path).valid

    def validate(
        self,
        path: str,
        allowed_modifications: Optional[List[Dict[str, Any]]] = None,
        allowed_segment_growth: Optional[Dict[int, int]] = None,
    ) -> ValidationResult:
        checks: Dict[str, bool] = {
            "lief_parse": self._validate_lief_parse(path),
            "elf_structure": False,
            "architecture": False,
        }
        metrics: Dict[str, Any] = {}

        if checks["lief_parse"]:
            checks["elf_structure"] = self._validate_elf_structure(path)
            checks["architecture"] = self._validate_architecture(path)

        if self.baseline is not None:
            invariant_checks, invariant_metrics = self._validate_preservation(
                path, allowed_segment_growth=allowed_segment_growth
            )
            checks.update(invariant_checks)
            metrics.update(invariant_metrics)

        if self.use_radare2:
            checks["radare2"] = self._validate_radare2(path)
            
        if allowed_modifications:
            if self.baseline is None:
                return ValidationResult(
                    valid=False,
                    path=path,
                    checks=checks,
                    metrics=metrics,
                    reason="Cannot validate instruction changes without a baseline",
                )

            change_validation = self._validate_instruction_changes(
                Path(self.baseline.path),
                Path(path),
                allowed_modifications,
            )

            metrics["instruction_changes"] = change_validation

            if not change_validation["valid"]:
                return ValidationResult(
                    valid=False,
                    path=path,
                    checks=checks,
                    metrics=metrics,
                    reason="Unexpected byte modifications detected",
                )
                
        valid = all(checks.values())
        failed = [name for name, passed in checks.items() if not passed]

        result = ValidationResult(
            valid=valid,
            path=path,
            checks=checks,
            metrics=metrics,
            reason="ok" if valid else "failed: " + ", ".join(failed),
        )

        self.results.append(result)
        self._write_report()
        return result

    @staticmethod
    def _validate_lief_parse(path: str) -> bool:
        try:
            return lief.parse(path) is not None
        except Exception:
            return False

    def _validate_architecture(self, path: str) -> bool:
        try:
            binary = lief.parse(path)
            return (
                binary is not None
                and binary.header.machine_type == self.expected_machine
            )
        except Exception:
            return False

    @staticmethod
    def _validate_elf_structure(path: str) -> bool:
        try:
            binary = lief.parse(path)
            if binary is None:
                return False

            load_segments = [
                seg
                for seg in binary.segments
                if seg.type == lief.ELF.Segment.TYPE.LOAD
            ]

            if not load_segments:
                return False

            for seg in load_segments:
                if int(seg.physical_size) > int(seg.virtual_size):
                    return False

            entrypoint = int(binary.header.entrypoint)

            for seg in load_segments:
                start = int(seg.virtual_address)
                end = start + int(seg.virtual_size)
                if start <= entrypoint < end:
                    return True

            if binary.header.file_type == lief.ELF.Header.FILE_TYPE.DYN:
                min_vaddr = min(int(seg.virtual_address) for seg in load_segments)
                adjusted = entrypoint + min_vaddr

                return any(
                    int(seg.virtual_address)
                    <= adjusted
                    < int(seg.virtual_address) + int(seg.virtual_size)
                    for seg in load_segments
                )

            return False

        except Exception:
            return False

    def _validate_preservation(
        self,
        path: str,
        allowed_segment_growth: Optional[Dict[int, int]] = None,
    ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        checks: Dict[str, bool] = {}
        metrics: Dict[str, Any] = {}

        assert self.baseline is not None

        try:
            with open(self.baseline.path, "rb") as handle:
                original_bytes = handle.read()

            with open(path, "rb") as handle:
                variant_bytes = handle.read()

            original_size = len(original_bytes)
            variant_size = len(variant_bytes)

            overlap = min(original_size, variant_size)
            changed = sum(
                1
                for a, b in zip(original_bytes[:overlap], variant_bytes[:overlap])
                if a != b
            )

            metrics.update(
                {
                    "original_size": original_size,
                    "variant_size": variant_size,
                    "appended_bytes": max(0, variant_size - original_size),
                    "changed_bytes_in_overlap": changed,
                    "percent_overlap_changed": (
                        100.0 * changed / overlap if overlap else 0.0
                    ),
                }
            )

            if self.allow_byte_modifications:
                checks["original_bytes_preserved"] = True
            else:
                checks["original_bytes_preserved"] = (
                    variant_size >= original_size
                    and variant_bytes[:original_size] == original_bytes
                )

            variant = lief.parse(path)
            if variant is None:
                return checks, metrics

            checks["entrypoint_preserved"] = (
                int(variant.header.entrypoint) == self.baseline.entrypoint
            )
            checks["segment_count_preserved"] = (
                len(variant.segments) == self.baseline.segment_count
            )

            if allowed_segment_growth:
                # A code-cave gene intentionally grows exactly one
                # PT_LOAD segment's (physical_size, virtual_size) by a
                # known, recorded amount. Compare against that expected
                # signature instead of exact byte-for-byte equality.
                expected = tuple(
                    (
                        vaddr, paddr,
                        physsz + allowed_segment_growth.get(vaddr, 0),
                        virtsz + allowed_segment_growth.get(vaddr, 0),
                        flags,
                    )
                    for (vaddr, paddr, physsz, virtsz, flags)
                    in self.baseline.load_segments
                )
                checks["load_segments_preserved"] = (
                    _load_segment_signature(variant) == expected
                )
            else:
                checks["load_segments_preserved"] = (
                    _load_segment_signature(variant) == self.baseline.load_segments
                )
            checks["section_count_preserved"] = (
                len(variant.sections) == self.baseline.section_count
            )
            checks["file_type_preserved"] = (
                variant.header.file_type == self.baseline.file_type
            )
            checks["machine_preserved"] = (
                variant.header.machine_type == self.baseline.machine
            )

        except Exception:
            for name in (
                "original_bytes_preserved",
                "entrypoint_preserved",
                "segment_count_preserved",
                "load_segments_preserved",
                "section_count_preserved",
                "file_type_preserved",
                "machine_preserved",
            ):
                checks.setdefault(name, False)

        return checks, metrics

    @staticmethod
    def _validate_radare2(path: str) -> bool:
        try:
            result = subprocess.run(
                ["r2", "-q", "-c", "aa;afl", path],
                capture_output=True,
                timeout=30,
                text=True,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def _write_report(self) -> None:
        if not self.report_path:
            return

        directory = os.path.dirname(os.path.abspath(self.report_path))
        os.makedirs(directory, exist_ok=True)

        with open(self.report_path, "w", encoding="utf-8") as handle:
            json.dump(
                [result.to_dict() for result in self.results],
                handle,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

@dataclass
class FitnessWeights:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2


@dataclass
class FitnessResult:
    fitness: float
    lgbm_delta: float
    svm_delta: float
    modification_cost: float
    baseline_lgbm: Optional[float]
    variant_lgbm: Optional[float]
    baseline_svm: Optional[float]
    variant_svm: Optional[float]
    baseline_lgbm_prediction: Optional[int] = None
    variant_lgbm_prediction: Optional[int] = None
    baseline_svm_prediction: Optional[int] = None
    variant_svm_prediction: Optional[int] = None
    explainability: Optional[Dict[str, Any]] = None
    lgbm_normalized_delta: float = 0.0
    svm_normalized_delta: float = 0.0
    lgbm_evasion: Optional[bool] = None
    svm_evasion: Optional[bool] = None
    baseline_lgbm_mpd: Optional[float] = None
    variant_lgbm_mpd: Optional[float] = None
    lgbm_mpd_delta: float = 0.0
    baseline_svm_mpd: Optional[float] = None
    variant_svm_mpd: Optional[float] = None
    svm_mpd_delta: float = 0.0
    lgbm_uncertainty_alert: Optional[bool] = None
    svm_uncertainty_alert: Optional[bool] = None
    combined_delta_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "lgbm_score_delta": self.lgbm_delta,
            "svm_score_delta": self.svm_delta,
            "modification_cost": self.modification_cost,
            "baseline_lgbm": self.baseline_lgbm,
            "variant_lgbm": self.variant_lgbm,
            "baseline_svm": self.baseline_svm,
            "variant_svm": self.variant_svm,
            "baseline_lgbm_prediction": self.baseline_lgbm_prediction,
            "variant_lgbm_prediction": self.variant_lgbm_prediction,
            "baseline_svm_prediction": self.baseline_svm_prediction,
            "variant_svm_prediction": self.variant_svm_prediction,
            "explainability": self.explainability,
            "lgbm_normalized_delta": self.lgbm_normalized_delta,
            "svm_normalized_delta": self.svm_normalized_delta,
            "lgbm_evasion": self.lgbm_evasion,
            "svm_evasion": self.svm_evasion,
            "joint_evasion": bool(self.lgbm_evasion is True and self.svm_evasion is True),
            "any_evasion": bool(self.lgbm_evasion is True or self.svm_evasion is True),
            "baseline_lgbm_mpd": self.baseline_lgbm_mpd,
            "variant_lgbm_mpd": self.variant_lgbm_mpd,
            "lgbm_mpd_delta": self.lgbm_mpd_delta,
            "baseline_svm_mpd": self.baseline_svm_mpd,
            "variant_svm_mpd": self.variant_svm_mpd,
            "svm_mpd_delta": self.svm_mpd_delta,
            "lgbm_uncertainty_alert": self.lgbm_uncertainty_alert,
            "svm_uncertainty_alert": self.svm_uncertainty_alert,
            "combined_delta_score": self.combined_delta_score,
        }


# ---------------------------------------------------------------------------
# Siamese SVM inference
# ---------------------------------------------------------------------------

class SiameseSVMScorer:
    """
    Scores a binary against the SAME fixed 10-row reference set used by
    ONNXDetector.predict_siamese_fixed_ref, so GA fitness and deployed
    inference agree exactly.

    score = mean(margin against malware refs) - mean(margin against benign refs)
    score > 0  => malware-like
    """

    def __init__(self, model_path: str, vectorizer_path: str, tfidf_dir: str):
        import joblib

        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)

        for p in (self.model_path, self.vectorizer_path):
            if not p.is_file():
                raise FileNotFoundError(f"Siamese SVM artifact not found: {p}")

        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict):
            raise TypeError("siamese_svm1.pkl must contain a dict with 'model' and 'scaler'.")

        self.model = artifact.get("model")
        self.scaler = artifact.get("scaler")
        if self.model is None or self.scaler is None:
            raise ValueError("siamese_svm1.pkl must contain both 'model' and 'scaler'.")

        self.vectorizer = joblib.load(self.vectorizer_path)

        self.reference_vectors, self.reference_labels, self.reference_indices = (
            load_fixed_reference(tfidf_dir)
        )

        print(
            "[SIAMESE] loaded:",
            f"model={self.model_path}",
            f"features={getattr(self.model, 'n_features_in_', '?')}",
            f"fixed_reference={len(self.reference_indices)} rows "
            f"({int((self.reference_labels==1).sum())} malware, "
            f"{int((self.reference_labels==0).sum())} benign)",
        )

    @staticmethod
    def _normalize_immediate(value_str):
        try:
            if value_str is None:
                value_str = "0"
            value = abs(int(value_str, 0))
            if value == 0:
                return "ZERO_IMM"
            if value <= 15:
                return "SMALL_IMM"
            if value <= 255:
                return "MED_IMM"
            if value > 65535:
                return "ADDR_IMM"
            return "LARGE_IMM"
        except (TypeError, ValueError):
            return "IMM"

    REGISTER_MAP = {
        "a0": "ARG_REG", "a1": "ARG_REG", "a2": "ARG_REG", "a3": "ARG_REG",
        "v0": "RET_REG", "v1": "RET_REG",
        "t0": "SCRATCH_REG", "t1": "SCRATCH_REG", "t2": "SCRATCH_REG", "t3": "SCRATCH_REG",
        "t4": "SCRATCH_REG", "t5": "SCRATCH_REG", "t6": "SCRATCH_REG", "t7": "SCRATCH_REG",
        "t8": "SCRATCH_REG", "t9": "SCRATCH_REG",
        "s0": "SAVED_REG", "s1": "SAVED_REG", "s2": "SAVED_REG", "s3": "SAVED_REG",
        "s4": "SAVED_REG", "s5": "SAVED_REG", "s6": "SAVED_REG", "s7": "SAVED_REG",
        "sp": "STACK_PTR", "ra": "RETURN_ADDR", "gp": "GLOBAL_PTR", "fp": "FRAME_PTR",
        "zero": "ZERO_REG", "at": "ASSEM_TEMP", "k0": "KERNEL_REG", "k1": "KERNEL_REG",
    }

    @classmethod
    def _normalize_operand(cls, operand: str) -> str:
        operand = operand.strip().lower()
        if operand.startswith("$"):
            operand = operand[1:]

        memory_match = re.match(r"(?:(-?(?:0x[\da-f]+|\d+))?)\(([\$\w]+)\)$", operand)
        if memory_match:
            offset_str, reg_str = memory_match.groups()
            if not offset_str:
                offset_str = "0"
            reg_str = reg_str.lstrip("$")
            reg_token = cls.REGISTER_MAP.get(reg_str, "REG")
            offset_token = cls._normalize_immediate(offset_str)
            return f"MEM_{offset_token}_{reg_token}"

        if operand in cls.REGISTER_MAP:
            return cls.REGISTER_MAP[operand]
        if re.fullmatch(r"0x[\da-f]+", operand):
            return "ADDR_IMM"
        if operand.startswith("sym.") or operand.startswith("fcn."):
            return "FUNC"
        if operand.startswith("str."):
            return "STRING"
        if re.fullmatch(r"-?(?:0x[\da-f]+|\d+)", operand):
            return cls._normalize_immediate(operand)
        return operand

    @classmethod
    def _normalize_instructions(cls, instructions):
        normalized_lines = []
        for instruction in instructions:
            disasm = instruction.disasm.strip()
            if not disasm:
                continue
            parts = disasm.split(None, 1)
            mnemonic = parts[0].lower()
            tokens = [mnemonic]
            if len(parts) > 1:
                operands = [v.strip() for v in parts[1].split(",")]
                for operand in operands:
                    tokens.append(cls._normalize_operand(operand))
            normalized_lines.append(" ".join(tokens))
        return normalized_lines

    @staticmethod
    def _make_bigram_document(normalized_lines):
        tokens = []
        for line in normalized_lines:
            tokens.extend(line.split())
        if len(tokens) < 2:
            return ""
        bigrams = ["_".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        return " ".join(bigrams)

    def _tfidf_for_binary(self, binary_path: str):
        instructions = Radare2InstructionExtractor(Path(binary_path)).extract()
        if not instructions:
            raise RuntimeError(
                f"No MIPS instructions extracted from {binary_path}; "
                "the ELF has no recoverable executable instruction stream"
            )
        normalized = self._normalize_instructions(instructions)
        document = self._make_bigram_document(normalized)
        if not document:
            raise RuntimeError(f"Could not build instruction bigrams from {binary_path}")
        return self.vectorizer.transform([document])

    def predict_details(self, binary_path: str):
        candidate = self._tfidf_for_binary(binary_path).toarray().astype(np.float32)
        diffs = np.abs(self.reference_vectors - candidate)  # (10, D)
        scaled = self.scaler.transform(diffs)
        margins = self.model.decision_function(scaled)  # (10,)

        malware_margin = float(margins[self.reference_labels == 1].mean())
        benign_margin = float(margins[self.reference_labels == 0].mean())
        score = malware_margin - benign_margin
        prediction = int(score > 0)

        return {
            "score": score,
            "prediction": prediction,
            "malware_score": malware_margin,
            "benign_score": benign_margin,
            "reference_indices": self.reference_indices.tolist(),
            "reference_labels": self.reference_labels.tolist(),
        }

    def __call__(self, binary_path: str) -> float:
        return float(self.predict_details(binary_path)["score"])


class BootstrapMPDScorer:
    """Paper-faithful bootstrap MPD scorer for GA variants.

    LightGBM: ensemble members consume the extracted tabular feature dict.
    Siamese SVM: calibrated ensemble members consume the same pairwise
    |candidate - reference| TF-IDF representation used by the uncertainty
    pipeline.

    MPD(x) = min(E[(q-0)^2], E[(q-1)^2]).
    The scorer returns MPD only; the GA converts it into an uncertainty delta.
    """

    def __init__(self, ensemble_path: str, kind: str,
                 siamese_scorer: Optional["SiameseSVMScorer"] = None,
                 threshold: Optional[float] = None,
                 feature_names: Optional[List[str]] = None):
        self.kind = kind
        self.ensemble_path = Path(ensemble_path)
        self.siamese_scorer = siamese_scorer
        self.threshold = threshold
        self.feature_names = list(feature_names) if feature_names else None
        if not self.ensemble_path.is_file():
            raise FileNotFoundError(f"MPD ensemble not found: {self.ensemble_path}")
        with open(self.ensemble_path, "rb") as f:
            self.ensemble = pickle.load(f)
        if not isinstance(self.ensemble, (list, tuple)) or not self.ensemble:
            raise ValueError(f"Expected a non-empty bootstrap ensemble in {self.ensemble_path}")
        print(f"[MPD] loaded {kind} ensemble: {len(self.ensemble)} models")

    @staticmethod
    def _mpd(probabilities: np.ndarray) -> float:
        q = np.asarray(probabilities, dtype=float).reshape(-1)
        u0 = float(np.mean((q - 0.0) ** 2))
        u1 = float(np.mean((q - 1.0) ** 2))
        return float(min(u0, u1))

    def __call__(self, value: Any) -> float:
        if self.kind == "lightgbm":
            # The ELF extractor returns a feature dict containing metadata and
            # non-model fields.  Bootstrap LightGBM models expect the exact
            # numeric feature matrix used during training, not the raw dict.
            if isinstance(value, dict):
                if not self.feature_names:
                    # Prefer the bootstrap model's fitted feature names when
                    # available; otherwise fail loudly instead of silently
                    # feeding metadata into LightGBM.
                    first_model = self.ensemble[0]
                    names = getattr(first_model, "feature_name_", None)
                    if names is None and hasattr(first_model, "feature_name"):
                        try:
                            names = first_model.feature_name()
                        except Exception:
                            names = None
                    if names:
                        self.feature_names = list(names)
                if not self.feature_names:
                    raise ValueError(
                        "LightGBM MPD ensemble needs feature_names; pass the "
                        "same feature order used by the trained LightGBM model."
                    )
                missing = [n for n in self.feature_names if n not in value]
                if missing:
                    raise ValueError(
                        "Feature extractor is missing LightGBM MPD features: "
                        + ", ".join(missing)
                    )
                X = pd.DataFrame(
                    [[value[n] for n in self.feature_names]],
                    columns=self.feature_names,
                )
            else:
                X = value
            probs = np.stack([m.predict_proba(X)[:, 1] for m in self.ensemble], axis=0)
            return self._mpd(probs[:, 0])

        if self.kind == "siamese":
            if self.siamese_scorer is None:
                raise ValueError("Siamese MPD scorer requires a SiameseSVMScorer")
            candidate = self.siamese_scorer._tfidf_for_binary(value).toarray().astype(np.float32)
            diffs = np.abs(self.siamese_scorer.reference_vectors - candidate)  # (10, D)
            scaled = self.siamese_scorer.scaler.transform(diffs)
            probs = np.stack([m.predict_proba(scaled)[:, 1] for m in self.ensemble], axis=0)
            # probs shape: (n_models, 10). Average per-reference MPD, then reduce over refs.
            mpd_per_ref = self._mpd(probs)  # scalar across all refs/models pooled
            return mpd_per_ref

        raise ValueError(f"Unknown MPD scorer kind: {self.kind}")

    def alert(self, mpd: Optional[float]) -> Optional[bool]:
        if mpd is None or self.threshold is None:
            return None
        return bool(mpd >= self.threshold)


class FitnessEvaluator:
    def __init__(
        self,
        weights: FitnessWeights,
        feature_extractor_fn: Callable[[str], Dict[str, float]],
        lightgbm_predict_fn: Optional[
            Callable[[Dict[str, float]], float]
        ] = None,
        siamese_svm_predict_fn: Optional[
            Callable[[str], float]
        ] = None,
        siamese_svm_details_fn: Optional[
            Callable[[str], Dict[str, Any]]
        ] = None,
        normalize_detector_deltas: bool = True,
        evasion_bonus: float = 0.50,
        lgbm_mpd_scorer: Optional[BootstrapMPDScorer] = None,
        svm_mpd_scorer: Optional[BootstrapMPDScorer] = None,
        uncertainty_weight: float = 0.0,
    ):
        self.weights = weights
        self.feature_extractor_fn = feature_extractor_fn
        self.lightgbm_predict_fn = lightgbm_predict_fn
        self.siamese_svm_predict_fn = siamese_svm_predict_fn
        self.siamese_svm_details_fn = siamese_svm_details_fn
        self.normalize_detector_deltas = bool(normalize_detector_deltas)
        self.evasion_bonus = float(evasion_bonus)
        self.lgbm_mpd_scorer = lgbm_mpd_scorer
        self.svm_mpd_scorer = svm_mpd_scorer
        self.uncertainty_weight = float(uncertainty_weight)
        self._baseline_lgbm_mpd: Optional[float] = None
        self._baseline_svm_mpd: Optional[float] = None

    @staticmethod
    def modification_cost(chromosome: Chromosome) -> float:
        cost = 0.0

        for gene in chromosome.genes:
            if gene.operator == Operator.PADDING:
                size = int(gene.parameters.get("size", 0))
                cost += size / 4096.0

            elif gene.operator == Operator.ELF_TRANSFORM:
                cost += 0.01

            elif gene.operator == Operator.INSTRUCTION_TRANSFORM:
                cost += 0.005

            elif gene.operator == Operator.CODE_CAVE_INSERTION:
                # Writes into unclaimed file space only (never executed in
                # the normal control-flow path), so cheaper than a control-
                # flow redirect, but scales with how much junk is inserted.
                n = int(gene.parameters.get("instruction_count", 0))
                cost += 0.02 + 0.01 * (n / 32.0)

            elif gene.operator == Operator.TRAMPOLINE_INSERTION:
                # Rewrites 2 live instructions, redirects control flow
                # through the cave, and grows a PT_LOAD segment -- more
                # invasive baseline, scaled by the injected junk length.
                n = int(gene.parameters.get("junk_instruction_count", 0))
                cost += 0.03 + 0.02 * (n / 16.0)

            elif gene.operator == Operator.OPAQUE_PREDICATE_INSERTION:
                # Same mechanism as trampoline insertion plus an added
                # conditional branch, so a touch more invasive.
                n = int(gene.parameters.get("junk_instruction_count", 0))
                cost += 0.03 + 0.02 * (n / 16.0)

            else:
                # Defensive: any future operator that reaches here without
                # an explicit cost rule would otherwise silently cost 0.0,
                # which previously let CODE_CAVE/TRAMPOLINE/OPAQUE_PREDICATE
                # genes through completely free. Fail loudly instead.
                logger.warning(
                    "modification_cost: no cost rule for operator %s; "
                    "defaulting to 0.10 instead of silently charging 0.0",
                    gene.operator,
                )
                cost += 0.10

        return float(cost)

    def evaluate(
        self,
        variant_path: str,
        baseline_lgbm: Optional[float],
        baseline_svm: Optional[float],
        chromosome: Chromosome,
        baseline_lgbm_prediction: Optional[int] = None,
        baseline_svm_prediction: Optional[int] = None,
    ) -> FitnessResult:
        features = self.feature_extractor_fn(variant_path)

        variant_lgbm: Optional[float] = None
        variant_lgbm_prediction: Optional[int] = None
        variant_svm: Optional[float] = None
        variant_svm_prediction: Optional[int] = None

        lgbm_delta = 0.0
        svm_delta = 0.0

        if self.lightgbm_predict_fn is not None:
            variant_lgbm = float(
                self.lightgbm_predict_fn(features)
            )
            lgbm_threshold = getattr(
                self.lightgbm_predict_fn, "best_threshold", None
            )
            if lgbm_threshold is not None:
                variant_lgbm_prediction = int(variant_lgbm >= lgbm_threshold)

            if baseline_lgbm is not None:
                lgbm_delta = float(
                    baseline_lgbm - variant_lgbm
                )

        if self.siamese_svm_predict_fn is not None:
            if self.siamese_svm_details_fn is not None:
                details = self.siamese_svm_details_fn(
                    variant_path
                )
                variant_svm = float(details["score"])
                variant_svm_prediction = int(
                    details["prediction"]
                )
            else:
                variant_svm = float(
                    self.siamese_svm_predict_fn(variant_path)
                )

            if baseline_svm is not None:
                svm_delta = float(
                    baseline_svm - variant_svm
                )

        explainability = self.explain_variant(
            baseline_features=getattr(self, "_baseline_features", {}),
            variant_features=features,
            variant_path=variant_path,
            top_k=10,
        ) if getattr(self, "_baseline_features", None) else None

        cost = self.modification_cost(chromosome)

        variant_lgbm_mpd = None
        variant_svm_mpd = None
        lgbm_mpd_delta = 0.0
        svm_mpd_delta = 0.0

        if self.lgbm_mpd_scorer is not None:
            variant_lgbm_mpd = float(self.lgbm_mpd_scorer(features))
            if self._baseline_lgbm_mpd is not None:
                lgbm_mpd_delta = variant_lgbm_mpd - self._baseline_lgbm_mpd

        if self.svm_mpd_scorer is not None:
            variant_svm_mpd = float(self.svm_mpd_scorer(variant_path))
            if self._baseline_svm_mpd is not None:
                svm_mpd_delta = variant_svm_mpd - self._baseline_svm_mpd

        # Positive detector delta = movement toward benign / lower malware score.
        # Positive MPD delta = increased model uncertainty. MPD is kept as a
        # separate signal so the paper can report evasion and uncertainty independently.
        uncertainty_norm = 0.0
        if self.uncertainty_weight != 0.0:
            mpd_deltas = []
            if self.lgbm_mpd_scorer is not None:
                mpd_deltas.append(lgbm_mpd_delta)
            if self.svm_mpd_scorer is not None:
                mpd_deltas.append(svm_mpd_delta)
            if mpd_deltas:
                uncertainty_norm = float(np.mean(mpd_deltas) / 0.5) if np is not None else float(sum(mpd_deltas) / len(mpd_deltas) / 0.5)

        lgbm_evasion = (
            bool(
                baseline_lgbm_prediction == 1
                and variant_lgbm_prediction == 0
            )
            if baseline_lgbm_prediction is not None
            and variant_lgbm_prediction is not None
            else None
        )
        svm_evasion = (
            bool(baseline_svm_prediction == 1 and variant_svm_prediction == 0)
            if baseline_svm_prediction is not None and variant_svm_prediction is not None
            else None
        )

        lgbm_uncertainty_alert = (
            self.lgbm_mpd_scorer.alert(variant_lgbm_mpd)
            if self.lgbm_mpd_scorer is not None else None
        )
        svm_uncertainty_alert = (
            self.svm_mpd_scorer.alert(variant_svm_mpd)
            if self.svm_mpd_scorer is not None else None
        )

        joint_evasion = bool(
            lgbm_evasion is True and svm_evasion is True
        )
        any_evasion = bool(
            lgbm_evasion is True or svm_evasion is True
        )

        # Do not reward score movement for a detector that already classifies
        # the baseline as benign.  Otherwise the GA can optimize a detector
        # that has already been evaded, which inflates fitness without
        # representing an adversarial transition.
        # Reward movement away from malware towards benign for both classifiers
        effective_lgbm_delta = lgbm_delta if baseline_lgbm_prediction == 1 else max(0.0, lgbm_delta)
        effective_svm_delta = svm_delta if baseline_svm_prediction == 1 else max(0.0, svm_delta)

        lgbm_norm = (
            effective_lgbm_delta / max(abs(float(baseline_lgbm)), 1e-6)
            if self.normalize_detector_deltas and baseline_lgbm is not None
            else effective_lgbm_delta
        )
        # Scale SVM delta to account for the difference between coarse tabular features
        # and fine-grained assembly TF-IDF shifts, ensuring SVM evasion pressure is active.
        svm_norm = (
            effective_svm_delta / max(abs(float(baseline_svm)), 1e-2)
            if self.normalize_detector_deltas and baseline_svm is not None
            else effective_svm_delta
        )

        has_structural_gene = any(
            g.operator != Operator.PADDING for g in chromosome.genes
        )

        bonus = 0.0
        if lgbm_evasion is True and svm_evasion is True:
            bonus += 1.2 * self.evasion_bonus
        elif lgbm_evasion is True or svm_evasion is True:
            bonus += 0.4 * self.evasion_bonus

        if not has_structural_gene:
            bonus = 0.0  # overlay-only chromosomes get no evasion bonus

        combined_delta_score = (
            self.weights.alpha * lgbm_norm
            + self.weights.beta * svm_norm
        )
        combined_delta_score = max(-2.0, min(2.0, combined_delta_score))

        fitness = (
            combined_delta_score
            - self.weights.gamma * cost
            + bonus
        )

        chromosome.modification_cost = cost
        chromosome.fitness = float(fitness)

        return FitnessResult(
            fitness=float(fitness),
            lgbm_delta=lgbm_delta,
            svm_delta=svm_delta,
            modification_cost=cost,
            baseline_lgbm=baseline_lgbm,
            variant_lgbm=variant_lgbm,
            baseline_svm=baseline_svm,
            variant_svm=variant_svm,
            baseline_lgbm_prediction=baseline_lgbm_prediction,
            variant_lgbm_prediction=variant_lgbm_prediction,
            baseline_svm_prediction=baseline_svm_prediction,
            variant_svm_prediction=variant_svm_prediction,
            explainability=explainability,
            lgbm_normalized_delta=float(lgbm_norm),
            svm_normalized_delta=float(svm_norm),
            lgbm_evasion=lgbm_evasion,
            svm_evasion=svm_evasion,
            baseline_lgbm_mpd=self._baseline_lgbm_mpd,
            variant_lgbm_mpd=variant_lgbm_mpd,
            lgbm_mpd_delta=float(lgbm_mpd_delta),
            baseline_svm_mpd=self._baseline_svm_mpd,
            variant_svm_mpd=variant_svm_mpd,
            svm_mpd_delta=float(svm_mpd_delta),
            lgbm_uncertainty_alert=lgbm_uncertainty_alert,
            svm_uncertainty_alert=svm_uncertainty_alert,
            combined_delta_score=float(combined_delta_score),
        )


    def explain_variant(
        self,
        baseline_features: Dict[str, float],
        variant_features: Dict[str, float],
        variant_path: Optional[str] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Return lightweight, reproducible GA explanations.

        The explanation has two layers:
        1. feature deltas: what the physical mutation actually changed;
        2. optional SHAP values for LightGBM: which changed/current features
           contributed most to the detector score.
        """
        changed = []
        for name, base_value in baseline_features.items():
            try:
                new_value = float(variant_features.get(name, base_value))
                base_value = float(base_value)
            except (TypeError, ValueError):
                continue
            delta = new_value - base_value
            if abs(delta) > 1e-12:
                changed.append({
                    "feature": name,
                    "baseline": base_value,
                    "variant": new_value,
                    "delta": delta,
                    "abs_delta": abs(delta),
                })
        changed.sort(key=lambda x: x["abs_delta"], reverse=True)

        result: Dict[str, Any] = {
            "changed_features": changed[:top_k],
            "changed_feature_count": len(changed),
            "shap_available": False,
            "top_shap": [],
        }

        predictor = self.lightgbm_predict_fn
        model = getattr(predictor, "model", None)
        feature_names = getattr(predictor, "feature_names", None)
        if model is not None and feature_names:
            try:
                import shap
                row = np.asarray(
                    [[float(variant_features[name]) for name in feature_names]],
                    dtype=float,
                )
                explainer = getattr(predictor, "shap_explainer", None)
                if explainer is None:
                    explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(row)
                if isinstance(values, list):
                    values = values[-1]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[0, :, -1]
                elif values.ndim == 2:
                    values = values[0]
                else:
                    values = values.reshape(-1)

                pairs = [
                    {
                        "feature": name,
                        "shap": float(value),
                        "abs_shap": abs(float(value)),
                        "variant_value": float(variant_features[name]),
                        "changed": bool(
                            abs(float(variant_features[name]) -
                                float(baseline_features.get(name, variant_features[name])))
                            > 1e-12
                        ),
                    }
                    for name, value in zip(feature_names, values)
                ]
                pairs.sort(key=lambda x: x["abs_shap"], reverse=True)
                result["shap_available"] = True
                result["top_shap"] = pairs[:top_k]
            except Exception as exc:
                result["shap_error"] = f"{type(exc).__name__}: {exc}"

        return result

    @staticmethod
    def default_feature_extractor(
        elf_feature_extractor_module_path: str,
    ) -> Callable[[str], Dict[str, float]]:
        import importlib.util

        module_name = f"ga_elf_feature_extractor_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            elf_feature_extractor_module_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load feature extractor: "
                f"{elf_feature_extractor_module_path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )
        spec.loader.exec_module(module)

        extract_features = getattr(
            module,
            "extract_features",
            None,
        )

        if not callable(extract_features):
            raise AttributeError(
                f"{elf_feature_extractor_module_path} must expose "
                "extract_features(path, include_opcode=False)"
            )

        def extract(path: str) -> Dict[str, float]:
            features = extract_features(
                path,
                include_opcode=False,
            )

            if not isinstance(features, dict):
                raise TypeError(
                    f"Feature extractor returned "
                    f"{type(features).__name__}; expected dict"
                )

            return features

        return extract

    @staticmethod
    def default_lightgbm_predict(
        model_path: str,
    ) -> Callable[[Dict[str, float]], float]:
        import joblib

        artifact = joblib.load(model_path)

        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
            feature_names = artifact.get("feature_names")
            threshold = artifact.get("best_threshold")
        else:
            model = artifact
            feature_names = getattr(
                model,
                "feature_name_",
                None,
            )
            threshold = None

            if feature_names is None and hasattr(
                model,
                "feature_name",
            ):
                try:
                    feature_names = model.feature_name()
                except Exception:
                    feature_names = None

        if not callable(
            getattr(model, "predict_proba", None)
        ):
            raise TypeError(
                f"Unsupported LightGBM model type: "
                f"{type(model).__name__}"
            )

        if not feature_names:
            raise ValueError(
                "LightGBM model artifact does not contain "
                "feature_names."
            )

        feature_names = list(feature_names)

        def predict(features: Dict[str, float]) -> float:
            missing = [
                name
                for name in feature_names
                if name not in features
            ]

            if missing:
                raise ValueError(
                    "Feature extractor is missing LightGBM features: "
                    + ", ".join(missing)
                )

            row = [
                [features[name] for name in feature_names]
            ]

            probabilities = model.predict_proba(
                pd.DataFrame(row, columns=feature_names)
            )

            if getattr(probabilities, "ndim", 0) != 2:
                raise ValueError(
                    "LightGBM predict_proba returned invalid shape"
                )

            if probabilities.shape[1] < 2:
                raise ValueError(
                    "LightGBM model does not expose binary probabilities"
                )

            score = float(probabilities[0][1])

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Malware probability outside [0, 1]: {score}"
                )

            return score

        # Expose the fitted model for SHAP and cache TreeExplainer once.
        # This avoids rebuilding the explainer for every GA chromosome.
        predict.feature_names = feature_names
        predict.best_threshold = threshold
        predict.model = model
        predict.model_path = str(model_path)
        predict.shap_explainer = None
        try:
            import shap
            predict.shap_explainer = shap.TreeExplainer(model)
            print("[INFO] SHAP TreeExplainer cached for GA evaluations")
        except Exception as exc:
            print(f"[INFO] SHAP unavailable for GA: {type(exc).__name__}: {exc}")
        return predict


# ---------------------------------------------------------------------------
# Gene randomization
# ---------------------------------------------------------------------------

class GeneRandomizer:
    """
    Generate legal genes for the current binary.

    Picks among every mutation technique that is currently enabled --
    instruction-level substitutions, the EI_PAD structural transform, and
    styled overlay padding -- using weighted sampling, and falls through to
    the next technique if the chosen one has no legal candidate left (e.g.
    a binary with no safe instructions, or EI_PAD already used). This is
    what lets a single chromosome stack several *different* obfuscation
    techniques instead of being limited to one gene type per run.
    """

    # Relative sampling weights when a technique is enabled. Padding is
    # always available as the guaranteed fallback.
    _DEFAULT_WEIGHTS = {
        "instruction": 0.30,
        "reorder": 0.20,
        "code_cave": 0.15,
        "trampoline": 0.12,
        "padding": 0.08,
        "elf_transform": 0.05,
        "opaque_predicate": 0.10,
    }

    def __init__(
        self,
        mask: MutationMask,
        binary_path: Optional[str] = None,
        rng: Optional[random.Random] = None,
        instruction_mutation_enabled: bool = False,
        elf_transform_enabled: bool = False,
        code_cave_enabled: bool = False,
        trampoline_enabled: bool = False,
        opaque_predicate_enabled: bool = False,
        min_junk_instructions: int = 4,
        max_junk_instructions: int = 32,
        padding_style: str = "mixed",
    ):
        self.mask = mask
        self.binary_path = binary_path
        self.rng = rng or random.Random()
        self.instruction_mutation_enabled = instruction_mutation_enabled
        self.elf_transform_enabled = elf_transform_enabled
        self.code_cave_enabled = code_cave_enabled
        self.trampoline_enabled = trampoline_enabled
        self.opaque_predicate_enabled = opaque_predicate_enabled
        self.min_junk_instructions = min_junk_instructions
        self.max_junk_instructions = max_junk_instructions
        self.padding_style = padding_style
        self.operator_weights = dict(self._DEFAULT_WEIGHTS)
        self._instruction_candidates_cache: Optional[List[MIPSInstruction]] = None
        self._trampoline_pairs_cache: Optional[
            List[Tuple[MIPSInstruction, MIPSInstruction]]
        ] = None

    def _instruction_candidates(self) -> List[MIPSInstruction]:
        # Cache candidates for the lifetime of this randomizer: they are
        # computed from the *original* source binary, which does not
        # change across GA generations (materialized variants are new
        # files), so re-running radare2 extraction for every gene draw
        # would be wasteful.
        if self._instruction_candidates_cache is None:
            self._instruction_candidates_cache = (
                find_safe_mips_instruction_candidates(self.binary_path)
            )
        return self._instruction_candidates_cache

    def _trampoline_pairs(self) -> List[Tuple[MIPSInstruction, MIPSInstruction]]:
        if self._trampoline_pairs_cache is None:
            self._trampoline_pairs_cache = (
                find_trampoline_insertion_points(self.binary_path)
            )
        return self._trampoline_pairs_cache

    def _build_reorder_gene(self) -> Optional[Gene]:
        if not (self.instruction_mutation_enabled and self.binary_path):
            return None
        try:
            pairs = self._trampoline_pairs()
            gene = build_reorder_gene(self.binary_path, self.rng, pairs=pairs)
            if gene is None:
                _debug("[DEBUG] no usable adjacent-reorder pair found")
            return gene
        except Exception as e:
            _debug("[DEBUG] reorder gene error:", type(e).__name__, str(e))
            return None

    def _build_instruction_gene(self) -> Optional[Gene]:
        if not (self.instruction_mutation_enabled and self.binary_path):
            return None
        try:
            candidates = self._instruction_candidates()
            gene = build_mips_instruction_gene(
                self.binary_path, self.rng, candidates=candidates
            )
            if gene is None:
                _debug("[DEBUG] no mutatable instruction found")
            else:
                _debug(
                    "[DEBUG] instruction gene:",
                    gene.parameters.get("transform"),
                    hex(int(gene.location)),
                    gene.parameters.get("disasm"),
                )
            return gene
        except Exception as e:
            _debug(
                "[DEBUG] instruction mutation error:",
                type(e).__name__,
                str(e),
            )
            return None

    def _build_elf_transform_gene(self) -> Optional[Gene]:
        if not (self.elf_transform_enabled and self.binary_path):
            return None
        try:
            return build_ei_pad_gene(self.binary_path, self.rng)
        except Exception as e:
            _debug(
                "[DEBUG] elf transform error:",
                type(e).__name__,
                str(e),
            )
            return None

    def _build_code_cave_gene(self) -> Optional[Gene]:
        if not (self.code_cave_enabled and self.binary_path):
            return None
        try:
            gene = build_code_cave_gene(
                self.binary_path,
                self.rng,
                min_junk_instructions=self.min_junk_instructions,
                max_junk_instructions=self.max_junk_instructions,
            )
            if gene is None:
                _debug("[DEBUG] no usable code cave found")
            else:
                _debug(
                    "[DEBUG] code cave gene:",
                    gene.parameters.get("instruction_count"),
                    "junk instructions at offset",
                    hex(int(gene.location)),
                )
            return gene
        except Exception as e:
            _debug(
                "[DEBUG] code cave error:",
                type(e).__name__,
                str(e),
            )
            return None

    def _build_trampoline_gene(self) -> Optional[Gene]:
        if not (self.trampoline_enabled and self.binary_path):
            return None
        try:
            pairs = self._trampoline_pairs()
            gene = build_trampoline_gene(
                self.binary_path,
                self.rng,
                min_junk_instructions=self.min_junk_instructions,
                max_junk_instructions=self.max_junk_instructions,
                pairs=pairs,
            )
            if gene is None:
                _debug("[DEBUG] no usable trampoline insertion point found")
            else:
                _debug(
                    "[DEBUG] trampoline gene:",
                    gene.parameters.get("junk_instruction_count"),
                    "junk instructions, redirect at",
                    hex(int(gene.location)),
                    "-> trampoline",
                    hex(gene.parameters["trampoline_vaddr"]),
                )
            return gene
        except Exception as e:
            _debug(
                "[DEBUG] trampoline error:",
                type(e).__name__,
                str(e),
            )
            return None

    def _build_opaque_predicate_gene(self) -> Optional[Gene]:
        if not (self.opaque_predicate_enabled and self.binary_path):
            return None
        try:
            pairs = self._trampoline_pairs()
            gene = build_opaque_predicate_gene(
                self.binary_path,
                self.rng,
                min_junk_instructions=self.min_junk_instructions,
                max_junk_instructions=self.max_junk_instructions,
                pairs=pairs,
            )
            if gene is None:
                _debug("[DEBUG] no usable opaque-predicate site found")
            else:
                _debug(
                    "[DEBUG] opaque predicate gene:",
                    gene.parameters.get("predicate_kind"),
                    gene.parameters.get("junk_instruction_count"),
                    "dead junk words, redirect at",
                    hex(int(gene.location)),
                )
            return gene
        except Exception as e:
            _debug(
                "[DEBUG] opaque predicate error:",
                type(e).__name__,
                str(e),
            )
            return None

    def _build_padding_gene(self) -> Optional[Gene]:
        upper = int(self.mask.max_padding_bytes)
        lower = min(16, upper)
        if upper <= 0:
            return None
        size = self.rng.randint(lower, upper)

        style = self.padding_style
        if style == "mixed":
            # Favor string_like and low_entropy which reduce entropy and mimic benign structure
            style = self.rng.choice(["string_like", "low_entropy", "string_like", "low_entropy", "random"])
        elif style not in PADDING_STYLES:
            style = "string_like"

        payload = generate_padding_payload(self.rng, size, style)

        return Gene(
            operator=Operator.PADDING,
            location="overlay",
            parameters={
                "region": "overlay",
                "size": size,
                "style": style,
                "payload_hex": payload.hex(),
            },
        )

    def random_gene(self) -> Optional[Gene]:
        builders: List[Tuple[str, Callable[[], Optional[Gene]]]] = []

        if self.instruction_mutation_enabled and self.binary_path:
            builders.append(("instruction", self._build_instruction_gene))
        if self.elf_transform_enabled and self.binary_path:
            builders.append(("elf_transform", self._build_elf_transform_gene))
        if self.code_cave_enabled and self.binary_path:
            builders.append(("code_cave", self._build_code_cave_gene))
        if self.trampoline_enabled and self.binary_path:
            builders.append(("trampoline", self._build_trampoline_gene))
        if self.opaque_predicate_enabled and self.binary_path:
            builders.append(("opaque_predicate", self._build_opaque_predicate_gene))
        if self.instruction_mutation_enabled and self.binary_path:
            builders.append(("instruction", self._build_instruction_gene))
            builders.append(("reorder", self._build_reorder_gene))
        builders.append(("padding", self._build_padding_gene))

        # Weighted shuffle: repeatedly draw (without replacement) using the
        # configured weights, so the most-favored enabled technique is
        # usually tried first but every enabled technique gets a fair shot
        # if earlier picks come back empty (e.g. no legal instructions
        # left, or EI_PAD already consumed by a sibling gene).
        remaining = list(builders)
        order: List[Callable[[], Optional[Gene]]] = []
        while remaining:
            weights = [
                self.operator_weights.get(name, self._DEFAULT_WEIGHTS.get(name, 0.1))
                for name, _ in remaining
            ]
            choice = self.rng.choices(remaining, weights=weights, k=1)[0]
            order.append(choice[1])
            remaining.remove(choice)

        for build in order:
            gene = build()
            if gene is not None:
                return gene

        return None


# ---------------------------------------------------------------------------
# GA driver
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 15
    tournament_size: int = 3
    mutation_rate: float = 0.30
    min_mutation_rate: float = 0.05
    mutation_decay: float = 0.95
    gene_deletion_rate: float = 0.10
    crossover_rate: float = 0.60
    elitism: int = 2
    max_genes_per_chromosome: int = 8
    max_padding_bytes: int = 4096
    work_dir: str = "./ga_work"
    seed: Optional[int] = None
    instruction_mutation_enabled: bool = False
    elf_transform_enabled: bool = False
    code_cave_enabled: bool = False
    trampoline_enabled: bool = False
    opaque_predicate_enabled: bool = False
    min_junk_instructions: int = 4
    max_junk_instructions: int = 32
    padding_style: str = "mixed"
    parallel_workers: int = 1
    checkpoint_every: int = 1
    diversity_ratio: float = 0.20
    stagnation_generations: int = 4
    immigrant_fraction: float = 0.15
    adaptive_operator_weights: bool = True
    explanation_top_k: int = 10
    normalize_detector_deltas: bool = True
    evasion_bonus: float = 0.50


class GeneticAlgorithm:
    def __init__(
        self,
        source_binary_path: str,
        config: GAConfig,
        fitness_evaluator: FitnessEvaluator,
        validator: Validator,
    ):
        self.source_binary_path = source_binary_path
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.validator = validator

        # Keep all stochastic GA operations on one RNG.  The previous
        # implementation seeded the module-level RNG but generated padding
        # with that global state, which made otherwise identical seeded runs
        # non-reproducible when other code consumed random numbers.
        self.rng = random.Random(config.seed)
        if config.seed is not None:
            random.seed(config.seed)
            if np is not None:
                np.random.seed(config.seed)

        if not os.path.isfile(source_binary_path):
            raise FileNotFoundError(source_binary_path)

        os.makedirs(config.work_dir, exist_ok=True)

        self.baseline_snapshot = snapshot_binary(source_binary_path)

        probe = lief.parse(source_binary_path)
        if probe is None:
            raise ValueError(f"Could not parse {source_binary_path}")

        self.expected_machine = probe.header.machine_type

        print(
            f"[BASELINE] size={self.baseline_snapshot.size} "
            f"sha256={self.baseline_snapshot.sha256}"
        )
        print(
            f"[BASELINE] entrypoint="
            f"{hex(self.baseline_snapshot.entrypoint)} "
            f"segments={self.baseline_snapshot.segment_count} "
            f"sections={self.baseline_snapshot.section_count}"
        )

        baseline_validation = validator.validate(source_binary_path)
        print(
            "[BASELINE validation]",
            json.dumps(baseline_validation.to_dict(), indent=2),
        )

        if not baseline_validation.valid:
            raise ValueError(
                "Source binary failed baseline validation: "
                + baseline_validation.reason
            )

        self.mask = MutationMask(
            probe,
            max_padding_bytes=config.max_padding_bytes,
        )
        self.randomizer = GeneRandomizer(
            self.mask,
            binary_path=self.source_binary_path,
            rng=self.rng,
            instruction_mutation_enabled=config.instruction_mutation_enabled,
            elf_transform_enabled=config.elf_transform_enabled,
            code_cave_enabled=config.code_cave_enabled,
            trampoline_enabled=config.trampoline_enabled,
            opaque_predicate_enabled=config.opaque_predicate_enabled,
            min_junk_instructions=config.min_junk_instructions,
            max_junk_instructions=config.max_junk_instructions,
            padding_style=config.padding_style,
        )

        self._rejection_reasons: Dict[str, int] = {}
        self.generation_history: List[Dict[str, Any]] = []
        self.operator_stats: Dict[str, Dict[str, float]] = {}
        self.best_fitness_ever = -1e12
        self.stagnation = 0

        if not self.mask.sectionless:
            print(
                "[INFO] Binary has sections. The rewritten GA still uses "
                "overlay-only mutation to preserve existing ELF bytes."
            )

        self.original_score_lgbm: Optional[float] = None
        self.original_lgbm_prediction: Optional[int] = None
        self.original_score_svm: Optional[float] = None
        self.original_svm_prediction: Optional[int] = None
        self.original_lgbm_mpd: Optional[float] = None
        self.original_svm_mpd: Optional[float] = None
        self.baseline_features: Dict[str, float] = {}
        self._compute_baseline_scores()

        # Per-binary statistics collected during chromosome evaluation.
        self.stats = {
            "file": str(self.source_binary_path),
            "baseline": {
                "lgbm": self.original_score_lgbm,
                "svm": self.original_score_svm,
                "svm_prediction": self.original_svm_prediction,
            },
            "operator_usage": {},
            "generations": [],
        }

    def _compute_baseline_scores(self) -> None:
        self.baseline_features = (
            self.fitness_evaluator.feature_extractor_fn(
                self.source_binary_path
            )
        )
        self.fitness_evaluator._baseline_features = self.baseline_features

        if self.fitness_evaluator.lightgbm_predict_fn is not None:
            self.original_score_lgbm = float(
                self.fitness_evaluator.lightgbm_predict_fn(
                    self.baseline_features
                )
            )
            threshold = getattr(
                self.fitness_evaluator.lightgbm_predict_fn,
                "best_threshold",
                None,
            )
            if threshold is not None:
                self.original_lgbm_prediction = int(
                    self.original_score_lgbm >= threshold
                )

        if self.fitness_evaluator.siamese_svm_predict_fn is not None:
            try:
                if self.fitness_evaluator.siamese_svm_details_fn is not None:
                    details = (
                        self.fitness_evaluator.siamese_svm_details_fn(
                            self.source_binary_path
                        )
                    )
                    self.original_score_svm = float(
                        details["score"]
                    )
                    self.original_svm_prediction = int(
                        details["prediction"]
                    )
                else:
                    self.original_score_svm = float(
                        self.fitness_evaluator.siamese_svm_predict_fn(
                            self.source_binary_path
                        )
                    )
            except RuntimeError as exc:
                # Some valid IoT ELFs are so stripped/corrupted that no
                # executable instruction stream can be recovered. Do not
                # abort a structural LightGBM GA just because the assembly
                # branch is unavailable for that sample. The limitation is
                # recorded explicitly and the Siamese/MPD branch is disabled
                # for this run. If LightGBM is also unavailable, continuing
                # would leave the GA with no detector objective, so fail with
                # a clear error instead.
                if self.fitness_evaluator.lightgbm_predict_fn is None:
                    raise RuntimeError(
                        "Siamese SVM could not score the baseline and no "
                        "LightGBM detector is available: " + str(exc)
                    ) from exc
                logger.warning(
                    "Siamese baseline unavailable for %s: %s -- continuing "
                    "with LightGBM-only fitness for this binary",
                    self.source_binary_path, exc,
                )
                self.original_score_svm = None
                self.original_svm_prediction = None
                self.fitness_evaluator.siamese_svm_predict_fn = None
                self.fitness_evaluator.siamese_svm_details_fn = None
                self.fitness_evaluator.svm_mpd_scorer = None
                self.fitness_evaluator._baseline_svm_mpd = None

        if self.fitness_evaluator.lgbm_mpd_scorer is not None:
            self.original_lgbm_mpd = float(
                self.fitness_evaluator.lgbm_mpd_scorer(self.baseline_features)
            )
            self.fitness_evaluator._baseline_lgbm_mpd = self.original_lgbm_mpd

        if self.fitness_evaluator.svm_mpd_scorer is not None:
            self.original_svm_mpd = float(
                self.fitness_evaluator.svm_mpd_scorer(self.source_binary_path)
            )
            self.fitness_evaluator._baseline_svm_mpd = self.original_svm_mpd

        print(
            "[BASELINE MODEL]",
            f"LightGBM={self.original_score_lgbm}",
            f"LightGBMPrediction={self.original_lgbm_prediction}",
            f"SiameseSVM={self.original_score_svm}",
            f"SiamesePrediction={self.original_svm_prediction}",
        )

    def _new_random_chromosome(self) -> Optional[Chromosome]:
        # Stack multiple genes (potentially mixing techniques: an
        # instruction substitution + styled padding + the EI_PAD
        # transform) per chromosome instead of capping at one, so the GA
        # can actually search combinations of techniques rather than only
        # ever evaluating one mutation at a time.
        min_genes = 2 if self.config.instruction_mutation_enabled else 1
        gene_count = self.rng.randint(min_genes, max(min_genes, self.config.max_genes_per_chromosome))
        genes = []

        # Ensure multi-objective chromosomes can combine structural padding and assembly mutations
        if self.config.instruction_mutation_enabled and self.randomizer.binary_path:
            pad_gene = self.randomizer._build_padding_gene()
            if pad_gene is not None:
                genes.append(pad_gene)
            for _ in range(max(1, gene_count - len(genes))):
                gene = self.randomizer.random_gene()
                if gene is not None:
                    genes.append(gene)
        else:
            for _ in range(gene_count):
                gene = self.randomizer.random_gene()
                if gene is not None:
                    genes.append(gene)

        if not genes:
            return None
        chromosome = Chromosome(
            genes=genes,
            source_binary=self.source_binary_path,
        )

        allowed = self.mask.allows_chromosome(
            chromosome,
            self.config.max_genes_per_chromosome,
        )

        _debug("[DEBUG] chromosome allowed:", allowed)

        if not allowed:
            _debug("[DEBUG] chromosome rejected by MutationMask")

        return chromosome if allowed else None
        
    def _init_population(self) -> List[Chromosome]:
        population: List[Chromosome] = []

        attempts = 0
        max_attempts = max(
            self.config.population_size * 20,
            20,
        )

        while (
            len(population) < self.config.population_size
            and attempts < max_attempts
        ):
            attempts += 1
            chromosome = self._new_random_chromosome()

            if chromosome is not None:
                population.append(chromosome)

        if len(population) != self.config.population_size:
            raise RuntimeError(
                "Could not create the requested population: "
                f"{len(population)}/{self.config.population_size}"
            )

        print(
            f"[GA] initialized population={len(population)} "
            f"requested={self.config.population_size}"
        )

        return population

    def _overlay_bytes_for(self, chromosome: Chromosome) -> bytes:
        chunks = bytearray()

        for gene in chromosome.genes:
            if gene.operator != Operator.PADDING:
                continue

            if gene.parameters.get("region") != "overlay":
                raise ValueError(
                    "Only region='overlay' padding is supported."
                )

            size = int(gene.parameters.get("size", 0))

            if not self.mask.allows_padding("overlay", size, None):
                raise ValueError(
                    f"Illegal overlay padding size: {size}"
                )

            if "payload_hex" in gene.parameters:
                payload = bytes.fromhex(gene.parameters["payload_hex"])
            else:
                style = gene.parameters.get("style", "string_like")
                payload = generate_padding_payload(self.rng, size, style)
                gene.parameters["payload_hex"] = payload.hex()

            chunks.extend(payload)

        return bytes(chunks)

    def _materialize(
        self,
        chromosome: Chromosome,
    ) -> Optional[str]:
        logger.debug(
            "materialize start: genes=%d first=%s",
            len(chromosome.genes),
            chromosome.genes[0].parameters if chromosome.genes else None,
        )
        if not self.mask.allows_chromosome(
            chromosome,
            self.config.max_genes_per_chromosome,
        ):
            allowed, reason = self.mask.explain_chromosome(
                chromosome,
                self.config.max_genes_per_chromosome,
            )
            logger.debug("materialize rejected: reason=%s", reason)
            return None

        try:
            out_path = os.path.join(
                self.config.work_dir,
                f"variant_{uuid.uuid4().hex}.elf",
            )
            shutil.copyfile(self.source_binary_path, out_path)

            for gene in chromosome.genes:
                if gene.operator == Operator.INSTRUCTION_TRANSFORM:
                    apply_exact_instruction_patch(
                        Path(out_path),
                        gene.parameters,
                    )
                elif (
                    gene.operator == Operator.ELF_TRANSFORM
                    and gene.parameters.get("field") == "ei_pad"
                ):
                    apply_ei_pad_patch(
                        Path(out_path),
                        gene.parameters,
                    )

            allowed_modifications = []
            allowed_segment_growth: Dict[int, int] = {}
            for gene in chromosome.genes:
                if gene.operator == Operator.INSTRUCTION_TRANSFORM:
                    allowed_modifications.append({
                        "file_offset": gene.parameters["file_offset"],
                        "size": gene.parameters["size"],
                        "original_bytes": gene.parameters["original_bytes"],
                        "new_bytes": gene.parameters["new_bytes"],
                    })
                elif (
                    gene.operator == Operator.ELF_TRANSFORM
                    and gene.parameters.get("field") == "ei_pad"
                ):
                    allowed_modifications.append({
                        "file_offset": gene.parameters["file_offset"],
                        "size": gene.parameters["size"],
                        "original_bytes": gene.parameters["original_bytes"],
                        "new_bytes": gene.parameters["new_bytes"],
                    })
                elif gene.operator == Operator.CODE_CAVE_INSERTION:
                    # Applied here (not in the earlier loop) so its two
                    # writes -- cave bytes + program-header entry -- are
                    # captured directly as modification records instead of
                    # being reconstructed a second time.
                    modifications = apply_code_cave_patch(
                        Path(out_path),
                        gene.parameters,
                    )
                    allowed_modifications.extend(modifications)
                    allowed_segment_growth[gene.parameters["segment_vaddr"]] = (
                        allowed_segment_growth.get(
                            gene.parameters["segment_vaddr"], 0
                        )
                        + len(bytes.fromhex(gene.parameters["new_bytes"]))
                    )
                elif gene.operator in (
                    Operator.TRAMPOLINE_INSERTION,
                    Operator.OPAQUE_PREDICATE_INSERTION,
                ):
                    modifications = apply_trampoline_patch(
                        Path(out_path),
                        gene.parameters,
                    )
                    allowed_modifications.extend(modifications)
                    allowed_segment_growth[gene.parameters["segment_vaddr"]] = (
                        allowed_segment_growth.get(
                            gene.parameters["segment_vaddr"], 0
                        )
                        + len(bytes.fromhex(gene.parameters["trampoline_bytes"]))
                    )

            overlay = self._overlay_bytes_for(chromosome)
            if overlay:
                with open(out_path, "ab") as handle:
                    handle.write(overlay)

            chromosome.genes_applied = [
                gene.to_dict()
                for gene in chromosome.genes
            ]

            validation = self.validator.validate(
                out_path,
                allowed_modifications=allowed_modifications,
                allowed_segment_growth=allowed_segment_growth or None,
            )

            chromosome.validation = validation.to_dict()

            logger.debug("materialize validation: %s", validation.to_dict())

            if not validation.valid:
                logger.debug("materialize rejected: %s", validation.to_dict())
                os.remove(out_path)
                return None

            chromosome.variant_path = out_path
            return out_path

        except Exception as exc:
            chromosome.validation = {
                "valid": False,
                "reason": f"materialization error: {exc}",
            }
            logger.debug(
                "materialize error: %s: %s",
                type(exc).__name__,
                str(exc),
            )
            return None

    @staticmethod
    def _fitness(chromosome: Chromosome) -> float:
        return (
            chromosome.fitness
            if chromosome.fitness is not None
            else -1e12
        )

    def _tournament_select(
        self,
        population: List[Chromosome],
    ) -> Chromosome:
        size = min(
            self.config.tournament_size,
            len(population),
        )

        contenders = self.rng.sample(population, size)
        return max(contenders, key=self._fitness)

    @staticmethod
    def _gene_key(gene: Gene) -> Tuple[Any, ...]:
        return (
            gene.operator.value,
            gene.location,
            gene.parameters.get("region"),
            gene.parameters.get("size"),
            gene.parameters.get("style"),
            gene.parameters.get("field"),
            gene.parameters.get("transform"),
        )

    def _crossover(
        self,
        parent_a: Chromosome,
        parent_b: Chromosome,
    ) -> Chromosome:
        if self.rng.random() > self.config.crossover_rate:
            child = self.rng.choice([parent_a, parent_b]).copy()
            return child

        combined = [
            gene.copy()
            for gene in (parent_a.genes + parent_b.genes)
        ]
        self.rng.shuffle(combined)

        deduped: List[Gene] = []
        seen = set()

        for gene in combined:
            if not self.mask.allows_gene(gene):
                continue

            key = self._gene_key(gene)

            if key in seen:
                continue

            seen.add(key)
            deduped.append(gene)

            if len(deduped) >= self.config.max_genes_per_chromosome:
                break

        if not deduped:
            fallback = self._new_random_chromosome()
            if fallback is not None:
                return fallback

        return Chromosome(
            genes=deduped,
            source_binary=self.source_binary_path,
        )

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        child = chromosome.copy()

        if (
            child.genes
            and self.rng.random() < self.config.gene_deletion_rate
        ):
            child.genes.pop(self.rng.randrange(len(child.genes)))

        if (
            len(child.genes)
            < self.config.max_genes_per_chromosome
            and self.rng.random() < self.config.mutation_rate
        ):
            new_gene = self.randomizer.random_gene()

            if new_gene is not None:
                candidate = child.copy()
                candidate.genes.append(new_gene)

                if self.mask.allows_chromosome(
                    candidate,
                    self.config.max_genes_per_chromosome,
                ):
                    child = candidate

        if not child.genes:
            fallback = self._new_random_chromosome()
            if fallback is not None:
                child = fallback

        return child

    def _decay_mutation_rate(self) -> None:
        self.config.mutation_rate = max(
            self.config.min_mutation_rate,
            self.config.mutation_rate * self.config.mutation_decay,
        )

    def _evaluate_chromosome(
        self,
        chromosome: Chromosome,
    ) -> None:
        variant_path = self._materialize(chromosome)
        print(
            "[EVAL SOURCE]",
            __file__,
            "_evaluate_chromosome reached",
        )
        if variant_path is None:
            chromosome.fitness = -1e12
            chromosome.modification_cost = float("inf")
            return

        result = self.fitness_evaluator.evaluate(
            variant_path=variant_path,
            baseline_lgbm=self.original_score_lgbm,
            baseline_svm=self.original_score_svm,
            chromosome=chromosome,
            baseline_lgbm_prediction=self.original_lgbm_prediction,
            baseline_svm_prediction=self.original_svm_prediction,
        )

        chromosome.fitness_result = result.to_dict()

        # Attribute this evaluated variant's detector-score change to each gene.
        for gene in chromosome.genes:
            op_name = gene.operator.value if hasattr(gene.operator, "value") else str(gene.operator)
            entry = self.stats["operator_usage"].setdefault(
                op_name, {"used": 0, "success": 0}
            )
            entry["used"] += 1
            if (
                (result.lgbm_delta is not None and result.lgbm_delta > 0)
                or (result.svm_delta is not None and abs(result.svm_delta) > 1e-3)
            ):
                entry["success"] += 1

        print(
            "[VARIANT]",
            os.path.basename(variant_path),
            f"fitness={result.fitness:.6f}",
            f"lgbm={result.variant_lgbm}",
            f"svm={result.variant_svm}",
            f"svm_prediction={result.variant_svm_prediction}",
            f"lgbm_delta={result.lgbm_delta:.6f}",
            f"svm_delta={result.svm_delta:.6f}",
            f"cost={result.modification_cost:.6f}",
            f"genes={len(chromosome.genes)}",
        )

    def _evaluate_population(
        self,
        population: List[Chromosome],
    ) -> None:
        pending = [
            chromosome for chromosome in population
            if chromosome.fitness is None
        ]
        workers = max(1, int(self.config.parallel_workers))
        if workers == 1 or len(pending) <= 1:
            for chromosome in pending:
                self._evaluate_chromosome(chromosome)
            return

        # Feature extraction/disassembly is subprocess-heavy, so threads give
        # useful overlap without duplicating the loaded model in memory.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._evaluate_chromosome, chromosome)
                for chromosome in pending
            ]
            for future in as_completed(futures):
                future.result()


    @staticmethod
    def _chromosome_signature(chromosome: Chromosome) -> str:
        payload = [
            gene.to_dict() for gene in chromosome.genes
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _update_operator_stats(self, population: List[Chromosome]) -> None:
        for chromosome in population:
            result = chromosome.fitness_result or {}
            # Use the same normalized/weighted detector objective that the
            # GA actually optimizes. Summing raw LightGBM and SVM margins is
            # scale-dependent and can make the larger-magnitude detector
            # dominate adaptive operator weighting.
            gain = max(
                0.0,
                float(result.get("combined_delta_score", 0.0) or 0.0),
            )
            # A chromosome-level gain cannot be causally assigned to every
            # gene when chromosomes contain multiple operators. Distribute
            # the observed gain across genes and explicitly label these as
            # operator-associated statistics. This prevents stacked padding
            # from receiving the full chromosome gain multiple times.
            per_gene_gain = gain / max(1, len(chromosome.genes))
            for gene in chromosome.genes:
                key = gene.operator.value
                stat = self.operator_stats.setdefault(
                    key, {
                        "uses": 0.0, "successful": 0.0, "gain": 0.0,
                        "lgbm_evasions": 0.0, "svm_evasions": 0.0,
                        "joint_evasions": 0.0,
                    }
                )
                stat["uses"] += 1.0
                if per_gene_gain > 0:
                    stat["successful"] += 1.0
                    stat["gain"] += per_gene_gain
                if result.get("lgbm_evasion") is True:
                    stat["lgbm_evasions"] += 1.0
                if result.get("svm_evasion") is True:
                    stat["svm_evasions"] += 1.0
                if result.get("joint_evasion") is True:
                    stat["joint_evasions"] += 1.0

        if not self.config.adaptive_operator_weights:
            return

        # Feed observed success back into GeneRandomizer. Keep a floor so
        # exploration never collapses to one operator.
        base = getattr(self.randomizer, "_DEFAULT_WEIGHTS", {})
        for name in list(base):
            stat = self.operator_stats.get(name, {})
            uses = stat.get("uses", 0.0)
            avg_gain = stat.get("gain", 0.0) / max(1.0, uses)
            success = stat.get("successful", 0.0) / max(1.0, uses)
            self.randomizer.operator_weights[name] = max(
                0.10, float(base.get(name, 0.1)) *
                (1.0 + min(3.0, 4.0 * success + 2.0 * avg_gain))
            )

    def _make_diverse_population(self, population: List[Chromosome]) -> List[Chromosome]:
        """Remove exact duplicate chromosomes and inject random immigrants."""
        unique = []
        seen = set()
        for chromosome in population:
            sig = self._chromosome_signature(chromosome)
            if sig not in seen:
                seen.add(sig)
                unique.append(chromosome)

        target = self.config.population_size
        immigrant_target = max(
            1, int(round(target * self.config.immigrant_fraction))
        )
        while len(unique) < target:
            candidate = self._new_random_chromosome()
            if candidate is None:
                break
            sig = self._chromosome_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(candidate)

        # Explicitly preserve a small exploration fraction even when
        # crossover repeatedly recreates the same high-fitness chromosome.
        if len(unique) >= target and immigrant_target > 0:
            ranked = sorted(unique, key=self._fitness, reverse=True)
            elites = ranked[:max(1, target - immigrant_target)]
            immigrants = []
            attempts = 0
            while len(immigrants) < immigrant_target and attempts < target * 10:
                attempts += 1
                candidate = self._new_random_chromosome()
                if candidate is None:
                    continue
                sig = self._chromosome_signature(candidate)
                if sig not in seen:
                    seen.add(sig)
                    immigrants.append(candidate)
            unique = elites + immigrants

        return unique[:target]

    def _save_checkpoint(self, population: List[Chromosome], generation: int) -> None:
        if self.config.checkpoint_every <= 0:
            return
        if generation % self.config.checkpoint_every != 0:
            return
        path = os.path.join(
            self.config.work_dir,
            f"checkpoint_gen_{generation:04d}.json",
        )
        payload = {
            "generation": generation,
            "best_fitness": self._fitness(max(population, key=self._fitness)),
            "mutation_rate": self.config.mutation_rate,
            "operator_stats": self.operator_stats,
            "population": [c.to_dict() for c in population],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def evolve(self) -> List[Chromosome]:
        population = self._init_population()
        self._evaluate_population(population)
        self._update_operator_stats(population)

        for generation in range(self.config.generations + 1):
            population.sort(key=self._fitness, reverse=True)
            best = population[0]

            if self._fitness(best) > self.best_fitness_ever + 1e-12:
                self.best_fitness_ever = self._fitness(best)
                self.stagnation = 0
            else:
                self.stagnation += 1

            best_result = best.fitness_result or {}
            history_item = {
                "generation": generation,
                "best_fitness": self._fitness(best),
                "best_lgbm": best_result.get("variant_lgbm"),
                "best_svm": best_result.get("variant_svm"),
                "best_lgbm_delta": best_result.get("lgbm_score_delta"),
                "best_svm_delta": best_result.get("svm_score_delta"),
                "best_lgbm_evasion": best_result.get("lgbm_evasion"),
                "best_svm_evasion": best_result.get("svm_evasion"),
                "best_joint_evasion": best_result.get("joint_evasion"),
                "best_cost": best_result.get("modification_cost"),
                "best_genes": len(best.genes),
                "unique_population": len({
                    self._chromosome_signature(c) for c in population
                }),
                "mutation_rate": self.config.mutation_rate,
                "stagnation": self.stagnation,
            }
            self.generation_history.append(history_item)

            print(
                f"[gen {generation:02d}] best_fitness={self._fitness(best):.6f} "
                f"lgbm={best_result.get('variant_lgbm')} "
                f"svm={best_result.get('variant_svm')} "
                f"unique={history_item['unique_population']}/{len(population)} "
                f"mutation_rate={self.config.mutation_rate:.4f} "
                f"stagnation={self.stagnation}"
            )

            self.stats["generations"].append({
                "generation": generation,
                "best_fitness": float(self._fitness(best)),
                "best_lgbm": best_result.get("variant_lgbm"),
                "best_svm": best_result.get("variant_svm"),
            })

            if generation == self.config.generations:
                break

            elite_count = min(
                max(0, self.config.elitism),
                len(population),
                self.config.population_size,
            )
            next_population = [
                population[i].copy() for i in range(elite_count)
            ]

            # If the search plateaus, temporarily increase exploration by
            # injecting fresh chromosomes rather than repeatedly cloning the
            # same weak local optimum.
            immigrant_count = 0
            if self.stagnation >= self.config.stagnation_generations:
                immigrant_count = max(
                    1,
                    int(round(
                        self.config.population_size *
                        self.config.immigrant_fraction
                    )),
                )

            while len(next_population) < self.config.population_size:
                if immigrant_count > 0 and len(next_population) >= (
                    self.config.population_size - immigrant_count
                ):
                    child = self._new_random_chromosome()
                    if child is None:
                        continue
                    immigrant_count -= 1
                else:
                    parent_a = self._tournament_select(population)
                    parent_b = self._tournament_select(population)
                    child = self._mutate(
                        self._crossover(parent_a, parent_b)
                    )

                if not self.mask.allows_chromosome(
                    child, self.config.max_genes_per_chromosome
                ):
                    _allowed, reason = self.mask.explain_chromosome(
                        child, self.config.max_genes_per_chromosome
                    )
                    self._rejection_reasons[reason] = (
                        self._rejection_reasons.get(reason, 0) + 1
                    )
                    continue

                next_population.append(child)

            population = self._make_diverse_population(next_population)
            self._evaluate_population(population)
            self._update_operator_stats(population)
            self._decay_mutation_rate()
            self._save_checkpoint(population, generation + 1)

        population.sort(key=self._fitness, reverse=True)
        return population

    def save_results(
        self,
        population: List[Chromosome],
        output_dir: str,
        top_k: int = 5,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        ranked = sorted(
            population,
            key=self._fitness,
            reverse=True,
        )

        selected = ranked[: max(0, top_k)]

        results = {
            "source_binary": self.source_binary_path,
            "baseline": {
                "size": self.baseline_snapshot.size,
                "sha256": self.baseline_snapshot.sha256,
                "entrypoint": hex(self.baseline_snapshot.entrypoint),
                "segment_count": self.baseline_snapshot.segment_count,
                "section_count": self.baseline_snapshot.section_count,
                "lightgbm_score": self.original_score_lgbm,
                "lightgbm_prediction": self.original_lgbm_prediction,
                "lightgbm_threshold": getattr(self.fitness_evaluator.lightgbm_predict_fn, "best_threshold", None),
                "siamese_svm_score": self.original_score_svm,
                "siamese_svm_prediction": self.original_svm_prediction,
                "lightgbm_mpd": self.original_lgbm_mpd,
                "siamese_svm_mpd": self.original_svm_mpd,
            },
            "config": {
                "population_size": self.config.population_size,
                "generations": self.config.generations,
                "max_genes_per_chromosome": (
                    self.config.max_genes_per_chromosome
                ),
                "max_padding_bytes": self.config.max_padding_bytes,
                "seed": self.config.seed,
                "instruction_mutation_enabled": (
                    self.config.instruction_mutation_enabled
                ),
                "elf_transform_enabled": (
                    self.config.elf_transform_enabled
                ),
                "code_cave_enabled": self.config.code_cave_enabled,
                "trampoline_enabled": self.config.trampoline_enabled,
                "opaque_predicate_enabled": (
                    self.config.opaque_predicate_enabled
                ),
                "min_junk_instructions": (
                    self.config.min_junk_instructions
                ),
                "max_junk_instructions": (
                    self.config.max_junk_instructions
                ),
                "padding_style": self.config.padding_style,
                "parallel_workers": self.config.parallel_workers,
                "checkpoint_every": self.config.checkpoint_every,
                "diversity_ratio": self.config.diversity_ratio,
                "stagnation_generations": self.config.stagnation_generations,
                "immigrant_fraction": self.config.immigrant_fraction,
                "adaptive_operator_weights": self.config.adaptive_operator_weights,
                "generation_history": self.generation_history,
                "operator_stats": self.operator_stats,
                "normalize_detector_deltas": self.fitness_evaluator.normalize_detector_deltas,
                "evasion_bonus": self.fitness_evaluator.evasion_bonus,
                "evasion_definition": "baseline malware prediction 1 -> variant benign prediction 0",
                "operator_stats_note": "gain/evasion counts are chromosome-associated and gain is divided across genes in multi-gene chromosomes; they are not causal per-operator effects",
                "uncertainty_weight": self.fitness_evaluator.uncertainty_weight,
                "mpd_enabled": {
                    "lightgbm": self.fitness_evaluator.lgbm_mpd_scorer is not None,
                    "siamese_svm": self.fitness_evaluator.svm_mpd_scorer is not None,
                },
                "fitness_weights": {
                    "alpha": self.fitness_evaluator.weights.alpha,
                    "beta": self.fitness_evaluator.weights.beta,
                    "gamma": self.fitness_evaluator.weights.gamma,
                },
            },
            "results": [chromosome.to_dict() for chromosome in selected],
        }

        report_path = os.path.join(
            output_dir,
            "ga_results.json",
        )

        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Polymorphic (multi-stage) hardening
# ---------------------------------------------------------------------------
#
# A single GA run explores combinations of a handful of genes against the
# *original* binary. Polymorphic hardening instead chains N independent GA
# runs: stage k's winning variant becomes stage k+1's starting binary, so
# mutations accumulate across stages (new instruction candidates, a fresh
# EI_PAD state, a fresh code-cave search against the now-larger segment)
# instead of being capped at whatever fits in one chromosome. Each stage
# gets its own Validator baselined against its own immediate parent, so
# "byte-identical except whitelisted regions" is still enforced at every
# single step -- the cumulative result is just as auditable as any one
# stage, it's simply audited one link at a time.
#
# This is also where to lean on the Siamese SVM specifically: pass an
# svm_weight_schedule to push FitnessWeights.beta up (and alpha down) for
# later stages once LightGBM evasion has already been achieved, so the GA
# stops "spending" genes on a detector that's already beaten and focuses
# search on the one that isn't.

def run_polymorphic_hardening(
    initial_binary_path: str,
    stages: int,
    config_template: GAConfig,
    evaluator: FitnessEvaluator,
    expected_machine: Any,
    use_radare2: bool,
    allow_byte_modifications: bool,
    output_dir: str,
    top_k: int = 5,
    svm_weight_schedule: Optional[List[float]] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    current_path = initial_binary_path
    stage_reports: List[Dict[str, Any]] = []
    base_weights = evaluator.weights

    for stage_idx in range(stages):
        stage_dir = os.path.join(output_dir, f"stage_{stage_idx:02d}")
        os.makedirs(stage_dir, exist_ok=True)

        if svm_weight_schedule:
            beta = svm_weight_schedule[
                min(stage_idx, len(svm_weight_schedule) - 1)
            ]
            remaining = max(0.0, 1.0 - beta)
            # Keep alpha:gamma proportion from the base weights while
            # reallocating the freed-up weight from alpha to beta.
            denom = base_weights.alpha + base_weights.gamma
            alpha = (
                remaining * (base_weights.alpha / denom) if denom else 0.0
            )
            gamma = (
                remaining * (base_weights.gamma / denom) if denom else 0.0
            )
            evaluator.weights = FitnessWeights(
                alpha=alpha, beta=beta, gamma=gamma
            )

        print(
            f"\n[POLYMORPHIC] === stage {stage_idx:02d} === "
            f"source={current_path} "
            f"weights={evaluator.weights}"
        )

        stage_config = replace(
            config_template,
            work_dir=os.path.join(stage_dir, "_work"),
        )

        probe = lief.parse(current_path)
        if probe is None:
            print(
                f"[POLYMORPHIC] stage {stage_idx:02d}: "
                f"LIEF could not parse {current_path}; stopping"
            )
            break

        validator = Validator(
            expected_machine=expected_machine,
            baseline_path=current_path,
            use_radare2=use_radare2,
            report_path=os.path.join(stage_dir, "validation_report.json"),
            allow_byte_modifications=allow_byte_modifications,
        )

        ga = GeneticAlgorithm(
            source_binary_path=current_path,
            config=stage_config,
            fitness_evaluator=evaluator,
            validator=validator,
        )

        population = ga.evolve()
        ga.save_results(population=population, output_dir=stage_dir, top_k=top_k)

        best = population[0] if population else None
        if best is None or best.variant_path is None or best.fitness is None:
            print(
                f"[POLYMORPHIC] stage {stage_idx:02d}: "
                "no valid variant produced; stopping early"
            )
            break

        stage_reports.append(
            {
                "stage": stage_idx,
                "source_binary": current_path,
                "best_variant": best.variant_path,
                "fitness": best.fitness,
                "genes_this_stage": [g.to_dict() for g in best.genes],
                "validation": best.validation,
                "weights": {
                    "alpha": evaluator.weights.alpha,
                    "beta": evaluator.weights.beta,
                    "gamma": evaluator.weights.gamma,
                },
            }
        )

        print(
            f"[POLYMORPHIC] stage {stage_idx:02d} complete: "
            f"fitness={best.fitness:.6f} "
            f"variant={best.variant_path}"
        )

        current_path = best.variant_path

    evaluator.weights = base_weights

    report = {
        "initial_binary": initial_binary_path,
        "stages_completed": len(stage_reports),
        "stages_requested": stages,
        "stage_reports": stage_reports,
        "final_variant": current_path,
    }

    report_path = os.path.join(output_dir, "polymorphic_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n[POLYMORPHIC] final variant: {current_path}")
    print(f"[POLYMORPHIC] report: {report_path}")

    report["report_path"] = report_path
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "GA-based physical ELF mutation generator with strict "
            "overlay-preservation validation with optional conservative MIPS mutation."
        )
    )

    parser.add_argument(
        "--binary",
        default=None,
        help="Path to a single source ELF binary. Mutually exclusive with --binary-dir.",
    )
    parser.add_argument(
        "--binary-dir",
        default=None,
        help=(
            "Path to a directory of ELF binaries to process in batch. "
            "Every regular file in it is triaged (ELF magic, architecture, "
            "UPX-packing check) and run through the same GA pipeline as "
            "--binary; files that fail triage or error out are skipped "
            "and recorded in batch_summary.json rather than aborting the "
            "whole run. Mutually exclusive with --binary."
        ),
    )
    parser.add_argument(
        "--file-pattern",
        default="*",
        help="Glob pattern applied to --binary-dir (default: '*', i.e. every file).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories in --binary-dir.",
    )
    parser.add_argument(
        "--no-auto-unpack",
        action="store_true",
        help=(
            "Disable automatic `upx -d` unpacking of UPX-packed inputs "
            "(auto-unpack is ON by default, requires the `upx` CLI on "
            "PATH). With this set, a packed file goes straight to "
            "--allow-packed's skip-or-force decision."
        ),
    )
    parser.add_argument(
        "--allow-packed",
        action="store_true",
        help=(
            "If a file is (still) UPX-packed after an auto-unpack attempt "
            "(or with auto-unpack disabled/unavailable), mutate the packed "
            "stub anyway instead of skipping. Off by default because "
            "mutating a packed stub does not meaningfully mutate the "
            "actual (compressed) payload."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging (replaces the old always-on [DEBUG] prints).",
    )
    parser.add_argument(
        "--feature-extractor-module",
        required=True,
        help="Path to elf_feature_extractor.py",
    )
    parser.add_argument(
        "--lightgbm-model",
        help="Path to trained LightGBM joblib/pickle artifact",
    )
    parser.add_argument(
        "--siamese-model",
        default=str(PROJECT_ROOT / "models" / "siamese_svm1.pkl"),
        help="Path to trained Siamese SVM joblib artifact",
    )
    parser.add_argument(
        "--tfidf-vectorizer",
        default=str(
            PROJECT_ROOT
            / "data"
            / "tfidf"
            / "tfidf_vectorizer.pkl"
        ),
        help="Path to the fitted TF-IDF vectorizer used during training",
    )

    parser.add_argument(
        "--tfidf-dir",
        default=str(PROJECT_ROOT / "data" / "tfidf"),
        help="Directory containing tfidf_matrix.npz and labels.pkl (for the fixed reference set)",
    )

    parser.add_argument(
        "--tfidf-matrix",
        default=str(
            PROJECT_ROOT
            / "data"
            / "tfidf"
            / "tfidf_matrix.npz"
        ),
        help="Path to the saved TF-IDF matrix containing reference samples",
    )
    parser.add_argument(
        "--tfidf-labels",
        default=str(
            PROJECT_ROOT
            / "data"
            / "tfidf"
            / "labels.pkl"
        ),
        help="Path to labels.pkl corresponding to the TF-IDF matrix",
    )
    parser.add_argument(
        "--siamese-reference-index",
        type=int,
        default=None,
        help=(
            "TF-IDF row to use as the Siamese reference. "
            "Defaults to the first label-1 malware sample."
        ),
    )
    parser.add_argument(
        "--lightgbm-mpd-ensemble",
        default=None,
        help="Path to LightGBM bootstrap ensemble.pkl for per-variant MPD uncertainty.",
    )
    parser.add_argument(
        "--siamese-mpd-ensemble",
        default=None,
        help="Path to calibrated Siamese bootstrap ensemble_svm.pkl for per-variant MPD uncertainty.",
    )
    parser.add_argument(
        "--lightgbm-mpd-threshold",
        type=float,
        default=None,
        help="MPD alert threshold for LightGBM uncertainty (e.g. P95/P99).",
    )
    parser.add_argument(
        "--siamese-mpd-threshold",
        type=float,
        default=None,
        help="MPD alert threshold for Siamese uncertainty (e.g. P95/P99).",
    )
    parser.add_argument(
        "--uncertainty-weight",
        type=float,
        default=0.0,
        help="Penalty weight for normalized MPD increase. Positive values reward confident evasion; 0 keeps MPD diagnostic-only.",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--output-dir",
        default="./ga_output",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-padding-bytes",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--enable-instruction-mutation",
        action="store_true",
        help="Enable conservative same-size MIPS instruction mutations.",
    )
    parser.add_argument(
        "--enable-elf-transform",
        action="store_true",
        help=(
            "Enable the EI_PAD (ELF reserved header bytes) structural "
            "mutation."
        ),
    )
    parser.add_argument(
        "--enable-code-cave",
        action="store_true",
        help=(
            "Enable Phase 1 relocation-free length-changing mutation: "
            "fill existing PT_LOAD alignment gaps with junk MIPS "
            "instructions and grow the segment's p_filesz/p_memsz to "
            "cover them."
        ),
    )
    parser.add_argument(
        "--enable-trampoline-insertion",
        action="store_true",
        help=(
            "Enable Phase 2 mid-function junk insertion via detour/"
            "trampoline: redirect a safe consecutive instruction pair "
            "through a code-cave trampoline, preserving execution order "
            "via MIPS delay-slot relocation, without shifting any other "
            "address in the binary. Requires radare2. See the module "
            "docstring above find_trampoline_insertion_points() for the "
            "soundness conditions and the known indirect-jump caveat."
        ),
    )
    parser.add_argument(
        "--enable-opaque-predicates",
        action="store_true",
        help=(
            "Enable Phase 3: same trampoline redirect as "
            "--enable-trampoline-insertion, but the cave body opens with "
            "a tautological MIPS branch (beq $zero,$zero,...) wrapped "
            "around the dead junk, so the static instruction stream "
            "contains a real conditional branch into what looks like a "
            "live alternate path. Shares the same candidate pairs and "
            "soundness caveats as --enable-trampoline-insertion."
        ),
    )
    parser.add_argument(
        "--min-junk-instructions",
        type=int,
        default=4,
        help="Minimum junk instructions per code-cave/trampoline gene.",
    )
    parser.add_argument(
        "--max-junk-instructions",
        type=int,
        default=32,
        help="Maximum junk instructions per code-cave/trampoline gene.",
    )
    parser.add_argument(
        "--polymorphic-stages",
        type=int,
        default=1,
        help=(
            "Number of chained GA runs. Stage k's winning variant becomes "
            "stage k+1's starting binary, so mutations accumulate across "
            "stages instead of being limited to one chromosome. Default 1 "
            "= a single ordinary run (current behavior)."
        ),
    )
    parser.add_argument(
        "--polymorphic-svm-weight",
        type=float,
        default=None,
        help=(
            "If set, ramps FitnessWeights.beta (Siamese SVM weight) "
            "linearly from the default up to this value across the "
            "polymorphic stages, reallocating from alpha/gamma -- use "
            "this once LightGBM evasion is already solved and you want "
            "later stages to spend their genes on the SVM instead."
        ),
    )
    parser.add_argument(
        "--padding-style",
        choices=["mixed", *PADDING_STYLES],
        default="mixed",
        help=(
            "Overlay padding byte pattern: 'random' (original behavior), "
            "'low_entropy' (repeating pattern), 'string_like' (printable "
            "ASCII filler), or 'mixed' (GA samples a style per gene)."
        ),
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--evasion-bonus",
        type=float,
        default=0.50,
        help="Fitness bonus per detector when the candidate crosses into the benign/evasion decision.",
    )
    parser.add_argument(
        "--no-normalized-fitness",
        action="store_true",
        help="Use raw detector-score deltas instead of baseline-normalized deltas.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of concurrent chromosome evaluations. Start with 4-8 on a multi-core workstation.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write a resumable generation checkpoint every N generations.",
    )
    parser.add_argument(
        "--diversity-ratio",
        type=float,
        default=0.20,
        help="Target fraction of population reserved for exploration.",
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=4,
        help="Generations without improvement before injecting random immigrants.",
    )
    parser.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.15,
        help="Population fraction replaced by fresh chromosomes during stagnation.",
    )
    parser.add_argument(
        "--no-adaptive-operator-weights",
        action="store_true",
        help="Disable feedback-driven operator selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--validation-report",
        help="JSON validation report path",
    )
    parser.add_argument(
        "--no-radare2",
        action="store_true",
        help="Disable optional radare2 validation",
    )

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if bool(args.binary) == bool(args.binary_dir):
        parser.error(
            "pass exactly one of --binary (single file) or --binary-dir "
            "(batch mode), not both and not neither"
        )

    if args.binary_dir and not os.path.isdir(args.binary_dir):
        parser.error(f"--binary-dir does not exist or is not a directory: {args.binary_dir}")

    if args.population_size <= 0:
        parser.error("--population-size must be > 0")

    if args.generations < 0:
        parser.error("--generations must be >= 0")

    if args.max_padding_bytes <= 0:
        parser.error("--max-padding-bytes must be > 0")

    if args.max_genes <= 0:
        parser.error("--max-genes must be > 0")
    if args.evasion_bonus < 0:
        parser.error("--evasion-bonus must be >= 0")
    if args.uncertainty_weight < 0:
        parser.error("--uncertainty-weight must be >= 0")
    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be > 0")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be > 0")
    if not 0.0 <= args.diversity_ratio <= 1.0:
        parser.error("--diversity-ratio must be between 0 and 1")
    if args.stagnation_generations <= 0:
        parser.error("--stagnation-generations must be > 0")
    if not 0.0 <= args.immigrant_fraction <= 1.0:
        parser.error("--immigrant-fraction must be between 0 and 1")

    if args.min_junk_instructions <= 0:
        parser.error("--min-junk-instructions must be > 0")

    if args.max_junk_instructions < args.min_junk_instructions:
        parser.error(
            "--max-junk-instructions must be >= --min-junk-instructions"
        )

    if args.polymorphic_stages <= 0:
        parser.error("--polymorphic-stages must be > 0")

    if (
        args.polymorphic_svm_weight is not None
        and not (0.0 < args.polymorphic_svm_weight < 1.0)
    ):
        parser.error("--polymorphic-svm-weight must be between 0 and 1")

    if args.binary and not os.path.isfile(args.binary):
        parser.error(f"Binary does not exist: {args.binary}")

    if not os.path.isfile(args.feature_extractor_module):
        parser.error(
            f"Feature extractor does not exist: "
            f"{args.feature_extractor_module}"
        )

    if args.lightgbm_model and not os.path.isfile(args.lightgbm_model):
        parser.error(
            f"LightGBM model does not exist: "
            f"{args.lightgbm_model}"
        )

    if args.siamese_model and not os.path.isfile(args.siamese_model):
        parser.error(
            f"Siamese SVM model does not exist: "
            f"{args.siamese_model}"
        )

    if args.tfidf_vectorizer and not os.path.isfile(
        args.tfidf_vectorizer
    ):
        parser.error(
            f"TF-IDF vectorizer does not exist: "
            f"{args.tfidf_vectorizer}"
        )

    if args.tfidf_matrix and not os.path.isfile(args.tfidf_matrix):
        parser.error(
            f"TF-IDF matrix does not exist: "
            f"{args.tfidf_matrix}"
        )

    if args.tfidf_labels and not os.path.isfile(args.tfidf_labels):
        parser.error(
            f"TF-IDF labels do not exist: "
            f"{args.tfidf_labels}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    feature_fn = FitnessEvaluator.default_feature_extractor(
        args.feature_extractor_module
    )

    lgbm_fn = None
    if args.lightgbm_model:
        lgbm_fn = FitnessEvaluator.default_lightgbm_predict(
            args.lightgbm_model
        )

        threshold = getattr(lgbm_fn, "best_threshold", None)
        threshold_text = (
            f", threshold={threshold:.6f}"
            if threshold is not None
            else ""
        )

        print(
            f"[INFO] LightGBM loaded: "
            f"{len(lgbm_fn.feature_names)} features{threshold_text}"
        )

    siamese_fn = None
    siamese_details_fn = None
    siamese_scorer = None

    siamese_requested = any(
        value is not None
        for value in (
            args.siamese_model,
            args.tfidf_vectorizer,
            args.tfidf_matrix,
            args.tfidf_labels,
        )
    )

    if siamese_requested:
        missing_args = [
            name
            for name, value in (
                ("--siamese-model", args.siamese_model),
                ("--tfidf-vectorizer", args.tfidf_vectorizer),
                ("--tfidf-matrix", args.tfidf_matrix),
                ("--tfidf-labels", args.tfidf_labels),
            )
            if value is None
        ]

        if missing_args:
            parser.error(
                "Siamese evaluation requires all four arguments: "
                + ", ".join(missing_args)
            )

        siamese_scorer = SiameseSVMScorer(
            model_path=args.siamese_model,
            vectorizer_path=args.tfidf_vectorizer,
            tfidf_dir=args.tfidf_dir,
        )
        siamese_fn = siamese_scorer
        siamese_details_fn = siamese_scorer.predict_details

    lgbm_mpd_scorer = None
    svm_mpd_scorer = None

    if args.lightgbm_mpd_ensemble:
        if lgbm_fn is None:
            parser.error("--lightgbm-mpd-ensemble requires --lightgbm-model")
        lgbm_mpd_scorer = BootstrapMPDScorer(
            args.lightgbm_mpd_ensemble, "lightgbm",
            threshold=args.lightgbm_mpd_threshold,
            feature_names=lgbm_fn.feature_names,
        )

    if args.siamese_mpd_ensemble:
        if siamese_scorer is None:
            parser.error("--siamese-mpd-ensemble requires the Siamese model/vectorizer/matrix/labels")
        svm_mpd_scorer = BootstrapMPDScorer(
            args.siamese_mpd_ensemble, "siamese",
            siamese_scorer=siamese_scorer,
            threshold=args.siamese_mpd_threshold,
        )

    if lgbm_fn is None and siamese_fn is None:
        parser.error(
            "At least one detector is required: provide --lightgbm-model "
            "and/or the complete Siamese SVM artifact set."
        )

    evaluator = FitnessEvaluator(
        weights=FitnessWeights(),
        feature_extractor_fn=feature_fn,
        lightgbm_predict_fn=lgbm_fn,
        siamese_svm_predict_fn=siamese_fn,
        siamese_svm_details_fn=siamese_details_fn,
        normalize_detector_deltas=not args.no_normalized_fitness,
        evasion_bonus=args.evasion_bonus,
        lgbm_mpd_scorer=lgbm_mpd_scorer,
        svm_mpd_scorer=svm_mpd_scorer,
        uncertainty_weight=args.uncertainty_weight,
    )

    def _run_one_binary(binary_path: str, out_dir: str) -> Dict[str, Any]:
        """Run triage + the full GA pipeline for a single binary.

        Returns a small summary dict. Raises BinaryTriageError for files
        that should be *skipped* (bad magic, wrong arch, packed), and lets
        any other exception propagate so the caller can decide how to log
        it (single-file mode: crash; batch mode: record and move on).
        """
        probe, mutate_path = triage_binary(
            binary_path,
            expected_machine=None,
            allow_packed=args.allow_packed,
            auto_unpack=not args.no_auto_unpack,
            unpack_work_dir=os.path.join(out_dir, "_unpacked"),
        )
        was_unpacked = mutate_path != binary_path
        if was_unpacked:
            logger.info(
                "%s: running GA on auto-unpacked copy %s (original left untouched)",
                binary_path, mutate_path,
            )

        os.makedirs(out_dir, exist_ok=True)

        validation_report = (
            args.validation_report
            or os.path.join(out_dir, "validation_report.json")
        )

        validator = Validator(
            expected_machine=probe.header.machine_type,
            baseline_path=mutate_path,
            use_radare2=not args.no_radare2,
            report_path=validation_report,
            allow_byte_modifications=(
                args.enable_instruction_mutation
                or args.enable_elf_transform
                or args.enable_code_cave
                or args.enable_trampoline_insertion
                or args.enable_opaque_predicates
            ),
        )

        config = GAConfig(
            population_size=args.population_size,
            generations=args.generations,
            max_genes_per_chromosome=args.max_genes,
            max_padding_bytes=args.max_padding_bytes,
            work_dir=os.path.join(out_dir, "_work"),
            seed=args.seed,
            instruction_mutation_enabled=args.enable_instruction_mutation,
            elf_transform_enabled=args.enable_elf_transform,
            code_cave_enabled=args.enable_code_cave,
            trampoline_enabled=args.enable_trampoline_insertion,
            opaque_predicate_enabled=args.enable_opaque_predicates,
            min_junk_instructions=args.min_junk_instructions,
            max_junk_instructions=args.max_junk_instructions,
            padding_style=args.padding_style,
            parallel_workers=args.parallel_workers,
            checkpoint_every=args.checkpoint_every,
            diversity_ratio=args.diversity_ratio,
            stagnation_generations=args.stagnation_generations,
            immigrant_fraction=args.immigrant_fraction,
            adaptive_operator_weights=not args.no_adaptive_operator_weights,
        )

        if args.polymorphic_stages > 1:
            svm_schedule = None
            if args.polymorphic_svm_weight is not None:
                start_beta = FitnessWeights().beta
                end_beta = args.polymorphic_svm_weight
                steps = args.polymorphic_stages
                svm_schedule = [
                    start_beta + (end_beta - start_beta) * (i / max(1, steps - 1))
                    for i in range(steps)
                ]

            run_polymorphic_hardening(
                initial_binary_path=mutate_path,
                stages=args.polymorphic_stages,
                config_template=config,
                evaluator=evaluator,
                expected_machine=probe.header.machine_type,
                use_radare2=not args.no_radare2,
                allow_byte_modifications=validator.allow_byte_modifications,
                output_dir=out_dir,
                top_k=args.top_k,
                svm_weight_schedule=svm_schedule,
            )
            return {
                "binary": binary_path,
                "mutated_path": mutate_path,
                "was_upx_unpacked": was_unpacked,
                "status": "ok",
                "mode": "polymorphic",
                "output_dir": out_dir,
            }

        ga = GeneticAlgorithm(
            source_binary_path=mutate_path,
            config=config,
            fitness_evaluator=evaluator,
            validator=validator,
        )

        population = ga.evolve()

        # Final per-binary statistics and JSON report.
        final_best = max(population, key=ga._fitness) if population else None
        most_used_op = max(
            ga.stats["operator_usage"].items(),
            key=lambda kv: kv[1]["used"],
            default=(None, {"used": 0, "success": 0}),
        )
        if final_best is not None:
            final_result = final_best.fitness_result or {}
            final_lgbm = final_result.get("variant_lgbm")
            final_svm = final_result.get("variant_svm")
            baseline_lgbm = ga.original_score_lgbm
            baseline_svm = ga.original_score_svm
            ga.stats["final"] = {
                "lgbm": final_lgbm,
                "svm": final_svm,
                "lgbm_delta": (
                    float(final_lgbm) - float(baseline_lgbm)
                    if final_lgbm is not None and baseline_lgbm is not None else None
                ),
                "svm_delta": (
                    float(final_svm) - float(baseline_svm)
                    if final_svm is not None and baseline_svm is not None else None
                ),
                "fitness": float(ga._fitness(final_best)),
                "genes": len(final_best.genes),
            }
        else:
            ga.stats["final"] = None

        ga.stats["most_used_operator"] = {
            "operator": most_used_op[0],
            "used": most_used_op[1]["used"],
            "success": most_used_op[1]["success"],
        }

        print("=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        print(f"File            : {ga.stats['file']}")
        print(f"Initial LGBM    : {ga.stats['baseline']['lgbm']}")
        print(f"Initial SVM     : {ga.stats['baseline']['svm']}")
        if ga.stats["final"] is not None:
            final_stats = ga.stats["final"]
            print(f"Final LGBM      : {final_stats['lgbm']}  (delta={final_stats['lgbm_delta']})")
            print(f"Final SVM       : {final_stats['svm']}  (delta={final_stats['svm_delta']})")
            print(f"Best fitness    : {final_stats['fitness']:.6f}")
        print(
            f"Most used gene  : {most_used_op[0]} "
            f"(used={most_used_op[1]['used']}, success={most_used_op[1]['success']})"
        )

        stats_path = Path(out_dir) / f"{Path(mutate_path).stem}_stats.json"
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(ga.stats, handle, indent=2)
        print(f"Stats saved to  : {stats_path}")

        results_path = ga.save_results(
            population=population,
            output_dir=out_dir,
            top_k=args.top_k,
        )

        print(f"\n[FINAL RANKING] {binary_path}" + (" (auto-unpacked)" if was_unpacked else ""))
        for index, chromosome in enumerate(population[: args.top_k], start=1):
            print(
                f"{index:02d}. "
                f"fitness={ga._fitness(chromosome):.6f} "
                f"genes={len(chromosome.genes)} "
                f"cost={chromosome.modification_cost} "
                f"path={chromosome.variant_path}"
            )

        print(f"[RESULTS] {results_path}")
        print(f"[VALIDATION REPORT] {validation_report}")

        best_fitness = ga._fitness(population[0]) if population else None
        return {
            "binary": binary_path,
            "mutated_path": mutate_path,
            "was_upx_unpacked": was_unpacked,
            "status": "ok",
            "mode": "single",
            "best_fitness": best_fitness,
            "results_path": results_path,
            "output_dir": out_dir,
        }

    # -----------------------------------------------------------------
    # Single-file mode: unchanged behavior, errors are fatal.
    # -----------------------------------------------------------------
    if args.binary:
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            _run_one_binary(args.binary, args.output_dir)
        except BinaryTriageError as exc:
            parser.error(f"{args.binary}: {exc}")
        return

    # -----------------------------------------------------------------
    # Batch mode: one subdirectory per input file under --output-dir,
    # failures are isolated per-file (triage skip vs. hard error are
    # both recorded) so one bad sample can't take down the whole batch.
    # -----------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    root = Path(args.binary_dir)
    iterator = root.rglob(args.file_pattern) if args.recursive else root.glob(args.file_pattern)
    candidates = sorted(p for p in iterator if p.is_file())
    if not candidates:
        parser.error(
            f"--binary-dir {args.binary_dir} matched no files "
            f"(pattern: {args.file_pattern!r})"
        )

    logger.info("Batch mode: %d candidate file(s) in %s", len(candidates), args.binary_dir)

    summary: List[Dict[str, Any]] = []
    for path in candidates:
        if args.recursive:
            try:
                relative_name = path.relative_to(root)
            except ValueError:
                relative_name = Path(path.name)
            # Keep nested input layout while making the output root separate.
            sample_out_dir = os.path.join(args.output_dir, str(relative_name))
            sample_out_dir = os.path.splitext(sample_out_dir)[0]
        else:
            sample_out_dir = os.path.join(args.output_dir, path.name)
            sample_out_dir = os.path.splitext(sample_out_dir)[0]
        logger.info("=== [%s] ===", path.name)
        try:
            result = _run_one_binary(str(path), sample_out_dir)
        except BinaryTriageError as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            summary.append({"binary": str(path), "status": "skipped", "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 -- isolate one bad file from the batch
            logger.error("failed on %s: %s", path.name, exc, exc_info=args.verbose)
            summary.append({"binary": str(path), "status": "error", "reason": str(exc)})
            continue
        summary.append(result)

    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ok = sum(1 for r in summary if r["status"] == "ok")
    skipped = sum(1 for r in summary if r["status"] == "skipped")
    errored = sum(1 for r in summary if r["status"] == "error")
    print(
        f"\n[BATCH DONE] {ok} ok, {skipped} skipped, {errored} errored "
        f"out of {len(summary)} file(s)"
    )
    print(f"[BATCH SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
