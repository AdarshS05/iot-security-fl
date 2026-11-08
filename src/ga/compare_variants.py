"""
compare_variants.py
====================

Compares each GA-generated variant binary against its original source
binary and produces:

  1. A byte-level diff summary (size delta, changed regions, overlay bytes)
  2. Structural diff via LIEF (segments, sections, entrypoint, header fields)
  3. Fitness / gene metadata pulled from manifest.json
  4. A single consolidated report (JSON + human-readable text)

Usage:
    python3 compare_variants.py \
        --original ../../data/raw/malware_dataset/2b50f3cbc1fdb69ca8b84de24bb0ccb2b4150d32328cfcc4c49946b1c34667bb \
        --ga-output-dir ./ga_output \
        --report-out ./ga_output/comparison_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import lief


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def byte_diff(original: bytes, variant: bytes) -> Dict[str, Any]:
    """Compares two byte strings and reports where they differ.

    Handles size differences (e.g. overlay appends) by comparing the
    overlapping prefix, then reporting the trailing appended region
    separately.
    """
    min_len = min(len(original), len(variant))
    diff_ranges: List[Dict[str, int]] = []

    in_diff = False
    start = 0
    for i in range(min_len):
        differs = original[i] != variant[i]
        if differs and not in_diff:
            in_diff = True
            start = i
        elif not differs and in_diff:
            in_diff = False
            diff_ranges.append({"start": start, "end": i, "length": i - start})
    if in_diff:
        diff_ranges.append({"start": start, "end": min_len, "length": min_len - start})

    changed_bytes_in_overlap = sum(r["length"] for r in diff_ranges)
    appended_bytes = max(0, len(variant) - len(original))
    truncated_bytes = max(0, len(original) - len(variant))

    return {
        "original_size": len(original),
        "variant_size": len(variant),
        "size_delta": len(variant) - len(original),
        "overlap_length": min_len,
        "changed_byte_ranges_in_overlap": diff_ranges,
        "num_changed_regions": len(diff_ranges),
        "changed_bytes_in_overlap": changed_bytes_in_overlap,
        "appended_bytes": appended_bytes,
        "truncated_bytes": truncated_bytes,
        "percent_overlap_changed": (
            round(100.0 * changed_bytes_in_overlap / min_len, 4) if min_len else 0.0
        ),
    }


def preview_hex(data: bytes, offset: int, length: int, context: int = 8) -> Dict[str, str]:
    """Small hex preview around a changed region, for quick eyeballing."""
    lo = max(0, offset - context)
    hi = min(len(data), offset + length + context)
    return {
        "offset": hex(offset),
        "hex": data[lo:hi].hex(sep=" "),
    }


# --------------------------------------------------------------------------
# Structural (ELF-level) diff
# --------------------------------------------------------------------------

def describe_elf(path: str) -> Optional[Dict[str, Any]]:
    b = lief.parse(path)
    if b is None:
        return None

    segments = []
    for seg in b.segments:
        segments.append({
            "type": str(seg.type),
            "flags": str(seg.flags),
            "virtual_address": hex(int(seg.virtual_address)),
            "virtual_size": int(seg.virtual_size),
            "physical_size": int(seg.physical_size),
            "file_offset": int(seg.file_offset),
        })

    sections = []
    try:
        for sec in b.sections:
            sections.append({
                "name": sec.name,
                "type": str(sec.type),
                "size": int(sec.size),
                "virtual_address": hex(int(sec.virtual_address)),
            })
    except Exception:
        pass  # sectionless / stripped binary

    return {
        "entrypoint": hex(int(b.header.entrypoint)),
        "machine_type": str(b.header.machine_type),
        "file_type": str(b.header.file_type),
        "num_segments": len(segments),
        "num_sections": len(sections),
        "has_section_headers": len(sections) > 0,
        "segments": segments,
        "sections": sections,
    }


def diff_elf_structure(orig_desc: Optional[Dict[str, Any]],
                        var_desc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if orig_desc is None or var_desc is None:
        return {"comparable": False, "reason": "one or both files failed LIEF parse"}

    result: Dict[str, Any] = {"comparable": True, "differences": []}

    for key in ("entrypoint", "machine_type", "file_type", "num_segments",
                "num_sections", "has_section_headers"):
        if orig_desc.get(key) != var_desc.get(key):
            result["differences"].append({
                "field": key,
                "original": orig_desc.get(key),
                "variant": var_desc.get(key),
            })

    orig_segs = orig_desc["segments"]
    var_segs = var_desc["segments"]
    if len(orig_segs) == len(var_segs):
        for i, (o, v) in enumerate(zip(orig_segs, var_segs)):
            for field_name in ("virtual_size", "physical_size", "virtual_address"):
                if o[field_name] != v[field_name]:
                    result["differences"].append({
                        "field": f"segment[{i}].{field_name}",
                        "original": o[field_name],
                        "variant": v[field_name],
                    })
    else:
        result["differences"].append({
            "field": "segment_count",
            "original": len(orig_segs),
            "variant": len(var_segs),
        })

    orig_sec_names = {s["name"] for s in orig_desc["sections"]}
    var_sec_names = {s["name"] for s in var_desc["sections"]}
    added = sorted(var_sec_names - orig_sec_names)
    removed = sorted(orig_sec_names - var_sec_names)
    if added:
        result["sections_added"] = added
    if removed:
        result["sections_removed"] = removed

    result["structurally_identical"] = (
        len(result["differences"]) == 0 and not added and not removed
    )
    return result


# --------------------------------------------------------------------------
# Manifest lookup
# --------------------------------------------------------------------------

def load_manifest(ga_output_dir: str) -> List[Dict[str, Any]]:
    manifest_path = os.path.join(ga_output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_variant_files(ga_output_dir: str) -> List[str]:
    return sorted(
        os.path.join(ga_output_dir, fn)
        for fn in os.listdir(ga_output_dir)
        if fn.startswith("variant_") and fn.endswith(".bin")
    )


# --------------------------------------------------------------------------
# Main comparison driver
# --------------------------------------------------------------------------

def compare_one(original_path: str, original_bytes: bytes,
                 original_hash: str, original_elf: Optional[Dict[str, Any]],
                 variant_path: str, manifest_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    variant_bytes = read_bytes(variant_path)
    variant_hash = sha256(variant_path)
    variant_elf = describe_elf(variant_path)

    report: Dict[str, Any] = {
        "variant_file": os.path.basename(variant_path),
        "original_file": os.path.basename(original_path),
        "identical_bytes": original_hash == variant_hash,
        "original_sha256": original_hash,
        "variant_sha256": variant_hash,
        "byte_diff": byte_diff(original_bytes, variant_bytes),
        "structural_diff": diff_elf_structure(original_elf, variant_elf),
    }

    # Small hex previews for the first few changed regions (helps eyeballing)
    previews = []
    for r in report["byte_diff"]["changed_byte_ranges_in_overlap"][:5]:
        previews.append({
            "original": preview_hex(original_bytes, r["start"], r["length"]),
            "variant": preview_hex(variant_bytes, r["start"], r["length"]),
        })
    if previews:
        report["changed_region_previews"] = previews

    if manifest_entry:
        report["fitness"] = manifest_entry.get("fitness")
        report["modification_cost"] = manifest_entry.get("modification_cost")
        report["genes_applied"] = manifest_entry.get("genes")

    return report


def summarize(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(reports)
    if n == 0:
        return {"num_variants": 0}

    size_deltas = [r["byte_diff"]["size_delta"] for r in reports]
    changed_bytes = [r["byte_diff"]["changed_bytes_in_overlap"] for r in reports]
    fitnesses = [r["fitness"] for r in reports if r.get("fitness") is not None]
    identical_count = sum(1 for r in reports if r["identical_bytes"])
    structurally_identical_count = sum(
        1 for r in reports
        if r["structural_diff"].get("structurally_identical")
    )
    gene_ops: Dict[str, int] = {}
    for r in reports:
        for g in (r.get("genes_applied") or []):
            gene_ops[g["operator"]] = gene_ops.get(g["operator"], 0) + 1

    return {
        "num_variants": n,
        "identical_to_original": identical_count,
        "structurally_identical_to_original": structurally_identical_count,
        "size_delta_bytes": {
            "min": min(size_deltas), "max": max(size_deltas),
            "avg": round(sum(size_deltas) / n, 2),
        },
        "changed_bytes_in_overlap": {
            "min": min(changed_bytes), "max": max(changed_bytes),
            "avg": round(sum(changed_bytes) / n, 2),
        },
        "fitness": (
            {"min": min(fitnesses), "max": max(fitnesses),
             "avg": round(sum(fitnesses) / len(fitnesses), 4)}
            if fitnesses else None
        ),
        "gene_operator_counts_across_variants": gene_ops,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare GA-generated ELF variants against the original binary")
    parser.add_argument("--original", required=True, help="Path to original source binary")
    parser.add_argument("--ga-output-dir", required=True,
                         help="Directory containing variant_*.bin and manifest.json")
    parser.add_argument("--report-out", default=None,
                         help="Where to write the JSON report "
                              "(default: <ga-output-dir>/comparison_report.json)")
    parser.add_argument("--text-summary", action="store_true",
                         help="Also print a human-readable summary to stdout")
    args = parser.parse_args()

    if not os.path.isfile(args.original):
        parser.error(f"Original binary not found: {args.original}")
    if not os.path.isdir(args.ga_output_dir):
        parser.error(f"GA output dir not found: {args.ga_output_dir}")

    report_out = args.report_out or os.path.join(args.ga_output_dir, "comparison_report.json")

    original_bytes = read_bytes(args.original)
    original_hash = sha256(args.original)
    original_elf = describe_elf(args.original)

    manifest = load_manifest(args.ga_output_dir)
    manifest_by_path = {os.path.basename(m["path"]): m for m in manifest}

    variant_paths = find_variant_files(args.ga_output_dir)
    if not variant_paths:
        parser.error(f"No variant_*.bin files found in {args.ga_output_dir}")

    reports = []
    for vp in variant_paths:
        entry = manifest_by_path.get(os.path.basename(vp))
        reports.append(compare_one(
            args.original, original_bytes, original_hash, original_elf, vp, entry
        ))

    full_report = {
        "original_binary": args.original,
        "original_sha256": original_hash,
        "original_size": len(original_bytes),
        "original_elf_summary": {
            "entrypoint": original_elf.get("entrypoint") if original_elf else None,
            "machine_type": original_elf.get("machine_type") if original_elf else None,
            "file_type": original_elf.get("file_type") if original_elf else None,
            "has_section_headers": original_elf.get("has_section_headers") if original_elf else None,
            "num_segments": original_elf.get("num_segments") if original_elf else None,
        },
        "variants": reports,
        "summary": summarize(reports),
    }

    os.makedirs(os.path.dirname(os.path.abspath(report_out)), exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Wrote comparison report: {report_out}")

    if args.text_summary:
        print("\n=== SUMMARY ===")
        s = full_report["summary"]
        print(f"Variants compared:              {s['num_variants']}")
        print(f"Byte-identical to original:     {s['identical_to_original']}")
        print(f"Structurally identical (ELF):   {s['structurally_identical_to_original']}")
        print(f"Size delta (bytes):              min={s['size_delta_bytes']['min']} "
              f"max={s['size_delta_bytes']['max']} avg={s['size_delta_bytes']['avg']}")
        print(f"Changed bytes in overlap region: min={s['changed_bytes_in_overlap']['min']} "
              f"max={s['changed_bytes_in_overlap']['max']} avg={s['changed_bytes_in_overlap']['avg']}")
        if s.get("fitness"):
            print(f"Fitness:                         min={s['fitness']['min']:.4f} "
                  f"max={s['fitness']['max']:.4f} avg={s['fitness']['avg']:.4f}")
        print(f"Gene operator usage:             {s['gene_operator_counts_across_variants']}")

        print("\n=== PER-VARIANT ===")
        for r in full_report["variants"]:
            print(f"\n- {r['variant_file']}")
            print(f"    identical_bytes:   {r['identical_bytes']}")
            print(f"    size_delta:        {r['byte_diff']['size_delta']} bytes "
                  f"(appended={r['byte_diff']['appended_bytes']}, "
                  f"changed_in_overlap={r['byte_diff']['changed_bytes_in_overlap']})")
            print(f"    structurally_identical: "
                  f"{r['structural_diff'].get('structurally_identical')}")
            if r.get("fitness") is not None:
                print(f"    fitness:           {r['fitness']:.4f}  "
                      f"mod_cost: {r.get('modification_cost')}")
            if r.get("genes_applied"):
                ops = [g["operator"] for g in r["genes_applied"]]
                print(f"    genes:             {ops}")


if __name__ == "__main__":
    main()
