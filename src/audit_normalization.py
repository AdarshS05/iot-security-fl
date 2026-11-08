#!/usr/bin/env python3
"""Compare structure and file counts in normalized benign/malware directories."""
import argparse
from collections import Counter
from pathlib import Path

def inspect(folder):
    files = sorted(folder.glob("*.asm"))
    mn, tok = Counter(), Counter()
    instructions, empty = 0, []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            empty.append(f.name)
        for line in content.splitlines():
            parts = line.split()
            if parts:
                instructions += 1
                mn[parts[0]] += 1
                tok.update(parts[1:])
    return {"files": files, "mn": mn, "tok": tok, "instructions": instructions, "empty": empty}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benign", required=True, type=Path)
    p.add_argument("--malware", required=True, type=Path)
    p.add_argument("--expected-benign", type=int, default=500)
    p.add_argument("--expected-malware", type=int, default=497)
    a = p.parse_args()
    b, m = inspect(a.benign), inspect(a.malware)

    for label, x, expected in [("BENIGN", b, a.expected_benign), ("MALWARE", m, a.expected_malware)]:
        print(f"\n{label}: files={len(x['files'])}/{expected}, instructions={x['instructions']:,}, "
              f"unique_mnemonics={len(x['mn'])}, unique_operand_tokens={len(x['tok'])}, empty={len(x['empty'])}")

    bn, mn = set(b["mn"]), set(m["mn"])
    bt, mt = set(b["tok"]), set(m["tok"])
    print("\nCOMPARISON")
    print(f"Shared mnemonics: {len(bn & mn)}")
    print(f"Benign-only mnemonics: {len(bn - mn)}")
    print(f"Malware-only mnemonics: {len(mn - bn)}")
    print(f"Shared operand tokens: {len(bt & mt)}")
    print(f"Benign-only operand tokens: {len(bt - mt)}")
    print(f"Malware-only operand tokens: {len(mt - bt)}")
    print(f"Benign-only filenames: {len({x.name for x in b['files']} - {x.name for x in m['files']})}")
    print(f"Malware-only filenames: {len({x.name for x in m['files']} - {x.name for x in b['files']})}")
    failures = []
    if len(b["files"]) != a.expected_benign: failures.append("Benign file count mismatch")
    if len(m["files"]) != a.expected_malware: failures.append("Malware file count mismatch")
    if b["empty"] or m["empty"]: failures.append("Empty normalized files found")
    print("\nRESULT:", "PASS" if not failures else "CHECK REQUIRED")
    for item in failures: print("-", item)

if __name__ == "__main__":
    main()
