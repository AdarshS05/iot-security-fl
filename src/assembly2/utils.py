from __future__ import annotations
import hashlib, json, math, re
from pathlib import Path
from collections import Counter


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())


def safe_json(value):
    try:
        return json.dumps(value, default=str)
    except Exception:
        return '{}'


def parse_json_cmd(r2, command: str):
    try:
        value = r2.cmdj(command)
        return value if value is not None else None
    except Exception:
        return None


def normalize_instruction(line: str, mode: str) -> str:
    """Normalize one disassembly line without deleting instruction identity."""
    line = line.strip().lower()
    if not line:
        return ''
    # Strip address prefix emitted by some disassemblers: 00401234: opcode ...
    line = re.sub(r'^(?:0x)?[0-9a-f]{6,16}:\s*', '', line)
    parts = line.split(None, 1)
    mnemonic = parts[0]
    operands = parts[1] if len(parts) > 1 else ''
    if mode == 'n0_aggressive':
        # Diagnostic baseline only: deliberately broad abstraction.
        operands = re.sub(r'[-+]?0x[0-9a-f]+', ' IMM ', operands)
        operands = re.sub(r'(?<![a-z0-9_])\d{4,}(?![a-z0-9_])', ' IMM ', operands)
        operands = re.sub(r'[-+]?\d+\s*\(', ' MEM(', operands)
    elif mode == 'n1_conservative':
        # Keep constants and offsets; abstract only likely absolute branch targets.
        if mnemonic in {'j','jal','jr','beq','bne','blez','bgtz','bltz','bgez','b'}:
            operands = re.sub(r'(?<![a-z0-9_])0x[0-9a-f]{5,}(?![a-z0-9_])', ' TARGET ', operands)
    elif mode == 'n2_selective':
        # Preserve small constants and ordinary stack/global offsets.
        def hex_sub(m):
            token = m.group(0)
            try: return ' IMM ' if int(token, 16) >= 0x1000 else token
            except ValueError: return token
        operands = re.sub(r'0x[0-9a-f]+', hex_sub, operands)
        def dec_sub(m):
            token = m.group(0)
            try: return ' IMM ' if abs(int(token)) >= 4096 else token
            except ValueError: return token
        operands = re.sub(r'(?<![a-z0-9_])-?\d+(?![a-z0-9_])', dec_sub, operands)
    else:
        raise ValueError(f'Unknown normalization mode: {mode}')
    operands = re.sub(r'\s+', ' ', operands).strip()
    return f'{mnemonic} {operands}'.strip()
