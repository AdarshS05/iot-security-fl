#!/usr/bin/env python3
"""Batch radare2 function-based extraction with per-sample triage metadata.

Does not execute samples, perform linear sweeps, or automatically unpack files.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import sha256_file, entropy, safe_json, parse_json_cmd


def flatten_functions(functions, r2):
    records, seen, duplicate_addrs, conflicts = [], {}, 0, 0
    for fn in functions or []:
        addr = fn.get('addr', fn.get('offset'))
        if addr is None:
            continue
        try:
            addr_i = int(addr)
        except Exception:
            continue
        pdf = parse_json_cmd(r2, f'pdfj @ {addr_i}') or {}
        ops = pdf.get('ops') or pdf.get('blocks', [])
        for op in ops:
            op_addr = op.get('addr', op.get('offset'))
            if op_addr is None:
                continue
            try: op_addr = int(op_addr)
            except Exception: continue
            text = op.get('disasm') or op.get('opcode') or op.get('mnemonic') or ''
            text = re.sub(r'\s+', ' ', str(text)).strip()
            if not text:
                continue
            if op_addr in seen:
                if seen[op_addr] == text: duplicate_addrs += 1
                else: conflicts += 1
                continue
            seen[op_addr] = text
            records.append((op_addr, text, int(op.get('size') or 0), str(fn.get('name',''))))
    records.sort(key=lambda x: x[0])
    return records, duplicate_addrs, conflicts


def get_entrypoint(r2):
    eps = parse_json_cmd(r2, 'iej')
    candidates = eps if isinstance(eps, list) else ([eps] if isinstance(eps, dict) else [])
    for item in candidates:
        if not isinstance(item, dict): continue
        for key in ('vaddr','addr','paddr','offset'):
            value = item.get(key)
            if value is not None:
                try: return int(value)
                except Exception: pass
    # Fallback to textual flags for radare2 builds that do not expose iej JSON.
    try:
        for line in r2.cmd('f~entry').splitlines():
            m = re.search(r'(0x[0-9a-fA-F]+)', line)
            if m: return int(m.group(1),16)
    except Exception: pass
    return None


def triage(path: Path, raw: bytes, r2):
    info = parse_json_cmd(r2, 'ij') or {}
    bininfo = info.get('bin') or {}
    sections = parse_json_cmd(r2, 'iSj') or []
    segments = parse_json_cmd(r2, 'iSSj') or []
    entrypoint = get_entrypoint(r2)
    imports = parse_json_cmd(r2, 'iij') or []
    strings = parse_json_cmd(r2, 'izj') or []
    names = [str(x.get('name','')) for x in sections if isinstance(x,dict)]
    upx_markers = [m.decode('ascii','ignore') for m in (b'UPX!',b'UPX0',b'UPX1',b'.upx') if m in raw]
    # Entropy is a triage feature, not a packer verdict.
    chunks = [entropy(raw[i:i+4096]) for i in range(0, len(raw), 4096) if raw[i:i+4096]]
    exec_segments = [s for s in segments if isinstance(s,dict) and ('x' in str(s.get('perm', s.get('flags',''))).lower())]
    entry = entrypoint
    upx_names = [n for n in names if 'upx' in n.lower()]
    indicators = []
    if upx_markers or upx_names: indicators.append('UPX signature/name present')
    if chunks and max(chunks) >= 7.2: indicators.append('one or more high-entropy 4-KiB blocks')
    if not sections and segments: indicators.append('ELF segments present but section table absent')
    if len(exec_segments) == 0: indicators.append('no executable segment identified by radare2 metadata')
    return {
        'format': bininfo.get('class'), 'arch': bininfo.get('arch'), 'bits': bininfo.get('bits'),
        'machine': bininfo.get('machine'), 'os': bininfo.get('os'),
        'entrypoint': entry, 'section_count': len(sections), 'segment_count': len(segments),
        'executable_segment_count': len(exec_segments), 'section_names': names,
        'upx_markers': upx_markers, 'upx_section_names': upx_names,
        'entropy_mean_4k': round(sum(chunks)/len(chunks),4) if chunks else 0,
        'entropy_max_4k': round(max(chunks),4) if chunks else 0,
        'packer_triage': 'suspected' if indicators else 'no_obvious_indicator',
        'packer_indicators': indicators,
        'imports_count': len(imports) if isinstance(imports,list) else None,
        'strings_count': len(strings) if isinstance(strings,list) else None,
        'raw_r2_bin_info': bininfo,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', required=True, help='Folder containing raw binaries; scanned recursively')
    ap.add_argument('--source', required=True, choices=['C','F','other'])
    ap.add_argument('--label', required=True, choices=['benign','malware'])
    ap.add_argument('--out', required=True, help='Dedicated output folder for this source/label run')
    ap.add_argument('--extensions', default='', help='Optional comma-separated extension allow-list')
    ap.add_argument('--limit', type=int, default=0, help='Debug limit; 0 means all')
    args = ap.parse_args()
    try: import r2pipe
    except ImportError: raise SystemExit('Missing r2pipe. Install with: pip install r2pipe')
    inroot, outroot = Path(args.input_root).resolve(), Path(args.out).resolve()
    asm_dir = outroot/'assembly'; func_dir = outroot/'functions'; report_dir = outroot/'reports'
    for d in (asm_dir,func_dir,report_dir): d.mkdir(parents=True,exist_ok=True)
    files = sorted(p for p in inroot.rglob('*') if p.is_file() and not p.is_symlink())
    if args.extensions:
        allowed = {'.'+x.lower().lstrip('.') for x in args.extensions.split(',')}
        files = [p for p in files if p.suffix.lower() in allowed]
    if args.limit: files = files[:args.limit]
    manifest=[]
    for idx,path in enumerate(files,1):
        sample_id = sha256_file(path)
        base = f'{args.source}_{args.label}_{sample_id[:16]}'
        asm_path, fn_path = asm_dir/f'{base}.asm', func_dir/f'{base}.json'
        row = {'sample_id':sample_id,'binary_sha256':sample_id,'source':args.source,'label':args.label,
               'binary_path':str(path),'assembly_path':str(asm_path),'functions_path':str(fn_path),
               'extraction_status':'failed','instruction_count':0,'function_count':0,
               'duplicate_address_records':0,'conflicting_address_records':0,'error':''}
        r2=None
        try:
            raw=path.read_bytes()
            # radare2 analysis flags: apply relocations, use quiet JSON-oriented batch operation.
            r2=r2pipe.open(str(path), flags=['-2'])
            r2.cmd('e bin.relocs.apply=true')
            r2.cmd('aaa')
            functions=parse_json_cmd(r2,'aflj') or []
            # If function discovery found nothing, try defining/analyzing only the ELF entry point.
            # Never linear-sweep from a segment base: it may point at ELF headers or embedded data.
            if not functions:
                ep=get_entrypoint(r2)
                if ep is not None:
                    r2.cmd(f'af @ {ep}')
                    functions=parse_json_cmd(r2,'aflj') or []
            records,dups,conflicts=flatten_functions(functions,r2)
            meta=triage(path,raw,r2)
            row.update(meta)
            row['function_count']=len(functions)
            row['instruction_count']=len(records)
            row['duplicate_address_records']=dups
            row['conflicting_address_records']=conflicts
            if records:
                with asm_path.open('w',encoding='utf-8') as f:
                    for addr,text,size,fn_name in records:
                        f.write(f'{text}\n')
                with fn_path.open('w',encoding='utf-8') as f:
                    json.dump([{'address':hex(a),'disasm':t,'size':s,'function':n} for a,t,s,n in records],f,indent=2)
                row['assembly_sha256']=sha256_file(asm_path)
                row['extraction_status']='success'
            else:
                row['extraction_status']='no_instructions'
                row['error']='No function instructions found; no linear sweep fallback was attempted.'
            (report_dir/f'{base}.json').write_text(json.dumps(row,indent=2,default=str),encoding='utf-8')
        except Exception as e:
            row['error']=f'{type(e).__name__}: {e}'
            (report_dir/f'{base}.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finally:
            if r2 is not None:
                try:r2.quit()
                except Exception:pass
        manifest.append(row)
        print(f'[{idx}/{len(files)}] {path.name}: {row["extraction_status"]} ({row["instruction_count"]} insns)')
    if manifest:
        keys=sorted({k for r in manifest for k in r})
        with (outroot/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
            for r in manifest:w.writerow({k:(json.dumps(r[k]) if isinstance(r.get(k),(dict,list)) else r.get(k,'')) for k in keys})
    print(f'Finished: {len(manifest)} files; manifest: {outroot/"manifest.csv"}')

if __name__=='__main__': main()
