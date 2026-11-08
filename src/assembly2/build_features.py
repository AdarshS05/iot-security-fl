#!/usr/bin/env python3
"""Merge extraction manifests, normalize, audit collisions, split by binary, and build TF-IDF."""
from __future__ import annotations
import argparse,csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils import normalize_instruction,sha256_file
MODES=['n0_aggressive','n1_conservative','n2_selective']


def read_manifest(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=sorted({k for r in rows for k in r})
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifests',nargs='+',required=True,help='One or more extraction manifest.csv paths')
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-features',type=int,default=30000)
    ap.add_argument('--ngram-max',type=int,default=2)
    ap.add_argument('--allow-single-class',action='store_true',help='Only for exploratory malware-only corpus; no binary classifier will be trained')
    args=ap.parse_args();out=Path(args.out).resolve()
    for d in ['manifests','assembly/normalized','features','splits','reports']: (out/d).mkdir(parents=True,exist_ok=True)
    rows=[]
    for mp in map(Path,args.manifests):
        if not mp.exists():raise SystemExit(f'Manifest not found: {mp}')
        rows.extend(read_manifest(mp))
    # Preserve only successful, non-empty extracted records.
    rows=[r for r in rows if r.get('extraction_status')=='success' and r.get('assembly_path') and Path(r['assembly_path']).is_file()]
    # De-duplicate exact raw binaries only. Never deduplicate distinct binaries on assembly content.
    seen={};unique=[]
    for r in rows:
        key=r.get('binary_sha256') or r.get('sample_id')
        if key in seen:
            if seen[key]['label']!=r['label']:raise SystemExit(f'Conflicting labels for identical binary SHA256: {key}')
            continue
        seen[key]=r;unique.append(r)
    rows=unique
    if not rows:raise SystemExit('No successful assembly files found in the supplied manifests.')
    labels={r['label'] for r in rows}
    if len(labels)<2 and not args.allow_single_class:
        raise SystemExit('Only one class found. Add the existing benign manifest for binary classification, or use --allow-single-class for malware-only exploratory TF-IDF.')
    samples=[]
    for r in rows:
        raw=Path(r['assembly_path']).read_text(encoding='utf-8',errors='replace').splitlines()
        sample={'sample_id':r.get('sample_id'), 'binary_sha256':r.get('binary_sha256'), 'source':r.get('source'), 'label':r.get('label'), 'assembly_path':r['assembly_path'], 'assembly_sha256':r.get('assembly_sha256') or sha256_file(Path(r['assembly_path']))}
        for mode in MODES:
            toks=[normalize_instruction(x,mode) for x in raw]
            toks=[x for x in toks if x]
            dest=out/'assembly'/'normalized'/mode/f'{sample["sample_id"]}.asm'
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(toks)+'\n',encoding='utf-8')
            sample[mode+'_path']=str(dest);sample[mode+'_instruction_count']=len(toks);sample[mode+'_normalized_sha256']=hashlib.sha256('\n'.join(toks).encode()).hexdigest()
        samples.append(sample)
    write_csv(out/'manifests'/'combined_manifest.csv',samples)
    # Duplicate audit per mode, showing both within-source and cross-source collisions.
    dup_rows=[]
    for mode in MODES:
        groups=defaultdict(list)
        for s in samples:groups[s[mode+'_normalized_sha256']].append(s)
        for h,grp in groups.items():
            if len(grp)>1:
                dup_rows.append({'mode':mode,'normalized_sha256':h,'group_size':len(grp),'sources':';'.join(sorted({x['source'] for x in grp})),
                                 'labels':';'.join(sorted({x['label'] for x in grp})), 'sample_ids':';'.join(x['sample_id'] for x in grp)})
    write_csv(out/'reports'/'normalization_duplicate_groups.csv',dup_rows)
    summary=[]
    for mode in MODES:
        hashes=[s[mode+'_normalized_sha256'] for s in samples];counts=Counter(hashes)
        summary.append({'mode':mode,'samples':len(samples),'unique_normalized':len(counts),'duplicate_groups':sum(v>1 for v in counts.values()),
                        'samples_in_duplicate_groups':sum(v for v in counts.values() if v>1),'largest_group':max(counts.values()),
                        'mean_instruction_count':round(float(np.mean([s[mode+'_instruction_count'] for s in samples])),2)})
    write_csv(out/'reports'/'normalization_comparison.csv',summary)
    # One fixed binary-level split shared by all normalization variants.
    y=np.array([s['label'] for s in samples]);groups=np.array([s['binary_sha256'] for s in samples]);idx=np.arange(len(samples))
    split_rows=[]
    if len(set(y))>=2:
        outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed)
        trainval_i,test_i=next(outer.split(idx,y,groups))
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=args.seed)
        tr_rel,val_rel=next(inner.split(trainval_i,y[trainval_i],groups[trainval_i]))
        train_i=trainval_i[tr_rel];val_i=trainval_i[val_rel]
        for i,s in enumerate(samples):
            part='train' if i in set(train_i) else ('validation' if i in set(val_i) else 'test')
            split_rows.append({'sample_id':s['sample_id'],'binary_sha256':s['binary_sha256'],'source':s['source'],'label':s['label'],'split':part})
    else:
        # Malware-only: retain a reproducible unsupervised corpus; no validation metrics.
        for s in samples:split_rows.append({'sample_id':s['sample_id'],'binary_sha256':s['binary_sha256'],'source':s['source'],'label':s['label'],'split':'train'})
    write_csv(out/'splits'/'split_manifest.csv',split_rows)
    split_by_id={r['sample_id']:r['split'] for r in split_rows}
    for mode in MODES:
        texts=[]
        for s in samples:texts.append(Path(s[mode+'_path']).read_text(encoding='utf-8'))
        train_ids=[i for i,s in enumerate(samples) if split_by_id[s['sample_id']]=='train']
        vec=TfidfVectorizer(token_pattern=r'(?u)\b[\w.$:+-]+\b',lowercase=False,ngram_range=(1,args.ngram_max),max_features=args.max_features,sublinear_tf=True,norm='l2',dtype=np.float32)
        Xtr=vec.fit_transform([texts[i] for i in train_ids])
        mode_dir=out/'features'/mode;mode_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(vec,mode_dir/'tfidf_vectorizer.joblib')
        with (mode_dir/'feature_names.txt').open('w',encoding='utf-8') as f:f.write('\n'.join(vec.get_feature_names_out()))
        for part in ['train','validation','test']:
            ids=[i for i,s in enumerate(samples) if split_by_id[s['sample_id']]==part]
            if not ids:continue
            X=vec.transform([texts[i] for i in ids])
            save_npz(mode_dir/f'X_{part}.npz',X)
            with (mode_dir/f'{part}_samples.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','binary_sha256','source','label']);w.writeheader()
                for i in ids:w.writerow({k:samples[i][k] for k in ['sample_id','binary_sha256','source','label']})
        with (mode_dir/'metadata.json').open('w',encoding='utf-8') as f:
            json.dump({'mode':mode,'vectorizer':'TfidfVectorizer','fit_partition':'train','vocabulary_size':len(vec.vocabulary_),
                       'train_samples':len(train_ids),'max_features':args.max_features,'ngram_range':[1,args.ngram_max]},f,indent=2)
    print(f'Prepared {len(samples)} unique binaries in {out}')
    print(f'Classes: {dict(Counter(s["label"] for s in samples))}')
    print(f'Normalization report: {out/"reports"/"normalization_comparison.csv"}')
    print(f'Split manifest: {out/"splits"/"split_manifest.csv"}')

if __name__=='__main__':main()
