#!/usr/bin/env python3
"""Dual-pipeline, feature-space-only robustness GA for IoT malware detectors.

Supports a tabular classifier and the project's Siamese SVM pair representation.
No executable binaries are read, modified, or emitted. Siamese predictions are
1=same class as a fixed malware reference, 0=different class; this is not itself
a malware/benign prediction. Feature-space flips do not prove executable evasion.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
LOG=logging.getLogger('robustness_ga_dual')

@dataclass(frozen=True)
class GAConfig:
    population_size:int=30; generations:int=10; tournament_size:int=3
    crossover_rate:float=.65; mutation_rate:float=.20; elitism:int=2
    max_changed_features:int=8; max_relative_change:float=.10; seed:int=42; patience:int=4
    def validate(self):
        if self.population_size<4: raise ValueError('population-size must be >= 4')
        if self.generations<1: raise ValueError('generations must be >= 1')
        if not 2<=self.tournament_size<=self.population_size: raise ValueError('invalid tournament-size')
        if not 1<=self.elitism<self.population_size: raise ValueError('elitism must be in [1, population-size-1]')
        if not 0<=self.crossover_rate<=1 or not 0<=self.mutation_rate<=1: raise ValueError('rates must be in [0,1]')
        if self.max_changed_features<1 or self.max_relative_change<0: raise ValueError('invalid perturbation budget')

class TabularAdapter:
    def __init__(self,model,malware_label,threshold):
        self.model,self.threshold,self.malware_label=model,float(threshold),malware_label
        if not hasattr(model,'predict_proba'): raise TypeError('Tabular model must expose predict_proba()')
        classes=list(getattr(model,'classes_',[])); hits=[i for i,c in enumerate(classes) if str(c)==str(malware_label)]
        if len(hits)!=1: raise ValueError(f'malware label {malware_label!r} not found uniquely in model.classes_={classes!r}')
        self.col=hits[0]
    def score_prediction(self,x):
        p=np.asarray(self.model.predict_proba(x),float)
        if p.ndim!=2 or p.shape[1]<=self.col: raise ValueError(f'unexpected predict_proba shape {p.shape}')
        score=float(p[0,self.col])
        if not np.isfinite(score) or not 0<=score<=1: raise ValueError('invalid malware probability')
        return score,int(score>=self.threshold)
    def objective(self,score,base_score): return base_score-score
    def eligible(self,pred): return pred==1
    def flipped(self,pred): return pred==0

class SiameseAdapter:
    """Apply saved vectorizer/matrix reference + abs-difference + scaler + SVM."""
    def __init__(self,artifact,vectorizer,matrix,labels,reference_index):
        if not isinstance(artifact,dict) or artifact.get('model') is None or artifact.get('scaler') is None:
            raise ValueError("Siamese artifact must be a dict containing 'model' and 'scaler'")
        self.model,self.scaler,self.vectorizer=artifact['model'],artifact['scaler'],vectorizer
        self.matrix,self.labels=matrix.tocsr(),np.asarray(labels).reshape(-1)
        if self.matrix.shape[0]!=len(self.labels): raise ValueError('TF-IDF matrix rows do not match labels')
        expected=getattr(self.model,'n_features_in_',None); vocab=len(getattr(vectorizer,'vocabulary_',{}))
        if expected is None or int(expected)!=self.matrix.shape[1] or vocab!=self.matrix.shape[1]:
            raise ValueError(f'feature mismatch: matrix={self.matrix.shape[1]}, model={expected}, vectorizer={vocab}')
        if reference_index is None:
            refs=np.flatnonzero(self.labels==1)
            if not len(refs): raise ValueError('no label-1 malware reference in labels')
            reference_index=int(refs[0])
        if not 0<=int(reference_index)<self.matrix.shape[0]: raise ValueError('reference index out of range')
        self.reference_index=int(reference_index); self.reference_label=int(self.labels[self.reference_index])
        if self.reference_label!=1: raise ValueError(f'reference row {self.reference_index} must have label 1, got {self.reference_label}')
        self.reference=self.matrix[self.reference_index].toarray().astype(np.float32)
    def score_prediction(self,x):
        difference=np.abs(np.asarray(x,dtype=np.float32).reshape(1,-1)-self.reference)
        scaled=self.scaler.transform(difference)
        score=float(np.asarray(self.model.decision_function(scaled)).reshape(-1)[0])
        pred=int(np.asarray(self.model.predict(scaled)).reshape(-1)[0])
        if not np.isfinite(score): raise ValueError('Siamese SVM returned non-finite decision score')
        return score,pred
    def objective(self,score,base_score): return base_score-score
    def eligible(self,pred): return pred==1
    def flipped(self,pred): return pred==0

class FeatureSpaceGA:
    def __init__(self,adapter,lower,upper,cfg):
        cfg.validate(); self.adapter,self.lower,self.upper,self.cfg=adapter,np.asarray(lower,float),np.asarray(upper,float),cfg
        if self.lower.shape!=self.upper.shape or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or (self.lower>self.upper).any(): raise ValueError('invalid feature bounds')
        self.rng=np.random.default_rng(cfg.seed)
    def repair(self,x,base):
        x=np.asarray(x,float).copy(); span=np.maximum(np.abs(base),1.0); delta=self.cfg.max_relative_change*span
        lo,hi=np.minimum(self.lower,base),np.maximum(self.upper,base); x=np.clip(x,lo,hi); x=np.clip(x,base-delta,base+delta); x=np.clip(x,lo,hi)
        changed=np.flatnonzero(~np.isclose(x,base,rtol=1e-10,atol=1e-12))
        if len(changed)>self.cfg.max_changed_features:
            strength=np.abs(x[changed]-base[changed])/span[changed]; keep=set(changed[np.argsort(-strength,kind='stable')[:self.cfg.max_changed_features]].tolist())
            for j in changed:
                if int(j) not in keep: x[j]=base[j]
        return x
    def fitness(self,x,base,base_score):
        score,pred=self.adapter.score_prediction(x.reshape(1,-1)); rel=float(np.mean(np.abs(x-base)/np.maximum(np.abs(base),1.0)))
        changed=int(np.count_nonzero(~np.isclose(x,base,rtol=1e-10,atol=1e-12)))
        return self.adapter.objective(score,base_score)-.05*rel-.002*changed,score,pred
    def search(self,base,source_index):
        base=np.asarray(base,float).reshape(-1)
        if len(base)!=len(self.lower) or not np.isfinite(base).all(): raise ValueError(f'invalid feature row at {source_index}')
        bs,bp=self.adapter.score_prediction(base.reshape(1,-1))
        if not self.adapter.eligible(bp):
            return {'source_index':str(source_index),'baseline_score':bs,'candidate_score':bs,'baseline_prediction':bp,'candidate_prediction':bp,'initially_eligible':False,'prediction_flip':False,'score_delta_toward_target':0.,'changed_feature_count':0,'generations_searched':0,'candidate_features':base.tolist()}
        pop=[base.copy()]; ranges=np.maximum(self.upper-self.lower,1e-12)
        for _ in range(self.cfg.population_size-1):
            x=base.copy(); n=int(self.rng.integers(1,min(self.cfg.max_changed_features,len(base))+1)); ix=self.rng.choice(len(base),n,replace=False)
            x[ix]+=self.rng.normal(0,.05,n)*ranges[ix]; pop.append(self.repair(x,base))
        best=base.copy(); best_score=bs; best_pred=bp; best_fit=0.; stagnant=0; done=0
        for gen in range(1,self.cfg.generations+1):
            vals=[self.fitness(x,base,bs) for x in pop]; fits=np.asarray([v[0] for v in vals]); ix=int(np.argmax(fits))
            if fits[ix]>best_fit+1e-12: best,best_score,best_pred,best_fit=pop[ix].copy(),vals[ix][1],vals[ix][2],float(fits[ix]); stagnant=0
            else: stagnant+=1
            done=gen; order=np.argsort(-fits,kind='stable'); nxt=[pop[int(i)].copy() for i in order[:self.cfg.elitism]]
            def tournament():
                ids=self.rng.choice(len(pop),self.cfg.tournament_size,replace=False); return pop[int(max(ids,key=lambda k:(fits[k],-k)))].copy()
            while len(nxt)<self.cfg.population_size:
                a,b=tournament(),tournament(); child=a.copy()
                if self.rng.random()<self.cfg.crossover_rate:
                    mask=self.rng.random(len(base))<.5; child[mask]=b[mask]
                mask=self.rng.random(len(base))<self.cfg.mutation_rate
                if mask.any(): child[mask]+=self.rng.normal(0,.05,len(base))[mask]*ranges[mask]
                nxt.append(self.repair(child,base))
            pop=nxt
            if stagnant>=self.cfg.patience: break
        changed=int(np.count_nonzero(~np.isclose(best,base,rtol=1e-10,atol=1e-12)))
        return {'source_index':str(source_index),'baseline_score':float(bs),'candidate_score':float(best_score),'baseline_prediction':int(bp),'candidate_prediction':int(best_pred),'initially_eligible':True,'prediction_flip':bool(self.adapter.flipped(best_pred)),'score_delta_toward_target':float(bs-best_score),'changed_feature_count':changed,'generations_searched':done,'candidate_features':best.tolist()}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save_results(rows,out,name,report):
    out.mkdir(parents=True,exist_ok=True); jp=out/f'{name}_results.jsonl'; cp=out/f'{name}_summary.csv'
    with jp.open('w',encoding='utf8') as f:
        for r in rows: f.write(json.dumps(r,allow_nan=False,default=str)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='candidate_features'} for r in rows]).to_csv(cp,index=False)
    eligible=[r for r in rows if r.get('initially_eligible') and np.isfinite(r.get('baseline_score',np.nan))]; flips=[r for r in eligible if r.get('prediction_flip')]
    report.update({'rows_total':len(rows),'eligible_rows':len(eligible),'prediction_flips':len(flips),'flip_rate_among_eligible':len(flips)/len(eligible) if eligible else None,'errors':sum('error' in r for r in rows),'results_jsonl':str(jp),'summary_csv':str(cp)})
    (out/f'{name}_report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str),encoding='utf8'); return report

def _as_model_label(value, classes):
    """Resolve a CLI label against model.classes_ without guessing class order."""
    for cls in classes:
        if str(cls) == str(value):
            return cls
    # Allow a numeric CLI value to match numeric classes such as np.int64(1).
    try:
        numeric = float(value)
        for cls in classes:
            try:
                if float(cls) == numeric:
                    return cls
            except (TypeError, ValueError):
                pass
    except (TypeError, ValueError):
        pass
    raise ValueError(f"malware label {value!r} not found in model classes {classes!r}")

def run_tabular(a,cfg,out):
    ip,mp=Path(a.tabular_input),Path(a.tabular_model)
    if not ip.is_file(): raise FileNotFoundError(ip)
    if not mp.is_file(): raise FileNotFoundError(mp)
    raw_df=pd.read_csv(ip)
    if raw_df.empty: raise ValueError('tabular input CSV is empty')

    # This project serializes the LightGBM classifier and its metadata in a dict.
    artifact=joblib.load(mp)
    if isinstance(artifact,dict) and 'model' in artifact:
        model=artifact['model']
        saved_features=artifact.get('feature_names')
        saved_threshold=artifact.get('best_threshold')
    else:
        model=artifact
        saved_features=None
        saved_threshold=None
    if not hasattr(model,'predict_proba'):
        raise TypeError(f"Expected classifier with predict_proba(); got {type(model).__module__}.{type(model).__name__}")

    # Prefer the exact training-time feature schema stored with the model.
    if saved_features is None:
        saved_features=getattr(model,'feature_name_',None)
        if not saved_features:
            saved_features=getattr(model,'feature_names_in_',None)
    cols=list(saved_features) if saved_features is not None and len(saved_features) else None
    if cols is None:
        cols=[c for c in raw_df.columns if c != a.label_column]
    missing=[c for c in cols if c not in raw_df.columns]
    if missing:
        raise ValueError(
            'Input CSV is missing features required by the saved LightGBM model: '
            + ', '.join(missing)
            + '. This model expects: ' + ', '.join(cols)
            + '. Supply the dataset generated by the same feature-extraction pipeline; do not substitute unrelated columns.'
        )
    df=raw_df.loc[:,cols].copy()
    non_numeric=[c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError('Model feature columns must be numeric: '+', '.join(non_numeric))
    if df.isna().any().any():
        missing_values=df.columns[df.isna().any()].tolist()
        raise ValueError('Model feature columns contain missing values: '+', '.join(missing_values)+'; apply the same imputation used during training.')

    classes=list(getattr(model,'classes_',[]))
    malware_class=_as_model_label(a.malware_label,classes)
    threshold=a.tabular_threshold if a.tabular_threshold is not None else saved_threshold
    if threshold is None:
        raise ValueError('No --tabular-threshold supplied and the model artifact has no best_threshold.')
    threshold=float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f'Tabular probability threshold must be between 0 and 1; got {threshold}')
    adapter=TabularAdapter(model,malware_class,threshold)

    ref=pd.read_csv(a.tabular_reference).reindex(columns=cols) if a.tabular_reference else df
    if ref.empty or ref[cols].isna().any().any(): raise ValueError('invalid tabular reference CSV or missing reference features')
    bad_ref=[c for c in cols if not pd.api.types.is_numeric_dtype(ref[c])]
    if bad_ref: raise ValueError('Reference features must be numeric: '+', '.join(bad_ref))
    lo=ref[cols].quantile(a.lower_quantile).to_numpy(float)
    hi=ref[cols].quantile(a.upper_quantile).to_numpy(float)
    ga=FeatureSpaceGA(adapter,lo,hi,cfg); rows=[]
    for i,(_,r) in enumerate(df.iterrows()):
        try: rows.append(ga.search(r.to_numpy(float),i))
        except Exception as e: LOG.exception('tabular row %s failed',i); rows.append({'source_index':str(i),'error':str(e),'initially_eligible':False,'prediction_flip':False,'baseline_score':None,'candidate_score':None})
    rep={'pipeline':'LightGBM/tabular','interpretation':'Malware probability; flip means initially malware-positive to candidate benign-negative. Synthetic feature-space results only.','model_path':str(mp),'model_sha256':sha256(mp),'input_path':str(ip),'input_sha256':sha256(ip),'model_class':type(model).__name__,'artifact_type':type(artifact).__name__,'threshold':threshold,'threshold_source':'CLI' if a.tabular_threshold is not None else ('artifact.best_threshold' if saved_threshold is not None else 'none'),'malware_label':str(malware_class),'feature_columns':cols,'config':asdict(cfg),'seed':cfg.seed}
    return save_results(rows,out,'tabular',rep)

def run_siamese(a,cfg,out):
    mp,vp,xp,lp=map(Path,(a.siamese_model,a.tfidf_vectorizer,a.tfidf_matrix,a.tfidf_labels))
    for p in (mp,vp,xp,lp):
        if not p.is_file(): raise FileNotFoundError(p)
    artifact,vec=joblib.load(mp),joblib.load(vp); matrix=sparse.load_npz(xp).tocsr(); labels=joblib.load(lp)
    if isinstance(labels,dict): labels=labels['labels']
    elif isinstance(labels,tuple): labels=labels[1]
    labels=np.asarray(labels).reshape(-1); adapter=SiameseAdapter(artifact,vec,matrix,labels,a.siamese_reference_index)
    if a.svm_input_csv:
        df=pd.read_csv(a.svm_input_csv)
        if df.shape[1]!=matrix.shape[1] or any(not pd.api.types.is_numeric_dtype(df[c]) for c in df): raise ValueError('SVM input CSV must be numeric and match TF-IDF vocabulary width/order')
        candidates=df.to_numpy(float); ids=list(df.index)
        bdf=pd.read_csv(a.svm_bounds_csv) if a.svm_bounds_csv else (pd.DataFrame(matrix.toarray()) if matrix.shape[0]*matrix.shape[1]<=20_000_000 else None)
        if bdf is None: raise ValueError('large TF-IDF matrix requires --svm-bounds-csv with --svm-input-csv')
        if bdf.shape[1]!=matrix.shape[1]: raise ValueError('SVM bounds CSV vocabulary width mismatch')
        bounds=bdf.to_numpy(float)
    else:
        if matrix.shape[0]*matrix.shape[1]>20_000_000: raise ValueError('TF-IDF matrix too large to densify; pass a smaller --svm-input-csv and --svm-bounds-csv')
        candidates=matrix.toarray().astype(float); ids=list(range(matrix.shape[0])); bounds=candidates
    lo=np.quantile(bounds,a.lower_quantile,axis=0); hi=np.quantile(bounds,a.upper_quantile,axis=0); ga=FeatureSpaceGA(adapter,lo,hi,cfg); rows=[]
    for i,x in enumerate(candidates):
        try: rows.append(ga.search(x,ids[i]))
        except Exception as e: LOG.exception('Siamese row %s failed',ids[i]); rows.append({'source_index':str(ids[i]),'error':str(e),'initially_eligible':False,'prediction_flip':False,'baseline_score':None,'candidate_score':None})
    rep={'pipeline':'Siamese SVM / assembly TF-IDF pair representation','interpretation':'SVM output is 1=same class as the fixed malware reference, 0=different class. This is NOT a malware-vs-benign prediction. A flip is only a pair-label change; synthetic feature-space candidates are not binary variants.','model_path':str(mp),'model_sha256':sha256(mp),'vectorizer_path':str(vp),'vectorizer_sha256':sha256(vp),'matrix_path':str(xp),'matrix_sha256':sha256(xp),'labels_path':str(lp),'labels_sha256':sha256(lp),'reference_index':adapter.reference_index,'reference_label':adapter.reference_label,'candidate_source':str(a.svm_input_csv or xp),'feature_count':int(matrix.shape[1]),'config':asdict(cfg),'seed':cfg.seed}
    fp=xp.parent/'filenames.pkl'
    if fp.is_file():
        try:
            fs=joblib.load(fp)
            if adapter.reference_index<len(fs): rep['reference_filename']=str(fs[adapter.reference_index])
        except Exception: pass
    return save_results(rows,out,'siamese',rep)

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter); p.add_argument('--output-dir',default='robustness_ga_output'); p.add_argument('--pipelines',choices=['tabular','siamese','both'],default='both')
    p.add_argument('--tabular-model'); p.add_argument('--tabular-input'); p.add_argument('--tabular-reference'); p.add_argument('--label-column',default='label'); p.add_argument('--malware-label',default='1'); p.add_argument('--tabular-threshold',type=float,default=None)
    p.add_argument('--siamese-model'); p.add_argument('--tfidf-vectorizer'); p.add_argument('--tfidf-matrix'); p.add_argument('--tfidf-labels'); p.add_argument('--siamese-reference-index',type=int,default=None); p.add_argument('--svm-input-csv'); p.add_argument('--svm-bounds-csv')
    p.add_argument('--lower-quantile',type=float,default=.01); p.add_argument('--upper-quantile',type=float,default=.99); p.add_argument('--population-size',type=int,default=30); p.add_argument('--generations',type=int,default=10); p.add_argument('--tournament-size',type=int,default=3); p.add_argument('--crossover-rate',type=float,default=.65); p.add_argument('--mutation-rate',type=float,default=.20); p.add_argument('--elitism',type=int,default=2); p.add_argument('--max-changed-features',type=int,default=8); p.add_argument('--max-relative-change',type=float,default=.10); p.add_argument('--seed',type=int,default=42); p.add_argument('--patience',type=int,default=4); p.add_argument('--verbose',action='store_true'); return p

def main():
    a=parser().parse_args()
    if not 0<a.lower_quantile<a.upper_quantile<1: raise ValueError('require 0 < lower-quantile < upper-quantile < 1')
    if a.pipelines in ('tabular','both') and not all([a.tabular_model,a.tabular_input]): raise ValueError('tabular/both requires --tabular-model and --tabular-input')
    if a.pipelines in ('siamese','both') and not all([a.siamese_model,a.tfidf_vectorizer,a.tfidf_matrix,a.tfidf_labels]): raise ValueError('siamese/both requires --siamese-model, --tfidf-vectorizer, --tfidf-matrix, --tfidf-labels')
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format='[%(levelname)s] %(message)s')
    cfg=GAConfig(a.population_size,a.generations,a.tournament_size,a.crossover_rate,a.mutation_rate,a.elitism,a.max_changed_features,a.max_relative_change,a.seed,a.patience); cfg.validate(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); reps=[]; start=time.time()
    if a.pipelines in ('tabular','both'): reps.append(run_tabular(a,cfg,out))
    if a.pipelines in ('siamese','both'): reps.append(run_siamese(a,cfg,out))
    combined={'tool':'dual_pipeline_feature_space_robustness_ga','warning':'No executable binaries are modified or emitted. Feature-space flips do not prove executable evasion. Siamese output is pairwise same/different class, not malware/benign.','elapsed_seconds':round(time.time()-start,3),'python_version':sys.version.split()[0],'numpy_version':np.__version__,'pandas_version':pd.__version__,'reports':reps}
    (out/'combined_report.json').write_text(json.dumps(combined,indent=2,allow_nan=False,default=str),encoding='utf8'); print(json.dumps(combined,indent=2,allow_nan=False,default=str)); return 0 if all(r.get('errors',0)==0 for r in reps) else 2
if __name__=='__main__': raise SystemExit(main())
