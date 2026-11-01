"""WESAD wrist-only, fixed-hyperparameter LOSO experiment. Python 3.11/3.12.
Only unpickle trusted official WESAD files. No fabricated fallback data.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, pickle, time, zipfile
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

SEED=42
SUBJECTS=[2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
META=['subject','start_s','end_s','condition','target']
ROOT=Path(__file__).resolve().parent

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def stats(x,prefix,fs):
    x=np.asarray(x,dtype=float).ravel()
    return {prefix+'_mean':float(x.mean()),prefix+'_std':float(x.std()),
            prefix+'_range':float(np.ptp(x)),
            prefix+'_slope':float(np.polyfit(np.arange(len(x))/fs,x,1)[0])}

def features(wrist,start,end):
    eda=np.asarray(wrist['EDA'][round(start*4):round(end*4)]).ravel()
    temp=np.asarray(wrist['TEMP'][round(start*4):round(end*4)]).ravel()
    bvp=np.asarray(wrist['BVP'][round(start*64):round(end*64)]).ravel()
    acc=np.asarray(wrist['ACC'][round(start*32):round(end*32)],dtype=float)/64.0
    if any(not np.isfinite(a).all() for a in [eda,temp,bvp,acc]):
        raise ValueError('Nonfinite raw signal; do not silently impute raw sensor data')
    ef=sosfiltfilt(butter(2,1,fs=4,output='sos'),eda)
    bf=sosfiltfilt(butter(2,[0.7,3.5],fs=64,btype='bandpass',output='sos'),bvp)
    f={**stats(ef,'eda',4),**stats(temp,'temp',4)}
    f['eda_positive_derivative_mean']=float(np.maximum(np.diff(ef)*4,0).mean())
    mag=np.linalg.norm(acc,axis=1)
    f['acc_magnitude_mean_g']=float(mag.mean())
    f['acc_magnitude_std_g']=float(mag.std())
    f['acc_jerk_mean_g_s']=float(np.linalg.norm(np.diff(acc,axis=0)*32,axis=1).mean())
    f['bvp_filtered_std']=float(bf.std())
    peaks,_=find_peaks(bf,distance=int(64*0.30),prominence=max(float(bf.std())*0.25,1e-8))
    intervals=np.diff(peaks)/64
    valid=(intervals>=0.33)&(intervals<=1.5)
    good=intervals[valid]
    ok=len(good)>=20 and valid.mean()>=0.8
    f['pulse_bpm']=float(60/np.median(good)) if ok else np.nan
    f['pulse_interval_std_ms']=float(good.std()*1000) if ok else np.nan
    adjacent=valid[:-1]&valid[1:]
    differences=np.diff(intervals)[adjacent]
    f['pulse_rmssd_ms']=float(np.sqrt(np.mean(differences**2))*1000) if ok and len(differences)>=10 else np.nan
    f['pulse_quality_ok']=float(ok)
    return f

def extract(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(args.zip) if args.zip else None
    rows=[];audit=[]
    try:
        for sid in SUBJECTS:
            name=f'S{sid}'
            if z:
                matches=[n for n in z.namelist() if n.endswith(f'/{name}/{name}.pkl')]
                if len(matches)!=1: raise ValueError(f'{name}: expected one pickle, got {matches}')
                with z.open(matches[0]) as f: d=pickle.load(f,encoding='latin1')
            else:
                p=Path(args.data)/name/f'{name}.pkl'
                with p.open('rb') as f: d=pickle.load(f,encoding='latin1')
            labels=np.asarray(d['label']).astype(int).ravel()
            edges=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
            n=0
            for a,b in zip(edges[:-1],edges[1:]):
                label=int(labels[a])
                if label not in [1,2,3]: continue
                start=a/700+5; stop=b/700-5
                while start+60<=stop:
                    end=start+60
                    assert np.all(labels[round(start*700):round(end*700)]==label)
                    rows.append(dict(subject=name,start_s=start,end_s=end,condition=label,target=int(label==2),**features(d['signal']['wrist'],start,end)))
                    n+=1;start=end
            audit.append({'subject':name,'windows':n,'raw_duration_s':len(labels)/700})
            print(f'Extracted {name}: {n} windows',flush=True)
            del d,labels
    finally:
        if z:z.close()
    df=pd.DataFrame(rows);df.to_csv(out/'features.csv',index=False)
    dump(out/'extraction_audit.json',{'window_s':60,'stride_s':60,'boundary_trim_s':5,'label_fs':700,'subjects':audit,'feature_names':[c for c in df if c not in META]})

def models():
    return {'dummy':make_pipeline(SimpleImputer(strategy='median'),DummyClassifier(strategy='most_frequent')),
      'logistic':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=SEED)),
      'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,class_weight='balanced',random_state=SEED,n_jobs=2))}

def metrics(y,p):
    pred=(np.asarray(p)>=0.5).astype(int)
    return dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0),stress_recall=recall_score(y,pred,zero_division=0),stress_precision=precision_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p),brier=brier_score_loss(y,p))

def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'folds').mkdir(exist_ok=True)
    df=pd.read_csv(args.features);cols=[c for c in df if c not in META]
    if set(df.subject)!=set(f'S{s}' for s in SUBJECTS): raise ValueError('Full experiment requires all 15 official subjects')
    X=df[cols].to_numpy();y=df.target.to_numpy();groups=df.subject.to_numpy()
    predictions=[];scores=[]
    for subject in sorted(set(groups),key=lambda s:int(s[1:])):
        tr=np.flatnonzero(groups!=subject);te=np.flatnonzero(groups==subject)
        assert not set(groups[tr])&set(groups[te])
        for name,m in models().items():
            t=time.perf_counter();m.fit(X[tr],y[tr]);p=m.predict_proba(X[te])[:,1]
            scores.append(dict(subject=subject,model=name,n_train=len(tr),n_test=len(te),train_seconds=time.perf_counter()-t,**metrics(y[te],p)))
            for i,prob in zip(te,p):predictions.append(dict(row_id=int(i),subject=subject,model=name,target=int(y[i]),probability=float(prob),prediction=int(prob>=0.5)))
            if name=='random_forest':joblib.dump(dict(model=m,train_indices=tr,test_indices=te,features=cols),out/'folds'/f'{subject}.joblib')
        print(f'LOSO completed: {subject}',flush=True)
    scores=pd.DataFrame(scores);pred=pd.DataFrame(predictions)
    scores.to_csv(out/'fold_metrics.csv',index=False);pred.to_csv(out/'oof_predictions.csv',index=False)
    rng=np.random.default_rng(SEED);summary={}
    for name,g in scores.groupby('model'):
        vals=g.macro_f1.to_numpy();boot=np.mean(rng.choice(vals,(10000,len(vals)),replace=True),axis=1)
        q=pred[pred.model==name]
        summary[name]={'subject_mean':{k:float(g[k].mean()) for k in metrics([0,1],[0.2,0.8])},'macro_f1_subject_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'pooled':metrics(q.target,q.probability),'confusion_matrix_true_rows_pred_columns':confusion_matrix(q.target,q.prediction,labels=[0,1]).tolist()}
    dump(out/'summary.json',summary)
    dump(out/'environment.json',{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']})
    print(json.dumps(summary,indent=2),flush=True)

def explain(args):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from alibi.explainers import AnchorTabular
    out=Path(args.out);df=pd.read_csv(args.features);oof=pd.read_csv(out/'oof_predictions.csv')
    q=oof[oof.model=='random_forest'].copy()
    q['bucket']=np.select([(q.target==1)&(q.prediction==1),(q.target==0)&(q.prediction==0),(q.target==0)&(q.prediction==1)],['TP','TN','FP'],default='FN')
    # Pre-specified stratified sample: up to 4/confusion cell, prefer different subjects.
    selected=[]
    for bucket in ['TP','TN','FP','FN']:
        candidates=q[q.bucket==bucket].sample(frac=1,random_state=SEED)
        diverse=candidates.drop_duplicates('subject')
        ids=diverse.head(4).row_id.tolist()
        ids+=(candidates[~candidates.row_id.isin(ids)].head(4-len(ids)).row_id.tolist())
        selected+=ids
    q=q.set_index('row_id');cases=[]
    cached={}
    if args.resume and (out/'explanations.json').exists():
        cached={x['row_id']:x for x in json.loads((out/'explanations.json').read_text())}
    for rid in selected:
        meta=q.loc[rid];fold=joblib.load(out/'folds'/f'{meta.subject}.joblib');pipe=fold['model'];cols=fold['features']
        train_raw=df.iloc[fold['train_indices']][cols].to_numpy()
        train_x=pipe[0].transform(train_raw);x=pipe[0].transform(df.iloc[[rid]][cols].to_numpy())[0];rf=pipe[-1]
        # RF raw tree output is already class probability. Path-dependent Tree SHAP
        # uses training tree path counts, not held-out examples, as its reference.
        t=time.perf_counter();expl=shap.TreeExplainer(rf,feature_perturbation='tree_path_dependent',model_output='raw')
        sv=expl.shap_values(x.reshape(1,-1),check_additivity=True)
        values=sv[1][0] if isinstance(sv,list) else np.asarray(sv)[0,:,1]
        base=float(np.asarray(expl.expected_value)[1]);prob=float(rf.predict_proba(x.reshape(1,-1))[0,1])
        error=abs(base+values.sum()-prob)
        if error>1e-6:raise ValueError(f'SHAP additivity failed {error}')
        st=time.perf_counter()-t
        top=np.argsort(-np.abs(values))[:3]
        shap_result={'method':'tree_path_dependent','model_output':'raw (RF class probability)','base':base,'values':dict(zip(cols,map(float,values))),'top3':[{'feature':cols[i],'value':float(values[i])} for i in top],'additivity_error':float(error),'seconds':st}
        if rid in cached:
            case=cached[rid]
            if abs(case['probability']-prob)>1e-12:raise ValueError('Resume model probability changed')
            case['shap']=shap_result;cases.append(case);dump(out/'explanations.json',cases)
            print(f'Revalidated SHAP {case["case_id"]}; reused unchanged LIME/Anchor results',flush=True)
            continue
        t=time.perf_counter();le=LimeTabularExplainer(train_x,feature_names=cols,class_names=['non-stress','stress'],mode='classification',random_state=SEED,discretize_continuous=True)
        lx=le.explain_instance(x,rf.predict_proba,labels=(1,),num_features=3,num_samples=5000)
        lt=time.perf_counter()-t
        t=time.perf_counter();np.random.seed(SEED)
        ae=AnchorTabular(rf.predict,cols,seed=SEED);ae.fit(train_x,disc_perc=(25,50,75))
        ax=ae.explain(x,threshold=.95,delta=.1,tau=.15,batch_size=100,beam_size=2,max_anchor_size=3,min_samples_start=100,binary_cache_size=1000)
        at=time.perf_counter()-t
        cases.append(dict(case_id=f'C{len(cases)+1:02}',row_id=int(rid),subject=meta.subject,target=int(meta.target),prediction=int(meta.prediction),bucket=meta.bucket,probability=prob,features=dict(zip(cols,map(float,x))),imputed_features=[c for c,v in zip(cols,df.iloc[rid][cols]) if pd.isna(v)],shap=shap_result,lime={'top3':lx.as_list(label=1),'local_r2':float(lx.score),'local_prediction':float(lx.local_pred[0]),'probability_error':float(abs(lx.local_pred[0]-prob)),'seconds':lt},anchor={'rule':ax.data['anchor'],'precision':float(ax.data['precision']),'coverage':float(ax.data['coverage']),'threshold_met':bool(ax.data['precision']>=.95),'seconds':at}))
        dump(out/'explanations.json',cases)
        print(f'Explained {cases[-1]["case_id"]} {meta.subject} {meta.bucket}: anchor={ax.data["precision"]:.3f}',flush=True)
    dump(out/'explanation_environment.json',{p:importlib.metadata.version(p) for p in ['shap','lime','alibi']})

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract');g=e.add_mutually_exclusive_group(required=True);g.add_argument('--zip');g.add_argument('--data');e.add_argument('--out',default=str(ROOT/'results'));e.set_defaults(func=extract)
    for command,fn in [('train',train),('explain',explain)]:
        e=sub.add_parser(command);e.add_argument('--features',default=str(ROOT/'results/features.csv'));e.add_argument('--out',default=str(ROOT/'results'));e.set_defaults(func=fn)
        if command=='explain':e.add_argument('--resume',action='store_true',help='Recompute SHAP and reuse existing LIME/Anchor results for unchanged predictions')
    a=p.parse_args();a.func(a)
if __name__=='__main__':main()
