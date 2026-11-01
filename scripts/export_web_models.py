"""Export real LOSO forests and train-only references for browser inference."""
from pathlib import Path
import json, shutil, argparse
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'wesad_experiment'/'results');args=parser.parse_args()
SOURCE=args.source
OUT=ROOT/'web'/'data';(OUT/'models').mkdir(parents=True,exist_ok=True)
f=pd.read_csv(SOURCE/'features.csv')
cases=json.loads((SOURCE/'explanations.json').read_text())
for subject in sorted({c['subject'] for c in cases}):
    fold=joblib.load(SOURCE/'folds'/f'{subject}.joblib');pipe=fold['model'];cols=fold['features']
    train=pipe[0].transform(f.iloc[fold['train_indices']][cols].to_numpy())
    trees=[]
    for estimator in pipe[-1].estimators_:
        t=estimator.tree_
        v=(t.value[:,0,1]/t.value[:,0,:].sum(axis=1)).tolist()
        def expectation(i):
            if t.children_left[i]==-1:return v[i]
            l=t.children_left[i];r=t.children_right[i]
            v[i]=(expectation(l)*t.weighted_n_node_samples[l]+expectation(r)*t.weighted_n_node_samples[r])/(t.weighted_n_node_samples[l]+t.weighted_n_node_samples[r]);return v[i]
        expectation(0)
        trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'cover':t.weighted_n_node_samples.tolist(),'value':v})
    model={'schema':1,'subject':subject,'training_subjects':14,'features':cols,'trees':trees,'base':sum(t['value'][0] for t in trees)/len(trees),'train':train.tolist(),'bounds':np.stack([train.min(axis=0),train.max(axis=0)],axis=1).tolist(),'quantiles':np.quantile(train,[.25,.5,.75],axis=0).T.tolist(),'imputer':pipe[0].statistics_.tolist()}
    (OUT/'models'/f'{subject}.json').write_text(json.dumps(model,separators=(',',':'),allow_nan=False))
    print(subject,len(trees),(OUT/'models'/f'{subject}.json').stat().st_size)
for name in ['explanations.json','summary.json','verification.json']:
    shutil.copy2(SOURCE/name,OUT/name)
# Preserve Python predictions/SHAP for all original 16 examples as parity fixtures.
fixtures=[{'case_id':c['case_id'],'subject':c['subject'],'x':[c['features'][k] for k in cols],'probability':c['probability'],'shap':[c['shap']['values'][k] for k in cols]} for c in cases]
(OUT/'parity.json').write_text(json.dumps(fixtures,separators=(',',':')))
shutil.copy2(ROOT/'wesad_experiment'/'RESULTS_KO.md',OUT/'RESULTS_KO.md')
