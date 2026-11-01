"""Check data split, output consistency and explanation provenance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

def main():
    out=Path(__file__).parent/'results'
    f=pd.read_csv(out/'features.csv');p=pd.read_csv(out/'oof_predictions.csv');m=pd.read_csv(out/'fold_metrics.csv')
    assert len(m)==45 and f.subject.nunique()==15
    assert set(f.condition)=={1,2,3} and np.all((f.condition==2).astype(int)==f.target)
    assert np.allclose(f.end_s-f.start_s,60)
    for s,g in f.groupby('subject'):
        g=g.sort_values('start_s');assert np.all(g.start_s.to_numpy()[1:]>=g.end_s.to_numpy()[:-1]-1e-8)
        fold=joblib.load(out/'folds'/f'{s}.joblib')
        assert set(f.iloc[fold['train_indices']].subject)==set(f.subject)-{s}
        assert set(f.iloc[fold['test_indices']].subject)=={s}
        assert len(fold['train_indices'])+len(fold['test_indices'])==len(f)
        assert set(fold['features']).isdisjoint(['subject','start_s','end_s','condition','target'])
    for model,g in p.groupby('model'):
        assert len(g)==len(f) and g.row_id.nunique()==len(f)
        for s,h in g.groupby('subject'):
            assert abs(f1_score(h.target,h.prediction,average='macro')-m[(m.model==model)&(m.subject==s)].macro_f1.iloc[0])<1e-12
    cases=json.loads((out/'explanations.json').read_text())
    for c in cases:
        r=p[(p.row_id==c['row_id'])&(p.model=='random_forest')].iloc[0]
        assert c['subject']==r.subject and c['target']==r.target
        assert abs(c['probability']-r.probability)<1e-12
        assert c['shap']['additivity_error']<1e-6
        assert 0<=c['anchor']['precision']<=1 and 0<=c['anchor']['coverage']<=1
    report={'status':'passed','subjects':int(f.subject.nunique()),'windows':len(f),'features':len(f.columns)-5,'fold_model_pairs':len(m),'oof_rows':len(p),'explanation_cases':len(cases),'pulse_quality_failed_windows':int((f.pulse_quality_ok==0).sum()),'checks':['official MD5 separately matched','60-second non-overlapping windows','all 15 LOSO train/test subject disjointness','no metadata in model features','one OOF prediction per window/model','independent Macro-F1 recomputation','explanations match held-out fold probabilities','SHAP probability additivity','Anchor metric bounds']}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
