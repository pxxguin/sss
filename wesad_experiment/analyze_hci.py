"""Analyze genuinely collected participant responses, never the empty template."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare,wilcoxon

def main():
    p=argparse.ArgumentParser();p.add_argument('responses');p.add_argument('--key',default=str(Path(__file__).parent/'results/hci_researcher_key.csv'));p.add_argument('--out',default='hci_analysis');a=p.parse_args()
    d=pd.read_csv(a.responses);key=pd.read_csv(a.key)
    if d.judgment.isna().all():raise ValueError('Empty template: actual participant responses are required.')
    d=d[d.judgment.notna()].copy()
    if not d.judgment.isin(['correct','incorrect','unsure']).all():raise ValueError('judgment must be correct/incorrect/unsure')
    if not d.condition.isin(list('ABCD')).all():raise ValueError('Unknown condition')
    if d.duplicated(['participant_id','case_id']).any():raise ValueError('Duplicate participant/case')
    if not set(d.case_id).issubset(set(key.case_id)):raise ValueError('Unknown case IDs; do not silently drop responses')
    d=d.merge(key[['case_id','target','prediction']],on='case_id',validate='many_to_one')
    d['accepted']=(d.judgment=='correct').astype(float);d['model_wrong']=d.target!=d.prediction
    tables={}
    for name,s in [('wrong_acceptance',d[d.model_wrong]),('wrong_acceptance_excluding_unsure',d[d.model_wrong&(d.judgment!='unsure')]),('correct_acceptance',d[~d.model_wrong])]:
        tables[name]=s.pivot_table(index='participant_id',columns='condition',values='accepted',aggfunc='mean').reindex(columns=list('ABCD'))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report={}
    for name,t in tables.items():
        t.to_csv(out/f'{name}.csv');complete=t.dropna();report[name]={'n_complete_participants':len(complete),'excluded_participant_count':int(d.participant_id.nunique()-len(complete)),'condition_means':{k:float(v) for k,v in complete.mean().items()} if len(complete) else {}}
        if len(complete)<5:
            report[name]['tests']='Not run: fewer than 5 complete participant pairs';continue
        arr=complete.to_numpy();fr=friedmanchisquare(*arr.T) if not np.all(arr==arr[:,[0]]) else None
        report[name]['friedman_p']=float(fr.pvalue) if fr else 1.0
        pairs=[]
        for c in 'BCD':
            diff=complete[c]-complete['A'];pval=1.0 if np.all(diff==0) else float(wilcoxon(diff,zero_method='wilcox',alternative='two-sided').pvalue)
            pairs.append({'comparison':f'{c}-A','mean_paired_difference':float(diff.mean()),'p_raw':pval})
        order=np.argsort([r['p_raw'] for r in pairs]);running=0
        for rank,i in enumerate(order):
            running=max(running,min(1,(3-rank)*pairs[i]['p_raw']));pairs[i]['p_holm']=running
        report[name]['comparisons']=pairs
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
