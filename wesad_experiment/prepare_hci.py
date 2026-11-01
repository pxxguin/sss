"""Create counterbalanced schedules and an empty response sheet from actual OOF cases."""
import csv,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def main():
    cases=json.loads((OUT/'explanations.json').read_text())
    by={b:[x for x in cases if x['bucket']==b] for b in ['TP','TN','FP','FN']}
    if any(len(v)!=4 for v in by.values()):
        raise ValueError('HCI schedule requires exactly 4 actual examples in each confusion cell; redesign, do not fabricate cases.')
    bundles=[[by[b][i]['case_id'] for b in by] for i in range(4)]
    sequences=['ABDC','BCAD','CDBA','DACB']
    rows=[]
    for seq_id,sequence in enumerate(sequences):
        for offset in range(4):
            pid=f'P{seq_id*4+offset+1:02}'
            for block,condition in enumerate(sequence):
                ids=bundles[(block+offset)%4].copy();random.Random(42+int(pid[1:])*10+block).shuffle(ids)
                for trial,case_id in enumerate(ids):
                    rows.append(dict(participant_id=pid,sequence=sequence,block=block+1,condition=condition,trial_in_block=trial+1,case_id=case_id,judgment='',confidence_1_7='',accept_break_alert='',ease_1_7='',response_ms=''))
    path=OUT/'hci_response_template.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for pid in set(r['participant_id'] for r in rows):
        trials=[r for r in rows if r['participant_id']==pid]
        assert len(trials)==16 and len(set(r['case_id'] for r in trials))==16
    for c in cases:
        for condition in 'ABCD':assert sum(r['case_id']==c['case_id'] and r['condition']==condition for r in rows)==4
    # Researcher key: separate from participant-facing stimuli.
    with (OUT/'hci_researcher_key.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','subject','target','prediction','bucket','probability']);w.writeheader()
        w.writerows({k:c[k] for k in w.fieldnames} for c in cases)
    print('Prepared 16 participant schedules / 256 EMPTY response rows; no people tested.')
if __name__=='__main__':main()
