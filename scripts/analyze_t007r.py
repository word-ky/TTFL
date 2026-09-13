"""Independent prediction-count audit and cross-pool state comparison."""
import argparse
import csv
import json
import shutil
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();raw,out=Path(a.raw),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((raw/'summary.json').read_text());assert s['metadata']['clients']==100
    rows=json.loads((raw/'raw_records.json').read_text());pred=np.load(raw/'predictions.npz')
    y=np.concatenate([pred[f'c{i}_labels'] for i in range(100)])
    total=[int(np.sum(y==c)) for c in range(10)]
    clean=np.concatenate([pred[f'c{i}_clean'] for i in range(100)])
    def macro(counts):return sum((Fraction(100*n,d) for n,d in zip(counts,total)),Fraction(0))/10
    clean_counts=[int(np.sum((y==c)&(clean==y))) for c in range(10)];clean_macro=macro(clean_counts)
    audits=[];exact={};error=0.
    for r in s['rows']:
        group=sorted([x for x in rows if all(x[k]==r[k] for k in ('pool','target','context'))],key=lambda x:x['client'])
        ps=np.concatenate([pred[x['prediction_key']] for x in group])
        counts=[int(np.sum((y==c)&(ps==y))) for c in range(10)]
        acc=macro(counts);weighted=Fraction(100*sum(counts),len(y))
        error=max(error,abs(float(acc)-r['macro_class']),abs(float(weighted)-r['sample_weighted']))
        assert error<1e-10
        key=(r['pool'],r['target'],r['context']);exact[key]=acc
        audits.append(dict(pool=r['pool'],target=r['target'],context=r['context'],class_counts=total,
            correct_counts=counts,macro_class=float(acc),sample_weighted=float(weighted)))
    for g in s['gates']:
        v=lambda c:exact[g['pool'],g['target'],c]
        h=clean_macro-v('none');tau=max(Fraction(1,2),h/4)
        passed=h>0 and (v('correct_pair')-v('none'))/h>=Fraction(1,2) and v('correct_pair')-v('wrong_alt_source')>=tau and v('correct_pair')-v('noise_source')>=tau
        assert passed==g['passed']
    states=torch.load(raw/'states.pt',map_location='cpu',weights_only=True)
    similarity=[];repro=[]
    for target in s['passes']:
        for ctx in dict.fromkeys(r['context'] for r in s['rows']):
            vectors=[]
            for pool in ('A','B'):
                key=f'{pool}|'+('clean|clean_identity' if ctx=='clean_identity' else f'{target}|{ctx}')
                vectors.append(torch.zeros(192,dtype=torch.float64) if ctx=='none' else torch.cat(states[key]).double())
            x,z=vectors;den=x.norm()*z.norm()
            similarity.append(dict(target=target,context=ctx,cosine=float(x@z/den) if den>0 else None,
                l2_distance=float((x-z).norm()),max_abs_difference=float((x-z).abs().max())))
            arow=next(r for r in s['rows'] if (r['pool'],r['target'],r['context'])==('A',target,ctx))
            brow=next(r for r in s['rows'] if (r['pool'],r['target'],r['context'])==('B',target,ctx))
            for metric in ('macro_class','sample_weighted','macro_client'):
                repro.append(dict(target=target,context=ctx,metric=metric,pool_A=arow[metric],pool_B=brow[metric],abs_difference=abs(arow[metric]-brow[metric])))
    ds=json.loads((raw/'raw_restoration.json').read_text())
    flags=[d for d in ds if d['cap_fraction']>.1]
    verify=json.loads((raw/'verification.json').read_text());verify.update(independent_count_max_error_pp=error,
        independent_fraction_gate_match=True,clean_correct_counts=clean_counts,clean_class_counts=total)
    s.update(state_similarity=similarity,pool_reproducibility=repro,cap_flags=flags)
    for name in ('summary.csv','per_class.csv','per_client.csv','calibration_pools.json','restoration_diagnostics.csv'):
        shutil.copyfile(raw/name,out/name)
    write_csv(out/'state_similarity.csv',similarity);write_csv(out/'pool_reproducibility.csv',repro)
    for name,obj in [('summary.json',s),('verification.json',verify),('integer_prediction_audit.json',audits)]:
        (out/name).write_text(json.dumps(obj,indent=2))
    print(json.dumps(dict(passes=s['passes'],gates=s['gates'],rows=s['rows'],cap_flags=len(flags),verification=verify),indent=2))


if __name__=='__main__':main()
