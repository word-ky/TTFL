"""Optional T002 retrospective; correlations only, no retraining or p-values."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def ranks(x):
    order=np.argsort(x,kind='stable')
    result=np.empty(len(x),dtype=float)
    for value in np.unique(x):
        positions=np.flatnonzero(np.asarray(x)[order]==value)
        result[order[positions]]=positions.mean()+1
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--records',required=True)
    p.add_argument('--identities',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    old=json.loads(Path(a.records).read_text())
    ids=json.loads(Path(a.identities).read_text())
    rows=[]
    for cid in range(100):
        count=np.bincount(ids[str(cid)]['support_labels'],minlength=10)
        prob=count/count.sum()
        row=dict(client=cid,support_label_entropy_nats=float(-sum(prob[prob>0]*np.log(prob[prob>0]))),
                 max_class_fraction=float(prob.max()))
        for key,method,context in [('prior_correct_gain','prior','correct'),
                                   ('affine_correct_gain','affine','correct'),
                                   ('affine_shuffled_gain','affine','shuffled'),
                                   ('affine_noise_gain','affine','noise')]:
            row[key]=next(r['gain_pp'] for r in old if r['client']==cid and
                          r['method']==method and r['context']==context)
        rows.append(row)
    corrs=[]
    for xname in ('max_class_fraction','prior_correct_gain'):
        for yname in ('affine_correct_gain','affine_shuffled_gain','affine_noise_gain'):
            x=np.array([r[xname] for r in rows]); y=np.array([r[yname] for r in rows])
            corrs.append(dict(x=xname,y=yname,pearson=float(np.corrcoef(x,y)[0,1]),
                              spearman=float(np.corrcoef(ranks(x),ranks(y))[0,1])))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'prior_correlations.json').write_text(json.dumps(dict(correlations=corrs,
        setting='Retrospective old static-client clean-query PFLlib results; T002 saved labels identify exact old support. Correlation is descriptive, not causal; ties use mean ranks.'),indent=2))
    with (out/'prior_per_client.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(corrs,indent=2))


if __name__=='__main__':
    main()
