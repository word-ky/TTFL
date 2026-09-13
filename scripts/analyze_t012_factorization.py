"""Phase I: exact T011 argmax sets and fixed client-mismatch controls, no forward passes."""
import json,itertools
from pathlib import Path
import numpy as np
from report_t009 import write_csv
from report_t010 import ranks,corr
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
OUT=ROOT/'results/t012_context_amplitude'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
OFFSETS=[7,13,23,37,41,53,71,89]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((OLD/'candidate_query_utility.json').read_text());p=np.load(OLD/'candidate_predictions.npz')
    skew={r['client']:r for r in json.loads((OLD/'frozen_skew_labels_receipt.json').read_text())}
    qi={(r['client'],r['bank'],r['target'],r['candidate']):r for r in rows}
    for r in rows:
        y=p[f"c{r['client']}_labels"];pred=p[r['prediction_key']]
        assert np.bincount(y[y==pred],minlength=10).tolist()==r['class_correct']
        assert np.bincount(y,minlength=10).tolist()==r['class_total']
    sets={};vectors={}
    for i in range(100):
        for b in ('A','B'):
            for t in C:
                counts=[sum(qi[i,b,t,c]['class_correct']) for c in C];mx=max(counts)
                sets[i,b,t]={c for c,n in zip(C,counts) if n==mx}
                # Same client/target denominator and zero offset preserve ranks exactly.
                vectors[i,b,t]=np.array(counts)-counts[0]
    overlap=[];similarity=[];dominant=[];gates=[];quartiles=[]
    def sim(b,t,u,k):
        vals=[corr(ranks(vectors[i,b,t]),ranks(vectors[(i+k)%100,b,u])) for i in range(100)]
        vals=[v for v in vals if v is not None]
        return dict(valid_count=len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None)
    for b in ('A','B'):
        for t,u in itertools.combinations(C,2):
            same=sum(bool(sets[i,b,t]&sets[i,b,u]) for i in range(100))/100
            controls=[sum(bool(sets[i,b,t]&sets[(i+k)%100,b,u]) for i in range(100))/100 for k in OFFSETS]
            overlap.append(dict(bank=b,target1=t,target2=u,same_client_overlap=same,control_mean=float(np.mean(controls)),control_min=min(controls),control_max=max(controls),difference_pp=100*(same-float(np.mean(controls)))))
            similarity.append(dict(bank=b,target1=t,target2=u,same=sim(b,t,u,0),controls=[dict(offset=k,**sim(b,t,u,k)) for k in OFFSETS],control_overlap_by_offset=dict(zip(OFFSETS,controls))))
        for i in range(100):
            counts={c:sum(c in sets[i,b,t] for t in C) for c in C}
            dominant.append(dict(client=i,bank=b,quartile=skew[i]['entropy_quartile'],max_context_count=max(counts.values()),**counts))
        for q in ('all',1,2,3,4):
            g=[r for r in dominant if r['bank']==b and (q=='all' or r['quartile']==q)]
            quartiles.append(dict(bank=b,quartile=q,n=len(g),distribution={n:sum(r['max_context_count']==n for r in g) for n in range(1,6)},fraction_ge3=sum(r['max_context_count']>=3 for r in g)/len(g),fraction_ge4=sum(r['max_context_count']>=4 for r in g)/len(g),fraction_eq5=sum(r['max_context_count']==5 for r in g)/len(g)))
        oo=[r for r in overlap if r['bank']==b];same=float(np.mean([r['same_client_overlap'] for r in oo]));control=float(np.mean([r['control_mean'] for r in oo]));frac=next(r['fraction_ge3'] for r in quartiles if r['bank']==b and r['quartile']=='all')
        # Integer comparison for the predeclared 15pp overlap and 50% persistence gate.
        same_hits=sum(sum(bool(sets[i,b,t]&sets[i,b,u]) for i in range(100)) for t,u in itertools.combinations(C,2))
        ctrl_hits=sum(sum(bool(sets[i,b,t]&sets[(i+k)%100,b,u]) for i in range(100)) for t,u in itertools.combinations(C,2) for k in OFFSETS)
        gates.append(dict(bank=b,mean_same_overlap=same,mean_control_overlap=control,difference_pp=100*(same-control),fraction_ge3=frac,passed=(8*same_hits-ctrl_hits>=1200 and frac>=.5)))
    result=dict(offsets=OFFSETS,matrix_rows_reconstructed=len(rows),gates=gates,CLIENT_LOCK_STRONG=all(r['passed'] for r in gates),persistence=quartiles,utility_similarity=similarity)
    (OUT/'context_client_factorization.json').write_text(json.dumps(result,indent=2))
    write_csv(OUT/'context_pair_overlap.csv',overlap);write_csv(OUT/'dominant_state_by_client.csv',dominant)
    print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
