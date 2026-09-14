"""Independent exact reconstruction; no main geometry/gate imports."""
import argparse,csv,hashlib,sys,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata
from t023_common import load,save,sha,gzload
SALTS=[f'T013-S{i}' for i in range(4)];LAM=[F(1,4),F(1,2),F(3,4),F(1)]
def rows(p):return list(csv.DictReader(p.open()))
def mean(v):return sum(v,F())/len(v)
def distance(p,q):return sum(abs(p[k]-q[k]) for k in range(10))/2
def dot(p,a):return [sum(p[c]*a[c][s] for c in range(10)) for s in range(5)]

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','analysis','output','commit'):ap.add_argument('--'+k,required=True)
    args=ap.parse_args();p=Path(args.project);analysis=Path(args.analysis);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    inp=load(analysis/'input_hashes.json');assert all(sha(path)==digest for path,digest in inp.items())
    f=load(analysis/'semantic_shift_freeze.json');summary=load(analysis/'summary.json');freezehash=sha(analysis/'semantic_shift_freeze.json')
    assert freezehash==summary['semantic_freeze_sha256']==load(analysis/'preflight.json')['semantic_freeze_sha256'] and f['utility_values_loaded'] is False and f['input_hashes']==inp
    assert f['state_order']==[f's{i}' for i in range(5)] and f['strict_noop_index']==0 and list(map(F,f['lambdas']))==LAM
    assert summary['new_model_forwards']==0 and all(summary[k] is False for k in ('new_training','new_writer','new_federation'))
    p14=p/'results/t014_class_conditional_factorization';p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';hc=np.load(p13/'half_integer_counts.npz');ct=hc['total'];cc=hc['correct']
    source=load(p14/'composition_vectors.json');assert len(source)==800
    pi0={(r['salt'],r['train_half'],r['client']):[F(n,r['denominator']) for n in r['class_counts']] for r in source};assert len(pi0)==800
    for r in source:assert sum(r['class_counts'])==r['denominator']>0 and r['class_counts']==ct[SALTS.index(r['salt']),r['client'],r['train_half']].tolist()
    frozen=f['compositions'];assert len(frozen)==800 and {(r['salt'],r['train_half'],r['client']) for r in frozen}==set(pi0)
    pairs={};counter={};current={};maxjs=0.
    for s in SALTS:
        for h in (0,1):
            order=sorted(range(100),key=lambda i:hashlib.sha256(f'T024|pair|{s}|{h}|{i}'.encode()).hexdigest())
            for j,i in enumerate(order):pairs[s,h,i]=order[(j+1)%100];assert pairs[s,h,i]!=i
            assert len({pairs[s,h,i] for i in range(100)})==100
    for r in frozen:
        key=r['salt'],r['train_half'],r['client'];original=pi0[key];candidates=[original[-k:]+original[:-k] for k in range(1,10)];d=[distance(original,q) for q in candidates];offset=d.index(max(d))+1;q=candidates[offset-1]
        assert list(map(F,r['pi0']))==original and list(map(F,r['q']))==q and sum(original)==sum(q)==1 and r['offset']==offset and r['degenerate']==(max(d)==0) and F(r['counter_TV'])==max(d) and r['paired_client']==pairs[key]
        js=float(jensenshannon(list(map(float,original)),list(map(float,q)),base=2));maxjs=max(maxjs,abs(js-r['counter_JS_distance']));counter[key]=q
        assert len(r['shifts'])==4
        for j,lam in enumerate(LAM):
            cur=[original[c]+lam*(q[c]-original[c]) for c in range(10)];saved=r['shifts'][j]
            assert F(saved['lambda_'])==lam and list(map(F,saved['pi']))==cur and sum(cur)==1 and F(saved['TV'])==distance(original,cur)==lam*max(d)
            js=float(jensenshannon(list(map(float,original)),list(map(float,cur)),base=2));maxjs=max(maxjs,abs(js-saved['JS_distance']));current[key+(lam,)]=cur
    assert maxjs<=1e-12 and len(current)==3200
    raw=gzload(p14/'class_templates.json.gz')['rows'];assert len(raw)==1600;matrices={}
    for r in raw:
        s,b,h,i=r['salt'],r['bank'],r['train_half'],r['target_client'];si=SALTS.index(s);bi=['A','B'].index(b);owners=[j for j in range(100) if j!=i]
        den=[sum(int(ct[si,j,h,c]) for j in owners) for c in range(10)];assert den==r['other99_class_denominators'] and min(den)>0
        a=[[F(sum(int(cc[si,bi,j,0,h,state,c])-int(cc[si,bi,j,0,h,0,c]) for j in owners),den[c]) for state in range(5)] for c in range(10)]
        assert a==[[F(x) for x in row] for row in r['utility'][0]] and all(row[0]==0 for row in a);matrices[s,b,h,i]=a
    assert set(matrices)=={(s,b,h,i) for s in SALTS for b in ('A','B') for h in (0,1) for i in range(100)}
    saved=gzload(analysis/'cell_geometry.json.gz');assert len(saved)==6400 and {(r['salt'],r['bank'],r['train_half'],r['client'],F(r['lambda'])) for r in saved}=={(s,b,h,i,lam) for s in SALTS for b in ('A','B') for h in (0,1) for i in range(100) for lam in LAM}
    reconstructed=[];vectors=0
    for r in saved:
        s,b,h,i=r['salt'],r['bank'],r['train_half'],r['client'];lam=F(r['lambda']);key=s,h,i;a=matrices[s,b,h,i];paired=pairs[key]
        uh=dot(pi0[key],a);uc=dot(current[key+(lam,)],a);um=dot(current[(s,h,paired,lam)],a)
        ah=[k for k,v in enumerate(uh) if v==max(uh)];ac=[k for k,v in enumerate(uc) if v==max(uc)];am=[k for k,v in enumerate(um) if v==max(um)];hs,cs,ms=ah[0],ac[0],am[0]
        for name,vector in [('U_hist',uh),('U_current',uc),('U_mismatch',um)]:assert list(map(F,r[name]))==vector;vectors+=1
        for name,v in [('s_hist',hs),('s_current',cs),('s_mismatch',ms),('s_zero',0),('hist_argmax',ah),('current_argmax',ac),('mismatch_argmax',am),('state_switch',hs!=cs),('argmax_overlap',bool(set(ah)&set(ac)))]:assert r[name]==v
        exact=dict(oracle_current=uc[cs],historical=uc[hs],mismatched=uc[ms],zero=uc[0],stale_regret=uc[cs]-uc[hs],mismatch_regret=uc[cs]-uc[ms],oracle_vs_zero=uc[cs]-uc[0],historical_minus_zero=uc[hs]-uc[0],top_two_margin=sorted(uc,reverse=True)[0]-sorted(uc,reverse=True)[1])
        for name,v in exact.items():assert F(r[name])==v
        assert r['paired_client']==paired and r['degenerate']==(pi0[key]==counter[key]) and F(r['TV'])==distance(pi0[key],current[key+(lam,)])
        reconstructed.append(dict(r,**exact,**{'lambda':lam}))
    gates=load(analysis/'diagnostic_gates.json');head=[];specific=[];trend=[]
    for b in ('A','B'):
        active=[r for r in reconstructed if r['bank']==b and not r['degenerate']];severe=[r for r in active if r['lambda']==1];assert len(severe)==800
        avg=mean([r['stale_regret'] for r in severe]);fraction=F(sum(r['stale_regret']>=F(1,100) for r in severe),800);switch=F(sum(r['state_switch'] for r in severe),800)
        hp=avg>=F(1,100) and fraction>=F(3,10) and switch>=F(3,10);sm={s:mean([r['mismatch_regret'] for r in severe if r['salt']==s]) for s in SALTS};sp={s:x>=F(1,200) for s,x in sm.items()};mf=F(sum(r['mismatch_regret']>=F(1,100) for r in severe),800);ss=sum(sp.values())>=3 and mf>=F(1,4)
        means=[mean([r['stale_regret'] for r in active if r['lambda']==lam]) for lam in LAM];mono=all(means[j+1]-means[j]>=-F(1,10**12) for j in range(3));gap=means[3]-means[0];tp=mono and gap>=F(1,200)
        bg=gates['banks'][b];assert bg['headroom']==hp and bg['specificity']==ss and bg['trend']==tp and bg['non_degenerate_severe_cells']==800 and bg['trend_nondecreasing']==mono
        for k,v in [('stale_mean',avg),('stale_ge_1pp_fraction',fraction),('switch_rate',switch),('mismatch_ge_1pp_fraction',mf),('severe_minus_mild',gap)]:assert F(bg[k])==v
        assert [F(x) for x in bg['trend_means']]==means and {s:F(x) for s,x in bg['mismatch_salt_means'].items()}==sm and bg['mismatch_salt_pass']==sp;head.append(hp);specific.append(ss);trend.append(tp)
    assert gates['SEM_HEADROOM_A']==all(head) and gates['SEM_SPEC_A']==all(specific) and gates['SEM_TREND_A']==all(trend)
    diagnosis='T024-S' if all(head+specific+trend) else 'T024-H' if all(head) else 'T024-N';assert gates['diagnosis']==summary['diagnosis']==diagnosis
    aggregate=rows(analysis/'aggregate_by_lambda.csv');mismatch=rows(analysis/'mismatch_by_salt.csv');trans=load(analysis/'state_transitions.json');assert len(aggregate)==8 and len(mismatch)==32 and len(trans)==8
    for r in aggregate:
        group=[z for z in reconstructed if z['bank']==r['bank'] and z['lambda']==F(r['lambda_']) and not z['degenerate']];assert len(group)==int(r['cells'])==800
        assert float(F(sum(z['state_switch'] for z in group),800))==float(r['switch_rate']) and float(F(sum(z['argmax_overlap'] for z in group),800))==float(r['argmax_overlap_rate'])
        for k in ('stale_regret','mismatch_regret','oracle_vs_zero','historical_minus_zero','top_two_margin'):
            v=[z[k] for z in group];assert str(mean(v))==r[k+'_mean_exact'] and 100*float(mean(v))==float(r[k+'_mean_pp'])
            for name,q in [('median',.5),('p75',.75),('p90',.9)]:assert 100*float(np.quantile([float(x) for x in v],q))==float(r[k+'_'+name+'_pp'])
    for r in mismatch:
        v=[z['mismatch_regret'] for z in reconstructed if z['bank']==r['bank'] and z['salt']==r['salt'] and z['lambda']==F(r['lambda_']) and not z['degenerate']];assert len(v)==int(r['cells'])==200 and F(r['mean_exact'])==mean(v) and float(r['mean_pp'])==100*float(mean(v)) and float(r['fraction_ge_1pp'])==float(F(sum(x>=F(1,100) for x in v),200))
    for r in trans:
        group=[z for z in reconstructed if z['bank']==r['bank'] and z['lambda']==F(r['lambda_']) and not z['degenerate']];matrix=[[sum(z['s_hist']==i and z['s_current']==j for z in group) for j in range(5)] for i in range(5)];assert matrix==r['counts'] and r['cells']==800 and r['state_order']==f['state_order']
    maxrho=0.
    for r in load(analysis/'distance_correlations.json'):
        group=[z for z in reconstructed if z['bank']==r['bank'] and not z['degenerate']];x=rankdata([float(F(z['TV'])) if r['distance']=='TV' else z['JS_distance'] for z in group]);y=rankdata([float(z['stale_regret']) for z in group]);rho=float(np.corrcoef(x,y)[0,1]);delta=abs(rho-r['spearman']);assert delta<=1e-12;maxrho=max(maxrho,delta)
    assert all(sha(path)==digest for path,digest in inp.items()) and sha(analysis/'semantic_shift_freeze.json')==freezehash
    save(out/'independent_verification.json',dict(status='PASS',runtime=args.commit,analysis_runtime=summary['runtime'],analysis=str(analysis),source_compositions=800,current_compositions=3200,pairings=800,target_excluded_clean_templates=1600,utility_vectors=vectors,cell_rows=6400,all_choices_argmax_regrets_bits_exact=True,all_gates_exact=True,aggregate_rows=8,mismatch_rows=32,transition_matrices=8,max_exact_discrepancy=0,max_JS_distance_discrepancy=maxjs,max_Spearman_discrepancy=maxrho,semantic_freeze_sha256=freezehash,all_input_hashes_unchanged=True,new_model_forwards=0,seconds=time.time()-start))
    print('T024_INDEPENDENT_PASS',diagnosis,flush=True)

if __name__=='__main__':main()
