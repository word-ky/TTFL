"""Pure saved-evidence T024 geometry: freeze compositions before utility access."""
import argparse,csv,sys,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t023_common import load,save,sha,gzload,gzsave
from src.context.semantic_geometry import LAMBDAS,SALTS,tv,js_distance,counter_state,interpolate,pairing,evaluate,gate_summary,serial,mean

def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p14=p/'results/t014_class_conditional_factorization';p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    bankfile=p/'research_log/t023_receipts/20260914-212025-ttfl-t023-extract-gpu1/artifacts/t023_rotation_ssl_alignment/extraction_freeze.json'
    protocol=p/'results/t024_semantic_state_geometry/PROTOCOL_FREEZE.md';inputs={str(protocol):sha(protocol)};manifest_entries=0
    for task in ('t013','t014','t023'):
        manifest=p/f'research_log/{task}_artifact_manifest.json'
        for r in load(manifest):
            path=p/r['path'];assert sha(path)==r['sha256'];inputs[str(path)]=r['sha256'];manifest_entries+=1
        inputs[str(manifest)]=sha(manifest)
    f14=load(p14/'policy_freeze.json');f13=load(p13/'policy_freeze.json');split=load(p13/'split_freeze.json');bank=load(bankfile)
    for name,key in [('composition_vectors.json','composition_sha256'),('class_templates.json.gz','templates_sha256'),('policy_utilities.json.gz','utilities_sha256'),('policy_choices.csv','choices_sha256')]:assert sha(p14/name)==f14[key];inputs[str(p14/name)]=f14[key]
    for name,key in [('half_integer_counts.npz','counts_sha256'),('crossfit_choices.json','choices_sha256'),('half_argmax_sets.json','argmax_sha256')]:assert sha(p13/name)==f13[key]
    assert sha(p13/'query_halves.json')==split['sha256']
    inherited=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
    assert list(bank['state_digests'])==['A','B'] and all(list(v)==inherited for v in bank['state_digests'].values())
    assert bank['state_digests']['A'][inherited[0]]==bank['state_digests']['B'][inherited[0]]
    for path in [p14/'policy_freeze.json',p14/'metadata.json',p14/'preflight.json',p14/'verification.json',bankfile]:inputs[str(path)]=sha(path)
    assert load(p14/'metadata.json')['template_rows']==1600 and load(p14/'verification.json')['other99_template_excludes_target'] is True
    source=load(p14/'composition_vectors.json');assert len(source)==800 and {(r['salt'],r['client'],r['train_half']) for r in source}=={(s,i,h) for s in SALTS for i in range(100) for h in (0,1)}
    half=np.load(p13/'half_integer_counts.npz');ct=half['total'];frozen=[]
    pairs={(s,h):pairing(s,h,list(range(100))) for s in SALTS for h in (0,1)}
    for pair in pairs.values():assert set(pair)==set(pair.values())==set(range(100)) and all(i!=j for i,j in pair.items())
    for r in source:
        s,i,h=r['salt'],r['client'],r['train_half'];n=r['class_counts'];assert n==ct[SALTS.index(s),i,h].tolist() and sum(n)==r['denominator']>0
        pi=[F(x,sum(n)) for x in n];q,offset,deg=counter_state(pi);assert sum(pi)==sum(q)==1
        shifts=[]
        for lam in LAMBDAS:
            cur=interpolate(pi,q,lam);assert sum(cur)==1 and tv(pi,cur)==lam*tv(pi,q)
            shifts.append(dict(lambda_=lam,pi=cur,TV=tv(pi,cur),JS_distance=js_distance(pi,cur)))
        frozen.append(dict(salt=s,client=i,train_half=h,counts=n,pi0=pi,q=q,offset=offset,degenerate=deg,counter_TV=tv(pi,q),counter_JS_distance=js_distance(pi,q),paired_client=pairs[s,h][i],shifts=shifts))
    save(out/'input_hashes.json',inputs);save(out/'semantic_shift_freeze.json',serial(dict(status='FROZEN',lead='d55a0f9',runtime=a.commit,lambdas=LAMBDAS,compositions=frozen,utility_values_loaded=False,state_order=[f's{i}' for i in range(5)],strict_noop_index=0,bank_state_digests={b:list(v.values()) for b,v in bank['state_digests'].items()},input_hashes=inputs)))
    freezehash=sha(out/'semantic_shift_freeze.json')
    print('T024_SEMANTIC_FREEZE',freezehash,flush=True)
    # First access to class utility values occurs after the saved semantic freeze.
    raw=gzload(p14/'class_templates.json.gz')['rows'];assert len(raw)==1600
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):r for r in raw};expected={(s,b,h,i) for s in SALTS for b in ('A','B') for h in (0,1) for i in range(100)};assert set(templates)==expected
    cc=half['correct'];clean={}
    for (s,b,h,i),r in templates.items():
        si=SALTS.index(s);bi=('A','B').index(b);den=ct[si,:,h].sum(0)-ct[si,i,h];hits=cc[si,bi,:,0,h].sum(0)-cc[si,bi,i,0,h]
        assert r['other99_class_denominators']==den.tolist() and np.all(den>0)
        matrix=[[F(x) for x in row] for row in r['utility'][0]];assert len(matrix)==10 and all(len(v)==5 and v[0]==0 for v in matrix)
        reference=[[F(int(hits[k,c])-int(hits[0,c]),int(den[c])) for k in range(5)] for c in range(10)];assert matrix==reference;clean[s,b,h,i]=matrix
    index={(r['salt'],r['train_half'],r['client']):r for r in frozen};records=[]
    for row in frozen:
        s,h,i=row['salt'],row['train_half'],row['client'];other=index[s,h,row['paired_client']]
        for b in ('A','B'):
            for j,lam in enumerate(LAMBDAS):
                cur=row['shifts'][j];wrong=other['shifts'][j]['pi'];v=evaluate(row['pi0'],cur['pi'],wrong,clean[s,b,h,i])
                assert v['stale_regret']>=0 and v['mismatch_regret']>=0 and v['zero']==0
                records.append(dict(salt=s,bank=b,client=i,train_half=h,**{'lambda':lam},paired_client=row['paired_client'],degenerate=row['degenerate'],TV=cur['TV'],JS_distance=cur['JS_distance'],**v))
    assert len(records)==6400 and len({(r['salt'],r['bank'],r['client'],r['train_half'],r['lambda']) for r in records})==6400
    gzsave(out/'cell_geometry.json.gz',serial(records));g=gate_summary(records);save(out/'diagnostic_gates.json',serial(g))
    aggregates=[];transitions=[];mismatch=[];correlations=[]
    for b in ('A','B'):
        active=[r for r in records if r['bank']==b and not r['degenerate']]
        for lam in LAMBDAS:
            group=[r for r in active if r['lambda']==lam];ar=dict(bank=b,lambda_=str(lam),cells=len(group),switch_rate=float(F(sum(r['state_switch'] for r in group),len(group))),argmax_overlap_rate=float(F(sum(r['argmax_overlap'] for r in group),len(group))))
            for key in ('stale_regret','mismatch_regret','oracle_vs_zero','historical_minus_zero','top_two_margin'):
                vals=[r[key] for r in group];ar[key+'_mean_exact']=str(mean(vals));ar[key+'_mean_pp']=100*float(mean(vals))
                for name,q in [('median',.5),('p75',.75),('p90',.9)]:ar[key+'_'+name+'_pp']=100*float(np.quantile([float(x) for x in vals],q))
            aggregates.append(ar);matrix=np.zeros((5,5),dtype=int)
            for r in group:matrix[r['s_hist'],r['s_current']]+=1
            transitions.append(dict(bank=b,lambda_=str(lam),state_order=[f's{i}' for i in range(5)],counts=matrix.tolist(),cells=len(group)))
            for s in SALTS:
                vals=[r['mismatch_regret'] for r in group if r['salt']==s];mismatch.append(dict(bank=b,salt=s,lambda_=str(lam),cells=len(vals),mean_exact=str(mean(vals)),mean_pp=100*float(mean(vals)),fraction_ge_1pp=float(F(sum(x>=F(1,100) for x in vals),len(vals)))))
        for name in ('TV','JS_distance'):
            x=[float(r[name]) for r in active];y=[float(r['stale_regret']) for r in active];rho=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
            correlations.append(dict(bank=b,distance=name,spearman=rho,cells=len(active),unit='source cell x lambda, repeated compositions'))
    writecsv(out/'aggregate_by_lambda.csv',aggregates);writecsv(out/'mismatch_by_salt.csv',mismatch);save(out/'state_transitions.json',transitions);save(out/'distance_correlations.json',correlations)
    save(out/'preflight.json',dict(status='PASS',manifest_entries=manifest_entries,source_compositions=800,template_cells=1600,clean_templates_reconstructed=1600,degenerate_compositions=sum(r['degenerate'] for r in frozen),source_counts_exact=True,other99_exclusion_verified=True,zero_state_index=0,state_order=[f's{i}' for i in range(5)],semantic_freeze_sha256=freezehash,utility_loaded_after_freeze=True))
    assert sha(out/'semantic_shift_freeze.json')==freezehash and all(sha(path)==digest for path,digest in inputs.items())
    save(out/'summary.json',serial(dict(status='COMPLETE',lead='d55a0f9',runtime=a.commit,diagnosis=g['diagnosis'],gates=g,rows=len(records),source_compositions=800,templates=1600,semantic_freeze_sha256=freezehash,visual_context='clean',utility_is_accuracy_gain_relative_to_noop=True,target_composition_privileged=True,new_model_forwards=0,new_training=False,new_writer=False,new_federation=False,seconds=time.time()-start)))
    print('T024_GEOMETRY_COMPLETE',g['diagnosis'],flush=True)

if __name__=='__main__':main()
