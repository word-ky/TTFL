"""T017R exact tie envelope, then unchanged deterministic finite-K decomposition."""
import argparse,csv,gzip,hashlib,json,sys,time
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.confusion_prevalence import probabilities,project_simplex
from src.context.matched_channel import emission_pools,replica_seed,bootstrap_observation,exact_utilities,mismatch_residual
from src.context.exact_template_lookup import compile_template,exact_lookup
from report_t009 import write_csv
from preflight_t017 import load,gzload,sha,save,gzsave,C,BANKS,SALTS
KS=[20,40,80,160];R=128
def macro(count,total):return sum((Fraction(100*int(x),int(y)) for x,y in zip(count,total)),Fraction())/10
def entropy(p):return -sum(v*np.log(v) for v in p if v>0)


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);out=Path(a.output);start=time.time()
    assert load(out/'noise_free_tie_sanity.json')['passed'];protocol=load(out/'protocol_freeze.json');assert protocol['replicas']==R and protocol['diagnostic_K']==KS
    p13=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';p14=project/'results/t014_class_conditional_factorization'
    p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';p16=project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1))
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    truth={r['client']:r for r in load(p16/'support_truth.json')};cal=np.load(p16/'calibration_counts.npz');logits=np.load(p15/'support_logits.npz')
    actual={(r['client'],r['bank'],r['target']):r for r in gzload(p16/'source_mixtures.json.gz') if r['policy']=='BBSE-S-01'}
    p00={(r['client'],r['bank'],r['target'],r['salt'],r['train_half']):r for r in gzload(p15/'privileged_policy_choices.json.gz') if r['policy']=='P00'}
    zmetrics={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    ametrics={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json') if r['policy']=='BBSE-S-01'}
    pmetrics={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    diagnostics=[];compiled={};args={};truevalues={};canonical=np.zeros((100,2,5,8),dtype=np.uint8)
    for i in range(100):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                for si,s in enumerate(SALTS):
                    for train in (0,1):
                        ti8=2*si+train;key=(i,b,t,s,train);template=templates[s,b,train,i][ti];values,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],template)
                        assert [str(v) for v in values]==[str(Fraction(x)) for x in p00[key]['utility']] and [C[x] for x in arg]==p00[key]['argmax']
                        compiled[key]=compile_template(template);args[key]=arg;truevalues[key]=values;canonical[i,bi,ti,ti8]=arg[0]
                        levels=sorted(set(values),reverse=True);ordered=sorted(values,reverse=True)
                        selected,du=exact_lookup(actual[i,b,t]['pi'],*compiled[key],truth[i]['counts'])
                        diagnostics.append(dict(client=i,bank=b,context=t,salt=s,train_half=train,argmax=arg,argmax_size=len(arg),best_tied=len(arg)>1,
                            legacy_margin=str(ordered[0]-ordered[1]),distinct_level_margin=str(levels[0]-levels[1]) if len(levels)>1 else None,
                            true_utilities=[str(v) for v in values],state_true_regret=[str(max(values)-v) for v in values],actual_state=selected,actual_regret=str(max(values)-values[selected]),actual_canonical_agreement=selected==arg[0],actual_optimal_set_agreement=selected in arg,
                            actual_L1=float(np.abs(np.array(actual[i,b,t]['pi'])-truth[i]['pi']).sum()),actual_DU=du))
    ordered=sorted(diagnostics,key=lambda r:(Fraction(r['legacy_margin']),r['client'],r['bank'],r['context'],r['salt'],r['train_half']))
    for rank,r in enumerate(ordered):r['margin_quartile']=rank//2000+1
    gzsave(out/'template_diagnostics.json.gz',diagnostics)
    envelope=[];flips=[]
    for si,s in enumerate(SALTS):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                lo=hi=canonical_metric=Fraction();ties=0
                for i in range(100):
                    for train in (0,1):
                        arg=args[i,b,t,s,train];vals=[macro(cc[si,bi,i,ti,1-train,state],totals) for state in arg]
                        lo+=min(vals);hi+=max(vals);canonical_metric+=vals[0];ties+=len(arg)>1
                key=(s,b,t);assert canonical_metric==pmetrics[key];zero=zmetrics[key];actual_gain=ametrics[key]-zero
                statuses={name:(actual_gain/(den-zero)>=Fraction(4,5) if den>zero else False) for name,den in [('min',lo),('canonical',canonical_metric),('max',hi)]}
                if len(set(statuses.values()))>1:flips.append(dict(salt=s,bank=b,context=t,statuses=statuses))
                envelope.append(dict(salt=s,bank=b,context=t,canonical=str(canonical_metric),minimum=str(lo),maximum=str(hi),canonical_accuracy=float(canonical_metric),minimum_accuracy=float(lo),maximum_accuracy=float(hi),canonical_minus_min_pp=float(canonical_metric-lo),max_minus_canonical_pp=float(hi-canonical_metric),width_pp=float(hi-lo),tied_episodes=ties,historical_actual80_canonical=statuses['canonical'],historical_actual80_min=statuses['min'],historical_actual80_max=statuses['max']))
    write_csv(out/'p00_tie_envelope.csv',envelope);save(out/'p00_tie_envelope_summary.json',dict(episodes=8000,tied_episodes=sum(r['best_tied'] for r in diagnostics),tied_fraction=sum(r['best_tied'] for r in diagnostics)/8000,max_width_pp=max(r['width_pp'] for r in envelope),historical80_flips=flips,passed=not flips,
        tied_by_client={str(i):sum(r['best_tied'] and r['client']==i for r in diagnostics) for i in range(100)}))
    assert not flips,'Tie envelope changes a historical 80% decision; stop before bootstrap'
    print('T017R_TIE_ENVELOPE_PASS',flush=True)
    shape=(4,100,2,5,R);qraw=np.zeros(shape+(10,));piraw=np.zeros_like(qraw);zraw=np.zeros_like(qraw);choices=np.zeros(shape+(8,),dtype=np.uint8);du=np.zeros(shape+(8,))
    l1=np.zeros(shape);js=np.zeros(shape);top=np.zeros(shape,dtype=np.uint8);negative=np.zeros(shape,dtype=np.uint8);projection=np.zeros(shape)
    poolprob={(i,b,t):probabilities(logits[f'{i}|{b}|{t}|{t}']) for i in range(100) for b in BANKS for t in C};replayed=0
    save(out/'bootstrap_start.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),runtime=a.commit,protocol_sha256=sha(out/'protocol_freeze.json'),tie_envelope_sha256=sha(out/'p00_tie_envelope.csv'),replicas=R,keys=KS,finite_K_no_tie_override=True))
    for i in range(100):
        counts=truth[i]['counts'];true=np.array(truth[i]['pi'])
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                pools,used=emission_pools(i,range(100),lambda j:(poolprob[j,b,t],np.array(truth[j]['labels_in_frozen_support_order'])))
                assert i not in used;matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];inverse=np.linalg.pinv(matrix)
                for ki,K in enumerate(KS):
                    for replica in range(R):
                        seed=replica_seed(i,b,t,K,replica);q,expanded=bootstrap_observation(pools,counts,K,seed);assert np.array_equal(expanded,np.array(counts)*(K//20))
                        z=inverse@q;pi=project_simplex(z);assert np.all(pi>=0) and abs(pi.sum()-1)<1e-10
                        loc=(ki,i,bi,ti,replica);qraw[loc]=q;piraw[loc]=pi;zraw[loc]=z;l1[loc]=np.abs(pi-true).sum();js[loc]=entropy((pi+true)/2)-(entropy(pi)+entropy(true))/2;top[loc]=pi.argmax()==true.argmax();negative[loc]=sum(z<0);projection[loc]=np.abs(pi-z).sum()
                        for si,s in enumerate(SALTS):
                            for train in (0,1):
                                key=(i,b,t,s,train);slot=2*si+train;selected,error=exact_lookup(pi,*compiled[key],counts);choices[loc+(slot,)]=selected;du[loc+(slot,)]=error
                                if replica==0 and K==20:
                                    values,arg=exact_utilities(pi,templates[s,b,train,i][ti]);assert selected==arg[0] and error==float(max(abs(x-y) for x,y in zip(values,truevalues[key])))
                        if replica==0:
                            q2,n2=bootstrap_observation(pools,counts,K,seed);assert np.array_equal(q,q2);replayed+=1
        print('T017R_CLIENT_DONE',i,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(out/'bootstrap_arrays.npz',q=qraw,pi=piraw,z=zraw,choices=choices,DU=du,L1=l1,JS=js,top=top,negative=negative,projection_L1=projection)
    aggregate_counts=np.zeros((4,2,5,4,R,10),dtype=np.int64);query_regret_hits=np.zeros(shape+(8,),dtype=np.int32);query_half_n=np.zeros((100,8),dtype=np.int32)
    for ki,K in enumerate(KS):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                for si,s in enumerate(SALTS):
                    for i in range(100):
                        for train in (0,1):
                            slot=si*2+train;h=1-train;selected=choices[ki,i,bi,ti,:,slot];table=cc[si,bi,i,ti,h];aggregate_counts[ki,bi,ti,si]+=table[selected]
                            query_regret_hits[ki,i,bi,ti,:,slot]=table.sum(-1).max()-table[selected].sum(-1);query_half_n[i,slot]=ct[si,i,h].sum()
    np.savez_compressed(out/'bootstrap_query_counts.npz',aggregate_counts=aggregate_counts,query_regret_hits=query_regret_hits,query_half_n=query_half_n)
    # Store aggregate replicas as compact JSON, with exact fraction strings.
    aggregate=[]
    for ki,K in enumerate(KS):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                for si,s in enumerate(SALTS):
                    zero=zmetrics[s,b,t];p=pmetrics[s,b,t]
                    for replica in range(R):
                        count=aggregate_counts[ki,bi,ti,si,replica];accuracy=macro(count,totals);capture=(accuracy-zero)/(p-zero)
                        aggregate.append(dict(K=K,bank=b,context=t,salt=s,replica=replica,macroclass_exact=str(accuracy),capture_exact=str(capture),macro_class=float(accuracy),capture=float(capture),sample_weighted=100*int(count.sum())/int(totals.sum())))
    gzsave(out/'bootstrap_aggregate_replicas.json.gz',aggregate)
    source_hashes=load(out/'historical_replay.json')['inputs'];assert all(sha(Path(p))==h for p,h in source_hashes.items())
    verification=load(out/'verification.json');verification.update(status='BOOTSTRAP_COMPLETE',bootstrap_replicas_executed=int(np.prod(shape)),state_choices=int(np.prod(choices.shape)),deterministic_replica_replays=replayed,exact_Fraction_lookup_checks=8000,
        aggregate_counts_from_unchanged_T013=True,query_regret_integer_numerators_saved=True,no_model_forwards=True,original_inputs_unchanged=True)
    save(out/'verification.json',verification)
    save(out/'bootstrap_receipt.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),runtime=a.commit,seconds=time.time()-start,numpy_version=np.__version__,shapes=dict(q=list(qraw.shape),choices=list(choices.shape),aggregate_counts=list(aggregate_counts.shape)),
        hashes={name:sha(out/name) for name in ('bootstrap_arrays.npz','bootstrap_query_counts.npz','bootstrap_aggregate_replicas.json.gz','template_diagnostics.json.gz','protocol_freeze.json')},axis_order='K,client,bank,context,replica,(class or salt*2+train_half)',no_bootstrap_tie_intervention=True))
    print('T017R_BOOTSTRAP_COMPLETE',time.time()-start,flush=True)


if __name__=='__main__':main()
