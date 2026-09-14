"""Unchanged T018 science after T018R sealed solver preflight passes."""
import argparse,gzip,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,gzload,sha,save,C,B,S
from report_t009 import write_csv
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.matched_channel import exact_utilities
from src.context.exact_template_lookup import compile_template,exact_lookup
from eval_t017_bootstrap import macro,entropy

def gzsave(p,x):p.write_bytes(gzip.compress(json.dumps(x,allow_nan=False).encode()))
def js(p,t):return entropy((p+t)/2)-(entropy(p)+entropy(t))/2
def quant(v):return dict(zip(('p05','median','p95'),[float(x) for x in np.quantile(v,[.05,.5,.95])]))


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit','preflight'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);pre=Path(a.preflight);start=time.time()
    assert load(pre/'solver_verification.json')['status']=='PASS'
    inputs=load(pre/'input_hashes.json');assert all(sha(Path(p))==h for p,h in inputs.items())
    p13=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    p14=project/'results/t014_class_conditional_factorization'
    p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p17=project/'runs/20260914-083609-ttfl-t017r-cached/artifacts/t017_channel_noise_decomposition'
    save(out/'protocol_freeze.json',dict(lead='c8aea5b',scientific_lead='1744440',runtime=a.commit,preflight=str(pre),preflight_sha256=sha(pre/'solver_verification.json'),
        estimator='same CLS-S objective; direct RHS face application only',K=20,replicas=128,resampling=False,new_model_forwards=0,
        source_q_sha256=sha(p17/'bootstrap_arrays.npz'),gate='80% capture both banks/all salts >=3/4 contexts',real_safety='no shifted regression >0.5pp',clean_safety='>=-0.5pp',
        selection='raw CLS prevalence, exact rational first argmax; no tie override'))
    cal=np.load(p16/'calibration_counts.npz');truth={r['client']:r for r in load(p16/'support_truth.json')}
    saved=np.load(p17/'bootstrap_arrays.npz');qraw=saved['q'][0];oldpi=saved['pi'][0];oldchoices=saved['choices'][0];olddu=saved['DU'][0]
    shape=(100,2,5,128);piraw=np.zeros(shape+(10,));updates=np.zeros(shape,dtype=np.uint8);kkt=np.zeros(shape);improvement=np.zeros(shape);sum_error=np.zeros(shape)
    # All matched observations are certified before any scientific metric is computed.
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];solver=ActiveSetCLS(matrix)
                for r in range(128):
                    loc=(i,bi,ti,r)
                    try:
                        pi,receipt=solver.solve(qraw[loc])
                        assert 2*receipt['objective']<=2*receipt['initial_objective']+1e-12
                    except Exception as error:
                        save(out/'blocker.json',dict(stage='matched_solver',cell=list(loc),error=repr(error),q=qraw[loc].tolist(),query_metrics_opened=False))
                        raise
                    piraw[loc]=pi;updates[loc]=receipt['updates'];kkt[loc]=receipt['direct_KKT'];improvement[loc]=receipt['initial_objective']-receipt['objective'];sum_error[loc]=receipt['sum_error']
        print('T018R_MATCHED_SOLVER_CLIENT',i,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(out/'matched_solver.npz',pi=piraw,updates=updates,KKT=kkt,sum_error=sum_error,objective_improvement=improvement)
    save(out/'matched_solver_verification.json',dict(cases=int(np.prod(shape)),max_KKT=float(kkt.max()),max_updates=int(updates.max()),
        min_objective_improvement=float(improvement.min()),source_q_sha256=sha(p17/'bootstrap_arrays.npz'),new_model_forwards=0,
        max_sum_error=float(sum_error.max()),diagnostics={name:dict(zip(('p95','p99','max'),[float(x) for x in np.quantile(arr,[.95,.99,1.])])) for name,arr in [('KKT',kkt),('sum_error',sum_error),('updates',updates),('objective_improvement',improvement)]},
        previous_blocker=dict(client=35,bank='B',context='brightness_dark',replica=7,pi=piraw[35,1,1,7].tolist(),KKT=float(kkt[35,1,1,7]),sum_error=float(sum_error[35,1,1,7]),updates=int(updates[35,1,1,7]))))
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1))
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    oldmetrics={(r['salt'],r['bank'],r['target'],r['policy']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json')}
    comp={};truevalues={};argsets={};selected=np.zeros(shape+(8,),dtype=np.uint8);du=np.zeros(shape+(8,));regret=np.zeros_like(du);oldregret=np.zeros_like(du)
    canonical=np.zeros(shape+(8,),dtype=np.uint8);optimal=np.zeros_like(canonical);oldcanonical=np.zeros_like(canonical);oldoptimal=np.zeros_like(canonical)
    l1=np.zeros(shape);jss=np.zeros(shape);top=np.zeros(shape,dtype=np.uint8);oldl1=np.zeros(shape);oldjs=np.zeros(shape);oldtop=np.zeros(shape,dtype=np.uint8)
    checks=0
    for i in range(100):
        tr=np.array(truth[i]['pi']);counts=truth[i]['counts']
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for h in (0,1):
                        key=(i,bi,ti,si,h);table=templates[s,b,h,i][ti];comp[key]=compile_template(table)
                        values,args=exact_utilities([Fraction(n,20) for n in counts],table);truevalues[key]=values;argsets[key]=args
                        for r in range(128):
                            loc=(i,bi,ti,r);slot=2*si+h;p=piraw[loc];state,error=exact_lookup(p,*comp[key],counts);os=int(oldchoices[loc+(slot,)])
                            selected[loc+(slot,)]=state;du[loc+(slot,)]=error;regret[loc+(slot,)]=float(max(values)-values[state]);oldregret[loc+(slot,)]=float(max(values)-values[os])
                            canonical[loc+(slot,)]=state==args[0];optimal[loc+(slot,)]=state in args
                            oldcanonical[loc+(slot,)]=os==args[0];oldoptimal[loc+(slot,)]=os in args
                            if r in (0,127):
                                vv,aa=exact_utilities(p,table);assert state==aa[0] and error==float(max(abs(x-y) for x,y in zip(vv,values)));checks+=1
                for r in range(128):
                    loc=(i,bi,ti,r);p=piraw[loc];op=oldpi[loc]
                    l1[loc]=np.abs(p-tr).sum();jss[loc]=js(p,tr);top[loc]=p.argmax()==tr.argmax()
                    oldl1[loc]=np.abs(op-tr).sum();oldjs[loc]=js(op,tr);oldtop[loc]=op.argmax()==tr.argmax()
        print('T018R_TEMPLATE_CLIENT',i,flush=True)
    ac=np.zeros((2,5,4,128,10),dtype=np.int64);oldac=np.load(p17/'bootstrap_query_counts.npz')['aggregate_counts'][0]
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            for si,s in enumerate(S):
                replay=np.zeros((128,10),dtype=np.int64)
                for i in range(100):
                    for h in (0,1):
                        ac[bi,ti,si]+=cc[si,bi,i,ti,1-h][selected[i,bi,ti,:,2*si+h]]
                        replay+=cc[si,bi,i,ti,1-h][oldchoices[i,bi,ti,:,2*si+h]]
                np.testing.assert_array_equal(replay,oldac[bi,ti,si])
    np.savez_compressed(out/'matched_choices.npz',choices=selected,DU=du,regret=regret,canonical=canonical,optimal=optimal,L1=l1,JS=jss,top=top,aggregate_counts=ac)
    replicas=[];matched=[];deltas=[]
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            for si,s in enumerate(S):
                key=(s,b,t);z=zero[key];den=p00[key]-z;group=[]
                for r in range(128):
                    acc=macro(ac[bi,ti,si,r],totals);oa=macro(oldac[bi,ti,si,r],totals)
                    row=dict(bank=b,context=t,salt=s,replica=r,macro_class=float(acc),macro_class_exact=str(acc),capture=float((acc-z)/den),capture_exact=str((acc-z)/den),
                        old_macro_class=float(oa),old_capture=float((oa-z)/den))
                    for name,new,old in [('L1',l1,oldl1),('JS',jss,oldjs),('dominant_agreement',top,oldtop)]:
                        row[name]=float(new[:,bi,ti,r].mean());row['old_'+name]=float(old[:,bi,ti,r].mean())
                    for name,new,old in [('DU',du,olddu),('true_template_regret',regret,oldregret),('canonical_agreement',canonical,oldcanonical),('optimal_set_agreement',optimal,oldoptimal)]:
                        row[name]=float(new[:,bi,ti,r,2*si:2*si+2].mean());row['old_'+name]=float(old[:,bi,ti,r,2*si:2*si+2].mean())
                    replicas.append(row);group.append(row)
                base=dict(bank=b,context=t,salt=s);m=base.copy();d=base.copy()
                for name in ('macro_class','capture','L1','JS','dominant_agreement','DU','true_template_regret','canonical_agreement','optimal_set_agreement'):
                    for label,val in quant([r[name] for r in group]).items():m[name+'_'+label]=val
                    for label,val in quant([r[name]-r['old_'+name] for r in group]).items():d[name+'_delta_'+label]=val
                matched.append(m);deltas.append(d)
    write_csv(out/'matched_k20_aggregate.csv',matched);write_csv(out/'matched_k20_delta_vs_t017.csv',deltas);gzsave(out/'matched_replicas.json.gz',replicas)
    save(out/'matched_freeze.json',dict(choices_sha256=sha(out/'matched_choices.npz'),aggregate_sha256=sha(out/'matched_k20_aggregate.csv'),lookup_Fraction_checks=checks,old_integer_counts_replayed=True))
    # Oracle results are persisted and frozen before the source-context path opens.
    observations={(r['client'],r['bank'],r['target'],r['policy']):r for r in gzload(p16/'source_mixtures.json.gz') if r['family']=='S'}
    numerical={(r['client'],r['bank'],r['context']):r for r in load(pre/'all_actual_solutions.json')}
    allactual=[];receipts=[];actual_tables={};subset=[]
    for mode in ('01','11'):
        episodes=[];aggregate=[]
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    hits=np.zeros(10,dtype=np.int64);group=[]
                    split={v:[np.zeros(10,dtype=np.int64),np.zeros(10,dtype=np.int64)] for v in ('context_correct','context_error')}
                    for i in range(100):
                        o=observations[i,b,t,'BBSE-S-'+mode];ci=C.index(o['matrix_context']);estimated=ci
                        if mode=='01':p=np.array(numerical[i,b,t]['pi']);rr=numerical[i,b,t]
                        else:
                            matrix=cal['soft_numerator'][i,bi,ci]/cal['soft_denominator'][i,bi,ci][None,:]
                            p,rr=ActiveSetCLS(matrix).solve(o['raw_soft']);assert 2*rr['objective']<=2*rr['initial_objective']+1e-12
                        for h in (0,1):
                            key=(i,bi,estimated,si,h);state,error=exact_lookup(p,*comp[key],truth[i]['counts'])
                            values=truevalues[i,bi,ti,si,h];args=argsets[i,bi,ti,si,h];counts=cc[si,bi,i,ti,1-h,state];hits+=counts
                            typ='context_correct' if estimated==ti else 'context_error';split[typ][0]+=counts;split[typ][1]+=ct[si,i,1-h]
                            row=dict(mode=mode,client=i,bank=b,context=t,salt=s,train_half=h,estimated_context=C[estimated],state=state,pi=p.tolist(),
                                L1=float(np.abs(p-truth[i]['pi']).sum()),JS=js(p,np.array(truth[i]['pi'])),dominant_agreement=bool(p.argmax()==np.argmax(truth[i]['pi'])),
                                canonical_agreement=state==args[0],optimal_set_agreement=state in args,true_template_regret=float(max(values)-values[state]),DU=error,
                                direct_KKT=rr['direct_KKT'],active_classes=rr['active_classes'],objective=rr['objective'],initial_objective=rr['initial_objective'])
                            episodes.append(row);group.append(row)
                    key=(s,b,t);acc=macro(hits,totals);old=oldmetrics[s,b,t,'BBSE-S-'+mode];cap=(acc-zero[key])/(p00[key]-zero[key])
                    row=dict(bank=b,context=t,salt=s,macro_class=float(acc),macro_class_exact=str(acc),capture=float(cap),capture_exact=str(cap),
                        delta_vs_BBSE_pp=float(acc-old),capture_delta_vs_BBSE=float((acc-old)/(p00[key]-zero[key])),clean_delta_vs_zero=float(acc-zero[key]))
                    for name in ('L1','JS','dominant_agreement','canonical_agreement','optimal_set_agreement','true_template_regret','DU','active_classes','objective'):
                        row[name]=float(np.mean([r[name] for r in group]))
                    aggregate.append(row);receipts.append(dict(mode=mode,bank=b,context=t,salt=s,class_correct=hits.tolist(),class_total=totals.tolist()))
                    if mode=='11':
                        for typ,(count,total) in split.items():
                            subset.append(dict(bank=b,context=t,salt=s,subset=typ,n=int(total.sum()),correct=int(count.sum()),sample_weighted=100*int(count.sum())/int(total.sum()) if total.sum() else None))
        name='actual_oracle_context' if mode=='01' else 'full_source'
        write_csv(out/(name+'.csv'),aggregate);gzsave(out/(name+'_episodes.json.gz'),episodes)
        save(out/(name+'_freeze.json'),dict(table_sha256=sha(out/(name+'.csv')),episodes_sha256=sha(out/(name+'_episodes.json.gz')),policy='CLS-S',mode=mode))
        allactual+=episodes;actual_tables[mode]=aggregate
    write_csv(out/'source_context_subsets.csv',subset);save(out/'integer_count_receipts.json',receipts)
    matched_pass={t:all(r['capture_median']>=.8 for r in matched if r['context']==t) for t in C[1:]}
    real_pass={t:all(r['capture']>=.8 for r in actual_tables['01'] if r['context']==t) for t in C[1:]}
    source_pass={t:all(r['capture']>=.8 for r in actual_tables['11'] if r['context']==t) for t in C[1:]}
    real_safe=all(r['delta_vs_BBSE_pp']>=-.5 for r in actual_tables['01'] if r['context']!='clean')
    clean_safe=all(r['clean_delta_vs_zero']>=-.5 for r in actual_tables['11'] if r['context']=='clean')
    ma=sum(matched_pass.values())>=3;ra=sum(real_pass.values())>=3 and real_safe;sa=sum(source_pass.values())>=3 and clean_safe
    blur=[]
    for m in matched:
        if m['context']!='gaussian_blur':continue
        real=next(r for r in actual_tables['01'] if r['context']==m['context'] and r['bank']==m['bank'] and r['salt']==m['salt'])
        src=next(r for r in actual_tables['11'] if r['context']==m['context'] and r['bank']==m['bank'] and r['salt']==m['salt'])
        blur.append(dict(bank=m['bank'],salt=m['salt'],matched_p05=m['capture_p05'],matched_median=m['capture_median'],matched_p95=m['capture_p95'],
            actual_oracle_capture=real['capture'],source_capture=src['capture'],actual_below_matched_p05=real['capture']<m['capture_p05'],source_minus_oracle_pp=src['macro_class']-real['macro_class']))
    write_csv(out/'blur_decomposition.csv',blur)
    assert all(sha(Path(p))==h for p,h in inputs.items())
    save(out/'input_hashes.json',inputs)
    save(out/'summary.json',dict(status='COMPLETE',runtime=a.commit,seconds=time.time()-start,CLS_MATCH_A=ma,CLS_REAL_A=ra,CLS_SRC_A=sa,
        outcome='A' if ma and ra else 'B' if ma else 'C',matched_pass=matched_pass,real_pass=real_pass,source_pass=source_pass,real_regression_safety=real_safe,source_clean_safety=clean_safe,
        matched_cases=128000,matched_choices=1024000,actual_choices=len(allactual),new_model_forwards=0,resampling=False,source_hashes_unchanged=True,
        source_q_sha256=sha(p17/'bootstrap_arrays.npz'),max_matched_KKT=float(kkt.max()),max_matched_updates=int(updates.max()),exact_lookup_checks=checks))
    print('T018R_SCIENCE_COMPLETE',json.dumps(load(out/'summary.json')),flush=True)


if __name__=='__main__':main()
