"""Freeze T019 null percentiles and utility attribution, then open query counts."""
import argparse,csv,json,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,sha,save,C,B,S
from report_t009 import write_csv
from src.context.observation_audit import null_comparison
from eval_t017_bootstrap import macro


def add_comparison(row,name,actual,null):
    row.update({name+'_'+key:value for key,value in null_comparison(actual,null).items()})


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);out=Path(a.output);start=time.time()
    freeze=load(out/'phaseB_choices_freeze.json');assert all(sha(out/n)==h for n,h in freeze['hashes'].items())
    real=dict(np.load(out/'real_decomposition.npz'));null=dict(np.load(out/'null_observations.npz'));u=dict(np.load(out/'utility_choices.npz'))
    regrets={tuple(map(int,k.split('|'))):[Fraction(x) for x in v] for k,v in load(out/'true_template_regrets.json').items()}
    counts=real['counts'];actual=u['actual_scores'];ns=u['null_scores'];rep=u['repair_scores']
    posterior=[];task=[];paired=[];classes=[];attribution=[];local=[];pairclass=[];marginrows=[]
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=(i,bi,ti);pr=dict(client=i,bank=b,context=t)
                for j,name in enumerate(('residual_L1','residual_L2','weighted_class_L1')):add_comparison(pr,name,real['norm'][loc][j],null['norm'][loc][:,j])
                for j,name in enumerate(('prevalence_L1','prevalence_JS','dominant_agreement')):add_comparison(pr,name,u['actual_semantic'][loc][j],u['null_semantic'][loc][:,j])
                posterior.append(pr)
                for y in np.flatnonzero(counts[i]):
                    cr=dict(client=i,bank=b,context=t,label=int(y),count=int(counts[i,y]));add_comparison(cr,'class_d_L1',np.abs(real['d'][loc][y]).sum(),null['class_norm_L1'][loc][:,y]);classes.append(cr)
                    if ti:
                        delta=real['d'][loc][y]-real['d'][i,bi,0,y]
                        pairclass.append(dict(client=i,bank=b,context=t,label=int(y),count=int(counts[i,y]),weight=float(counts[i,y]/20),delta_d=json.dumps(delta.tolist()),delta_L1=float(np.abs(delta).sum()),delta_L2=float(np.linalg.norm(delta))))
                for si,s in enumerate(S):
                    sl=slice(2*si,2*si+2);am=actual[loc][sl].mean(0);nm=ns[loc][:,sl].mean(1)
                    row=dict(client=i,bank=b,context=t,salt=s,actual_optimal_agreement=float(am[2]),null_optimal_agreement=float(nm[:,2].mean()),
                        actual_canonical_agreement=float(am[3]),null_canonical_agreement=float(nm[:,3].mean()),mean_utility_margin=float(u['margin'][loc][sl].mean()))
                    add_comparison(row,'DU',am[0],nm[:,0]);add_comparison(row,'regret',am[1],nm[:,1]);row['task_excess']=row['DU_above_p95'] or row['regret_above_p95'];task.append(row)
                    if ti:
                        actualdiff=real['residual'][loc]-real['residual'][i,bi,0]
                        pp=dict(client=i,bank=b,context=t,salt=s,coverage=1.,observed_classes=int(np.sum(counts[i]>0)),weight_sum=1.,
                            clean_residual_L1=float(real['norm'][i,bi,0,0]),shift_residual_L1=float(real['norm'][loc][0]))
                        add_comparison(pp,'paired_residual_L1',np.abs(actualdiff).sum(),null['paired_norm'][loc][:,0])
                        add_comparison(pp,'paired_residual_L2',np.linalg.norm(actualdiff),null['paired_norm'][loc][:,1])
                        actual_delta=am[:2]-actual[i,bi,0,sl,:2].mean(0);null_delta=null['paired_task_change'][loc][:,sl].mean(1)
                        add_comparison(pp,'delta_DU',actual_delta[0],null_delta[:,0]);add_comparison(pp,'delta_regret',actual_delta[1],null_delta[:,1])
                        pp['context_excess']=pp['paired_residual_L1_above_p95'] or pp['delta_DU_above_p95'];paired.append(pp)
                    real_regret=sum((regrets[i,bi,ti,slot][int(u['actual_choices'][loc][slot])] for slot in range(2*si,2*si+2)),Fraction())/2
                    candidates=[]
                    for y in np.flatnonzero(counts[i]):
                        repair_regret=sum((regrets[i,bi,ti,slot][int(u['repair_choices'][loc][y,slot])] for slot in range(2*si,2*si+2)),Fraction())/2
                        reduction=real_regret-repair_regret;fraction=reduction/real_regret if real_regret>0 else None;rm=rep[loc][y,sl].mean(0)
                        ar=dict(client=i,bank=b,context=t,salt=s,label=int(y),count=int(counts[i,y]),real_regret_exact=str(real_regret),repair_regret_exact=str(repair_regret),
                            regret_reduction_exact=str(reduction),recovery_fraction_exact=str(fraction) if fraction is not None else None,recovery_fraction=float(fraction) if fraction is not None else None,
                            zero_gap=real_regret==0,prevalence_L1_change=float(u['repair_semantic'][loc][y,0]-u['actual_semantic'][loc][0]),DU_change=float(rm[0]-am[0]),
                            actual_optimal_agreement=float(am[2]),repair_optimal_agreement=float(rm[2]),enters_optimal_set=bool(rm[2]>am[2]),repaired_states=json.dumps(u['repair_choices'][loc][y,sl].tolist()))
                        candidates.append((reduction,int(y),ar))
                    best=sorted(candidates,key=lambda v:(-v[0],v[1]))[0]
                    for reduction,y,ar in candidates:ar['best_utility_class']=y==best[1];attribution.append(ar)
                    local.append(dict(client=i,bank=b,context=t,salt=s,eligible=real_regret>0,best_class=best[1],best_fraction=best[2]['recovery_fraction'],
                        at_least_half=bool(real_regret>0 and best[0]/real_regret>=Fraction(1,2))))
    write_csv(out/'matched_exact_count_null_cells.csv',posterior);write_csv(out/'per_class_null_summary.csv',classes)
    write_csv(out/'task_actual_vs_null_cells.csv',task);write_csv(out/'one_class_repair_attribution.csv',attribution);write_csv(out/'localization_clients.csv',local)
    write_csv(out/'paired_context_cells.csv',paired);write_csv(out/'paired_class_residuals.csv',pairclass)
    postsummary=[];tasksummary=[];pairsummary=[];localsummary=[]
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            group=[r for r in posterior if r['bank']==b and r['context']==t];r=dict(bank=b,context=t)
            for name in ('residual_L1','residual_L2','weighted_class_L1','prevalence_L1','prevalence_JS','dominant_agreement'):
                r[name+'_actual_median']=float(np.median([x[name+'_actual'] for x in group]));r[name+'_null_median_median']=float(np.median([x[name+'_null_median'] for x in group]));r[name+'_exceedance_fraction']=float(np.mean([x[name+'_above_p95'] for x in group]))
            postsummary.append(r)
            for si,s in enumerate(S):
                group=[r for r in task if r['bank']==b and r['context']==t and r['salt']==s];r=dict(bank=b,context=t,salt=s)
                for name in ('DU','regret'):
                    for label,q in [('median',.5),('p90',.9)]:r[name+'_actual_'+label]=float(np.quantile([x[name+'_actual'] for x in group],q))
                    r[name+'_actual_mean']=float(np.mean([x[name+'_actual'] for x in group]));r[name+'_null_median_mean']=float(np.mean([x[name+'_null_median'] for x in group]));r[name+'_exceedance_fraction']=float(np.mean([x[name+'_above_p95'] for x in group]))
                r.update(task_exceedance_fraction=float(np.mean([x['task_excess'] for x in group])),actual_optimal_agreement=float(np.mean([x['actual_optimal_agreement'] for x in group])),
                    null_optimal_agreement=float(np.mean([x['null_optimal_agreement'] for x in group])),actual_canonical_agreement=float(np.mean([x['actual_canonical_agreement'] for x in group])),null_canonical_agreement=float(np.mean([x['null_canonical_agreement'] for x in group])))
                tasksummary.append(r)
                ordered=sorted(group,key=lambda x:(x['mean_utility_margin'],x['client']))
                for q in range(4):
                    g=ordered[q*25:(q+1)*25];marginrows.append(dict(bank=b,context=t,salt=s,quartile=q+1,n=25,margin_min=min(x['mean_utility_margin'] for x in g),margin_max=max(x['mean_utility_margin'] for x in g),
                        zero_margin_clients=sum(x['mean_utility_margin']==0 for x in g),actual_optimal_error=float(np.mean([1-x['actual_optimal_agreement'] for x in g])),null_optimal_error=float(np.mean([1-x['null_optimal_agreement'] for x in g])),
                        actual_regret_mean=float(np.mean([x['regret_actual'] for x in g])),null_median_regret_mean=float(np.mean([x['regret_null_median'] for x in g])),task_exceedance_fraction=float(np.mean([x['task_excess'] for x in g]))))
                if ti:
                    group=[r for r in paired if r['bank']==b and r['context']==t and r['salt']==s];r=dict(bank=b,context=t,salt=s,eligible_clients=len(group))
                    for name in ('paired_residual_L1','paired_residual_L2','delta_DU','delta_regret'):
                        r[name+'_actual_median']=float(np.median([x[name+'_actual'] for x in group]));r[name+'_null_median_median']=float(np.median([x[name+'_null_median'] for x in group]));r[name+'_exceedance_fraction']=float(np.mean([x[name+'_above_p95'] for x in group]))
                    r['context_exceedance_fraction']=float(np.mean([x['context_excess'] for x in group]));pairsummary.append(r)
                group=[r for r in local if r['bank']==b and r['context']==t and r['salt']==s];eligible=[r for r in group if r['eligible']]
                localsummary.append(dict(bank=b,context=t,salt=s,eligible_clients=len(eligible),zero_gap_clients=100-len(eligible),fraction_best_class_recovers_half=float(np.mean([r['at_least_half'] for r in eligible])) if eligible else None,
                    best_class_histogram=json.dumps({str(y):sum(r['best_class']==y for r in eligible) for y in range(10)})))
    write_csv(out/'matched_exact_count_null_summary.csv',postsummary);write_csv(out/'task_actual_vs_null_summary.csv',tasksummary);write_csv(out/'paired_context_summary.csv',pairsummary)
    write_csv(out/'localization_summary.csv',localsummary);write_csv(out/'utility_margin_quartiles.csv',marginrows)
    taskpass={t:all(r['task_exceedance_fraction']>=.2 for r in tasksummary if r['context']==t) for t in C[1:]}
    ctxpass={t:all(r['context_exceedance_fraction']>=.2 for r in pairsummary if r['context']==t) for t in C[1:]}
    localpass={t:all(r['fraction_best_class_recovers_half'] is not None and r['fraction_best_class_recovers_half']>=.25 for r in localsummary if r['context']==t) for t in C}
    clean_excess=[]
    for b in B:
        for s in S:
            gg=[r for r in task if r['bank']==b and r['context']=='clean' and r['salt']==s]
            pmap={r['client']:r for r in posterior if r['bank']==b and r['context']=='clean'}
            clean_excess.append(dict(bank=b,salt=s,fraction=float(np.mean([r['task_excess'] or pmap[r['client']]['residual_L1_above_p95'] for r in gg]))))
    taskgate=sum(taskpass[t] for t in C[2:])>=2;ctxgate=sum(ctxpass[t] for t in C[2:])>=2;generic=all(r['fraction']>=.2 for r in clean_excess)
    gates=dict(TASK_MISMATCH_A=taskgate,CTX_SPEC_A=ctxgate,task_context_pass=taskpass,context_specific_pass=ctxpass,LOCAL_A=localpass,substantial_clean_excess=generic,clean_excess=clean_excess,
        diagnosis='T019-C' if taskgate and ctxgate else 'T019-G' if taskgate and not ctxgate and generic else 'T019-X',
        diagnosis_note='S requires a clear descriptive state-boundary explanation; mixed evidence is retained as X, not forced.')
    save(out/'diagnostic_gates.json',gates)
    mechanism_files=['matched_exact_count_null_cells.csv','per_class_null_summary.csv','task_actual_vs_null_cells.csv','one_class_repair_attribution.csv','localization_clients.csv','paired_context_cells.csv','paired_class_residuals.csv',
        'matched_exact_count_null_summary.csv','task_actual_vs_null_summary.csv','paired_context_summary.csv','localization_summary.csv','utility_margin_quartiles.csv','diagnostic_gates.json']
    save(out/'mechanism_freeze.json',dict(runtime=a.commit,query_metrics_opened=False,choices_freeze_sha256=sha(out/'phaseB_choices_freeze.json'),hashes={n:sha(out/n) for n in mechanism_files}))
    print('T019_MECHANISM_FROZEN',json.dumps(gates),flush=True)
    # Query-count lookup is first opened after all residual/null/utility quantities above are hashed.
    p13=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p18=project/'runs/20260914-132634-ttfl-t018r2-science/artifacts/t018r2_constrained_prevalence'
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];den=ct[0].sum((0,1))
    zsaved={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    psaved={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    oldactual={(r['salt'],r['bank'],r['context']):r for r in csv.DictReader((p18/'actual_oracle_context.csv').open())}
    capture=[];clientcapture=[];repairquery=[];aggregate=[];aggcounts=np.zeros((2,5,4,128,10),dtype=np.int64);realcounts=np.zeros((2,5,4,10),dtype=np.int64)
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            for si,s in enumerate(S):
                zcounts=np.zeros(10,dtype=np.int64);pcounts=np.zeros(10,dtype=np.int64)
                for i in range(100):
                    zh=np.zeros(10,dtype=np.int64);ph=zh.copy();ah=zh.copy();nh=np.zeros((128,10),dtype=np.int64)
                    for h in (0,1):
                        slot=2*si+h;table=cc[si,bi,i,ti,1-h];canonical=regrets[i,bi,ti,slot].index(Fraction())
                        zh+=table[0];ph+=table[canonical];ah+=table[u['actual_choices'][i,bi,ti,slot]];nh+=table[u['null_choices'][i,bi,ti,:,slot]]
                    zcounts+=zh;pcounts+=ph;realcounts[bi,ti,si]+=ah;aggcounts[bi,ti,si]+=nh
                    z=macro(zh,den);p=macro(ph,den);acc=macro(ah,den);valid=p>z
                    row=dict(client=i,bank=b,context=t,salt=s,valid_positive_P00_gain=valid,P00_gain_contribution_exact=str(p-z),actual_macro_contribution_exact=str(acc))
                    if valid:
                        cap=(acc-z)/(p-z);nc=np.array([float((macro(hits,den)-z)/(p-z)) for hits in nh]);add_comparison(row,'capture_deficit',float(1-cap),1-nc)
                    else:
                        row.update({f'capture_deficit_{k}':None for k in ('actual','null_median','null_p95','percentile','above_p95')})
                    clientcapture.append(row)
                    for y in np.flatnonzero(counts[i]):
                        rh=np.zeros(10,dtype=np.int64)
                        for h in (0,1):rh+=cc[si,bi,i,ti,1-h,u['repair_choices'][i,bi,ti,y,2*si+h]]
                        racc=macro(rh,den);repairquery.append(dict(client=i,bank=b,context=t,salt=s,label=int(y),macro_contribution_change_exact=str(racc-acc),
                            capture_change=float((racc-acc)/(p-z)) if valid else None,valid_positive_P00_gain=valid))
                key=(s,b,t);za=macro(zcounts,den);pa=macro(pcounts,den);aa=macro(realcounts[bi,ti,si],den)
                assert za==zsaved[key] and pa==psaved[key] and aa==Fraction(oldactual[key]['macro_class_exact'])
                cap=(aa-za)/(pa-za);nullcaps=[];nullacc=[]
                for k in range(128):
                    na=macro(aggcounts[bi,ti,si,k],den);nc=(na-za)/(pa-za);nullcaps.append(float(nc));nullacc.append(float(na))
                    aggregate.append(dict(bank=b,context=t,salt=s,replica=k,macro_exact=str(na),capture_exact=str(nc)))
                cr=dict(bank=b,context=t,salt=s,actual_macro=float(aa),actual_macro_exact=str(aa),actual_capture=float(cap),actual_capture_exact=str(cap),
                    null_macro_p05=float(np.quantile(nullacc,.05)),null_macro_median=float(np.median(nullacc)),null_macro_p95=float(np.quantile(nullacc,.95)),
                    null_capture_p05=float(np.quantile(nullcaps,.05)),null_capture_median=float(np.median(nullcaps)),null_capture_p95=float(np.quantile(nullcaps,.95)),actual_below_null_p05=float(cap)<np.quantile(nullcaps,.05))
                valid=[r for r in clientcapture if r['bank']==b and r['context']==t and r['salt']==s and r['valid_positive_P00_gain']]
                cr.update(eligible_client_capture=len(valid),nonpositive_client_denominator=100-len(valid),capture_deficit_p95_exceedance=float(np.mean([r['capture_deficit_above_p95'] for r in valid])) if valid else None)
                capture.append(cr)
    write_csv(out/'capture_summary.csv',capture);write_csv(out/'client_capture_deficit.csv',clientcapture);write_csv(out/'class_repair_query_capture.csv',repairquery)
    save(out/'null_capture_replicas.json',aggregate);np.savez_compressed(out/'query_count_receipt.npz',null_counts=aggcounts,actual_counts=realcounts,total=den)
    sourceblur=[r for r in csv.DictReader((p18/'blur_decomposition.csv').open())];write_csv(out/'historical_source_blur_penalty.csv',sourceblur)
    assert all(sha(Path(p))==h for p,h in load(out/'input_hashes.json').items())
    assert all(sha(out/n)==h for n,h in load(out/'mechanism_freeze.json')['hashes'].items())
    save(out/'summary.json',dict(status='COMPLETE',runtime=a.commit,preparation_runtime=freeze['runtime'],seconds=time.time()-start,**gates,
        source_hashes_unchanged=True,query_opened_only_after_mechanism_freeze=True,new_model_forwards=0,actual_T018R2_macro_exact=True))
    print('T019_COMPLETE',json.dumps(load(out/'summary.json')),flush=True)


if __name__=='__main__':main()
