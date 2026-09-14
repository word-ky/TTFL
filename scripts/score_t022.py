"""Inherited T018 integer scoring after the complete T022 actual/matched freezes."""
import argparse,csv,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,save,sha,gzload,C,B,S
from phase_a_t021 import gzsave
from eval_t017_bootstrap import macro
from src.context.matched_channel import exact_utilities
from src.context.representation_gates import inherited_gates as original_gates,regret_gate,diagnosis as original_diagnosis

def inherited_gates(*args):
    return {k.replace("REP_","RESP_"):v for k,v in original_gates(*args).items()}

def diagnosis(gates):
    old={r:{k.replace("RESP_","REP_"):v for k,v in g.items()} for r,g in gates.items()}
    return {"T021-L":"T022-RL","T021-H":"T022-RH","T021-M":"T022-M","T021-N":"T022-N","T021-X":"T022-X"}[original_diagnosis(old)]
from report_t009 import write_csv


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','matched','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);matched=Path(a.matched);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    mf=load(matched/'matched_freeze.json');phase=Path(mf['phase_a']);af=load(phase/'phaseA_choices_freeze.json')
    assert mf['status']==af['status']=='PASS'
    for folder,freeze in [(matched,mf),(phase,af)]:assert all(sha(folder/name)==digest for name,digest in freeze['hashes'].items())
    save(out/'scoring_start.json',dict(runtime=a.commit,matched=str(matched),phase_a=str(phase),matched_freeze_sha256=sha(matched/'matched_freeze.json'),phase_a_freeze_sha256=sha(phase/'phaseA_choices_freeze.json'),
        actual_choices_frozen=True,matched_choices_frozen=True,target_composition_and_query_scoring_now_authorized=True))
    p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';p14=p/'results/t014_class_conditional_factorization'
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p18=p/'runs/20260914-132634-ttfl-t018r2-science/artifacts/t018r2_constrained_prevalence'
    inputs=load(phase/'input_hashes.json');inputs.update({str(path):sha(path) for path in [p13/'half_integer_counts.npz',p13/'integer_count_receipts.json',p15/'integer_count_receipts.json',p16/'integer_count_receipts.json',p18/'actual_oracle_context_episodes.json.gz',p18/'integer_count_receipts.json',p18/'matched_k20_aggregate.csv',p18/'summary.json']})
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1))
    truth={r['client']:r for r in load(p16/'support_truth.json')}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    bbse={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json') if r['policy']=='BBSE-S-01'}
    bbse_source={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json') if r['policy']=='BBSE-S-11'}
    historical={(r['salt'],r['bank'],r['context']):macro(r['class_correct'],r['class_total']) for r in load(p18/'integer_count_receipts.json') if r['mode']=='01'}
    tv={};ru={};optimal={}
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for h in (0,1):
                        values,args=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],templates[s,b,h,i][ti]);key=i,bi,ti,2*si+h
                        tv[key]=values;ru[key]=[max(values)-v for v in values];optimal[key]=args
    point=np.zeros((100,2,5,8),dtype=np.uint8);pointpi=np.zeros((100,2,5,10))
    for r in gzload(p18/'actual_oracle_context_episodes.json.gz'):
        loc=r['client'],B.index(r['bank']),C.index(r['context']);slot=2*S.index(r['salt'])+r['train_half'];point[loc+(slot,)]=r['state'];pointpi[loc]=r['pi']
        assert float(ru[loc+(slot,)][r['state']])==r['true_template_regret']
    actual_data={r:dict(np.load(phase/f'{r}_actual.npz')) for r in ('L','H')}
    states={('P','oracle'):point};prevalence={('P','oracle'):pointpi}
    for rep in ('L','H'):
        for k,policy in enumerate(('oracle','source','clean_prototype')):states[rep,policy]=actual_data[rep]['choices'][k];prevalence[rep,policy]=actual_data[rep]['pi'][k]
    absolute_phase=p/'runs/20260914-183528-ttfl-t021-phase-a/artifacts/t021_frozen_representation_observability'
    absolute_score=p/'runs/20260914-184639-ttfl-t021-score/artifacts/t021_frozen_representation_observability'
    absolute_counts={(r['representation'],r['bank'],r['context'],r['salt']):r['class_correct'] for r in load(absolute_score/'integer_count_receipts.json') if r['policy']=='oracle'}
    for rep in ('L','H'):
        old=dict(np.load(absolute_phase/f'{rep}_actual.npz'));states['T021_'+rep,'oracle']=old['choices'][0];prevalence['T021_'+rep,'oracle']=old['pi'][0]
    inputs.update({str(path):sha(path) for path in [absolute_phase/'L_actual.npz',absolute_phase/'H_actual.npz',absolute_score/'integer_count_receipts.json',absolute_score/'matched_aggregate.csv']})
    actual_rows=[];counts_receipts=[];paired=[];episodes=[]
    def score_actual(rep,policy):
        rows=[];selected=states[rep,policy];pi=prevalence[rep,policy]
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    count=np.zeros(10,dtype=np.int64);regrets=[];hits=0;can=0;L1=[]
                    for i in range(100):
                        for h in (0,1):
                            slot=2*si+h;state=int(selected[i,bi,ti,slot]);key=i,bi,ti,slot
                            count+=cc[si,bi,i,ti,1-h,state];reg=ru[key][state];regrets.append(reg);hits+=state in optimal[key];can+=state==optimal[key][0]
                            l1=float(np.abs(pi[i,bi,ti]-truth[i]['pi']).sum());L1.append(l1)
                            episodes.append(dict(representation=rep,policy=policy,client=i,bank=b,context=t,salt=s,train_half=h,state=state,
                                true_template_regret_exact=str(reg),true_template_regret=float(reg),optimal_set=state in optimal[key],canonical=state==optimal[key][0],L1=l1))
                    key=s,b,t;acc=macro(count,totals);cap=(acc-zero[key])/(p00[key]-zero[key]);mean=sum(regrets,Fraction())/len(regrets)
                    row=dict(representation=rep,policy=policy,bank=b,context=t,salt=s,macro_class=float(acc),macro_class_exact=str(acc),capture=float(cap),capture_exact=str(cap),
                        delta_vs_BBSE_pp=float(acc-(bbse_source if policy=='source' else bbse)[key]),clean_delta_vs_zero=float(acc-zero[key]),true_template_regret=float(mean),true_template_regret_exact=str(mean),
                        regret_p90=float(np.quantile([float(x) for x in regrets],.9)),optimal_set_agreement=hits/200,canonical_agreement=can/200,L1=float(np.mean(L1)))
                    if rep=='P':assert acc==historical[key]
                    if rep.startswith('T021_'):np.testing.assert_array_equal(count,absolute_counts[rep[5:],b,t,s])
                    rows.append(row);counts_receipts.append(dict(representation=rep,policy=policy,bank=b,context=t,salt=s,class_correct=count.tolist(),class_total=totals.tolist()))
        actual_rows.extend(rows);return rows
    actual_tables={('P','oracle'):score_actual('P','oracle')}
    for rep in ('L','H'):
        for policy in ('oracle','clean_prototype'):actual_tables[rep,policy]=score_actual(rep,policy)
    for rep in ('L','H'):actual_tables['T021_'+rep,'oracle']=score_actual('T021_'+rep,'oracle')
    def paired_regret(rep,baseline_rep,baseline_policy,comparison):
        rows=[];new=states[rep,'oracle'];old=states[baseline_rep,baseline_policy]
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    nv=[];ov=[];better=equal=worse=0
                    for i in range(100):
                        nr=[];orr=[]
                        for h in (0,1):
                            slot=2*si+h;key=i,bi,ti,slot;nr.append(ru[key][new[key]]);orr.append(ru[key][old[key]])
                        delta=sum(nr,Fraction())-sum(orr,Fraction());better+=delta<0;equal+=delta==0;worse+=delta>0;nv+=nr;ov+=orr
                    nm=sum(nv,Fraction())/200;om=sum(ov,Fraction())/200
                    rows.append(dict(representation=rep,comparison=comparison,bank=b,context=t,salt=s,new_mean=float(nm),baseline_mean=float(om),new_mean_exact=str(nm),baseline_mean_exact=str(om),
                        new_p90=float(np.quantile([float(x) for x in nv],.9)),baseline_p90=float(np.quantile([float(x) for x in ov],.9)),mean_delta_pp=float(100*(nm-om)),
                        relative_mean_reduction=float((om-nm)/om) if om>0 else None,clients_better=int(better),clients_equal=int(equal),clients_worse=int(worse)))
        paired.extend(rows);return rows
    regret_rows={rep:paired_regret(rep,'P','oracle','vs_P') for rep in ('L','H')}
    ctx_rows={rep:paired_regret(rep,rep,'clean_prototype','context_vs_clean_prototype') for rep in ('L','H')}
    for rep in ('L','H'):paired_regret(rep,'T021_'+rep,'oracle','vs_T021_absolute')
    all_matched=[];matched_replicas=[]
    for rep in ('L','H'):
        selected=np.load(matched/f'{rep}_matched.npz')['choices'];aggregate=np.zeros((2,5,4,128,10),dtype=np.int64)
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for i in range(100):
                        for h in (0,1):aggregate[bi,ti,si]+=cc[si,bi,i,ti,1-h][selected[i,bi,ti,:,2*si+h]]
                    group=[];key=s,b,t
                    for r in range(128):
                        acc=macro(aggregate[bi,ti,si,r],totals);cap=(acc-zero[key])/(p00[key]-zero[key])
                        row=dict(representation=rep,bank=b,context=t,salt=s,replica=r,macro_class=float(acc),macro_class_exact=str(acc),capture=float(cap),capture_exact=str(cap));group.append(row);matched_replicas.append(row)
                    all_matched.append(dict(representation=rep,bank=b,context=t,salt=s,**{f'{name}_{label}':float(np.quantile([r[name] for r in group],q)) for name in ('macro_class','capture') for label,q in [('p05',.05),('median',.5),('p95',.95)]}))
        np.savez_compressed(out/f'{rep}_matched_query_counts.npz',aggregate_counts=aggregate,totals=totals)
    pg=[dict(r,representation='P') for r in csv.DictReader((p18/'matched_k20_aggregate.csv').open())]
    for r in pg:
        for k in ('capture_median','capture_p05','capture_p95','macro_class_median','macro_class_p05','macro_class_p95'):r[k]=float(r[k])
    absolute_matched=[dict(r,representation='T021_'+r['representation']) for r in csv.DictReader((absolute_score/'matched_aggregate.csv').open()) if r['representation'] in ('L','H')]
    for r in absolute_matched:
        for k in ('capture_median','capture_p05','capture_p95','macro_class_median','macro_class_p05','macro_class_p95'):r[k]=float(r[k])
    pg+=absolute_matched
    gates={}
    for rep in ('L','H'):
        mm=[r for r in all_matched if r['representation']==rep];g=inherited_gates(mm,actual_tables[rep,'oracle'])
        if g['RESP_REAL_A']:
            src=score_actual(rep,'source');actual_tables[rep,'source']=src;g=inherited_gates(mm,actual_tables[rep,'oracle'],src)
        g['RESP_REGRET_A']=regret_gate(regret_rows[rep]);g['RESP_CTX_A']=regret_gate(ctx_rows[rep]);gates[rep]=g
    write_csv(out/'matched_aggregate.csv',all_matched+[ {k:r[k] for k in all_matched[0]} for r in pg]);gzsave(out/'matched_replicas.json.gz',matched_replicas)
    write_csv(out/'actual_aggregate.csv',actual_rows);save(out/'integer_count_receipts.json',counts_receipts);gzsave(out/'actual_regret_episodes.json.gz',episodes);write_csv(out/'paired_regret_context.csv',paired)
    source_penalty=[]
    for rep in ('L','H'):
        if (rep,'source') in actual_tables:
            for r in actual_tables[rep,'source']:
                if r['context']=='gaussian_blur':
                    oracle=next(x for x in actual_tables[rep,'oracle'] if x['bank']==r['bank'] and x['context']==r['context'] and x['salt']==r['salt'])
                    source_penalty.append(dict(representation=rep,bank=r['bank'],salt=r['salt'],oracle_capture=oracle['capture'],source_capture=r['capture'],source_minus_oracle_pp=r['macro_class']-oracle['macro_class']))
    if source_penalty:write_csv(out/'source_blur_penalty.csv',source_penalty)
    main_diagnosis=diagnosis(gates)
    save(out/'diagnostic_gates.json',dict(diagnosis=main_diagnosis,representations=gates,flags={r:dict(CTX='CTX+' if gates[r]['RESP_CTX_A']['passed'] else 'CTX-',SRC=('SRC+' if gates[r]['RESP_SRC_A'] else 'SRC-') if isinstance(gates[r]['RESP_SRC_A'],bool) else 'NOT_EXECUTED') for r in ('L','H')}))
    assert all(sha(Path(path))==digest for path,digest in inputs.items());save(out/'input_hashes.json',inputs)
    save(out/'summary.json',dict(status='COMPLETE',runtime=a.commit,diagnosis=main_diagnosis,representations=gates,new_model_forwards=0,source_hashes_unchanged=True,seconds=time.time()-start,
        phase_a_choices=48000,matched_solutions=256000,matched_choices=2048000,scored_actual_rows=len(actual_rows),scored_actual_episodes=len(episodes),matched='fixedR128exactcounts',
        safety_inheritance='T018 real shifted >=BBSE-S-01 minus0.5pp; source clean >=zero minus0.5pp',source_scoring='only representations passing RESP_REAL_A'))
    print('T022_SCIENCE_COMPLETE',json.dumps(load(out/'summary.json')),flush=True)


if __name__=='__main__':main()
