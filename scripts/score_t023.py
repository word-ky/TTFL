"""Frozen rotation choices scored only through historical utility/integer receipts."""
import argparse,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t023_common import *
from src.context.matched_channel import exact_utilities
from src.context.rotation_alignment import spearman,gates
from eval_t017_bootstrap import macro
from report_t009 import write_csv

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','phase-a','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);phase=Path(a.phase_a);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    freeze=load(phase/'phaseA_choices_freeze.json');assert freeze['status']=='PASS' and all(sha(phase/name)==digest for name,digest in freeze['hashes'].items())
    assert all(freeze[k] is False for k in ('target_class_labels_used','privileged_utility_loaded','query_outcomes_scored','model_parameters_updated'))
    ext=Path(freeze['extraction']);ef=load(ext/'extraction_freeze.json');assert sha(ext/'extraction_freeze.json')==freeze['extraction_freeze_sha256']
    assert all(sha(ext/name)==digest for name,digest in ef['hashes'].items()) and all(sha(path)==digest for path,digest in load(phase/'input_hashes.json').items())
    save(out/'scoring_start.json',dict(runtime=a.commit,phase_a=str(phase),phase_a_runtime=freeze['runtime'],phase_a_freeze_sha256=sha(phase/'phaseA_choices_freeze.json'),all_choices_frozen=True,phase_a_sealed_flags_verified=True,privileged_scoring_now_authorized=True))
    p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';p14=p/'results/t014_class_conditional_factorization'
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics';p18=p/'runs/20260914-132634-ttfl-t018r2-science/artifacts/t018r2_constrained_prevalence'
    files=[p13/'half_integer_counts.npz',p13/'integer_count_receipts.json',p14/'class_templates.json.gz',p15/'integer_count_receipts.json',p16/'support_truth.json',p16/'integer_count_receipts.json',p18/'actual_oracle_context_episodes.json.gz',p18/'integer_count_receipts.json']
    inputs=load(phase/'input_hashes.json');inputs.update({str(path):sha(path) for path in files});inputs.update({str(phase/name):digest for name,digest in freeze['hashes'].items()})
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];totals=hc['total'][0].sum((0,1));truth={r['client']:r for r in load(p16/'support_truth.json')}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    bbse={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p16/'integer_count_receipts.json') if r['policy']=='BBSE-S-01'}
    historical={(r['salt'],r['bank'],r['context']):r['class_correct'] for r in load(p18/'integer_count_receipts.json') if r['mode']=='01'}
    data=dict(np.load(phase/'phaseA_scores.npz'));point=np.zeros((100,2,5,8),dtype=np.uint8)
    for r in gzload(p18/'actual_oracle_context_episodes.json.gz'):point[r['client'],B.index(r['bank']),C.index(r['context']),2*S.index(r['salt'])+r['train_half']]=r['state']
    states={'ROT-CURRENT':np.repeat(data['current_choices'][...,None],8,axis=-1),'ROT-CLEAN-SURROGATE':np.repeat(data['clean_choices'][...,None],8,axis=-1),'P':point}
    utilities={};regrets={};optimal={}
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for h in (0,1):
                        v,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],templates[s,b,h,i][ti]);key=i,bi,ti,2*si+h;utilities[key]=v;regrets[key]=[max(v)-z for z in v];optimal[key]=arg
    actual=[];receipts=[];episodes=[];rankrows=[];histograms=[]
    for policy,choices in states.items():
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                if policy!='P':
                    ci=ti if policy=='ROT-CURRENT' else 0;masks=data['current_masks' if policy=='ROT-CURRENT' else 'clean_masks'][:,bi,ti]
                    histograms.append(dict(policy=policy,bank=b,context=t,client_state_counts=np.bincount(choices[:,bi,ti,0],minlength=5).tolist(),exact_tie_cells=int(sum(int(mask).bit_count()>1 for mask in masks)),cells=100))
                for si,s in enumerate(S):
                    count=np.zeros(10,dtype=np.int64);rr=[];hits=can=0
                    for i in range(100):
                        for h in (0,1):
                            slot=2*si+h;key=i,bi,ti,slot;state=int(choices[key]);reg=regrets[key][state];rr.append(reg);hits+=state in optimal[key];can+=state==optimal[key][0];count+=cc[si,bi,i,ti,1-h,state]
                            episodes.append(dict(policy=policy,client=i,bank=b,context=t,salt=s,train_half=h,state=state,true_template_regret_exact=str(reg),true_template_regret=float(reg),optimal_set=state in optimal[key],canonical=state==optimal[key][0]))
                            if policy!='P':rankrows.append(dict(policy=policy,client=i,bank=b,context=t,salt=s,train_half=h,spearman=spearman(data['scores'][i,bi,ci],utilities[key])))
                    key=s,b,t;acc=macro(count,totals);cap=(acc-zero[key])/(p00[key]-zero[key]);mean=sum(rr,Fraction())/200
                    if policy=='P':np.testing.assert_array_equal(count,historical[key])
                    actual.append(dict(policy=policy,bank=b,context=t,salt=s,macro_class=float(acc),macro_class_exact=str(acc),capture=float(cap),capture_exact=str(cap),delta_vs_BBSE_pp=float(acc-bbse[key]),clean_delta_vs_zero=float(acc-zero[key]),true_template_regret=float(mean),true_template_regret_exact=str(mean),regret_p90=float(np.quantile([float(r) for r in rr],.9)),optimal_set_agreement=hits/200,canonical_agreement=can/200))
                    receipts.append(dict(policy=policy,bank=b,context=t,salt=s,class_correct=count.tolist(),class_total=totals.tolist()))
    paired=[]
    for policy,baseline,comparison in [('ROT-CURRENT','P','vs_P'),('ROT-CLEAN-SURROGATE','P','vs_P'),('ROT-CURRENT','ROT-CLEAN-SURROGATE','current_vs_clean')]:
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    nv=[];ov=[];deltas=[]
                    for i in range(100):
                        nn=[];oo=[]
                        for h in (0,1):
                            key=i,bi,ti,2*si+h;nn.append(regrets[key][states[policy][key]]);oo.append(regrets[key][states[baseline][key]])
                        nv+=nn;ov+=oo;deltas.append(sum(nn,Fraction())-sum(oo,Fraction()))
                    nm=sum(nv,Fraction())/200;om=sum(ov,Fraction())/200
                    paired.append(dict(policy=policy,comparison=comparison,bank=b,context=t,salt=s,new_mean=float(nm),baseline_mean=float(om),new_mean_exact=str(nm),baseline_mean_exact=str(om),new_p90=float(np.quantile([float(v) for v in nv],.9)),baseline_p90=float(np.quantile([float(v) for v in ov],.9)),relative_mean_reduction=float((om-nm)/om) if om else None,clients_better=sum(v<0 for v in deltas),clients_equal=sum(v==0 for v in deltas),clients_worse=sum(v>0 for v in deltas)))
    ranksummary=[]
    for policy in ('ROT-CURRENT','ROT-CLEAN-SURROGATE'):
        for b in B:
            for t in C:
                group=[r['spearman'] for r in rankrows if r['policy']==policy and r['bank']==b and r['context']==t];v=[r for r in group if r is not None]
                ranksummary.append(dict(policy=policy,bank=b,context=t,defined=len(v),undefined=len(group)-len(v),median=float(np.median(v)) if v else None,p25=float(np.quantile(v,.25)) if v else None,p75=float(np.quantile(v,.75)) if v else None,fraction_positive=float(np.mean(np.array(v)>0)) if v else None,unit='client x salt x half;100 clients x8 utility vectors'))
    g=gates([r for r in actual if r['policy']=='ROT-CURRENT'],[r for r in paired if r['policy']=='ROT-CURRENT' and r['comparison']=='vs_P'],[r for r in paired if r['comparison']=='current_vs_clean'])
    write_csv(out/'actual_aggregate.csv',actual);save(out/'integer_count_receipts.json',receipts);gzsave(out/'actual_regret_episodes.json.gz',episodes);write_csv(out/'paired_regret_context.csv',paired);gzsave(out/'rank_correlations.json.gz',rankrows);write_csv(out/'rank_summary.csv',ranksummary);save(out/'state_choice_histograms.json',histograms);save(out/'diagnostic_gates.json',g)
    assert all(sha(path)==digest for path,digest in inputs.items());save(out/'input_hashes.json',inputs)
    save(out/'summary.json',dict(status='COMPLETE',lead='89466b3',runtime=a.commit,diagnosis=g['diagnosis'],CTX=g['CTX'],gates=g,actual_rows=len(actual),regret_episodes=len(episodes),rank_correlations=len(rankrows),new_query_forwards=0,new_model_forwards=0,model_parameters_updated=False,seconds=time.time()-start))
    print('T023_SCIENCE_COMPLETE',g,flush=True)

if __name__=='__main__':main()
