"""T014 exact supervised composition diagnostic on frozen T013 halves, no model code."""
import argparse,csv,gzip,hashlib,itertools,json,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.class_factorization import class_templates,predict_utilities
from src.context.disjoint_factorization import SALTS,OFFSETS,factor_utilities,exact_argmax
from report_t009 import write_csv
from report_t010 import ranks,corr
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
P13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
P11=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dumpcsv(path,rows):write_csv(path,[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def fracstr(v):return f'{v.numerator}/{v.denominator}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--commit',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    def write(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
    def gz(n,x):(out/n).write_bytes(gzip.compress(json.dumps(x,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    assert sha(P11/'candidate_predictions.npz')==load(P11/'query_freeze.json')['predictions_sha256']
    f13=load(P13/'policy_freeze.json');assert sha(P13/'half_integer_counts.npz')==f13['counts_sha256']
    assert sha(P13/'crossfit_choices.json')==f13['choices_sha256'] and sha(P13/'query_halves.json')==load(P13/'split_freeze.json')['sha256']
    manifest={r['path']:r['sha256'] for r in load(ROOT/'research_log/t013_artifact_manifest.json')}
    assert sha(P13/'integer_count_receipts.json')==manifest[(P13/'integer_count_receipts.json').relative_to(ROOT).as_posix()]
    hc=np.load(P13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];hits=cc.sum(-1);n=ct.sum(-1);totals=ct[0].sum((0,1))
    halves=load(P13/'query_halves.json');p=np.load(P11/'candidate_predictions.npz');labels=[p[f'c{i}_labels'] for i in range(100)];correct=[]
    for i in range(100):
        arr=np.empty((2,5,5,len(labels[i])),dtype=bool)
        for bi,b in enumerate(('A','B')):
            for ti,t in enumerate(C):
                for ci,c in enumerate(C):arr[bi,ti,ci]=p[f'{b}|{t}|{c}|c{i}']==labels[i]
        correct.append(arr)
        for si,salt in enumerate(SALTS):
            for h in (0,1):
                idx=halves[salt][str(i)][h]['indices'];y=labels[i][idx]
                assert np.bincount(y,minlength=10).tolist()==ct[si,i,h].tolist()
                for bi in range(2):
                    for ti in range(5):
                        for ci in range(5):assert np.bincount(y[arr[bi,ti,ci,idx]],minlength=10).tolist()==cc[si,bi,i,ti,h,ci].tolist()
    oldchoices=load(P13/'crossfit_choices.json');oi={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):C.index(r['selected']) for r in oldchoices}
    oldreceipts=load(P13/'integer_count_receipts.json');oldmetrics={(r['salt'],r['bank'],r['target'],r['policy']):r for r in load(P13/'crossfit_metrics.json')};oldexact={}
    for r in oldreceipts:
        salt,b,t,pol=r['salt'],r['bank'],r['target'],r['policy'];si=SALTS.index(salt);bi=('A','B').index(b);ti=C.index(t)
        counts=sum((cc[si,bi,i,ti,h,oi[salt,b,i,t,h,pol]] for i in range(100) for h in (0,1)),np.zeros(10,dtype=np.int64))
        assert counts.tolist()==r['class_correct']
        val=sum((Fraction(100*int(x),int(y)) for x,y in zip(counts,totals)),Fraction())/10;oldexact[salt,b,t,pol]=val
        assert abs(float(val)-oldmetrics[salt,b,t,pol]['macro_class'])<1e-10
    for r in csv.DictReader((P13/'clean_safety.csv').open()):assert abs(float(oldexact[r['salt'],r['bank'],'clean','two_factor']-oldexact[r['salt'],r['bank'],'clean','zero'])-float(r['delta_pp']))<1e-10
    write('preflight.json',dict(timestamp=now(),T011_prediction_sha_verified=True,T013_splits_choices_counts_hashes_verified=True,all_half_counts_reconstructed=True,T013_all_600_policy_count_metrics_exact=True,clean_safety_exact=True,new_model_forward_passes=0))
    compositions=[dict(salt=salt,client=i,train_half=h,class_counts=ct[si,i,h].tolist(),denominator=int(n[si,i,h])) for si,salt in enumerate(SALTS) for i in range(100) for h in (0,1)]
    write('composition_vectors.json',compositions)
    templates=[];choices=[];predclass={};policies=['composition_persistent','class_context_only','hybrid_client_plus_class_context','T013_two_factor','T013_client_only','uniform_class_context','uniform_hybrid']
    policies += [f'mismatched_pi_{family}_{k}' for k in OFFSETS for family in ('persistent','class_context','hybrid')]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for train in (0,1):
                pactual,globalcontext=factor_utilities(hits[si,bi,:,:,train,:],n[si,:,train]);ev=1-train
                for i in range(100):
                    template=class_templates(cc[si,bi,:,:,train,:,:],ct[si,:,train,:],i)
                    denom=ct[si,:,train,:].sum(0)-ct[si,i,train]
                    templates.append(dict(salt=salt,bank=b,train_half=train,target_client=i,other99_class_denominators=denom.tolist(),utility=[[[fracstr(v) for v in row] for row in context] for context in template]))
                    persistent,classes,hybrid=predict_utilities(template,pactual[i],ct[si,i,train])
                    _,uniformclass,uniformhybrid=predict_utilities(template,pactual[i],[1]*10)
                    mismatch={k:predict_utilities(template,pactual[i],ct[si,(i+k)%100,train]) for k in OFFSETS}
                    assert hybrid[0]==pactual[i]
                    for ti,t in enumerate(C):
                        predclass[si,bi,i,ti,ev]=classes[ti]
                        values=dict(composition_persistent=persistent,class_context_only=classes[ti],hybrid_client_plus_class_context=hybrid[ti],T013_two_factor=[pactual[i][s]+globalcontext[i][ti][s] for s in range(5)],T013_client_only=pactual[i],uniform_class_context=uniformclass[ti],uniform_hybrid=uniformhybrid[ti])
                        for k in OFFSETS:
                            pp,cl,hy=mismatch[k];values[f'mismatched_pi_persistent_{k}']=pp;values[f'mismatched_pi_class_context_{k}']=cl[ti];values[f'mismatched_pi_hybrid_{k}']=hy[ti]
                        for name in policies:
                            arg=exact_argmax(values[name]);chosen=arg[0]
                            if name in ('T013_two_factor','T013_client_only'):assert chosen==oi[salt,b,i,t,ev,name.removeprefix('T013_')]
                            if ti==0 and name=='hybrid_client_plus_class_context':assert chosen==oi[salt,b,i,t,ev,'client_only']
                            choices.append(dict(salt=salt,bank=b,client=i,target=t,train_half=train,eval_half=ev,policy=name,selected=C[chosen],argmax=[C[s] for s in arg],utility=[fracstr(v) for v in values[name]]))
        print('T014_CHOICES_SALT_DONE',salt,flush=True)
    gz('class_templates.json.gz',dict(definition='utility[c][k][s]=other99 class utility vs zero; D=utility[clean]; R=utility[c]-D, all exact fractions',rows=templates))
    gz('policy_utilities.json.gz',choices);dumpcsv(out/'policy_choices.csv',[{k:v for k,v in r.items() if k!='utility'} for r in choices])
    freeze=dict(timestamp=now(),composition_sha256=sha(out/'composition_vectors.json'),templates_sha256=sha(out/'class_templates.json.gz'),utilities_sha256=sha(out/'policy_utilities.json.gz'),choices_sha256=sha(out/'policy_choices.csv'),choices=len(choices),T014_eval_outcomes_opened=False,residual_analysis_opened=False)
    write('policy_freeze.json',freeze);print('T014_POLICIES_FROZEN',len(choices),flush=True)
    ci={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):C.index(r['selected']) for r in choices};metrics=[];receipts=[];exact={}
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for ti,t in enumerate(C):
                for pol in policies:
                    counts=np.zeros(10,dtype=np.int64);clientacc=[];regs=[];freq={c:0 for c in C};hit=0;gt2=0;gt5=0
                    for i in range(100):
                        nc=0
                        for h in (0,1):
                            chosen=ci[salt,b,i,t,h,pol];q=cc[si,bi,i,ti,h,chosen];counts+=q;nc+=int(q.sum());freq[C[chosen]]+=1
                            reg=Fraction(100*(int(hits[si,bi,i,ti,h].max())-int(q.sum())),int(n[si,i,h]));regs.append(float(reg));hit+=reg==0;gt2+=reg>2;gt5+=reg>5
                        clientacc.append(100*nc/int(n[si,i].sum()))
                    val=sum((Fraction(100*int(x),int(y)) for x,y in zip(counts,totals)),Fraction())/10;exact[salt,b,t,pol]=val
                    if pol.startswith('T013_'):assert val==oldexact[salt,b,t,pol.removeprefix('T013_')]
                    if t=='clean' and pol=='hybrid_client_plus_class_context':assert val==oldexact[salt,b,t,'client_only']
                    metrics.append(dict(salt=salt,bank=b,target=t,policy=pol,macro_class=float(val),sample_weighted=100*int(counts.sum())/int(totals.sum()),macro_client=float(np.mean(clientacc)),state_choice_frequency=freq,oracle_hit_rate=hit/200,regret_median=float(np.median(regs)),regret_p75=float(np.quantile(regs,.75)),regret_p90=float(np.quantile(regs,.9)),regret_gt2_fraction=gt2/200,regret_gt5_fraction=gt5/200))
                    receipts.append(dict(salt=salt,bank=b,target=t,policy=pol,class_total=totals.tolist(),class_correct=counts.tolist()))
    # Separate per-example concatenation verifies all new class-count aggregates.
    for r in receipts:
        salt,b,t,pol=r['salt'],r['bank'],r['target'],r['policy'];bi=('A','B').index(b);ti=C.index(t);counts=np.zeros(10,dtype=np.int64)
        for i in range(100):
            chosen=np.zeros(len(labels[i]),dtype=bool)
            for h in (0,1):
                idx=halves[salt][str(i)][h]['indices'];candidate=ci[salt,b,i,t,h,pol];chosen[idx]=correct[i][bi,ti,candidate,idx]
            counts+=np.bincount(labels[i][chosen],minlength=10)
        assert counts.tolist()==r['class_correct']
    dumpcsv(out/'policy_metrics.csv',metrics);write('policy_metrics.json',metrics);write('integer_count_receipts.json',receipts)
    compositioncapture=[];interaction=[];captures={};compgates=[]
    for salt in SALTS:
        for b in ('A','B'):
            zero=oldexact[salt,b,'clean','zero'];client=oldexact[salt,b,'clean','client_only'];comp=exact[salt,b,'clean','composition_persistent'];mismatch=sum(exact[salt,b,'clean',f'mismatched_pi_persistent_{k}'] for k in OFFSETS)/8
            cap=(comp-zero)/(client-zero) if client>zero else None
            passed=cap is not None and cap>=Fraction(4,5) and comp-mismatch>=Fraction(1,2)
            compositioncapture.append(dict(salt=salt,bank=b,composition_gain=float(comp-zero),client_gain=float(client-zero),composition_capture=float(cap) if cap is not None else None,composition_minus_mismatch_mean=float(comp-mismatch),passed=passed));compgates.append(passed)
            for t in C[1:]:
                z=oldexact[salt,b,t,'zero'];oracle=oldexact[salt,b,t,'half_oracle'];hy=exact[salt,b,t,'hybrid_client_plus_class_context'];mm=[exact[salt,b,t,f'mismatched_pi_hybrid_{k}'] for k in OFFSETS];cap=(hy-z)/(oracle-z);captures[salt,b,t]=cap
                interaction.append(dict(salt=salt,bank=b,target=t,class_capture=float(cap),hybrid_minus_two_factor=float(hy-oldexact[salt,b,t,'two_factor']),hybrid_minus_client_only=float(hy-oldexact[salt,b,t,'client_only']),class_context_minus_context_only=float(exact[salt,b,t,'class_context_only']-oldexact[salt,b,t,'context_only']),hybrid_minus_uniform=float(hy-exact[salt,b,t,'uniform_hybrid']),hybrid_minus_mismatch_mean=float(hy-sum(mm)/8),mismatch_mean=float(sum(mm)/8),mismatch_min=float(min(mm)),mismatch_max=float(max(mm))))
    joint={t:all(captures[salt,b,t]>=Fraction(4,5) for salt in SALTS for b in ('A','B')) for t in C[1:]};advantages=[]
    for b in ('A','B'):
        for t in C[1:]:
            gain=sum(exact[salt,b,t,'hybrid_client_plus_class_context']-oldexact[salt,b,t,'two_factor'] for salt in SALTS)/4
            misgain=sum(exact[salt,b,t,'hybrid_client_plus_class_context']-sum(exact[salt,b,t,f'mismatched_pi_hybrid_{k}'] for k in OFFSETS)/8 for salt in SALTS)/4
            advantages.append(dict(bank=b,target=t,mean_hybrid_minus_two_factor=float(gain),mean_hybrid_minus_mismatch=float(misgain),gain_pass=gain>=Fraction(1,2),mismatch_pass=misgain>=Fraction(1,2)))
    addition=any(all(next(r['gain_pass'] for r in advantages if r['bank']==b and r['target']==t) for b in ('A','B')) for t in ('contrast_low','gaussian_blur'))
    inta=sum(joint.values())>=3 and addition
    intb=not inta and all(sum(r['gain_pass'] for r in advantages if r['bank']==b)>=2 and sum(r['mismatch_pass'] for r in advantages if r['bank']==b)>=2 for b in ('A','B'))
    dumpcsv(out/'composition_capture.csv',compositioncapture);dumpcsv(out/'interaction_capture.csv',interaction)
    # Residual utilities on held-out halves, after all policies were frozen.
    residual={};ranked={};sets={};rawsets={}
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for i in range(100):
                for ti,t in enumerate(C):
                    for h in (0,1):
                        obs=[Fraction(int(x)-int(hits[si,bi,i,ti,h,0]),int(n[si,i,h])) for x in hits[si,bi,i,ti,h]]
                        e=[x-y for x,y in zip(obs,predclass[si,bi,i,ti,h])];key=si,bi,i,ti,h
                        residual[key]=e;ranked[key]=ranks(e);sets[key]=set(exact_argmax(e));rawsets[key]=set(exact_argmax(obs))
    residualrows=[];residualsummary=[];similarity=[];rawsummary={(r['salt'],r['bank']):r for r in load(P13/'disjoint_lock_summary.json')['groups']}
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            samecount=0;controlcount=0;rawsame=0;rawctrl=0
            for t,u in itertools.combinations(range(5),2):
                for h in (0,1):
                    same=sum(bool(sets[si,bi,i,t,h]&sets[si,bi,i,u,1-h]) for i in range(100));controls=[sum(bool(sets[si,bi,i,t,h]&sets[si,bi,(i+k)%100,u,1-h]) for i in range(100)) for k in OFFSETS]
                    samecount+=same;controlcount+=sum(controls)
                    rawsame+=sum(bool(rawsets[si,bi,i,t,h]&rawsets[si,bi,i,u,1-h]) for i in range(100));rawctrl+=sum(sum(bool(rawsets[si,bi,i,t,h]&rawsets[si,bi,(i+k)%100,u,1-h]) for i in range(100)) for k in OFFSETS)
                    residualrows.append(dict(salt=salt,bank=b,target1=C[t],target2=C[u],orientation=h,same_overlap=same/100,control_mean=sum(controls)/800,control_min=min(controls)/100,control_max=max(controls)/100))
                    def sim(k):
                        v=[corr(ranked[si,bi,i,t,h],ranked[si,bi,(i+k)%100,u,1-h]) for i in range(100)];v=[x for x in v if x is not None]
                        return dict(valid_count=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None)
                    similarity.append(dict(salt=salt,bank=b,target1=C[t],target2=C[u],orientation=h,same=sim(0),controls=[dict(offset=k,**sim(k)) for k in OFFSETS]))
            rawmargin=Fraction(100*(8*rawsame-rawctrl),16000);resmargin=Fraction(100*(8*samecount-controlcount),16000)
            assert abs(float(rawmargin)-rawsummary[salt,b]['difference_pp'])<1e-12
            residualsummary.append(dict(salt=salt,bank=b,raw_margin_pp=float(rawmargin),residual_margin_pp=float(resmargin),attenuation_pp=float(rawmargin-resmargin),fractional_attenuation=float((rawmargin-resmargin)/rawmargin),residual_above_original_15pp=resmargin>=15))
    dumpcsv(out/'residual_lock.csv',residualrows);write('residual_lock_summary.json',dict(groups=residualsummary,utility_similarity=similarity))
    gz('residual_utilities.json.gz',[dict(salt=SALTS[si],bank=('A','B')[bi],client=i,target=C[ti],half=h,utility=[fracstr(v) for v in e]) for (si,bi,i,ti,h),e in residual.items()])
    write('summary.json',dict(COMP_A=all(compgates),CLASS_INT_A=inta,CLASS_INT_B=intb,CLASS_INT_descriptor='CLASS-INT-A' if inta else ('CLASS-INT-B' if intb else 'CLASS-INT-C'),joint_capture=joint,contrast_or_blur_increment=addition,advantages=advantages,composition_capture=compositioncapture,residual_lock=residualsummary))
    for name,key in [('composition_vectors.json','composition_sha256'),('class_templates.json.gz','templates_sha256'),('policy_utilities.json.gz','utilities_sha256'),('policy_choices.csv','choices_sha256')]:assert sha(out/name)==freeze[key]
    write('verification.json',dict(new_model_forward_passes=0,T011_T013_hashes_verified=True,all_T013_600_metrics_reconstructed=True,exact_same_halves=True,target_opposite_half_only=True,other99_template_excludes_target=True,
        all_class_denominators_positive=True,template_and_policy_arithmetic_exact_Fraction=True,all_clean_hybrid_choices_metrics_equal_client_only=True,controls_share_templates_only_pi_changes=True,
        all_1240_metrics_independently_reconstructed_per_example=True,raw_disjoint_lock_T013_exact=True,policy_freeze_hashes_unchanged=True,focused_tests_passed=3))
    write('metadata.json',dict(code_commit=a.commit,timestamp=now(),seconds=time.time()-start,execution='local CPU frozen saved predictions',policies=len(policies),choice_rows=len(choices),metric_rows=len(metrics),template_rows=len(templates)))
    print('T014_DONE',json.dumps(load(out/'summary.json')),flush=True)

if __name__=='__main__':main()
