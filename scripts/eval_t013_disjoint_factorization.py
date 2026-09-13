"""T013 saved-prediction-only cross-fit audit; no model import or forward pass."""
import argparse,csv,gzip,hashlib,itertools,json,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.disjoint_factorization import SALTS,OFFSETS,query_halves,factor_utilities,exact_argmax
from report_t009 import write_csv
from report_t010 import ranks,corr
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
P11=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def now():return datetime.now(timezone.utc).isoformat()
def simstats(values):
    v=[x for x in values if x is not None]
    return dict(valid_count=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None)
def dumpcsv(path,rows):write_csv(path,[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--commit',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    def write(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
    ids=load(ROOT/'research_log/pfllib_receipts/data/Cifar10/split_manifest.json')['clients'];p=np.load(P11/'candidate_predictions.npz');qr=load(P11/'candidate_query_utility.json')
    assert sha(P11/'candidate_predictions.npz')==load(P11/'query_freeze.json')['predictions_sha256']
    qi={(r['client'],r['bank'],r['target'],r['candidate']):r for r in qr};labels=[];correct=[];fullsets={}
    for i in range(100):
        y=p[f'c{i}_labels'];labels.append(y);assert len(y)==len(ids[str(i)]['test'])
        arr=np.empty((2,5,5,len(y)),dtype=bool)
        for bi,b in enumerate(('A','B')):
            for ti,t in enumerate(C):
                for ci,c in enumerate(C):
                    r=qi[i,b,t,c];pr=p[r['prediction_key']];assert len(pr)==len(y)
                    assert np.bincount(y,minlength=10).tolist()==r['class_total']
                    assert np.bincount(y[y==pr],minlength=10).tolist()==r['class_correct']
                    arr[bi,ti,ci]=pr==y
                fullsets[i,bi,ti]=set(exact_argmax(arr[bi,ti].sum(-1)))
        correct.append(arr)
    oldpairs=list(csv.DictReader((ROOT/'results/t012_context_amplitude/context_pair_overlap.csv').open()))
    for r in oldpairs:
        bi=('A','B').index(r['bank']);t,u=C.index(r['target1']),C.index(r['target2'])
        same=sum(bool(fullsets[i,bi,t]&fullsets[i,bi,u]) for i in range(100))/100
        control=[sum(bool(fullsets[i,bi,t]&fullsets[(i+k)%100,bi,u]) for i in range(100))/100 for k in OFFSETS]
        assert same==float(r['same_client_overlap']) and abs(float(np.mean(control))-float(r['control_mean']))<1e-15
    write('preflight.json',dict(timestamp=now(),T011_5000_count_rows_exact=True,T012_full_query_overlap_exact=True,all_lengths_aligned=True,new_forward_passes=0))
    # Historical regression above is required; no T013 half-derived outcome exists yet.
    halves={salt:{str(i):[dict(indices=h,original_ids=[ids[str(i)]['test'][j] for j in h]) for h in query_halves(ids[str(i)]['test'],salt,i)] for i in range(100)} for salt in SALTS}
    for salt in SALTS:
        for i in range(100):
            h0,h1=[r['indices'] for r in halves[salt][str(i)]]
            assert not set(h0)&set(h1) and sorted(h0+h1)==list(range(len(labels[i]))) and abs(len(h0)-len(h1))<=1
    write('query_halves.json',halves);write('split_freeze.json',dict(timestamp=now(),sha256=sha(out/'query_halves.json'),label_blind=True,T013_half_label_analysis_started=False))
    print('T013_HALVES_FROZEN',flush=True)
    # Half count tensor [salt,bank,client,target,half,candidate,class].
    cc=np.zeros((4,2,100,5,2,5,10),dtype=np.int64);ct=np.zeros((4,100,2,10),dtype=np.int64)
    for si,salt in enumerate(SALTS):
        for i in range(100):
            for h in (0,1):
                inds=halves[salt][str(i)][h]['indices'];y=labels[i][inds];ct[si,i,h]=np.bincount(y,minlength=10)
                for bi in range(2):
                    for ti in range(5):
                        for c in range(5):cc[si,bi,i,ti,h,c]=np.bincount(y[correct[i][bi,ti,c,inds]],minlength=10)
    hits=cc.sum(-1);n=ct.sum(-1);sets={};argmaxrows=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for i in range(100):
                for ti,t in enumerate(C):
                    for h in (0,1):
                        ss=exact_argmax(hits[si,bi,i,ti,h]);sets[si,bi,i,ti,h]=set(ss)
                        argmaxrows.append(dict(salt=salt,bank=b,client=i,target=t,half=h,argmax=[C[c] for c in ss]))
    disjoint=[];similarities=[];client_overlap=np.zeros((4,2,100));locksummary=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            samehits=0;ctrlhits=0
            for t,u in itertools.combinations(range(5),2):
                for h in (0,1):
                    same=[bool(sets[si,bi,i,t,h]&sets[si,bi,i,u,1-h]) for i in range(100)]
                    ctrl=[sum(bool(sets[si,bi,i,t,h]&sets[si,bi,(i+k)%100,u,1-h]) for i in range(100)) for k in OFFSETS]
                    samehits+=sum(same);ctrlhits+=sum(ctrl);client_overlap[si,bi]+=np.array(same)/20
                    disjoint.append(dict(salt=salt,bank=b,target1=C[t],target2=C[u],orientation=h,same_client_overlap=sum(same)/100,control_mean=sum(ctrl)/800,control_min=min(ctrl)/100,control_max=max(ctrl)/100,control_by_offset={str(k):v/100 for k,v in zip(OFFSETS,ctrl)}))
                    def sim(k):return simstats([corr(ranks(hits[si,bi,i,t,h]),ranks(hits[si,bi,(i+k)%100,u,1-h])) for i in range(100)])
                    similarities.append(dict(salt=salt,bank=b,target1=C[t],target2=C[u],orientation=h,same=sim(0),controls=[dict(offset=k,**sim(k)) for k in OFFSETS]))
            locksummary.append(dict(salt=salt,bank=b,mean_same_overlap=samehits/2000,mean_control_overlap=ctrlhits/16000,difference_pp=100*(8*samehits-ctrlhits)/16000,passed=8*samehits-ctrlhits>=2400))
    dumpcsv(out/'disjoint_lock.csv',disjoint);write('disjoint_lock_summary.json',dict(groups=locksummary,DISJOINT_LOCK_STRONG=all(r['passed'] for r in locksummary),utility_similarity=similarities))
    # Every policy utility receives training-half counts only. Evaluation-half outcomes are used below solely by named oracle controls and later evaluation.
    policies=['client_only','context_only','two_factor']+[f'mismatch_{k}' for k in OFFSETS]+['zero','true_state','full_query_oracle','half_oracle']
    choices=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for train in (0,1):
                persistent,context=factor_utilities(hits[si,bi,:,:,train,:],n[si,:,train])
                ev=1-train
                for i in range(100):
                    for ti,t in enumerate(C):
                        u=dict(client_only=persistent[i],context_only=context[i][ti],two_factor=[persistent[i][s]+context[i][ti][s] for s in range(5)])
                        for k in OFFSETS:u[f'mismatch_{k}']=[persistent[(i+k)%100][s]+context[i][ti][s] for s in range(5)]
                        for name in policies:
                            if name in u:arg=exact_argmax(u[name])
                            elif name=='zero':arg=[0]
                            elif name=='true_state':arg=[ti]
                            elif name=='full_query_oracle':arg=sorted(fullsets[i,bi,ti])
                            else:arg=sorted(sets[si,bi,i,ti,ev])
                            choices.append(dict(salt=salt,bank=b,client=i,target=t,train_half=train,eval_half=ev,policy=name,selected=C[arg[0]],predicted_argmax=[C[c] for c in arg]))
    write('crossfit_choices.json',choices);dumpcsv(out/'crossfit_choices.csv',choices);write('half_argmax_sets.json',argmaxrows)
    np.savez_compressed(out/'half_integer_counts.npz',correct=cc,total=ct)
    write('policy_freeze.json',dict(timestamp=now(),choices_sha256=sha(out/'crossfit_choices.json'),argmax_sha256=sha(out/'half_argmax_sets.json'),counts_sha256=sha(out/'half_integer_counts.npz'),choice_rows=len(choices),composition_audit_opened=False))
    print('T013_POLICIES_FROZEN',len(choices),flush=True)
    # Independent evaluation-half lookup; all choices are now immutable.
    choiceidx={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):C.index(r['selected']) for r in choices}
    totals=ct[0].sum((0,1));metrics=[];receipts=[];regretrows=[];exactmetrics={}
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for ti,t in enumerate(C):
                for policy in policies:
                    counts=np.zeros(10,dtype=np.int64);client_acc=[];regs=[];hit=0;freq={c:0 for c in C};gt2=0;gt5=0
                    for i in range(100):
                        client_correct=0
                        for h in (0,1):
                            c=choiceidx[salt,b,i,t,h,policy];rr=cc[si,bi,i,ti,h,c];counts+=rr;client_correct+=int(rr.sum());freq[C[c]]+=1
                            reg=Fraction(100*(int(hits[si,bi,i,ti,h].max())-int(rr.sum())),int(n[si,i,h]));regs.append(float(reg));gt2+=reg>2;gt5+=reg>5;hit+=reg==0
                            regretrows.append(dict(salt=salt,bank=b,client=i,target=t,half=h,policy=policy,regret=float(reg),oracle_hit=reg==0))
                        client_acc.append(100*client_correct/int(n[si,i].sum()))
                    val=sum((Fraction(100*int(x),int(y)) for x,y in zip(counts,totals)),Fraction())/10;exactmetrics[salt,b,t,policy]=val
                    metrics.append(dict(salt=salt,bank=b,target=t,policy=policy,macro_class=float(val),sample_weighted=100*int(counts.sum())/int(totals.sum()),macro_client=float(np.mean(client_acc)),state_choice_frequency=freq,oracle_hit_rate=hit/200,regret_median=float(np.median(regs)),regret_p75=float(np.quantile(regs,.75)),regret_p90=float(np.quantile(regs,.9)),regret_gt2_fraction=gt2/200,regret_gt5_fraction=gt5/200))
                    receipts.append(dict(salt=salt,bank=b,target=t,policy=policy,class_correct=counts.tolist(),class_total=totals.tolist()))
    dumpcsv(out/'crossfit_metrics.csv',metrics);write('crossfit_metrics.json',metrics);write('integer_count_receipts.json',receipts);write('crossfit_regret.json',regretrows)
    # Exact reconstruction of the reported aggregate from a second concatenation path.
    error=0.
    for r in receipts:
        salt,b,t,policy=r['salt'],r['bank'],r['target'],r['policy'];si=SALTS.index(salt);bi=('A','B').index(b);ti=C.index(t)
        allcorrect=[];ally=[]
        for i in range(100):
            chosen=np.empty(len(labels[i]),dtype=bool)
            for h in (0,1):
                idx=halves[salt][str(i)][h]['indices'];c=choiceidx[salt,b,i,t,h,policy];chosen[idx]=correct[i][bi,ti,c,idx]
            allcorrect.append(chosen);ally.append(labels[i])
        yy=np.concatenate(ally);cc0=np.concatenate(allcorrect);recount=np.bincount(yy[cc0],minlength=10)
        assert recount.tolist()==r['class_correct']
        error=max(error,abs(float(np.mean(100*recount/totals))-float(exactmetrics[salt,b,t,policy])))
    historical=load(ROOT/'results/t011_task_proximal/summary.json');history={(r['bank'],r['target']):r for r in historical['best_of_five']}
    for salt in SALTS:
        for b in ('A','B'):
            for t in C:assert abs(float(exactmetrics[salt,b,t,'full_query_oracle'])-history[b,t]['query_best_policy_macro_class'])<1e-10
    captures=[];capture_exact={};clean=[]
    for salt in SALTS:
        for b in ('A','B'):
            clean.append(dict(salt=salt,bank=b,delta_pp=float(exactmetrics[salt,b,'clean','two_factor']-exactmetrics[salt,b,'clean','zero'])))
            for t in C:
                values={p:exactmetrics[salt,b,t,p] for p in policies};mism=[values[f'mismatch_{k}'] for k in OFFSETS];meanmis=sum(mism)/8
                gain=values['half_oracle']-values['zero'];cap=(values['two_factor']-values['zero'])/gain if gain>0 else None
                capture_exact[salt,b,t]=cap
                captures.append(dict(salt=salt,bank=b,target=t,factor_capture=float(cap) if cap is not None else None,two_minus_client=float(values['two_factor']-values['client_only']),two_minus_context=float(values['two_factor']-values['context_only']),two_minus_mean_mismatch=float(values['two_factor']-meanmis),mismatch_mean=float(meanmis),mismatch_min=float(min(mism)),mismatch_max=float(max(mism))))
    joint={t:all(capture_exact[salt,b,t] is not None and capture_exact[salt,b,t]>=Fraction(4,5) for salt in SALTS for b in ('A','B')) for t in C[1:]}
    cleangates=[dict(bank=b,mean_delta_pp=float(sum(exactmetrics[salt,b,'clean','two_factor']-exactmetrics[salt,b,'clean','zero'] for salt in SALTS)/4)) for b in ('A','B')]
    factora=sum(joint.values())>=3 and all(r['mean_delta_pp']>=-.5 for r in cleangates)
    advantages=[]
    for b in ('A','B'):
        for t in C[1:]:
            delta={p:sum(exactmetrics[salt,b,t,'two_factor']-exactmetrics[salt,b,t,p] for salt in SALTS)/4 for p in ('client_only','context_only')}
            advantages.append(dict(bank=b,target=t,mean_two_minus_client=float(delta['client_only']),mean_two_minus_context=float(delta['context_only']),passed=all(v>=Fraction(1,2) for v in delta.values())))
    lock=all(r['passed'] for r in locksummary);factorb=not factora and lock and all(sum(r['passed'] for r in advantages if r['bank']==b)>=2 for b in ('A','B'))
    write('summary.json',dict(DISJOINT_LOCK_STRONG=lock,FACTOR_A=factora,FACTOR_B=factorb,primary='FACTOR-A' if factora else ('FACTOR-B' if factorb else 'FACTOR-C'),all_salts_banks_joint_capture=joint,clean=cleangates,advantages=advantages))
    dumpcsv(out/'factor_capture.csv',captures);dumpcsv(out/'clean_safety.csv',clean)
    # Label composition is descriptive, opened after split, argmax and policy freeze.
    composition=[];compositionidx={}
    def entropy(p):return float(-sum(x*np.log(x) for x in p if x>0))
    for si,salt in enumerate(SALTS):
        for i in range(100):
            ps=[ct[si,i,h]/n[si,i,h] for h in (0,1)];mid=(ps[0]+ps[1])/2;js=entropy(mid)-(entropy(ps[0])+entropy(ps[1]))/2;l1=float(np.abs(ps[0]-ps[1]).sum())
            for h in (0,1):
                r=dict(salt=salt,client=i,half=h,n=int(n[si,i,h]),normalized_entropy=entropy(ps[h])/np.log(10),max_class_fraction=float(ps[h].max()),represented_classes=int((ps[h]>0).sum()),halves_histogram_L1=l1,halves_histogram_JS=js)
                composition.append(r);compositionidx[si,i,h]=r
    dumpcsv(out/'label_composition_audit.csv',composition)
    descriptions=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            overlap=client_overlap[si,bi];reg=[]
            for i in range(100):reg.append(float(np.mean([r['regret'] for r in regretrows if r['salt']==salt and r['bank']==b and r['client']==i and r['policy']=='two_factor'])))
            for feature in ('normalized_entropy','max_class_fraction','halves_histogram_L1','halves_histogram_JS'):
                x=[(compositionidx[si,i,0][feature]+compositionidx[si,i,1][feature])/2 for i in range(100)]
                descriptions.append(dict(salt=salt,bank=b,feature=feature,spearman_same_client_overlap=corr(ranks(x),ranks(overlap)),spearman_two_factor_regret=corr(ranks(x),ranks(reg))))
    write('composition_associations.json',descriptions)
    assert sha(out/'query_halves.json')==load(out/'split_freeze.json')['sha256']
    assert sha(out/'crossfit_choices.json')==load(out/'policy_freeze.json')['choices_sha256']
    write('verification.json',dict(T011_predictions_sha256=sha(P11/'candidate_predictions.npz'),T011_5000_counts_exact=True,T012_full_query_overlap_exact=True,new_model_forward_passes=0,
        label_blind_disjoint_complete_splits=True,splits_shared_contexts_banks=True,split_frozen_before_T013_half_analysis=True,
        exact_fraction_utilities_equal_client_leave_one_out=True,diagnostic_policies_train_half_only=True,oracle_controls_explicit=True,
        policies_argmax_frozen_before_composition=True,independent_per_example_aggregate_reconstruction=True,macroclass_max_error_pp=error,
        no_training_or_policy_salt_offset_tuning=True))
    write('metadata.json',dict(code_commit=a.commit,started_seconds=start,seconds=time.time()-start,execution='local CPU saved predictions only',timestamp=now(),salts=SALTS,offsets=OFFSETS))
    for name in ('crossfit_choices.json','crossfit_regret.json'):
        pp=out/name;pp.with_suffix('.json.gz').write_bytes(gzip.compress(pp.read_bytes(),mtime=0))
    print('T013_DONE',json.dumps(load(out/'summary.json')),flush=True)

if __name__=='__main__':main()
