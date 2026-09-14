"""Independent original-model replay, prototype/CLS/choice checks and count scoring."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,save,sha,gzload,C,B,S
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.data.covariate import corrupt,TARGETS
from src.eval.common import state_hash
from src.context.constrained_prevalence import ActiveSetCLS,face_reference,direct_kkt
from src.context.matched_channel import exact_utilities
from eval_t015_semantic_mixture import state_digest
from eval_t017_bootstrap import macro
from utils.data_utils import read_data
def rows(path):return list(csv.DictReader(path.open()))


def main():
    ap=argparse.ArgumentParser()
    for key in ('project','scoring','output','commit'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();p=Path(a.project);score=Path(a.scoring);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    stage=load(score/'scoring_start.json');phase=Path(stage['phase_a']);matched=Path(stage['matched']);ext=Path(load(phase/'phaseA_choices_freeze.json')['extraction'])
    for folder,filename in [(phase,'phaseA_choices_freeze.json'),(matched,'matched_freeze.json')]:
        assert all(sha(folder/name)==digest for name,digest in load(folder/filename)['hashes'].items())
    inputs=load(score/'input_hashes.json');assert all(sha(Path(path))==digest for path,digest in inputs.items())
    p7=p/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer';p9=p/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p13=p/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';p14=p/'results/t014_class_conditional_factorization'
    support=load(p9/'natural_support_manifest.json');source={(r['client'],r['bank'],r['true_context']):r['selected'] for r in load(p9/'source_decisions.json')}
    data={r:dict(np.load(ext/f'support_{r}.npz')) for r in ('L','H')};oldlogits=dict(np.load(p15/'support_logits.npz'))
    base=Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10');os.chdir(base/'system');torch.set_num_threads(2)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval();model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True));initial=state_hash(model)
    saved=torch.load(p7/'states.pt',map_location='cuda',weights_only=True);states={b:{'clean':new_state(model),**{t:saved[f'{b}|{t}|correct_pair'] for t in TARGETS}} for b in B}
    statehashes={b:{t:state_digest(st) for t,st in bank.items()} for b,bank in states.items()}
    captured=[];hook=model.fc.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()))
    forward_samples=forward_calls=0;maxlogit=maxfeature=0.
    for i in (0,11,22,33,44,55,64,77,87,99):
        selected=support['clients'][str(i)]['selected'];ids=[r['original_id'] for r in selected];idx=[r['train_index'] for r in selected]
        x=torch.as_tensor(read_data('Cifar10',i,is_train=True)['x'][idx],dtype=torch.float32).cuda()
        for t in C:
            tx=corrupt(x,t,ids,i)
            for b in B:
                for c in dict.fromkeys((t,source[i,b,t])):
                    with torch.no_grad():z=model(tx,states[b][c]).cpu().numpy()
                    h=captured.pop();key=f'{i}|{b}|{t}|{c}'
                    dl=float(np.max(np.abs(z-oldlogits[key])));dh=float(np.max(np.abs(h-data['H'][key])))
                    assert dl<=1e-6 and dh<=1e-6 and np.array_equal(z.argmax(1),oldlogits[key].argmax(1));np.testing.assert_array_equal(z,data['L'][key])
                    maxlogit=max(maxlogit,dl);maxfeature=max(maxfeature,dh);forward_samples+=20;forward_calls+=1
    hook.remove();assert forward_samples>=2000 and state_hash(model)==initial
    assert all(state_digest(st)==statehashes[b][c] for b,bank in states.items() for c,st in bank.items())
    print('T021_INDEPENDENT_FORWARD',forward_samples,flush=True)
    truth={r['client']:r for r in load(p16/'support_truth.json')};labels=np.array([truth[i]['labels_in_frozen_support_order'] for i in range(100)]);flatlabels=labels.reshape(-1)
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    prototypes={r:dict(np.load(phase/f'{r}_prototypes.npz')) for r in ('L','H')};actual={r:dict(np.load(phase/f'{r}_actual.npz')) for r in ('L','H')}
    nulls={r:dict(np.load(matched/f'{r}_matched.npz')) for r in ('L','H')}
    prototypechecks=meanchecks=choicechecks=clschecks=matchedchoicechecks=0;maxpi=maxkkt=0.
    for rep in ('L','H'):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for kind,context in [(0,t),(1,'clean')]:
                    stacked=np.stack([data[rep][f'{i}|{b}|{context}|{t}'] for i in range(100)]).astype(np.float64)
                    for i in (0,24,49,64,74,87,99):
                        keep=np.arange(100)!=i;x=stacked[keep].reshape(-1,stacked.shape[-1]);y=labels[keep].reshape(-1)
                        m=np.stack([x[y==k].mean(0) for k in range(10)],axis=1);n=np.bincount(y,minlength=10)
                        np.testing.assert_array_equal(m,prototypes[rep]['prototypes'][kind,i,bi,ti]);np.testing.assert_array_equal(n,prototypes[rep]['class_counts'][kind,i,bi,ti]);prototypechecks+=1
                for i in range(100):
                    for pol,policy in enumerate(('oracle','source','clean_same_state')):
                        c=source[i,b,t] if policy=='source' else t;ci=C.index(c);kind=int(policy=='clean_same_state');loc=pol,i,bi,ti
                        q=data[rep][f'{i}|{b}|{t}|{c}'].astype(np.float64).mean(0);np.testing.assert_array_equal(q,actual[rep]['means'][loc]);meanchecks+=1
                        m=prototypes[rep]['prototypes'][kind,i,bi,ci];pi=actual[rep]['pi'][loc];kk=direct_kkt(m,q,pi);assert kk<=1e-10;maxkkt=max(maxkkt,kk)
                        for si,s in enumerate(S):
                            for h in (0,1):
                                _,arg=exact_utilities(pi,templates[s,b,h,i][ci]);slot=2*si+h
                                assert arg[0]==actual[rep]['choices'][loc+(slot,)] and sum(1<<v for v in arg)==actual[rep]['argmax_masks'][loc+(slot,)];choicechecks+=1
                    ids=nulls[rep]['source_ids'][i,bi,ti];assert np.all(ids//20!=i)
                    np.testing.assert_array_equal(np.stack([(flatlabels[ids]==k).sum(1) for k in range(10)],axis=1),np.tile(truth[i]['counts'],(128,1)))
                    m=prototypes[rep]['prototypes'][0,i,bi,ti];solver=ActiveSetCLS(m)
                    stack=np.stack([data[rep][f'{j}|{b}|{t}|{t}'] for j in range(100)]).reshape(2000,-1).astype(np.float64)
                    otherids=np.flatnonzero(np.arange(2000)//20!=i)
                    for sample_idx,r in enumerate((0,127)):
                        seed=int(hashlib.sha256(f'T021|matched|{rep}|{i}|{b}|{t}|{r}'.encode()).hexdigest(),16);rng=np.random.Generator(np.random.PCG64(seed));draw=[]
                        for k,n in enumerate(truth[i]['counts']):
                            pool=otherids[flatlabels[otherids]==k];draw.extend(pool[rng.integers(0,len(pool),size=n)])
                        draw=np.array(draw);np.testing.assert_array_equal(draw,ids[r]);q=stack[draw].mean(0);np.testing.assert_array_equal(q,nulls[rep]['qsample'][i,bi,ti,sample_idx])
                        pi,receipt=solver.solve(q);savedpi=nulls[rep]['pi'][i,bi,ti,r];error=float(np.max(np.abs(pi-savedpi)));assert error<=1e-9;maxpi=max(maxpi,error);clschecks+=1
                        for si,s in enumerate(S):
                            for h in (0,1):
                                _,arg=exact_utilities(savedpi,templates[s,b,h,i][ti]);slot=2*si+h
                                assert arg[0]==nulls[rep]['choices'][i,bi,ti,r,slot] and sum(1<<v for v in arg)==nulls[rep]['argmax_masks'][i,bi,ti,r,slot];matchedchoicechecks+=1
                print('T021_INDEPENDENT_OBSERVER',rep,b,t,flush=True)
    fallbacks=load(matched/'solver_fallbacks.json');assert not load(phase/'solver_fallbacks.json');fallbackchecks=[]
    for r in fallbacks:
        rep=r['representation'];i=r['client'];bi=B.index(r['bank']);ti=C.index(r['context']);j=r['replica'];m=prototypes[rep]['prototypes'][0,i,bi,ti]
        ids=nulls[rep]['source_ids'][i,bi,ti,j];x=np.stack([data[rep][f'{k}|{r["bank"]}|{r["context"]}|{r["context"]}'] for k in range(100)]).reshape(2000,-1).astype(np.float64)
        q=x[ids].mean(0);ref,obj=face_reference(m,q);savedpi=nulls[rep]['pi'][i,bi,ti,j]
        assert np.max(np.abs(ref-savedpi))<=1e-9 and abs(obj-r['objective'])<=1e-12 and direct_kkt(m,q,ref)<=1e-10
        fallbackchecks.append(dict(representation=rep,client=i,bank=r['bank'],context=r['context'],replica=j,reference_KKT=direct_kkt(m,q,ref),objective_difference=abs(obj-r['objective']),max_pi_difference=float(np.max(np.abs(ref-savedpi)))))
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1))
    episodes=gzload(score/'actual_regret_episodes.json.gz');countmap={};regmap={};exactregrets=0
    for r in episodes:
        i=r['client'];bi=B.index(r['bank']);ti=C.index(r['context']);si=S.index(r['salt']);h=r['train_half'];state=r['state'];key=r['representation'],r['policy'],r['bank'],r['context'],r['salt']
        if key not in countmap:countmap[key]=np.zeros(10,dtype=np.int64);regmap[key]=[]
        countmap[key]+=cc[si,bi,i,ti,1-h,state]
        vals,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],templates[r['salt'],r['bank'],h,i][ti]);reg=max(vals)-vals[state]
        assert str(reg)==r['true_template_regret_exact'] and (state in arg)==r['optimal_set'];regmap[key].append((i,h,reg));exactregrets+=1
    for r in load(score/'integer_count_receipts.json'):
        key=r['representation'],r['policy'],r['bank'],r['context'],r['salt'];np.testing.assert_array_equal(countmap[key],r['class_correct']);np.testing.assert_array_equal(totals,r['class_total'])
    for r in rows(score/'actual_aggregate.csv'):
        key=r['representation'],r['policy'],r['bank'],r['context'],r['salt'];assert str(macro(countmap[key],totals))==r['macro_class_exact']
        assert str(sum((v for i,h,v in regmap[key]),Fraction())/200)==r['true_template_regret_exact']
    for r in rows(score/'paired_regret_context.csv'):
        key=r['representation'],'oracle',r['bank'],r['context'],r['salt'];basekey=('P','oracle') if r['comparison']=='vs_P' else (r['representation'],'clean_same_state')
        old=regmap[basekey+(r['bank'],r['context'],r['salt'])];new=regmap[key];nv=[v for i,h,v in new];ov=[v for i,h,v in old]
        assert str(sum(nv,Fraction())/200)==r['new_mean_exact'] and str(sum(ov,Fraction())/200)==r['baseline_mean_exact']
        assert float(np.quantile([float(v) for v in nv],.9))==float(r['new_p90']) and float(np.quantile([float(v) for v in ov],.9))==float(r['baseline_p90'])
        deltas=[sum((v for j,h,v in new if j==i),Fraction())-sum((v for j,h,v in old if j==i),Fraction()) for i in range(100)]
        assert sum(v<0 for v in deltas)==int(r['clients_better']) and sum(v==0 for v in deltas)==int(r['clients_equal']) and sum(v>0 for v in deltas)==int(r['clients_worse'])
    querychecks=0;matched_rows=gzload(score/'matched_replicas.json.gz');matched_acc={}
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    for rep in ('L','H'):
        ar=np.zeros((2,5,4,128,10),dtype=np.int64);saved=np.load(score/f'{rep}_matched_query_counts.npz')['aggregate_counts']
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for i in range(100):
                        for h in (0,1):ar[bi,ti,si]+=cc[si,bi,i,ti,1-h][nulls[rep]['choices'][i,bi,ti,:,2*si+h]]
                    for j in range(128):matched_acc[rep,b,t,s,j]=macro(ar[bi,ti,si,j],totals);querychecks+=1
        np.testing.assert_array_equal(ar,saved)
    for r in matched_rows:
        key=r['representation'],r['bank'],r['context'],r['salt'],r['replica'];acc=matched_acc[key];basekey=r['salt'],r['bank'],r['context']
        assert str(acc)==r['macro_class_exact'] and str((acc-zero[basekey])/(p00[basekey]-zero[basekey]))==r['capture_exact']
    for r in rows(score/'matched_aggregate.csv'):
        if r['representation']=='P':continue
        group=[x for x in matched_rows if all(x[k]==r[k] for k in ('representation','bank','context','salt'))]
        for name in ('capture','macro_class'):
            for label,q in [('p05',.05),('median',.5),('p95',.95)]:assert float(r[name+'_'+label])==float(np.quantile([x[name] for x in group],q))
    # Rebuild the predeclared gates from the independently checked numeric tables.
    gates=load(score/'diagnostic_gates.json')['representations'];ma=rows(score/'matched_aggregate.csv');aa=rows(score/'actual_aggregate.csv');pa=rows(score/'paired_regret_context.csv')
    for rep in ('L','H'):
        mpass={t:all(float(r['capture_median'])>=.8 for r in ma if r['representation']==rep and r['context']==t) for t in C[1:]}
        real=[r for r in aa if r['representation']==rep and r['policy']=='oracle'];rpass={t:all(float(r['capture'])>=.8 for r in real if r['context']==t) for t in C[1:]}
        safe=all(float(r['delta_vs_BBSE_pp'])>=-.5 for r in real if r['context']!='clean')
        assert mpass==gates[rep]['matched_pass'] and rpass==gates[rep]['real_pass'] and (sum(mpass.values())>=3)==gates[rep]['REP_MATCH_A'] and (sum(rpass.values())>=3 and safe)==gates[rep]['REP_REAL_A']
        for comparison,name in [('vs_P','REP_REGRET_A'),('context_vs_clean_same_state','REP_CTX_A')]:
            passes={t:all(float(r['baseline_mean'])>0 and float(r['new_mean'])<=.85*float(r['baseline_mean']) and float(r['new_p90'])<=1.05*float(r['baseline_p90']) for r in pa if r['representation']==rep and r['comparison']==comparison and r['context']==t) for t in C[1:]}
            assert passes==gates[rep][name]['context_pass'] and (sum(passes.values())>=3)==gates[rep][name]['passed']
        assert not gates[rep]['REP_REAL_A'] and gates[rep]['REP_SRC_A']=='NOT_EXECUTED' and not any(r['representation']==rep and r['policy']=='source' for r in aa)
    assert all(sha(Path(path))==digest for path,digest in inputs.items())
    files=[]
    for folder in (ext,phase,matched,score):
        files.extend(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in folder.iterdir() if path.is_file())
    save(out/'remote_artifact_manifest.json',files)
    save(out/'independent_verification.json',dict(status='PASS',runtime=a.commit,scoring=str(score),historical_logit_samples=forward_samples,independent_original_model_forwards=forward_calls,
        max_logit_difference=maxlogit,max_feature_difference=maxfeature,model_and_states_unchanged=True,target_excluded_prototype_checks=prototypechecks,
        actual_mean_reconstructions=meanchecks,all_phase_a_state_and_argmax_checks=choicechecks,matched_CLS_replays=clschecks,max_pi_difference=maxpi,max_actual_KKT=maxkkt,
        matched_sample_choice_checks=matchedchoicechecks,matched_all_draw_counts_and_exclusion_verified=True,all_fallback_reenumerations=fallbackchecks,
        exact_true_regret_reconstructions=exactregrets,actual_integer_count_rows=len(countmap),matched_integer_count_rows=querychecks,all_gates_reconstructed=True,
        all_paired_better_equal_worse_counts_reconstructed=True,upstream_input_hashes_unchanged=True,seconds=time.time()-start))
    print('T021_INDEPENDENT_PASS',json.dumps(load(out/'independent_verification.json')),flush=True)


if __name__=='__main__':main()
