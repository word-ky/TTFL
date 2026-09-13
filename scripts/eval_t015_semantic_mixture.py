"""T015 support-posterior phase A, privileged phase B, then historical query lookup."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.context.semantic_mixture import mixtures_from_logits,choose_utility,require_phaseA_freeze
from fractions import Fraction
import gzip
from src.context.source_signature import source_signature
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import evaluate,state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from scripts.eval_t008_source_identifiability import retrieval_gate
from scripts.eval_t009_natural_context import natural_id_gate,clean_safety_gate
from utils.data_utils import read_data

CONTEXTS=('clean',)+TARGETS


def state_digest(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state)).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','t007r','t008','t009','t010','output','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base,data,t007,t008,t009,t010,out=[Path(v).resolve() for v in (a.baseline,a.data_root,a.t007r,a.t008,a.t009,a.t010,a.output)]
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    def write(name,obj):(out/name).write_text(json.dumps(obj,indent=2))
    expected={'src/adaptation/paired_affine_oracle_neutral.py':'a6876677a72fe5f147a5ae8e30de4866344df86872f15633f0169d9436c96c0d',
        'src/data/covariate.py':'297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174',
        'src/context/source_signature.py':'84542fb2381c23073c7504fcc0bfc06546eacffa04b86809372a5ed27d8ca695',
        'src/models/pfllib.py':'22acbc42950652b62dd01fba765ddc7204e2dba665643e47bb9dc2edf79162d3'}
    for path,digest in expected.items():assert sha(ROOT/path)==digest
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    assert sha(t007/'states.pt')=='0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a'
    assert sha(t008/'prototypes.json')=='9138f7849f92fbfa820cb7c5eaf815fc209dd3178eca64cf4cc02ab9522b6347'
    assert sha(t009/'natural_support_manifest.json')=='ba139b530bc231b13dc5f0fe9a370a25050bb74f1264eadc938472f56216afac'
    support=json.loads((t009/'natural_support_manifest.json').read_text());assert support['chosen_K']==20
    shutil.copyfile(t009/'natural_support_manifest.json',out/'natural_support_manifest.json')
    split=json.loads((data/'Cifar10/split_manifest.json').read_text())
    pools=json.loads((t007/'calibration_pools.json').read_text());micro=json.loads((t008/'microbatches.json').read_text())
    proto=json.loads((t008/'prototypes.json').read_text())
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    saved=torch.load(t007/'states.pt',map_location='cuda',weights_only=True)
    states={b:{'clean':new_state(model),**{c:saved[f'{b}|{c}|correct_pair'] for c in TARGETS}} for b in ('A','B')}
    hashes={b:{c:state_digest(st) for c,st in bank.items()} for b,bank in states.items()}
    assert all(torch.isfinite(v).all() for bank in states.values() for st in bank.values() for v in st)
    project=Path('/home/wenchang/asdasdsad/wjq/TTFL')
    t011=project/'runs/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
    t013=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    t014=project/'results/t014_class_conditional_factorization'
    def load(path):return json.loads(path.read_text())
    def gz(name,obj):(out/name).write_bytes(gzip.compress(json.dumps(obj,separators=(',',':')).encode(),mtime=0))
    def csvout(name,rows):write_csv(out/name,[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
    assert sha(t009/'source_decisions.json')==load(t009/'selection_freeze.json')['decisions_sha256']
    assert sha(t011/'candidate_predictions.npz')==load(t011/'query_freeze.json')['predictions_sha256']
    assert sha(t014/'class_templates.json.gz')==load(t014/'policy_freeze.json')['templates_sha256']
    decisions=load(t009/'source_decisions.json');di={(r['client'],r['bank'],r['true_context']):r['selected'] for r in decisions};assert len(di)==1000
    oldconf=load(t009/'summary.json')['identification']
    for b in ('A','B'):
        matrix=[[sum(r['bank']==b and r['true_context']==t and r['selected']==c for r in decisions) for c in CONTEXTS] for t in CONTEXTS]
        assert matrix==next(r['confusion'] for r in oldconf if r['bank']==b)
    calibration_ids={r['original_id'] for pool in pools.values() for r in pool['samples']};query_ids={x for r in split['clients'].values() for x in r['test']}
    for i in range(100):
        ids0={r['original_id'] for r in support['clients'][str(i)]['selected']}
        assert len(ids0)==20 and not ids0&calibration_ids and not ids0&query_ids
    # Only historical aggregate offline receipts are opened for mandatory preflight; no individual target query outcome/label is loaded.
    def countmacro(r):return sum((Fraction(100*int(x),int(y)) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10
    receipts14=load(t014/'integer_count_receipts.json');metrics14={(r['salt'],r['bank'],r['target'],r['policy']):r for r in load(t014/'policy_metrics.json')}
    receipts13=load(t013/'integer_count_receipts.json');metrics13={(r['salt'],r['bank'],r['target'],r['policy']):r for r in load(t013/'crossfit_metrics.json')}
    checked=0
    for receipts,metrics,allowed in [(receipts14,metrics14,('class_context_only',)),(receipts13,metrics13,('zero','half_oracle'))]:
        for r in receipts:
            if r['policy'] in allowed:assert abs(float(countmacro(r))-metrics[r['salt'],r['bank'],r['target'],r['policy']]['macro_class'])<1e-10;checked+=1
    template_rows=json.loads(gzip.decompress((t014/'class_templates.json.gz').read_bytes()))['rows']
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in template_rows}
    assert len(templates)==1600 and all(all(x>0 for x in r['other99_class_denominators']) for r in template_rows)
    write('preflight.json',dict(checkpoint_states_support_context_predictions_templates_hashes_verified=True,T009_confusions_exact=True,offline_aggregate_regressions=checked,
        support_calibration_query_disjoint=True,template_target_keys_exclusion_provenance_unchanged=True,target_query_arrays_labels_opened=False))
    mixtures=[];source_logits={};forward_calls=0
    for cid in range(100):
        sr=support['clients'][str(cid)]['selected'];ids0=[r['original_id'] for r in sr];indices=[r['train_index'] for r in sr]
        x=torch.as_tensor(read_data('Cifar10',cid,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for context in CONTEXTS:
            tx=corrupt(x,context,ids0,cid)
            for bank in ('A','B'):
                predicted=di[cid,bank,context];logits={}
                for c in dict.fromkeys(('clean',context,predicted)):
                    with torch.no_grad():z=model(tx,states[bank][c])
                    assert torch.isfinite(z).all() and state_hash(model)==initial and state_digest(states[bank][c])==hashes[bank][c]
                    logits[c]=z;source_logits[f'{cid}|{bank}|{context}|{c}']=z.cpu().numpy();forward_calls+=1
                mix=mixtures_from_logits(logits,context,predicted)
                assert all(abs(sum(v)-1)<1e-12 and all(np.isfinite(y) and y>=0 for y in v) for v in mix.values())
                if context=='clean':assert mix['pi_zero_soft']==mix['pi_oracle_state_soft']
                if predicted=='clean':assert mix['pi_zero_soft']==mix['pi_source_state_soft']
                mixtures.append(dict(client=cid,bank=bank,true_context=context,source_context=predicted,**mix))
        if cid%20==19:print('T015_SUPPORT_DONE',cid,flush=True)
    mi={(r['client'],r['bank'],r['true_context']):r for r in mixtures};salts=('T013-S0','T013-S1','T013-S2','T013-S3')
    def choice_row(cid,bank,context,salt,train,policy,pi,ctx):
        selected,arg,values=choose_utility(pi,templates[salt,bank,train,cid][CONTEXTS.index(ctx)])
        return dict(client=cid,bank=bank,target=context,salt=salt,train_half=train,eval_half=1-train,policy=policy,template_context=ctx,
            selected=CONTEXTS[selected],argmax=[CONTEXTS[x] for x in arg],utility=[f'{x.numerator}/{x.denominator}' for x in values])
    phaseA=[]
    for r in mixtures:
        cid,bank,context,source=r['client'],r['bank'],r['true_context'],r['source_context']
        for salt in salts:
            for train in (0,1):
                for policy,pi,ctx in [('P01',r['pi_oracle_state_soft'],context),('P11',r['pi_source_state_soft'],source),('zero_soft',r['pi_zero_soft'],source),('zero_hard',r['pi_zero_hard'],source),('uniform',[Fraction(1,10)]*10,source)]:
                    phaseA.append(choice_row(cid,bank,context,salt,train,policy,pi,ctx))
    gz('source_mixtures.json.gz',mixtures);gz('phaseA_policy_choices.json.gz',phaseA);csvout('phaseA_policy_choices.csv',phaseA)
    pp=out/'phaseA_policy_choices.csv';pp.with_suffix('.csv.gz').write_bytes(gzip.compress(pp.read_bytes(),mtime=0))
    csvout('context_decisions.csv',[dict(client=r['client'],bank=r['bank'],true_context=r['true_context'],source_context=r['source_context']) for r in mixtures])
    np.savez_compressed(out/'support_logits.npz',**source_logits)
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),mixtures_sha256=sha(out/'source_mixtures.json.gz'),choices_sha256=sha(out/'phaseA_policy_choices.json.gz'),logits_sha256=sha(out/'support_logits.npz'),choices=len(phaseA),support_labels_opened=False,target_query_outcomes_opened=False)
    write('phaseA_freeze.json',freeze);print('T015_PHASE_A_FROZEN',json.dumps(freeze),flush=True)
    # P00/P10 cannot be reached without the verified Phase-A receipt; only now access support y.
    require_phaseA_freeze(out)
    truth=[];truepi={}
    for cid in range(100):
        indices=[r['train_index'] for r in support['clients'][str(cid)]['selected']]
        labels=np.array(read_data('Cifar10',cid,is_train=True)['y'])[indices];counts=np.bincount(labels,minlength=10)
        truepi[cid]=[Fraction(int(n),20) for n in counts];truth.append(dict(client=cid,counts=counts.tolist(),K=20,pi=[float(v) for v in truepi[cid]]))
    write('support_true_composition.json',truth);privileged=[]
    for r in mixtures:
        for salt in salts:
            for train in (0,1):
                for policy,ctx in [('P00',r['true_context']),('P10',r['source_context'])]:privileged.append(choice_row(r['client'],r['bank'],r['true_context'],salt,train,policy,truepi[r['client']],ctx))
    gz('privileged_policy_choices.json.gz',privileged);csvout('privileged_policy_choices.csv',privileged)
    pp=out/'privileged_policy_choices.csv';pp.with_suffix('.csv.gz').write_bytes(gzip.compress(pp.read_bytes(),mtime=0))
    write('phaseB_freeze.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),choices_sha256=sha(out/'privileged_policy_choices.json.gz'),support_truth_sha256=sha(out/'support_true_composition.json'),choices=len(privileged),target_query_outcomes_opened=False))
    print('T015_PHASE_B_FROZEN',flush=True)
    # Target query arrays and half counts are opened only after both policy freezes.
    predictions=np.load(t011/'candidate_predictions.npz');halves=load(t013/'query_halves.json');hc=np.load(t013/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];hits=cc.sum(-1);n=ct.sum(-1);totals=ct[0].sum((0,1))
    assert sha(t013/'half_integer_counts.npz')==load(t013/'policy_freeze.json')['counts_sha256']
    # Recheck reused template target exclusion from the now-available half counts.
    from src.context.class_factorization import class_templates
    for r in template_rows:
        si=salts.index(r['salt']);bi=('A','B').index(r['bank']);train=r['train_half'];i=r['target_client']
        assert class_templates(cc[si,bi,:,:,train,:,:],ct[si,:,train,:],i)==templates[r['salt'],r['bank'],train,i]
    allchoices=phaseA+privileged;ci={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):CONTEXTS.index(r['selected']) for r in allchoices};policies=('P00','P01','P10','P11','zero_soft','zero_hard','uniform');metrics=[];counts_receipts=[];halfregrets=[]
    for si,salt in enumerate(salts):
        for bi,bank in enumerate(('A','B')):
            for ti,target in enumerate(CONTEXTS):
                for policy in policies:
                    correct=np.zeros(10,dtype=np.int64);client_acc=[];regrets=[];freq={c:0 for c in CONTEXTS};hitrate=0;gt2=0;gt5=0
                    for cid in range(100):
                        y=predictions[f'c{cid}_labels'];full_pred=np.empty_like(y);nc=0
                        for half in (0,1):
                            chosen=ci[salt,bank,cid,target,half,policy];c=CONTEXTS[chosen];idx=halves[salt][str(cid)][half]['indices'];pr=predictions[f'{bank}|{target}|{c}|c{cid}'];full_pred[idx]=pr[idx]
                            count=np.bincount(y[idx][y[idx]==pr[idx]],minlength=10);assert np.array_equal(count,cc[si,bi,cid,ti,half,chosen])
                            correct+=count;nc+=int(count.sum());freq[c]+=1;reg=Fraction(100*(int(hits[si,bi,cid,ti,half].max())-int(count.sum())),int(n[si,cid,half]));regrets.append(float(reg));hitrate+=reg==0;gt2+=reg>2;gt5+=reg>5
                            halfregrets.append(dict(salt=salt,bank=bank,client=cid,target=target,half=half,policy=policy,selected=c,regret=float(reg)))
                        assert nc==int((full_pred==y).sum());client_acc.append(100*nc/len(y))
                    macro=sum((Fraction(100*int(x),int(y)) for x,y in zip(correct,totals)),Fraction())/10
                    metrics.append(dict(salt=salt,bank=bank,target=target,policy=policy,macro_class=float(macro),sample_weighted=100*int(correct.sum())/int(totals.sum()),macro_client=float(np.mean(client_acc)),state_choice_frequency=freq,oracle_hit_rate=hitrate/200,regret_median=float(np.median(regrets)),regret_p75=float(np.quantile(regrets,.75)),regret_p90=float(np.quantile(regrets,.9)),regret_gt2_fraction=gt2/200,regret_gt5_fraction=gt5/200))
                    counts_receipts.append(dict(salt=salt,bank=bank,target=target,policy=policy,class_correct=correct.tolist(),class_total=totals.tolist()))
    write('policy_metrics.json',metrics);csvout('policy_metrics.csv',metrics);write('integer_count_receipts.json',counts_receipts);gz('half_regrets.json.gz',halfregrets)
    shutil.copyfile(t009/'skew_labels_receipt.json',out/'frozen_skew_labels_receipt.json')
    assert sha(out/'source_mixtures.json.gz')==freeze['mixtures_sha256'] and sha(out/'phaseA_policy_choices.json.gz')==freeze['choices_sha256']
    write('verification.json',dict(all_frozen_hashes_verified=True,T009_context_confusions_exact=True,T014_offline_aggregate_regressions=checked,
        support_query_calibration_ids_disjoint=True,phaseA_no_support_labels_no_target_query_outcomes=True,phaseB_privileged_choices_before_query=True,
        model_states_unchanged_every_support_forward=True,forward_calls=forward_calls,query_forward_calls=0,mixtures_normalized_finite_nonnegative=True,
        clean_zero_mixture_branches_exact=True,T014_all1600_template_exclusions_reconstructed_after_freeze=True,all280_metrics_per_example_integer_reconstructed=True,no_new_fit_or_tuning=True))
    write('metadata.json',dict(code_commit=a.commit,gpu=torch.cuda.get_device_name(),seconds=time.time()-start,checkpoint_sha256=CHECKPOINT_SHA,states_sha256=sha(t007/'states.pt'),
        context_decisions_sha256=sha(t009/'source_decisions.json'),support_manifest_sha256=sha(t009/'natural_support_manifest.json'),templates_sha256=sha(t014/'class_templates.json.gz'),forward_calls=forward_calls))
    print('T015_DONE',json.dumps(dict(seconds=time.time()-start,forward_calls=forward_calls)),flush=True)

if __name__=='__main__':main()
