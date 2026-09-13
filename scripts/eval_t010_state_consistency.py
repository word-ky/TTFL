"""Frozen candidate states scored on source pixels, then mixed query evaluation."""
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
from src.context.state_consistency import post_state_signature,choose_state
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
    for name in ('baseline','data-root','t007r','t008','t009','output','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base,data,t007,t008,t009,out=[Path(v).resolve() for v in (a.baseline,a.data_root,a.t007r,a.t008,a.t009,a.output)]
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
    refs={b:[(torch.tensor(r['mu'],dtype=torch.float64,device='cuda'),torch.tensor(r['sigma'],dtype=torch.float64,device='cuda')) for r in p['reference']] for b,p in proto.items()}
    score_calls=0;zero_checks=0
    def score(source,bank,true):
        nonlocal score_calls,zero_checks
        historical=source_signature(model,source,refs[bank]);scores={}
        for context in CONTEXTS:
            psi=post_state_signature(model,source,states[bank][context],refs[bank])
            scores[context]=float(psi.square().sum())
            assert state_hash(model)==initial and state_digest(states[bank][context])==hashes[bank][context]
            score_calls+=1
            if context=='clean':
                assert torch.equal(psi,historical) and float(historical.square().sum())==scores[context]
                zero_checks+=1
        choice=choose_state(scores)
        return dict(**{key:v for key,v in choice.items() if key!='ranking'},
            true_rank=choice['ranking'].index(true)+1 if true in CONTEXTS else None,
            **{f'J_{c}':v for c,v in scores.items()})
    poolpixels={}
    for b,pool in pools.items():
        poolpixels[b]=torch.stack([torch.as_tensor(read_data('Cifar10',r['client_id'],is_train=True)['x'][r['train_index']],dtype=torch.float32) for r in pool['samples']]).cuda()
    balanced=[];ood=[]
    for bank,held in (('A','B'),('B','A')):
        for context in CONTEXTS+('noise_image',):
            tx=torch.cat([corrupt(poolpixels[held][i:i+1],context,[r['original_id']],r['client_id']) for i,r in enumerate(pools[held]['samples'])])
            for batch in micro[held]:
                r=dict(bank=bank,heldout=held,batch=batch['batch'],true_context=context,**score(tx[batch['indices']],bank,context))
                (ood if context=='noise_image' else balanced).append(r)
    write_csv(out/'balanced_sanity.csv',balanced);write_csv(out/'ood_state_scores.csv',ood)
    print('T010_BALANCED_SANITY_DONE',flush=True)
    selections=[]
    for cid in range(100):
        rows=support['clients'][str(cid)]['selected'];ids=[r['original_id'] for r in rows];indices=[r['train_index'] for r in rows]
        x=torch.as_tensor(read_data('Cifar10',cid,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for context in CONTEXTS:
            tx=corrupt(x,context,ids,cid)
            for bank in ('A','B'):selections.append(dict(client=cid,bank=bank,true_context=context,**score(tx,bank,context)))
        if cid%20==19:print('T010_SOURCE_CLIENT_DONE',cid,flush=True)
    write('state_scores.json',selections);write_csv(out/'state_scores.csv',selections)
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),decisions_sha256=sha(out/'state_scores.json'),
        count=1000,query_results_opened=False,label_statistics_opened=False)
    write('selection_freeze.json',freeze);print('T010_SELECTIONS_FROZEN',json.dumps(freeze),flush=True)
    # All support scoring is complete. Only now open historical query predictions and selected policies.
    old7rows=json.loads((t007/'raw_records.json').read_text());old7summary=json.loads((t007/'summary.json').read_text())
    old9rows=json.loads((t009/'mixed_query_records.json').read_text());old9summary=json.loads((t009/'summary.json').read_text())
    p7=np.load(t007/'predictions.npz');p9=np.load(t009/'mixed_predictions.npz')
    i7={(r['client'],r['pool'],r['target'],r['context']):r for r in old7rows};i9={(r['client'],r['bank'],r['target']):r for r in old9rows}
    m7={(r['pool'],r['target'],r['context']):r for r in old7summary['rows']};g7={(r['pool'],r['target']):r for r in old7summary['gates']}
    sel={(r['client'],r['bank'],r['true_context']):r for r in selections};mixed={};query_rows=[];new_evals=0;reuse7=0;reuse9=0
    for cid in range(100):
        y=p7[f'c{cid}_labels'];mixed[f'c{cid}_labels']=y;query=None
        for target in CONTEXTS:
            for bank in ('A','B'):
                choice=sel[cid,bank,target]['selected'];oldkey=None;origin=None
                if choice==i9[cid,bank,target]['selected']:
                    oldkey=i9[cid,bank,target]['prediction_key'];pred=p9[oldkey];origin='T009';reuse9+=1
                else:
                    if target=='clean' and choice=='clean':oldkey=f'c{cid}_clean'
                    elif target!='clean':
                        ctx='none' if choice=='clean' else ('correct_pair' if choice==target else ('wrong_alt_source' if choice==WRONG_ALT[target] else None))
                        if ctx:oldkey=i7[cid,bank,target,ctx]['prediction_key']
                    if oldkey is not None:pred=p7[oldkey];origin='T007R';reuse7+=1
                    else:
                        if query is None:query=read_data('Cifar10',cid,is_train=False)
                        assert np.array_equal(query['y'],y)
                        x=torch.as_tensor(query['x'],dtype=torch.float32).cuda();qy=torch.as_tensor(y,dtype=torch.long).cuda()
                        tx=corrupt(x,target,split['clients'][str(cid)]['test'],cid)
                        _,pred=evaluate(model,tx,qy,states[bank][choice]);pred=pred.numpy();new_evals+=1;origin='new'
                        assert state_hash(model)==initial
                key=f'{bank}|{target}|c{cid}';mixed[key]=pred
                query_rows.append(dict(client=cid,bank=bank,target=target,selected=choice,prediction_key=key,
                    reused_from=origin,reused_key=oldkey,n_query=len(y),accuracy=100*float(np.mean(pred==y)),
                    class_total=np.bincount(y,minlength=10).tolist(),class_correct=np.bincount(y[pred==y],minlength=10).tolist()))
    np.savez_compressed(out/'mixed_predictions.npz',**mixed);write('mixed_query_records.json',query_rows)
    write('query_freeze.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),predictions_sha256=sha(out/'mixed_predictions.npz')))
    all_y=np.concatenate([p7[f'c{i}_labels'] for i in range(100)]);totals=np.bincount(all_y,minlength=10)
    def aggregate(preds):
        correct=np.bincount(all_y[preds==all_y],minlength=10)
        return float(np.mean(100*correct/totals)),correct
    clean_macro,_=aggregate(np.concatenate([p7[f'c{i}_clean'] for i in range(100)]))
    old_error=0.
    for bank in ('A','B'):
        for target in TARGETS:
            for ctx in ('none','correct_pair','wrong_alt_source'):
                value,_=aggregate(np.concatenate([p7[i7[i,bank,target,ctx]['prediction_key']] for i in range(100)]))
                old_error=max(old_error,abs(value-m7[bank,target,ctx]['macro_class']))
    for r in old9summary['retrieval']+old9summary['clean_safety']:
        target=r.get('target','clean');value,_=aggregate(np.concatenate([p9[f"{r['bank']}|{target}|c{i}"] for i in range(100)]))
        old_error=max(old_error,abs(value-r.get('selected_mixed',r.get('selected'))))
    assert old_error<1e-10
    retrieval=[];cleans=[];count_receipts=[]
    for bank in ('A','B'):
        for target in CONTEXTS:
            rows=[r for r in query_rows if (r['bank'],r['target'])==(bank,target)]
            value,correct=aggregate(np.concatenate([mixed[f'{bank}|{target}|c{i}'] for i in range(100)]))
            extra=dict(sample_weighted=float(100*correct.sum()/totals.sum()),macro_client=float(np.mean([r['accuracy'] for r in rows])))
            count_receipts.append(dict(bank=bank,target=target,total=totals.tolist(),correct=correct.tolist(),macro_class=value,**extra))
            if target=='clean':cleans.append(dict(bank=bank,none=clean_macro,selected=value,delta_pp=value-clean_macro,passed=clean_safety_gate(clean_macro,value),**extra))
            else:
                none=m7[bank,target,'none']['macro_class'];oracle=m7[bank,target,'correct_pair']['macro_class'];wrong=m7[bank,target,'wrong_alt_source']['macro_class'];tau=g7[bank,target]['tau_pp']
                retrieval.append(dict(bank=bank,target=target,none=none,oracle=oracle,selected_mixed=value,wrong_alt=wrong,tau_pp=tau,
                    **extra,**retrieval_gate(none,oracle,value,wrong,tau)))
    idrows=[];balanced_summary=[]
    for bank in ('A','B'):
        group=[r for r in selections if r['bank']==bank]
        matrix=[[sum(r['true_context']==t and r['selected']==c for r in group) for c in CONTEXTS] for t in CONTEXTS]
        correct=[matrix[i][i] for i in range(5)]
        idrows.append(dict(bank=bank,correct=sum(correct),accuracy=sum(correct)/5,per_context_accuracy=dict(zip(CONTEXTS,correct)),
            passed=natural_id_gate(correct),confusion=matrix,median_true_rank=float(np.median([r['true_rank'] for r in group])),
            fraction_true_rank_1=float(np.mean([r['true_rank']==1 for r in group])),fraction_true_rank_le2=float(np.mean([r['true_rank']<=2 for r in group]))))
        write_csv(out/f'confusion_{bank}.csv',[dict(true_context=c,**dict(zip(CONTEXTS,matrix[i]))) for i,c in enumerate(CONTEXTS)])
        balanced_group=[r for r in balanced if r['bank']==bank]
        bm=[[sum(r['true_context']==t and r['selected']==c for r in balanced_group) for c in CONTEXTS] for t in CONTEXTS]
        balanced_summary.append(dict(bank=bank,correct=sum(bm[i][i] for i in range(5)),total=20,confusion=bm))
    # Post-query audit only: true-state and selected-state task utility vs zero.
    alignment=[]
    for r in selections:
        cid,bank,target=r['client'],r['bank'],r['true_context'];y=p7[f'c{cid}_labels']
        zero=p7[f'c{cid}_clean'] if target=='clean' else p7[i7[cid,bank,target,'none']['prediction_key']]
        true=zero if target=='clean' else p7[i7[cid,bank,target,'correct_pair']['prediction_key']]
        for role,context,pred in [('selected',r['selected'],mixed[f'{bank}|{target}|c{cid}']),('true',target,true)]:
            alignment.append(dict(client=cid,bank=bank,target=target,role=role,candidate=context,
                delta_J=r['J_clean']-r[f'J_{context}'],delta_accuracy=100*float(np.mean(pred==y)-np.mean(zero==y)),
                zero_accuracy=100*float(np.mean(zero==y)),candidate_accuracy=100*float(np.mean(pred==y))))
    write_csv(out/'restoration_task_rows.csv',alignment)
    # Existing frozen skew statistics are copied for reporting, never recomputed or passed to scores.
    shutil.copyfile(t009/'skew_labels_receipt.json',out/'frozen_skew_labels_receipt.json')
    write_csv(out/'retrieval.csv',retrieval);write_csv(out/'clean_safety.csv',cleans);write('integer_count_receipts.json',count_receipts)
    joint={t:all(r['passed'] for r in retrieval if r['target']==t) for t in TARGETS}
    scid=all(r['passed'] for r in idrows);scret=sum(joint.values())>=3 and all(r['passed'] for r in cleans)
    assert sha(out/'state_scores.json')==freeze['decisions_sha256']
    assert all(state_digest(states[b][c])==hashes[b][c] for b in states for c in CONTEXTS)
    metadata=dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,frozen_code_hashes=expected,
        support_manifest_sha256=sha(t009/'natural_support_manifest.json'),states_sha256=sha(t007/'states.pt'),
        prototypes_sha256=sha(t008/'prototypes.json'),helper_sha256=sha(ROOT/'src/context/state_consistency.py'),
        candidate_memory_hashes=hashes,gpu=torch.cuda.get_device_name(),seconds=time.time()-start)
    verify=dict(frozen_hashes_unchanged=True,score_calls=score_calls,zero_regression_checks=zero_checks,
        zero_psi_max_error=0.,zero_J_max_error=0.,all_candidate_states_unchanged=True,model_unchanged_every_episode=True,
        finite_scores_states=True,support_ids_exact_T009=True,no_label_query_inputs_to_score=True,
        selections_before_query=True,no_state_fits=True,no_query_tuning=True,historical_metrics_max_error_pp=old_error,
        query_reuse_T007R=reuse7,query_reuse_T009=reuse9,new_query_evaluations=new_evals)
    write('metadata.json',metadata);write('verification.json',verify)
    write('summary.json',dict(identification=idrows,balanced_sanity=balanced_summary,retrieval=retrieval,clean_safety=cleans,
        joint_shift_pass=joint,SC_ID_A=scid,SC_RET_A=scret))
    print('T010_DONE',json.dumps(dict(SC_ID_A=scid,SC_RET_A=scret,identification=idrows,balanced=balanced_summary,retrieval=retrieval,clean=cleans,verification=verify)),flush=True)


if __name__=='__main__':main()
