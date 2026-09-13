"""Natural client context decisions frozen before query/label-skew analysis."""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.natural_support import natural_support_manifest
from src.context.source_signature import source_signature,nearest_prototype
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import evaluate,state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from scripts.eval_t007r import POOL_SHA
from scripts.eval_t008_source_identifiability import retrieval_gate
from scripts.analyze_prior_receipts import ranks

CONTEXTS=('clean',)+TARGETS


def natural_id_gate(correct):return sum(correct)>=450 and min(correct)>=80


def clean_safety_gate(clean,selected):return selected>=clean-.5


def spearman(x,y):
    x,y=ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','t007r','t008','output','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base,data,old,t008,out=[Path(v).resolve() for v in (a.baseline,a.data_root,a.t007r,a.t008,a.output)]
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2))
    assert sha(old/'calibration_pools.json')==POOL_SHA
    pools=json.loads((old/'calibration_pools.json').read_text())
    split=json.loads((data/'Cifar10/split_manifest.json').read_text())
    calibration={r['original_id'] for pool in pools.values() for r in pool['samples']}
    support=natural_support_manifest(split['clients'],calibration)
    support['frozen_at']=datetime.now(timezone.utc).isoformat()
    write('natural_support_manifest.json',support)
    assert support['chosen_K'] is not None, 'No common K>=16: eligible counts persisted; stop'
    k=support['chosen_K'];sizes=[k]+([20] if support['secondary_K'] else [])
    allquery={sid for r in split['clients'].values() for sid in r['test']}
    assert not {r['original_id'] for c in support['clients'].values() for r in c['selected']}.intersection(calibration|allquery)
    support_sha=sha(out/'natural_support_manifest.json')
    # No image tensors or labels were loaded before the support manifest was persisted.
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    expected={'src/adaptation/paired_affine_oracle_neutral.py':'a6876677a72fe5f147a5ae8e30de4866344df86872f15633f0169d9436c96c0d',
        'src/data/covariate.py':'297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174',
        'src/context/source_signature.py':'84542fb2381c23073c7504fcc0bfc06546eacffa04b86809372a5ed27d8ca695'}
    for path,digest in expected.items():assert sha(ROOT/path)==digest
    assert sha(t008/'prototypes.json')=='9138f7849f92fbfa820cb7c5eaf815fc209dd3178eca64cf4cc02ab9522b6347'
    assert sha(old/'states.pt')=='0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a'
    proto=json.loads((t008/'prototypes.json').read_text())
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    refs={b:[(torch.tensor(r['mu'],dtype=torch.float64,device='cuda'),torch.tensor(r['sigma'],dtype=torch.float64,device='cuda')) for r in p['reference']] for b,p in proto.items()}
    prototypes={b:{c:torch.tensor(p['prototypes'][c]['vector'],dtype=torch.float64,device='cuda') for c in CONTEXTS} for b,p in proto.items()}
    decisions=[]
    for cid in range(100):
        rows=support['clients'][str(cid)]['selected'];indices=[r['train_index'] for r in rows];ids=[r['original_id'] for r in rows]
        # Access only x; support y is not inspected until the final audit phase below.
        x=torch.as_tensor(read_data('Cifar10',cid,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for context in CONTEXTS:
            tx=corrupt(x,context,ids,cid)
            for size in sizes:
                for bank in ('A','B'):
                    phi=source_signature(model,tx[:size],refs[bank]);choice=nearest_prototype(phi,prototypes[bank])
                    signed=min(v for c,v in choice['distances'].items() if c!=context)-choice['distances'][context]
                    decisions.append(dict(client=cid,bank=bank,K=size,primary=size==k,true_context=context,
                        **{key:v for key,v in choice.items() if key!='distances'},signed_true_margin=signed,
                        **{f'd_{c}':v for c,v in choice['distances'].items()},signature=phi.cpu().tolist()))
                    assert state_hash(model)==initial
    write('source_decisions.json',decisions)
    write_csv(out/'identification.csv',[{key:v for key,v in r.items() if key!='signature'} for r in decisions])
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),manifest_sha256=support_sha,
        decisions_sha256=sha(out/'source_decisions.json'),primary_decisions=1000,total_decisions=len(decisions),
        query_metrics_opened=False,support_label_statistics_computed=False)
    write('selection_freeze.json',freeze);print('T009_SELECTIONS_FROZEN',json.dumps(freeze),flush=True)
    idsummaries=[]
    for size in sizes:
        for bank in ('A','B'):
            group=[r for r in decisions if r['bank']==bank and r['K']==size]
            matrix=[[sum(r['true_context']==c and r['selected']==p for r in group) for p in CONTEXTS] for c in CONTEXTS]
            counts=[matrix[i][i] for i in range(5)]
            idsummaries.append(dict(bank=bank,K=size,primary=size==k,correct=sum(counts),total=500,accuracy=sum(counts)/5,
                per_context_accuracy=dict(zip(CONTEXTS,counts)),passed=natural_id_gate(counts),confusion=matrix,
                signed_margin_mean=float(np.mean([r['signed_true_margin'] for r in group])),
                signed_margin_median=float(np.median([r['signed_true_margin'] for r in group])),
                nearest_distance_quantiles=dict(zip(('p50','p90','p95','max'),map(float,np.percentile([r['distance'] for r in group],[50,90,95,100]))))))
            write_csv(out/(f'confusion_{bank}.csv' if size==k else f'confusion_{bank}_K20.csv'),
                [dict(true_context=c,**dict(zip(CONTEXTS,matrix[i]))) for i,c in enumerate(CONTEXTS)])
    # Query metrics/labels become accessible only after both banks and sizes are frozen.
    saved=torch.load(old/'states.pt',map_location='cuda',weights_only=True)
    assert all(torch.isfinite(v).all() for state in saved.values() for v in state)
    states={b:{'clean':new_state(model),**{c:saved[f'{b}|{c}|correct_pair'] for c in TARGETS}} for b in ('A','B')}
    old_summary=json.loads((old/'summary.json').read_text());oldrows=json.loads((old/'raw_records.json').read_text())
    pred_old=np.load(old/'predictions.npz')
    oi={(r['client'],r['pool'],r['target'],r['context']):r for r in oldrows}
    oldmetric={(r['pool'],r['target'],r['context']):r for r in old_summary['rows']}
    gates={(r['pool'],r['target']):r for r in old_summary['gates']}
    # Integer-count verification of all frozen baseline/oracle/wrong metrics, no extra model evaluation.
    old_error=0.
    for bank in ('A','B'):
        for target in TARGETS:
            for context in ('none','correct_pair','wrong_alt_source'):
                total=np.zeros(10,dtype=np.int64);correct=np.zeros(10,dtype=np.int64)
                for cid in range(100):
                    y=pred_old[f'c{cid}_labels'];pr=pred_old[oi[cid,bank,target,context]['prediction_key']]
                    total+=np.bincount(y,minlength=10);correct+=np.bincount(y[pr==y],minlength=10)
                old_error=max(old_error,abs(float(np.mean(100*correct/total))-oldmetric[bank,target,context]['macro_class']))
    assert old_error<1e-10
    primary=[r for r in decisions if r['primary']];decision={(r['client'],r['bank'],r['true_context']):r['selected'] for r in primary}
    predictions={};query_rows=[];new_evals=0;reused=0
    for cid in range(100):
        y=pred_old[f'c{cid}_labels'];predictions[f'c{cid}_labels']=y
        raw_query=None
        for target in CONTEXTS:
            for bank in ('A','B'):
                choice=decision[cid,bank,target];old_context=None;old_key=None
                if target=='clean' and choice=='clean':old_key=f'c{cid}_clean'
                elif target!='clean':
                    old_context='none' if choice=='clean' else ('correct_pair' if choice==target else ('wrong_alt_source' if choice==WRONG_ALT[target] else None))
                    if old_context:old_key=oi[cid,bank,target,old_context]['prediction_key']
                if old_key is not None:pr=pred_old[old_key];reused+=1
                else:
                    if raw_query is None:raw_query=read_data('Cifar10',cid,is_train=False)
                    assert np.array_equal(raw_query['y'],y)
                    x=torch.as_tensor(raw_query['x'],dtype=torch.float32).cuda();qy=torch.as_tensor(y,dtype=torch.long).cuda()
                    tx=corrupt(x,target,split['clients'][str(cid)]['test'],cid)
                    _,pr=evaluate(model,tx,qy,states[bank][choice]);pr=pr.numpy();new_evals+=1
                    assert state_hash(model)==initial
                key=f'{bank}|{target}|c{cid}';predictions[key]=pr
                query_rows.append(dict(client=cid,bank=bank,target=target,selected=choice,prediction_key=key,reused_old_key=old_key,
                    n_query=len(y),accuracy=100*float(np.mean(pr==y)),
                    class_total=np.bincount(y,minlength=10).tolist(),class_correct=np.bincount(y[pr==y],minlength=10).tolist()))
    retrieval=[];clean_results=[];count_receipts=[]
    clean_y=np.concatenate([pred_old[f'c{i}_labels'] for i in range(100)])
    clean_p=np.concatenate([pred_old[f'c{i}_clean'] for i in range(100)])
    clean_macro=float(np.mean([100*np.mean(clean_p[clean_y==c]==c) for c in range(10)]))
    for bank in ('A','B'):
        for target in CONTEXTS:
            group=[r for r in query_rows if (r['bank'],r['target'])==(bank,target)]
            total=np.sum([r['class_total'] for r in group],axis=0);correct=np.sum([r['class_correct'] for r in group],axis=0)
            macro=float(np.mean(100*correct/total));weighted=float(100*correct.sum()/total.sum());client=float(np.mean([r['accuracy'] for r in group]))
            receipt=dict(bank=bank,target=target,class_total=total.tolist(),class_correct=correct.tolist(),macro_class=macro,sample_weighted=weighted,macro_client=client)
            count_receipts.append(receipt)
            if target=='clean':
                clean_results.append(dict(bank=bank,none=clean_macro,selected=macro,delta_pp=macro-clean_macro,passed=clean_safety_gate(clean_macro,macro),sample_weighted=weighted,macro_client=client))
            else:
                none=oldmetric[bank,target,'none']['macro_class'];oracle=oldmetric[bank,target,'correct_pair']['macro_class'];wrong=oldmetric[bank,target,'wrong_alt_source']['macro_class'];tau=gates[bank,target]['tau_pp']
                retrieval.append(dict(bank=bank,target=target,none=none,oracle=oracle,selected_mixed=macro,wrong_alt=wrong,tau_pp=tau,
                    sample_weighted=weighted,macro_client=client,**retrieval_gate(none,oracle,macro,wrong,tau)))
    write('mixed_query_records.json',query_rows);np.savez_compressed(out/'mixed_predictions.npz',**predictions)
    write('integer_count_receipts.json',count_receipts);write_csv(out/'retrieval.csv',retrieval);write_csv(out/'clean_safety.csv',clean_results)
    # Post-freeze analysis only: first access to support y and its composition statistics.
    skew=[]
    for cid in range(100):
        indices=[r['train_index'] for r in support['clients'][str(cid)]['selected']]
        y=np.asarray(read_data('Cifar10',cid,is_train=True)['y'])[indices];counts=np.bincount(y,minlength=10);prob=counts/counts.sum()
        entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))/np.log(10))
        skew.append(dict(client=cid,K=k,normalized_entropy=entropy,max_class_fraction=float(prob.max()),represented_classes=int(np.sum(counts>0)),class_counts=counts.tolist()))
    # Equal-count entropy quartiles; deterministic client-ID tie break, no cutoffs chosen from outcomes.
    order=sorted(range(100),key=lambda cid:(skew[cid]['normalized_entropy'],cid))
    for rank,cid in enumerate(order):skew[cid]['entropy_quartile']=rank//25+1
    quartiles=[];correlations=[]
    for bank in ('A','B'):
        for context in CONTEXTS:
            group=[r for r in primary if (r['bank'],r['true_context'])==(bank,context)]
            for quartile in range(1,5):
                part=[r for r in group if skew[r['client']]['entropy_quartile']==quartile]
                quartiles.append(dict(bank=bank,context=context,quartile=quartile,clients=len(part),
                    accuracy=100*sum(r['selected']==context for r in part)/len(part),
                    median_signed_margin=float(np.median([r['signed_true_margin'] for r in part])),
                    entropy_min=min(skew[r['client']]['normalized_entropy'] for r in part),entropy_max=max(skew[r['client']]['normalized_entropy'] for r in part)))
            for feature in ('normalized_entropy','max_class_fraction','represented_classes'):
                correlations.append(dict(bank=bank,context=context,feature=feature,spearman=spearman([skew[r['client']][feature] for r in group],[r['signed_true_margin'] for r in group])))
    worst=[dict(**{key:val for key,val in r.items() if key!='signature'},skew=skew[r['client']]) for r in sorted(primary,key=lambda r:(r['signed_true_margin'],r['bank'],r['client'],r['true_context']))[:10]]
    write_csv(out/'skew_audit.csv',[{key:val for key,val in r.items() if key!='class_counts'} for r in skew])
    write('skew_labels_receipt.json',skew);write('skew_summary.json',dict(quartiles=quartiles,correlations=correlations,worst_ten=worst,quartile_rule='Sort entropy then client ID, consecutive groups of25; descriptive only'))
    ood=list(csv.DictReader((t008/'ood_noise.csv').open()));distance={}
    for bank in ('A','B'):
        known=[r['distance'] for r in primary if r['bank']==bank];noise=[float(r['distance']) for r in ood if r['bank']==bank]
        distance[bank]=dict(known=dict(zip(('p50','p90','p95','max'),map(float,np.percentile(known,[50,90,95,100])))),
            t008_noise=dict(min=min(noise),median=float(np.median(noise)),max=max(noise)))
    write('distance_ood_comparison.json',distance)
    joint={t:all(r['passed'] for r in retrieval if r['target']==t) for t in TARGETS}
    nid=all(r['passed'] for r in idsummaries if r['primary']);nret=sum(joint.values())>=3 and all(r['passed'] for r in clean_results)
    assert sha(out/'natural_support_manifest.json')==support_sha and sha(out/'source_decisions.json')==freeze['decisions_sha256']
    assert state_hash(model)==initial
    metadata=dict(code_commit=a.commit,K=k,secondary_K=support['secondary_K'],checkpoint_sha256=CHECKPOINT_SHA,
        frozen_code_hashes=expected,prototype_sha256=sha(t008/'prototypes.json'),bank_states_sha256=sha(old/'states.pt'),
        pool_sha256=POOL_SHA,support_manifest_sha256=support_sha,decision_sha256=freeze['decisions_sha256'],
        old_predictions_sha256=sha(old/'predictions.npz'),gpu=torch.cuda.get_device_name(),seconds=time.time()-start)
    verify=dict(frozen_hashes_unchanged=True,no_state_fits=True,support_pool_overlap=0,support_query_overlap=0,
        selection_labels_blind=True,manifest_before_features=True,primary_decisions=1000,total_decisions=len(decisions),
        all_selections_before_query_and_skew=True,same_ids_all_contexts_banks=True,source_only_signature_api=True,
        model_unchanged=True,all_finite=True,old_macroclass_max_error_pp=old_error,
        reused_query_client_evaluations=reused,new_query_client_evaluations=new_evals,
        no_query_tuning=True,manifest_decisions_unchanged=True)
    write('metadata.json',metadata);write('verification.json',verify)
    write('summary.json',dict(identification=idsummaries,retrieval=retrieval,clean_safety=clean_results,
        joint_retrieval_pass=joint,NID_A=nid,NRET_A=nret))
    print('T009_DONE',json.dumps(dict(K=k,NID_A=nid,NRET_A=nret,identification=idsummaries,retrieval=retrieval,clean_safety=clean_results,verification=verify)),flush=True)


if __name__=='__main__':main()
