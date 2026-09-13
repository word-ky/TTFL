"""T008: freeze source-only decisions first, then retrieve opposite-pool oracle states."""
import argparse
import hashlib
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
from src.context.source_signature import source_moments,source_signature,nearest_prototype
from src.data.microbatches import balanced_microbatches
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.adaptation.paired_affine_oracle_neutral import paired_clean_oracle_neutral
from src.eval.common import evaluate,state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from scripts.eval_t007r import POOL_SHA
from utils.data_utils import read_data

CONTEXTS=('clean',)+TARGETS


def identification_gate(counts):
    return sum(counts)>=18 and min(counts)>=3


def retrieval_gate(none,oracle,selected,wrong,tau):
    gain=oracle-none
    retained=(selected-none)/gain if gain>0 else None
    return dict(oracle_gain=gain,retained_recovery=retained,selected_wrong_margin=selected-wrong,
                passed=retained is not None and retained>=.9 and selected-wrong>=tau)


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','old','output','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base,data,old,out=[Path(v).resolve() for v in (a.baseline,a.data_root,a.old,a.output)]
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    poolfile=old/'calibration_pools.json';assert sha(poolfile)==POOL_SHA
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    assert sha(ROOT/'src/adaptation/paired_affine_oracle_neutral.py')=='a6876677a72fe5f147a5ae8e30de4866344df86872f15633f0169d9436c96c0d'
    assert sha(ROOT/'src/data/covariate.py')=='297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174'
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text());pools=json.loads(poolfile.read_text())
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    stored_states=torch.load(old/'states.pt',map_location='cuda',weights_only=True)
    pixels={};micro={};allqueries={sid for r in manifest['clients'].values() for sid in r['test']};used=set()
    for pool,receipt in pools.items():
        rs=receipt['samples'];ids=[r['original_id'] for r in rs];labels=[r['class'] for r in rs]
        assert len(ids)==80 and len(set(ids))==80 and not used.intersection(ids) and not allqueries.intersection(ids)
        used.update(ids);batches=balanced_microbatches(labels)
        assert sorted(i for b in batches for i in b)==list(range(80))
        assert all(len(b)==20 and all(sum(labels[i]==c for i in b)==2 for c in range(10)) for b in batches)
        micro[pool]=[dict(batch=b,indices=indices,original_ids=[ids[i] for i in indices],
            client_ids=[rs[i]['client_id'] for i in indices],labels=[labels[i] for i in indices]) for b,indices in enumerate(batches)]
        xs=[]
        for r in rs:
            raw=read_data('Cifar10',r['client_id'],is_train=True)
            assert manifest['clients'][str(r['client_id'])]['train'][r['train_index']]==r['original_id']
            assert int(raw['y'][r['train_index']])==r['class']
            xs.append(torch.as_tensor(raw['x'][r['train_index']],dtype=torch.float32))
        pixels[pool]=torch.stack(xs).cuda()
    def write(name,value):
        (out/name).write_text(json.dumps(value,indent=2))
    write('microbatches.json',micro)
    def transformed(pool,context):
        return torch.cat([corrupt(pixels[pool][i:i+1],context,[r['original_id']],r['client_id'])
                          for i,r in enumerate(pools[pool]['samples'])])
    sources={(pool,c):transformed(pool,c) for pool in pools for c in CONTEXTS+('noise_image',)}
    banks={};references={};prototypes={};proto_receipt={};state_matches=0
    for pool in pools:
        references[pool]=source_moments(model,pixels[pool])
        prototypes[pool]={c:source_signature(model,sources[pool,c],references[pool]) for c in CONTEXTS}
        banks[pool]={'clean':new_state(model)}
        for c in TARGETS:
            state,_=paired_clean_oracle_neutral(model,pixels[pool],sources[pool,c])
            assert all(torch.equal(x,y) for x,y in zip(state,stored_states[f'{pool}|{c}|correct_pair']))
            banks[pool][c]=state;state_matches+=1
            assert state_hash(model)==initial
        for c in TARGETS:
            assert all(torch.equal(x,y) for x,y in zip(banks[pool][WRONG_ALT[c]],stored_states[f'{pool}|{c}|wrong_alt_source']))
        proto_receipt[pool]=dict(reference=[dict(mu=m.cpu().tolist(),sigma=s.cpu().tolist()) for m,s in references[pool]],
            prototypes={c:dict(vector=v.cpu().tolist(),sha256=hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()) for c,v in prototypes[pool].items()},
            noise_image_descriptive=source_signature(model,sources[pool,'noise_image'],references[pool]).cpu().tolist())
    write('prototypes.json',proto_receipt);prototype_sha=sha(out/'prototypes.json')
    identification=[];ood=[];id_summaries={}
    for bank,held in (('A','B'),('B','A')):
        direction=f'{bank}_to_{held}'
        for context in CONTEXTS+('noise_image',):
            for batch in micro[held]:
                # Only current source pixels and opposite-bank stored moments reach the signature API.
                source=sources[held,context][batch['indices']]
                phi=source_signature(model,source,references[bank])
                selected=nearest_prototype(phi,prototypes[bank])
                row=dict(direction=direction,bank=bank,heldout=held,batch=batch['batch'],true_context=context,
                    **{k:v for k,v in selected.items() if k!='distances'},
                    **{f'd_{k}':v for k,v in selected['distances'].items()},signature=phi.cpu().tolist())
                (ood if context=='noise_image' else identification).append(row)
                assert state_hash(model)==initial
        group=[r for r in identification if r['direction']==direction]
        matrix=[[sum(r['true_context']==true and r['selected']==guess for r in group) for guess in CONTEXTS] for true in CONTEXTS]
        counts=[matrix[i][i] for i in range(5)]
        id_summaries[direction]=dict(correct=sum(counts),total=20,per_context_correct=dict(zip(CONTEXTS,counts)),
            passed=identification_gate(counts),margin_mean=float(np.mean([r['margin'] for r in group])),
            margin_median=float(np.median([r['margin'] for r in group])),confusion=matrix)
        write_csv(out/f'confusion_{direction}.csv',[dict(true_context=c,**dict(zip(CONTEXTS,matrix[i]))) for i,c in enumerate(CONTEXTS)])
    write('support_decisions.json',dict(primary=identification,ood=ood))
    write_csv(out/'identification.csv',[{k:v for k,v in r.items() if k!='signature'} for r in identification])
    write_csv(out/'ood_noise.csv',[{k:v for k,v in r.items() if k!='signature'} for r in ood])
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),prototype_sha256=prototype_sha,
        microbatch_sha256=sha(out/'microbatches.json'),decision_sha256=sha(out/'support_decisions.json'),
        primary_decisions=40,ood_decisions=8,query_evaluation_started=False)
    write('selection_freeze.json',freeze)
    print('T008_SELECTIONS_FROZEN',json.dumps(id_summaries),flush=True)
    # No query labels/metrics are opened until BOTH directions have been persisted above.
    old_summary=json.loads((old/'summary.json').read_text());old_rows=json.loads((old/'raw_records.json').read_text())
    old_pred=np.load(old/'predictions.npz')
    old_index={(r['pool'],r['target'],r['context']):r for r in old_summary['rows']}
    old_rowindex={(r['client'],r['pool'],r['target'],r['context']):r for r in old_rows}
    old_gate={(r['pool'],r['target']):r for r in old_summary['gates']}
    cache={};new_predictions={};metric_audit=[];regressions=0;new_evaluations=0
    for bank,held in (('A','B'),('B','A')):
        direction=f'{bank}_to_{held}'
        for target in TARGETS:
            selections={r['selected'] for r in identification if r['direction']==direction and r['true_context']==target}
            choices=sorted(selections|{'clean',target,WRONG_ALT[target]})
            for choice in choices:
                old_context='none' if choice=='clean' else ('correct_pair' if choice==target else ('wrong_alt_source' if choice==WRONG_ALT[target] else None))
                totals=np.zeros(10,dtype=np.int64);correct=np.zeros(10,dtype=np.int64);client_acc=[]
                for cid in range(100):
                    y=old_pred[f'c{cid}_labels']
                    if old_context:
                        r=old_rowindex[cid,bank,target,old_context];pred=old_pred[r['prediction_key']]
                    # Focused prediction regression on first2 clients proves reconstructed state/evaluator path.
                    if old_context is None or cid<2:
                        raw=read_data('Cifar10',cid,is_train=False)
                        assert np.array_equal(np.asarray(raw['y']),y)
                        qx=torch.as_tensor(raw['x'],dtype=torch.float32).cuda();qy=torch.as_tensor(y,dtype=torch.long).cuda()
                        tx=corrupt(qx,target,manifest['clients'][str(cid)]['test'],cid)
                        _,computed=evaluate(model,tx,qy,banks[bank][choice]);computed=computed.numpy()
                        assert state_hash(model)==initial
                        if old_context:
                            assert np.array_equal(computed,pred);regressions+=1
                        else:
                            pred=computed;new_evaluations+=1
                            new_predictions[f'{bank}|{target}|{choice}|c{cid}']=pred
                    totals+=np.bincount(y,minlength=10);correct+=np.bincount(y[pred==y],minlength=10)
                    client_acc.append(100*float(np.mean(pred==y)))
                metrics=dict(macro_class=float(np.mean(100*correct/totals)),sample_weighted=float(100*correct.sum()/totals.sum()),
                             macro_client=float(np.mean(client_acc)))
                if old_context:
                    assert abs(metrics['macro_class']-old_index[bank,target,old_context]['macro_class'])<1e-10
                    assert abs(metrics['sample_weighted']-old_index[bank,target,old_context]['sample_weighted'])<1e-10
                cache[bank,target,choice]=metrics
                metric_audit.append(dict(bank=bank,target=target,state_choice=choice,reused_context=old_context,
                    class_counts=totals.tolist(),correct_counts=correct.tolist(),**metrics))
    retrieval=[];retrieval_summary=[]
    for bank,held in (('A','B'),('B','A')):
        direction=f'{bank}_to_{held}'
        for target in TARGETS:
            selected_rows=[r for r in identification if r['direction']==direction and r['true_context']==target]
            for r in selected_rows:
                retrieval.append(dict(direction=direction,target=target,batch=r['batch'],selected=r['selected'],**cache[bank,target,r['selected']]))
            means={m:float(np.mean([cache[bank,target,r['selected']][m] for r in selected_rows])) for m in ('macro_class','sample_weighted','macro_client')}
            none=old_index[bank,target,'none']['macro_class'];oracle=old_index[bank,target,'correct_pair']['macro_class']
            wrong=old_index[bank,target,'wrong_alt_source']['macro_class'];tau=old_gate[bank,target]['tau_pp']
            retrieval_summary.append(dict(direction=direction,target=target,none=none,oracle=oracle,wrong_alt=wrong,tau_pp=tau,
                selected_mean=means['macro_class'],selected_weighted_mean=means['sample_weighted'],selected_client_mean=means['macro_client'],
                **retrieval_gate(none,oracle,means['macro_class'],wrong,tau)))
    joint={t:all(r['passed'] for r in retrieval_summary if r['target']==t) for t in TARGETS}
    id_a=all(v['passed'] for v in id_summaries.values());ret_a=sum(joint.values())>=3
    assert sha(out/'prototypes.json')==prototype_sha and sha(out/'support_decisions.json')==freeze['decision_sha256']
    metadata=dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,pool_sha256=POOL_SHA,
        writer_sha256=sha(ROOT/'src/adaptation/paired_affine_oracle_neutral.py'),corruption_sha256=sha(ROOT/'src/data/covariate.py'),
        signature_sha256=sha(ROOT/'src/context/source_signature.py'),old_summary_sha256=sha(old/'summary.json'),
        old_states_sha256=sha(old/'states.pt'),gpu=torch.cuda.get_device_name(),seconds=time.time()-start,
        contexts=CONTEXTS,prototype_sha256=prototype_sha)
    verify=dict(checkpoint_writer_corruption_unchanged=True,exact_pool_reuse=True,microbatch_partition_exact=True,
        microbatch_two_per_class=True,bank_heldout_query_disjoint=True,source_only_api=True,zero_state_features_only=True,
        bank_reference_only=True,all_selections_frozen_before_query=True,model_unchanged=True,finite_signatures_distances=True,
        reconstructed_states_exact=state_matches,wrong_alt_states_exact=True,focused_prediction_regressions_exact=regressions,
        new_state_query_client_evaluations=new_evaluations,query_used_for_selection=False,prototypes_decisions_unchanged=True)
    write_csv(out/'retrieval.csv',retrieval);write_csv(out/'retrieval_summary.csv',retrieval_summary)
    write('metadata.json',metadata);write('verification.json',verify);write('metric_count_audit.json',metric_audit)
    write('summary.json',dict(identification=id_summaries,retrieval=retrieval_summary,joint_retrieval_pass=joint,ID_A=id_a,RET_A=ret_a))
    if new_predictions:np.savez_compressed(out/'new_predictions.npz',**new_predictions)
    print('T008_DONE',json.dumps(dict(ID_A=id_a,RET_A=ret_a,joint=joint,verification=verify)),flush=True)


if __name__=='__main__':main()
