"""T011 prediction-consistency source freeze followed by full fixed-state query matrix."""
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
from src.context.prediction_consistency import score_state,select_pc_safe,js_divergence
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
    score_calls=0;zero_checks=0
    def score(source,bank,true):
        nonlocal score_calls,zero_checks
        scores={};zero_probs=None
        for context in CONTEXTS:
            metrics,raw=score_state(model,source,states[bank][context])
            if context=='clean':
                with torch.no_grad():assert torch.equal(raw['logits'],model(source))
                zero_probs=raw['probabilities'];zero_checks+=1
            metrics['js_to_zero_original']=float(js_divergence(raw['probabilities'],zero_probs).mean())
            assert all(np.isfinite(v) for v in metrics.values())
            assert state_hash(model)==initial and state_digest(states[bank][context])==hashes[bank][context]
            scores[context]=metrics;score_calls+=1
        choice=select_pc_safe(scores)
        ranked=sorted(CONTEXTS,key=lambda c:(scores[c]['consistency_js'],CONTEXTS.index(c)))
        choice['true_state_rank_by_consistency']=ranked.index(true)+1 if true in CONTEXTS else None
        return scores,choice
    poolpixels={}
    for b,pool in pools.items():
        poolpixels[b]=torch.stack([torch.as_tensor(read_data('Cifar10',r['client_id'],is_train=True)['x'][r['train_index']],dtype=torch.float32) for r in pool['samples']]).cuda()
    balanced=[];ood=[];sanity_scores=[]
    for bank,held in (('A','B'),('B','A')):
        for context in CONTEXTS+('noise_image',):
            tx=torch.cat([corrupt(poolpixels[held][i:i+1],context,[r['original_id']],r['client_id']) for i,r in enumerate(pools[held]['samples'])])
            for batch in micro[held]:
                metrics,choice=score(tx[batch['indices']],bank,context)
                identity=dict(bank=bank,heldout=held,batch=batch['batch'],true_context=context)
                (ood if context=='noise_image' else balanced).append(dict(**identity,**choice))
                for c,v in metrics.items():sanity_scores.append(dict(**identity,candidate=c,**v))
    write_csv(out/'balanced_sanity.csv',balanced);write_csv(out/'ood_selections.csv',ood)
    write_csv(out/'sanity_candidate_scores.csv',sanity_scores)
    print('T011_BALANCED_DONE',flush=True)
    selections=[];source_rows=[]
    for cid in range(100):
        rows=support['clients'][str(cid)]['selected'];ids=[r['original_id'] for r in rows];indices=[r['train_index'] for r in rows]
        x=torch.as_tensor(read_data('Cifar10',cid,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for context in CONTEXTS:
            tx=corrupt(x,context,ids,cid)
            for bank in ('A','B'):
                metrics,choice=score(tx,bank,context);identity=dict(client=cid,bank=bank,true_context=context)
                selections.append(dict(**identity,**choice))
                for c,v in metrics.items():source_rows.append(dict(**identity,candidate=c,**v))
        if cid%20==19:print('T011_SOURCE_CLIENT_DONE',cid,flush=True)
    write('source_candidate_scores.json',source_rows);write('selections.json',selections)
    write_csv(out/'source_candidate_scores.csv',source_rows);write_csv(out/'selections.csv',selections)
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),scores_sha256=sha(out/'source_candidate_scores.json'),
        decisions_sha256=sha(out/'selections.json'),score_rows=len(source_rows),decisions=len(selections),query_opened=False,labels_opened=False)
    assert len(source_rows)==5000 and len(selections)==1000
    write('source_freeze.json',freeze);print('T011_SOURCE_FROZEN',json.dumps(freeze),flush=True)
    # First access to historical query predictions, past choices, and moment scores occurs here.
    old7rows=json.loads((t007/'raw_records.json').read_text())
    i7={(r['client'],r['pool'],r['target'],r['context']):r for r in old7rows}
    old9rows=json.loads((t009/'mixed_query_records.json').read_text());old10rows=json.loads((t010/'mixed_query_records.json').read_text())
    p7=np.load(t007/'predictions.npz');p9=np.load(t009/'mixed_predictions.npz');p10=np.load(t010/'mixed_predictions.npz')
    reuse_index={}
    for origin,rows,pp in [('T009',old9rows,p9),('T010',old10rows,p10)]:
        for r in rows:reuse_index[r['client'],r['bank'],r['target'],r['selected']]=(origin,pp,r['prediction_key'])
    moments={(r['client'],r['bank'],r['true_context']):r for r in json.loads((t010/'state_scores.json').read_text())}
    joined=[dict(**r,moment_J_from_T010=moments[r['client'],r['bank'],r['true_context']]['J_'+r['candidate']]) for r in source_rows]
    write_csv(out/'source_scores_with_moments.csv',joined)
    predictions={};query_rows=[];new_evals=0;reused=0
    for cid in range(100):
        y=p7[f'c{cid}_labels'];predictions[f'c{cid}_labels']=y;query=None
        for target in CONTEXTS:
            tx=None
            for bank in ('A','B'):
                for choice in CONTEXTS:
                    origin='new';oldkey=None
                    if (cid,bank,target,choice) in reuse_index:
                        origin,pp,oldkey=reuse_index[cid,bank,target,choice];pred=pp[oldkey]
                    else:
                        if target=='clean' and choice=='clean':oldkey=f'c{cid}_clean'
                        elif target!='clean':
                            ctx='none' if choice=='clean' else ('correct_pair' if choice==target else ('wrong_alt_source' if choice==WRONG_ALT[target] else None))
                            if ctx:oldkey=i7[cid,bank,target,ctx]['prediction_key']
                        if oldkey is not None:pred=p7[oldkey];origin='T007R'
                    if origin=='new':
                        if query is None:
                            query=read_data('Cifar10',cid,is_train=False);assert np.array_equal(query['y'],y)
                        if tx is None:tx=corrupt(torch.as_tensor(query['x'],dtype=torch.float32).cuda(),target,split['clients'][str(cid)]['test'],cid)
                        _,pred=evaluate(model,tx,torch.as_tensor(y,dtype=torch.long).cuda(),states[bank][choice]);pred=pred.numpy();new_evals+=1
                    else:reused+=1
                    assert state_hash(model)==initial and state_digest(states[bank][choice])==hashes[bank][choice]
                    key=f'{bank}|{target}|{choice}|c{cid}';predictions[key]=pred
                    query_rows.append(dict(client=cid,bank=bank,target=target,candidate=choice,prediction_key=key,
                        reused_from=origin,reused_key=oldkey,n_query=len(y),query_accuracy=100*float(np.mean(pred==y)),
                        class_total=np.bincount(y,minlength=10).tolist(),class_correct=np.bincount(y[pred==y],minlength=10).tolist()))
        if cid%20==19:print('T011_QUERY_CLIENT_DONE',cid,flush=True)
    qi={(r['client'],r['bank'],r['target'],r['candidate']):r for r in query_rows}
    for r in query_rows:r['DeltaAcc_vs_zero']=r['query_accuracy']-qi[r['client'],r['bank'],r['target'],'clean']['query_accuracy']
    np.savez_compressed(out/'candidate_predictions.npz',**predictions);write('candidate_query_utility.json',query_rows)
    write_csv(out/'candidate_query_utility.csv',[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in query_rows])
    write('query_freeze.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),predictions_sha256=sha(out/'candidate_predictions.npz'),matrix_sha256=sha(out/'candidate_query_utility.json')))
    shutil.copyfile(t009/'skew_labels_receipt.json',out/'frozen_skew_labels_receipt.json')
    assert sha(out/'source_candidate_scores.json')==freeze['scores_sha256'] and sha(out/'selections.json')==freeze['decisions_sha256']
    write('metadata.json',dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,frozen_code_hashes=expected,
        support_manifest_sha256=sha(t009/'natural_support_manifest.json'),states_sha256=sha(t007/'states.pt'),
        prototypes_sha256=sha(t008/'prototypes.json'),t010_scores_sha256=sha(t010/'state_scores.json'),
        helper_sha256=sha(ROOT/'src/context/prediction_consistency.py'),candidate_memory_hashes=hashes,gpu=torch.cuda.get_device_name(),seconds=time.time()-start))
    write('verification.json',dict(frozen_hashes_unchanged=True,score_calls=score_calls,zero_regression_checks=zero_checks,zero_logit_max_error=0.,
        model_states_unchanged_every_episode=True,all_finite=True,source_label_query_free=True,source_frozen_before_query=True,
        no_fitting_or_tuning=True,query_rows=len(query_rows),query_reused=reused,new_query_evaluations=new_evals))
    print('T011_DONE',json.dumps(dict(new_query_evaluations=new_evals,reused=reused,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
