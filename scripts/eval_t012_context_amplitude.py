"""T012 true-direction amplitude source freeze then fixed-grid query diagnostic."""
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
from src.context.prediction_consistency import score_state
from src.context.state_consistency import post_state_signature
from src.context.amplitude_ray import ALPHAS,scale_state,choose_ray
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
    for path,digest in {'src/context/state_consistency.py':'497ef9d1428b36cdb1e539ab6d7f7499727e1d558e96e4db5de9629ef47e71e1',
        'src/context/prediction_consistency.py':'0f6aa26de21b4e0252e553ffe993968e7f71926c2678a2dc5c3c48d5905835d4'}.items():
        assert sha(ROOT/path)==digest;expected[path]=digest
    refs={b:[(torch.tensor(r['mu'],dtype=torch.float64,device='cuda'),torch.tensor(r['sigma'],dtype=torch.float64,device='cuda')) for r in p['reference']] for b,p in proto.items()}
    rays={b:{c:{a:scale_state(states[b][c],a) for a in ALPHAS} for c in TARGETS} for b in ('A','B')}
    rayhash={b:{c:{a:state_digest(st) for a,st in ray.items()} for c,ray in bank.items()} for b,bank in rays.items()}
    probe=torch.linspace(-1,1,2*3*32*32,device='cuda').reshape(2,3,32,32)
    with torch.no_grad():
        for bank in ('A','B'):
            for context in TARGETS:
                assert torch.equal(model(probe),model(probe,rays[bank][context][0.]))
                assert all(torch.equal(x,y) for x,y in zip(states[bank][context],rays[bank][context][1.]))
                assert torch.equal(model(probe,states[bank][context]),model(probe,rays[bank][context][1.]))
                for alpha,st in rays[bank][context].items():
                    assert all(torch.isfinite(v).all() for v in st) and torch.isfinite(model(probe,st)).all()
                assert state_digest(states[bank][context])==hashes[bank][context] and state_hash(model)==initial
    write('amplitude_preflight.json',dict(all_8_states_7_alphas_finite=True,zero_logit_error=0.,one_logit_error=0.,one_state_exact=True,immutable=True))
    print('T012_ENDPOINT_PREFLIGHT_PASS',flush=True)
    rows=[];choices=[];endpoint_checks=0
    for cid in range(100):
        support_rows=support['clients'][str(cid)]['selected'];ids=[r['original_id'] for r in support_rows];indices=[r['train_index'] for r in support_rows]
        x=torch.as_tensor(read_data('Cifar10',cid,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for context in TARGETS:
            tx=corrupt(x,context,ids,cid)
            for bank in ('A','B'):
                scores={}
                for alpha in ALPHAS:
                    st=rays[bank][context][alpha];psi=post_state_signature(model,tx,st,refs[bank]);pc,raw=score_state(model,tx,st)
                    assert torch.isfinite(raw['logits']).all() and torch.isfinite(raw['flipped_logits']).all()
                    assert state_hash(model)==initial and state_digest(st)==rayhash[bank][context][alpha] and state_digest(states[bank][context])==hashes[bank][context]
                    if alpha in (0.,1.):
                        with torch.no_grad():assert torch.equal(raw['logits'],model(tx,None if alpha==0 else states[bank][context]))
                        endpoint_checks+=1
                    scores[alpha]=dict(J=float(psi.square().sum()),**pc)
                    assert all(np.isfinite(v) for v in scores[alpha].values())
                    rows.append(dict(client=cid,bank=bank,target=context,alpha=alpha,**scores[alpha]))
                choices.append(dict(client=cid,bank=bank,target=context,**choose_ray(scores)))
        if cid%20==19:print('T012_SOURCE_CLIENT_DONE',cid,flush=True)
    write('amplitude_source_scores.json',rows);write_csv(out/'amplitude_source_scores.csv',rows)
    write('amplitude_source_choices.json',choices);write_csv(out/'amplitude_source_choices.csv',choices)
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),scores_sha256=sha(out/'amplitude_source_scores.json'),choices_sha256=sha(out/'amplitude_source_choices.json'),rows=len(rows),choices=len(choices),new_query_amplitude_opened=False)
    assert len(rows)==5600 and len(choices)==800
    write('source_freeze.json',freeze);print('T012_SOURCE_FROZEN',json.dumps(freeze),flush=True)
    # Only now open historical query arrays, then compute missing amplitude outcomes.
    oldrows=json.loads((t007/'raw_records.json').read_text());ix={(r['client'],r['pool'],r['target'],r['context']):r for r in oldrows};oldp=np.load(t007/'predictions.npz')
    si={(r['client'],r['bank'],r['target'],r['alpha']):r for r in rows};predictions={};queryrows=[];new_evals=0
    for cid in range(100):
        query=read_data('Cifar10',cid,is_train=False);y=oldp[f'c{cid}_labels'];assert np.array_equal(query['y'],y)
        predictions[f'c{cid}_labels']=y;x=torch.as_tensor(query['x'],dtype=torch.float32).cuda()
        for target in TARGETS:
            tx=corrupt(x,target,split['clients'][str(cid)]['test'],cid)
            for bank in ('A','B'):
                for alpha in ALPHAS:
                    oldkey=None
                    if alpha in (0.,1.):
                        oldkey=ix[cid,bank,target,'none' if alpha==0 else 'correct_pair']['prediction_key'];pred=oldp[oldkey]
                    else:
                        with torch.no_grad():logits=torch.cat([model(b,rays[bank][target][alpha]) for b in tx.split(256)])
                        assert torch.isfinite(logits).all();pred=logits.argmax(1).cpu().numpy();new_evals+=1
                    assert state_hash(model)==initial and state_digest(rays[bank][target][alpha])==rayhash[bank][target][alpha] and state_digest(states[bank][target])==hashes[bank][target]
                    key=f'{bank}|{target}|{alpha}|c{cid}';predictions[key]=pred;sr=si[cid,bank,target,alpha]
                    queryrows.append(dict(client=cid,bank=bank,target=target,alpha=alpha,prediction_key=key,reused_key=oldkey,
                        query_accuracy=100*float(np.mean(pred==y)),class_total=np.bincount(y,minlength=10).tolist(),class_correct=np.bincount(y[y==pred],minlength=10).tolist(),source_J=sr['J'],source_C=sr['consistency_js'],source_agreement=sr['flip_top1_agreement']))
        if cid%20==19:print('T012_QUERY_CLIENT_DONE',cid,flush=True)
    qi={(r['client'],r['bank'],r['target'],r['alpha']):r for r in queryrows}
    for r in queryrows:r['DeltaAcc_vs_alpha0']=r['query_accuracy']-qi[r['client'],r['bank'],r['target'],0.]['query_accuracy']
    np.savez_compressed(out/'amplitude_predictions.npz',**predictions);write('amplitude_query_utility.json',queryrows)
    write_csv(out/'amplitude_query_utility.csv',[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in queryrows])
    write('query_freeze.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),predictions_sha256=sha(out/'amplitude_predictions.npz'),matrix_sha256=sha(out/'amplitude_query_utility.json')))
    assert sha(out/'amplitude_source_scores.json')==freeze['scores_sha256'] and sha(out/'amplitude_source_choices.json')==freeze['choices_sha256']
    shutil.copyfile(t009/'skew_labels_receipt.json',out/'frozen_skew_labels_receipt.json')
    write('metadata.json',dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,frozen_code_hashes=expected,
        states_sha256=sha(t007/'states.pt'),support_manifest_sha256=sha(t009/'natural_support_manifest.json'),prototypes_sha256=sha(t008/'prototypes.json'),
        scaling_helper_sha256=sha(ROOT/'src/context/amplitude_ray.py'),gpu=torch.cuda.get_device_name(),seconds=time.time()-start,alphas=ALPHAS))
    write('verification.json',dict(frozen_hashes_unchanged=True,all_8_states_7_alphas_preflight=True,source_endpoint_regressions=endpoint_checks,
        zero_one_logit_max_error=0.,scaled_states_logits_finite=True,model_saved_scaled_states_unchanged_every_episode=True,
        exact_K20_manifest=True,source_label_query_free=True,source_frozen_before_query=True,no_state_fits_no_query_tuning=True,
        source_rows=len(rows),query_rows=len(queryrows),reused_endpoint_queries=1600,new_query_evaluations=new_evals))
    print('T012_DONE',json.dumps(dict(new_query_evaluations=new_evals,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
