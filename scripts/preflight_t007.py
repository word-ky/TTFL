"""T007 real-pool provenance and clean-identity prerequisite, before shift evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.balanced_pool import calibration_pools, balanced_pairings
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.adaptation.paired_affine_oracle import paired_clean_oracle, EPS
from src.eval.common import state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from utils.data_utils import read_data


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','output','commit'):p.add_argument('--'+name,required=True)
    a=p.parse_args();base,data,out=[Path(v).resolve() for v in (a.baseline,a.data_root,a.output)]
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text())
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    assert sha(ROOT/'src/adaptation/paired_affine_oracle.py')=='99824a64af46ad500c8be742c0775435bf7566188935eb59d39d7acca86a11dd'
    assert sha(ROOT/'src/data/covariate.py')=='297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174'
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    candidates=[];query_ids=set()
    for cid in range(100):
        raw=read_data('Cifar10',cid,is_train=True)
        ids=manifest['clients'][str(cid)]['train']
        assert len(ids)==len(raw['y'])
        candidates.extend(dict(client_id=cid,train_index=i,original_id=int(sid),**{'class':int(raw['y'][i])})
                          for i,sid in enumerate(ids))
        query_ids.update(manifest['clients'][str(cid)]['test'])
    pools=calibration_pools(candidates)
    assert not {r['original_id'] for rs in pools.values() for r in rs}.intersection(query_ids)
    pool_receipt={}
    for name,rs in pools.items():
        labels=[r['class'] for r in rs];perms=balanced_pairings(labels)
        pool_receipt[name]=dict(samples=rs,permutations=perms,
            histogram={str(c):labels.count(c) for c in range(10)},
            unique_clients=len({r['client_id'] for r in rs}),
            clients_per_class={str(c):len({r['client_id'] for r in rs if r['class']==c}) for c in range(10)})
    (out/'calibration_pools.json').write_text(json.dumps(pool_receipt,indent=2))
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model)
    assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    states,diagnostics,variances={},[],[]
    for name,rs in pools.items():
        pixels=[]
        for r in rs:
            raw=read_data('Cifar10',r['client_id'],is_train=True)
            assert int(raw['y'][r['train_index']])==r['class']
            pixels.append(torch.as_tensor(raw['x'][r['train_index']],dtype=torch.float32))
        x=torch.stack(pixels).cuda()
        with torch.no_grad():
            for perm in pool_receipt[name]['permutations'].values():
                pm=torch.tensor(perm,device='cuda');inv=pm.argsort()
                assert torch.equal(x[pm][inv],x)
                h,hp=x,x[pm]
                for block in (model.conv1,model.conv2):
                    h,hp=block(h),block(hp)
                    assert torch.equal(hp[inv],h)
            state,ds=paired_clean_oracle(model,x,x)
            states[name]=state
            diagnostics.extend(dict(pool=name,**d) for d in ds)
            source,target=x,x
            for layer,block in enumerate((model.conv1,model.conv2)):
                source,target=block(source),block(target)
                variance=source.double().var((0,2,3),unbiased=False)
                gamma,beta=state[layer*2:layer*2+2]
                corrected=source*(1+gamma[None,:,None,None])+beta[None,:,None,None]
                worst=int(gamma.abs().argmax())
                variances.append(dict(pool=name,layer=layer+1,min_variance=variance.min().item(),
                    max_abs_gamma=gamma.abs().max().item(),worst_channel=worst,
                    worst_channel_variance=variance[worst].item(),eps=EPS,
                    worst_channel_scale=(1+gamma[worst]).item(),
                    max_feature_diff=(corrected-target).abs().max().item()))
                source=corrected
        assert state_hash(model)==initial
    predictions,rows={},[];maxlogit={n:0. for n in states}
    counts={n:np.zeros(10,dtype=np.int64) for n in ('none','A','B')};totals=np.zeros(10,dtype=np.int64)
    with torch.no_grad():
        for cid in range(100):
            raw=read_data('Cifar10',cid,is_train=False)
            x=torch.as_tensor(raw['x'],dtype=torch.float32).cuda();labels=np.asarray(raw['y'],dtype=np.int64)
            predictions[f'c{cid}_labels']=labels
            preds={n:[] for n in counts}
            for b in x.split(256):
                baseline=model(b)
                assert torch.equal(baseline,model(b,new_state(model)))
                preds['none'].append(baseline.argmax(1).cpu().numpy())
                for name,state in states.items():
                    logits=model(b,state)
                    maxlogit[name]=max(maxlogit[name],(logits-baseline).abs().max().item())
                    preds[name].append(logits.argmax(1).cpu().numpy())
            totals+=np.bincount(labels,minlength=10)
            for name,parts in preds.items():
                pred=np.concatenate(parts);predictions[f'c{cid}_{name}']=pred
                counts[name]+=np.bincount(labels[pred==labels],minlength=10)
                rows.append(dict(client=cid,condition=name,n_query=len(labels),accuracy=100*float((pred==labels).mean())))
    metrics={n:dict(macro_class=float(np.mean(100*v/totals)),sample_weighted=float(100*v.sum()/totals.sum()),
              macro_client=float(np.mean([r['accuracy'] for r in rows if r['condition']==n])),
              per_class=(100*v/totals).tolist(),correct=v.tolist()) for n,v in counts.items()}
    assert state_hash(model)==initial
    payload=dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,
        writer_sha256=sha(ROOT/'src/adaptation/paired_affine_oracle.py'),
        corruption_sha256=sha(ROOT/'src/data/covariate.py'),gpu=torch.cuda.get_device_name(),
        query_class_counts=totals.tolist(),metrics=metrics,max_query_logit_diff=maxlogit,
        identity_macro_class_delta={n:metrics[n]['macro_class']-metrics['none']['macro_class'] for n in states},
        channel_diagnostics=variances,writer_episodes=2,shift_states_fitted=0,
        zero_state_exact=True,model_unchanged=True,query_pool_overlap=0,
        exact_target_id_image_feature_multisets=True,labels_only_pool_and_pairing_construction=True,
        pool_selection_before_query_evaluation=True)
    (out/'identity_preflight.json').write_text(json.dumps(payload,indent=2))
    (out/'restoration_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    write_csv(out/'per_client.csv',rows)
    np.savez_compressed(out/'identity_predictions.npz',**predictions)
    torch.save({n:[v.cpu() for v in state] for n,state in states.items()},out/'identity_states.pt')
    print('T007_IDENTITY_PREFLIGHT',json.dumps(payload),flush=True)


if __name__=='__main__':main()
