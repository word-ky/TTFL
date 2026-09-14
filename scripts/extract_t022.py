"""Reuse T021 paths, extract only missing states, freeze raw response signatures."""
import argparse,hashlib,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,save,sha,C,B
from t021_features import extract
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import state_hash
from src.data.covariate import corrupt,TARGETS
from src.context.state_response import state_response
from eval_t015_semantic_mixture import state_digest
from utils.data_utils import read_data


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    old=p/'runs/20260914-183139-ttfl-t021-extract-gpu1/artifacts/t021_frozen_representation_observability'
    prior=load(old/'extraction_freeze.json');inputs=load(old/'input_hashes.json');assert all(sha(Path(path))==h for path,h in inputs.items())
    for name,h in prior['representation_hashes'].items():assert sha(old/name)==h;inputs[str(old/name)]=h
    base=Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10');data=Path('/media/wenchang/F/wjq/TTFL/dataset')
    p7=p/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer';p9=p/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    support=load(p9/'natural_support_manifest.json');split=load(data/'Cifar10/split_manifest.json');pools=load(p7/'calibration_pools.json')
    calids={r['original_id'] for pool in pools.values() for r in pool['samples']};qids={x for r in split['clients'].values() for x in r['test']}
    for i in range(100):
        ids={r['original_id'] for r in support['clients'][str(i)]['selected']};assert len(ids)==20 and not ids&calids and not ids&qids
    cached={rep:dict(np.load(old/f'support_{rep}.npz')) for rep in ('L','H')};initial_cache={rep:{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in bank.items()} for rep,bank in cached.items()}
    full={rep:dict(bank) for rep,bank in cached.items()};new={rep:{} for rep in cached};signatures={rep:{} for rep in cached}
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval();model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True));initial=state_hash(model);assert initial==prior['model_hash']
    saved=torch.load(p7/'states.pt',map_location='cuda',weights_only=True);states={b:{'clean':new_state(model),**{t:saved[f'{b}|{t}|correct_pair'] for t in TARGETS}} for b in B}
    hashes={b:{c:state_digest(st) for c,st in bank.items()} for b,bank in states.items()};assert hashes==prior['state_digests']
    calls=checked=0
    for i in range(100):
        selected=support['clients'][str(i)]['selected'];ids=[r['original_id'] for r in selected];idx=[r['train_index'] for r in selected]
        x=torch.as_tensor(read_data('Cifar10',i,is_train=True)['x'][idx],dtype=torch.float32).cuda()
        for t in C:
            tx=corrupt(x,t,ids,i)
            for b in B:
                for c in C:
                    key=f'{i}|{b}|{t}|{c}'
                    if key not in full['H']:
                        with torch.no_grad():h,l=extract(model,tx,states[b][c])
                        assert torch.isfinite(h).all() and torch.isfinite(l).all()
                        for rep,v in [('H',h),('L',l)]:new[rep][key]=v.cpu().numpy();full[rep][key]=new[rep][key]
                        calls+=1
                    else:
                        for rep in ('L','H'):assert hashlib.sha256(full[rep][key].tobytes()).hexdigest()==initial_cache[rep][key]
                    assert state_digest(states[b][c])==hashes[b][c]
                for rep,d in [('L',10),('H',512)]:
                    values=np.stack([full[rep][f'{i}|{b}|{t}|{c}'] for c in C]);phi=state_response(values)
                    assert phi.shape==(20,4*d) and np.isfinite(phi).all()
                    for k in range(4):np.testing.assert_array_equal(phi[:,k*d:(k+1)*d],values[k+1]-values[0])
                    signatures[rep][f'{i}|{b}|{t}']=phi;checked+=20
        assert state_hash(model)==initial
        if i%20==19:print('T022_RESPONSE_CLIENT',i+1,flush=True)
    for rep in ('L','H'):
        assert len(full[rep])==5000 and len(signatures[rep])==1000
        np.savez_compressed(out/f'new_{rep}.npz',**new[rep]);np.savez_compressed(out/f'signature_{rep}.npz',**signatures[rep])
    assert all(sha(Path(path))==h for path,h in inputs.items())
    save(out/'input_hashes.json',inputs);save(out/'cache_reuse.json',dict(source=str(old),source_hashes=prior['representation_hashes'],cached_keys=initial_cache,reused_paths=len(cached['L']),reused_samples=len(cached['L'])*20,max_reuse_difference=0))
    save(out/'response_arithmetic.json',dict(status='PASS',state_order=C[1:],formula='concat(raw_state-raw_clean)',dtype='float32',signature_vectors_checked=checked,
        dimensions=dict(L=40,H=2048),all_blocks_exact=True,construction_independent_of_oracle_or_source_context=True))
    save(out/'extraction_freeze.json',dict(status='PASS',lead='06647a0',runtime=a.commit,task='T022',gpu=torch.cuda.get_device_name(),model_hash=initial,state_digests=hashes,
        reused_paths=len(cached['L']),new_forward_calls=calls,logical_state_paths=5000,logical_sample_outputs=100000,signature_vectors=40000,seconds=time.time()-start,
        representation_hashes={name:sha(out/name) for name in ('new_L.npz','new_H.npz','signature_L.npz','signature_H.npz')},target_labels_used=False,query_outcomes_scored=False,
        support_calibration_query_disjoint=True,model_states_unchanged=True))
    print('T022_EXTRACTION_COMPLETE',calls,flush=True)


if __name__=='__main__':main()
