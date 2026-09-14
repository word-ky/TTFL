"""Frozen support-only rotation extraction; target class labels are never accessed."""
import argparse,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t023_common import *
from t021_features import extract
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.eval.common import state_hash
from src.data.covariate import corrupt,TARGETS
from src.context.rotation_probe import folds
from eval_t015_semantic_mixture import state_digest
from utils.data_utils import read_data

def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    old=p/'runs/20260914-183139-ttfl-t021-extract-gpu1/artifacts/t021_frozen_representation_observability';prior=p/'runs/20260914-193443-ttfl-t022-extract-gpu1/artifacts/t022_state_response_semantics'
    p7=p/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer';p9=p/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    base=Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10')
    support=load(p9/'natural_support_manifest.json');previous=load(prior/'extraction_freeze.json')
    paths=[base/'global_state.pt',p7/'states.pt',p9/'natural_support_manifest.json',ROOT/'src/data/covariate.py',ROOT/'src/models/pfllib.py',prior/'extraction_freeze.json',old/'extraction_freeze.json']
    for rep in ('L','H'):paths.extend([old/f'support_{rep}.npz',prior/f'new_{rep}.npz'])
    inputs={str(path):sha(path) for path in paths}
    for folder in (old,prior):
        f=load(folder/'extraction_freeze.json')
        for name,digest in f['representation_hashes'].items():assert sha(folder/name)==digest
    cached={r:{**dict(np.load(old/f'support_{r}.npz')),**dict(np.load(prior/f'new_{r}.npz'))} for r in ('H','L')}
    os.chdir(base/'system');torch.set_num_threads(2);model=ContextFedAvgCNN(3,10,1600).cuda().eval();model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==previous['model_hash']
    saved=torch.load(p7/'states.pt',map_location='cuda',weights_only=True);states={b:{'clean':new_state(model),**{t:saved[f'{b}|{t}|correct_pair'] for t in TARGETS}} for b in B}
    hashes={b:{c:state_digest(st) for c,st in bank.items()} for b,bank in states.items()};assert hashes==previous['state_digests']
    foldrows={};rotation=[];replays=[];files=[];calls=replaycalls=0;maxh=maxl=0.
    for i in range(100):
        selected=support['clients'][str(i)]['selected'];ids=[r['original_id'] for r in selected];idx=[r['train_index'] for r in selected]
        fa,fb=folds(i,ids);assert len(fa)==len(fb)==10 and not set(fa)&set(fb)
        foldrows[str(i)]=dict(support_ids=ids,A_positions=fa.tolist(),B_positions=fb.tolist(),A_ids=[ids[k] for k in fa],B_ids=[ids[k] for k in fb])
        # Only x is selected from the inherited PFLlib storage container; no y field is accessed.
        x=torch.as_tensor(read_data('Cifar10',i,is_train=True)['x'][idx],dtype=torch.float32).cuda();arrays={};pixels={}
        for t in C:
            tx=corrupt(x,t,ids,i);pixels[t]=tx.cpu().numpy();views=[]
            for r in range(4):
                view=torch.rot90(tx,r,(-2,-1));v=view.cpu().numpy();np.testing.assert_array_equal(v,np.rot90(pixels[t],r,axes=(-2,-1)))
                rotation.append(dict(client=i,context=t,rotation=r,input_sha256=rawsha(pixels[t]),view_sha256=rawsha(v),samples=20,exact_quarter_turn=True));views.append(view)
            for b in B:
                hgrid=np.empty((5,20,4,512),dtype=np.float32);argmax=np.empty((5,20),dtype=np.uint8)
                for si,s in enumerate(C):
                    key=f'{i}|{b}|{t}|{s}';hgrid[si,:,0]=cached['H'][key];argmax[si]=cached['L'][key].argmax(1)
                    if i in (0,24,49,74,99):
                        with torch.no_grad():h,l=extract(model,views[0],states[b][s])
                        hv=h.cpu().numpy();lv=l.cpu().numpy();dh=float(np.max(np.abs(hv-cached['H'][key])));dl=float(np.max(np.abs(lv-cached['L'][key])))
                        assert dh<=1e-6 and dl<=1e-6 and np.array_equal(lv.argmax(1),argmax[si]);maxh=max(maxh,dh);maxl=max(maxl,dl);replaycalls+=1
                        replays.append(dict(key=key,samples=20,max_H_difference=dh,max_logit_difference=dl,argmax_identical=True))
                    for r in (1,2,3):
                        with torch.no_grad():h,l=extract(model,views[r],states[b][s])
                        assert np.isfinite(h.cpu().numpy()).all();hgrid[si,:,r]=h.cpu().numpy();calls+=1
                arrays[f'{b}|{t}']=hgrid;arrays[f'argmax|{b}|{t}']=argmax
        np.savez_compressed(out/f'H_{i:03d}.npz',**arrays);np.savez_compressed(out/f'pixels_{i:03d}.npz',**pixels)
        files.extend([f'H_{i:03d}.npz',f'pixels_{i:03d}.npz'])
        assert state_hash(model)==initial and all(state_digest(st)==hashes[b][s] for b,bank in states.items() for s,st in bank.items())
        if i%10==9:print('T023_EXTRACT_CLIENT',i+1,flush=True)
    save(out/'fold_membership.json',foldrows);gzsave(out/'rotation_arithmetic.json.gz',rotation);save(out/'H_replay.json',dict(status='PASS',samples=20*replaycalls,max_H_difference=maxh,max_logit_difference=maxl,rows=replays))
    assert all(sha(path)==digest for path,digest in inputs.items());save(out/'input_hashes.json',inputs)
    files+=['fold_membership.json','rotation_arithmetic.json.gz','H_replay.json']
    save(out/'extraction_freeze.json',dict(status='PASS',task='T023',lead='89466b3',runtime=a.commit,model_hash=initial,state_digests=hashes,hashes={name:sha(out/name) for name in files},gpu=torch.cuda.get_device_name(),new_rotation_forward_calls=calls,replay_forward_calls=replaycalls,reused_unrotated_paths=5000,new_query_forwards=0,target_class_labels_used=False,privileged_utility_loaded=False,model_parameters_updated=False,seconds=time.time()-start))
    print('T023_EXTRACTION_COMPLETE',calls,flush=True)

if __name__=='__main__':main()
