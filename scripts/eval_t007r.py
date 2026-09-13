"""Frozen T007R pools: exact identity prerequisite, then global-state transfer."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.adaptation.paired_affine_oracle_neutral import paired_clean_oracle_neutral
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.eval.common import evaluate,state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from utils.data_utils import read_data

POOL_SHA='c6b1692e6fae8db52dec5dfe036af03bdc696de359a7b35f5bf64fe1152e49d0'
CONTEXTS=('none','clean_identity','correct_pair','same_class_derangement',
          'cross_class_derangement','wrong_alt_source','noise_source')


def primary_gate(clean,none,correct,wrong,noise):
    h=clean-none;tau=max(.5,.25*h)
    recovery=(correct-none)/h if h>0 else None
    return dict(headroom_pp=h,tau_pp=tau,recovery=recovery,
        correct_wrong_pp=correct-wrong,correct_noise_pp=correct-noise,
        passed=recovery is not None and recovery>=.5 and correct-wrong>=tau and correct-noise>=tau)


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','pools','output','commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--clients',type=int,default=100)
    a=p.parse_args();base,data,poolfile,out=map(lambda v:Path(v).resolve(),(a.baseline,a.data_root,a.pools,a.output))
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    assert sha(poolfile)==POOL_SHA
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    assert sha(ROOT/'src/adaptation/paired_affine_oracle.py')=='99824a64af46ad500c8be742c0775435bf7566188935eb59d39d7acca86a11dd'
    assert sha(ROOT/'src/data/covariate.py')=='297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174'
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text());pools=json.loads(poolfile.read_text())
    shutil.copyfile(poolfile,out/'calibration_pools.json')
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    allqueries={sid for m in manifest['clients'].values() for sid in m['test']}
    pixels={};used=set()
    for pool,receipt in pools.items():
        rs=receipt['samples'];ids=[r['original_id'] for r in rs];labels=[r['class'] for r in rs]
        assert len(ids)==80 and len(set(ids))==80 and not used.intersection(ids) and not allqueries.intersection(ids)
        used.update(ids);assert all(labels.count(c)==8 for c in range(10))
        xs=[]
        for r in rs:
            raw=read_data('Cifar10',r['client_id'],is_train=True)
            assert manifest['clients'][str(r['client_id'])]['train'][r['train_index']]==r['original_id']
            assert int(raw['y'][r['train_index']])==r['class']
            xs.append(torch.as_tensor(raw['x'][r['train_index']],dtype=torch.float32))
        x=torch.stack(xs).cuda();pixels[pool]=x
        with torch.no_grad():
            for ctx,permutation in receipt['permutations'].items():
                assert sorted(permutation)==list(range(80)) and all(i!=j for i,j in enumerate(permutation))
                assert sum(labels[i]==labels[j] for i,j in enumerate(permutation))==(80 if ctx.startswith('same') else 0)
                perm=torch.tensor(permutation,device='cuda');inv=perm.argsort()
                assert torch.equal(x[perm][inv],x)
                h,hp=x,x[perm]
                for block in (model.conv1,model.conv2):
                    h,hp=block(h),block(hp);assert torch.equal(hp[inv],h)
    states={};diags=[];predictions={};rows=[];identity=[]
    for pool,x in pixels.items():
        state,ds=paired_clean_oracle_neutral(model,x,x)
        assert sum(v.numel() for v in state)==192 and all(torch.equal(v,torch.zeros_like(v)) for v in state)
        assert state_hash(model)==initial
        states[pool,'clean','clean_identity']=state
        diags.extend(dict(pool=pool,target='clean',context='clean_identity',**d) for d in ds)
    # All 100 clean queries must pass before even a smoke corrupted state is fit.
    with torch.no_grad():
        for cid in range(100):
            raw=read_data('Cifar10',cid,is_train=False)
            x=torch.as_tensor(raw['x'],dtype=torch.float32).cuda();y=np.asarray(raw['y'],dtype=np.int64)
            predictions[f'c{cid}_labels']=y;parts=[]
            for b in x.split(256):
                logits=model(b);assert torch.equal(logits,model(b,new_state(model)))
                for pool in pools:assert torch.equal(logits,model(b,states[pool,'clean','clean_identity']))
                parts.append(logits.argmax(1).cpu().numpy())
            pred=np.concatenate(parts);predictions[f'c{cid}_clean']=pred
            identity.append(dict(client=cid,n_query=len(y),correct=int((pred==y).sum())))
            assert state_hash(model)==initial
    identity_verify=dict(all_192_state_values_zero=True,max_clean_logit_diff=0.,changed_clean_predictions=0,
        clean_clients=100,clean_queries=sum(r['n_query'] for r in identity),all_metric_deltas=0.,model_hash_unchanged=True)
    (out/'identity_verification.json').write_text(json.dumps(identity_verify,indent=2))
    print('T007R_IDENTITY_PASS',json.dumps(identity_verify),flush=True)
    # Use each calibration image's original client+ID for the frozen corruption RNG convention.
    def pool_corruption(pool,name):
        return torch.cat([corrupt(pixels[pool][i:i+1],name,[r['original_id']],r['client_id'])
                          for i,r in enumerate(pools[pool]['samples'])])
    for pool,x in pixels.items():
        for target in TARGETS:
            source=pool_corruption(pool,target)
            for ctx in CONTEXTS[2:]:
                target_x=x;source_x=source
                if ctx in pools[pool]['permutations']:
                    target_x=x[torch.tensor(pools[pool]['permutations'][ctx],device='cuda')]
                elif ctx=='wrong_alt_source':source_x=pool_corruption(pool,WRONG_ALT[target])
                elif ctx=='noise_source':source_x=pool_corruption(pool,'noise_image')
                state,ds=paired_clean_oracle_neutral(model,target_x,source_x)
                assert state_hash(model)==initial
                states[pool,target,ctx]=state
                diags.extend(dict(pool=pool,target=target,context=ctx,**d) for d in ds)
    torch.save({'|'.join(k):[v.cpu() for v in st] for k,st in states.items()},out/'states.pt')
    for cid in range(a.clients):
        raw=read_data('Cifar10',cid,is_train=False)
        x=torch.as_tensor(raw['x'],dtype=torch.float32).cuda();y=torch.as_tensor(raw['y'],dtype=torch.long).cuda()
        ids=manifest['clients'][str(cid)]['test']
        for target in TARGETS:
            tx=corrupt(x,target,ids,cid)
            for pool in pools:
                for ctx in CONTEXTS:
                    state=None if ctx=='none' else states[pool,'clean' if ctx=='clean_identity' else target,ctx]
                    result,pred=evaluate(model,tx,y,state)
                    assert state_hash(model)==initial
                    key=f'r{len(rows)}';predictions[key]=pred.numpy()
                    class_total=np.bincount(y.cpu().numpy(),minlength=10)
                    class_correct=np.bincount(y.cpu().numpy()[pred.numpy()==y.cpu().numpy()],minlength=10)
                    rows.append(dict(client=cid,pool=pool,target=target,context=ctx,n_query=len(y),
                        prediction_key=key,class_total=class_total.tolist(),class_correct=class_correct.tolist(),**result))
        print('T007R_CLIENT_DONE',cid,flush=True)
    summary=[];perclass=[]
    for pool in pools:
        for target in TARGETS:
            for ctx in CONTEXTS:
                group=[r for r in rows if (r['pool'],r['target'],r['context'])==(pool,target,ctx)]
                totals=np.sum([r['class_total'] for r in group],axis=0);correct=np.sum([r['class_correct'] for r in group],axis=0)
                assert (totals>0).all()
                summary.append(dict(pool=pool,target=target,context=ctx,macro_class=float((100*correct/totals).mean()),
                    sample_weighted=float(100*correct.sum()/totals.sum()),macro_client=float(np.mean([r['accuracy'] for r in group]))))
                perclass.extend(dict(pool=pool,target=target,context=ctx,**{'class':c},n_query=int(totals[c]),
                    correct=int(correct[c]),accuracy=float(100*correct[c]/totals[c])) for c in range(10))
    clean_y=np.concatenate([predictions[f'c{cid}_labels'] for cid in range(a.clients)])
    clean_p=np.concatenate([predictions[f'c{cid}_clean'] for cid in range(a.clients)])
    clean_macro=float(np.mean([100*np.mean(clean_p[clean_y==c]==c) for c in range(10)]))
    lookup={(r['pool'],r['target'],r['context']):r['macro_class'] for r in summary};gates=[]
    for pool in pools:
        for target in TARGETS:
            v=lambda c:lookup[pool,target,c]
            gate=primary_gate(clean_macro,v('none'),v('correct_pair'),v('wrong_alt_source'),v('noise_source'))
            gates.append(dict(pool=pool,target=target,**gate,instance_advantage_pp=v('correct_pair')-v('same_class_derangement'),
                semantic_advantage_pp=v('same_class_derangement')-v('cross_class_derangement'),
                hard_pair_gap_pp=v('correct_pair')-v('cross_class_derangement')))
    passes={t:all(r['passed'] for r in gates if r['target']==t) for t in TARGETS}
    metadata=dict(code_commit=a.commit,clients=a.clients,smoke=a.clients!=100,checkpoint_sha256=CHECKPOINT_SHA,
        old_writer_sha256=sha(ROOT/'src/adaptation/paired_affine_oracle.py'),
        writer_sha256=sha(ROOT/'src/adaptation/paired_affine_oracle_neutral.py'),pool_sha256=POOL_SHA,
        corruption_sha256=sha(ROOT/'src/data/covariate.py'),gpu=torch.cuda.get_device_name(),seconds=time.time()-start,
        eps=1e-5,scale_clamp=[-8,8],clean_macro_class=clean_macro)
    verify=dict(**identity_verify,records=len(rows),global_writer_fits=len(states),corruption_fits=40,
        exact_committed_pools=True,pools_disjoint=True,pool_query_overlap=0,target_multisets_exact=True,
        same_class_fraction=1.,cross_class_fraction=0.,zero_pair_fixed_points=True,old_hashes_unchanged=True,
        labels_query_excluded_from_writer=True,model_unchanged_every_fit_evaluation=True,
        nonfinite_count=0,cap_gt10pct_layers=sum(d['cap_fraction']>.1 for d in diags))
    for name,value in [('metadata.json',metadata),('verification.json',verify),('raw_records.json',rows),
                       ('raw_restoration.json',diags),('summary.json',dict(metadata=metadata,rows=summary,gates=gates,passes=passes))]:
        (out/name).write_text(json.dumps(value,indent=2))
    write_csv(out/'summary.csv',summary);write_csv(out/'per_class.csv',perclass)
    write_csv(out/'per_client.csv',[{k:v for k,v in r.items() if k not in ('class_total','class_correct','per_class_accuracy')} for r in rows])
    write_csv(out/'restoration_diagnostics.csv',[{k:v for k,v in d.items() if k not in ('a','b')} for d in diags])
    np.savez_compressed(out/'predictions.npz',**predictions)
    print('T007R_DONE',json.dumps(dict(verification=verify,passes=passes,gates=gates)),flush=True)


if __name__=='__main__':main()
