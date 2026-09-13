"""T006 new semantic pairing controls with verified T005 receipt reuse."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.adaptation.paired_affine_oracle import paired_clean_oracle
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.data.semantic_pairing import semantic_pairings
from src.eval.common import evaluate,state_hash
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,write_csv
from utils.data_utils import read_data,read_client_data


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','t005','output','commit'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--clients',type=int,default=100)
    a=p.parse_args()
    base,data,old,out=map(lambda s:Path(s).resolve(),(a.baseline,a.data_root,a.t005,a.output))
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text())
    supports=json.loads((base/'support_indices.json').read_text())
    old_pairs=json.loads((old/'pairings.json').read_text())
    old_rows=json.loads((old/'raw_records.json').read_text())
    old_ds=json.loads((old/'raw_restoration.json').read_text())
    old_pred=np.load(old/'predictions.npz')
    old_summary=json.loads((old/'summary.json').read_text())
    old_verify=json.loads((old/'verification.json').read_text())
    writer_sha=sha(ROOT/'src/adaptation/paired_affine_oracle.py')
    assert writer_sha==old_verify['writer_sha256']
    assert sha(ROOT/'src/data/covariate.py')==old_summary['metadata']['corruption_sha256']
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model)
    assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    rows,ds,semantics,pairings,predictions=[],[],[],{},{}
    episodes,regressions=0,0
    start=time.time()
    mapping={'none':'none','oracle_correct_pair':'correct_pair','oracle_wrong_alt':'wrong_alt','oracle_noise_source':'noise_source'}
    for cid in range(a.clients):
        raw=read_data('Cifar10',cid,is_train=True)
        si=supports[str(cid)]['correct_train_indices']
        sx=torch.as_tensor(raw['x'][si],dtype=torch.float32).cuda()
        # Labels stop at pairing construction/receipt; the writer signature is unchanged.
        support_labels=torch.as_tensor(raw['y'][si],dtype=torch.long)
        choices=semantic_pairings(support_labels,cid)
        ids=[manifest['clients'][str(cid)]['train'][i] for i in si]
        query=read_client_data('Cifar10',cid,is_train=False)
        qx=torch.stack([x for x,y in query]).cuda();qy=torch.stack([y for x,y in query]).cuda()
        qids=manifest['clients'][str(cid)]['test']
        assert ids==old_pairs[str(cid)]['source_ids'] and qids==old_pairs[str(cid)]['query_ids']
        assert not set(ids).intersection(qids)
        assert np.array_equal(qy.cpu().numpy(),old_pred[f'c{cid}_labels'])
        predictions[f'c{cid}_labels']=qy.cpu().numpy()
        assert choices[1]['same_label_count']>=choices[0]['same_label_count']
        for choice in choices:
            perm=torch.tensor(choice['permutation'],device=sx.device)
            assert (perm!=torch.arange(len(sx),device=sx.device)).all()
            assert sorted([ids[i] for i in choice['permutation']])==sorted(ids)
            inverse=perm.argsort();assert torch.equal(sx[perm][inverse],sx)
            with torch.no_grad():
                h,hp=sx,sx[perm]
                for block in (model.conv1,model.conv2):
                    h,hp=block(h),block(hp)
                    assert torch.equal(hp[inverse],h)
            semantics.append(dict(client=cid,context=choice['context'],offset=choice['offset'],
                n_support=len(sx),n_query=len(qy),same_label_fraction=choice['same_label_fraction']))
        old_perm=torch.tensor(old_pairs[str(cid)]['permutation'])
        for ctx,offset,fraction in [('correct_pair',0,1.),('t005_original',old_pairs[str(cid)]['offset'],
                (support_labels==support_labels[old_perm]).double().mean().item())]:
            semantics.append(dict(client=cid,context=ctx,offset=offset,n_support=len(sx),n_query=len(qy),same_label_fraction=fraction))
        pairings[str(cid)]=dict(source_ids=ids,query_ids=qids,support_labels=support_labels.tolist(),choices=choices)
        for target in TARGETS:
            tx=corrupt(qx,target,qids,cid)
            assert torch.equal(tx,corrupt(qx,target,qids,cid))
            with torch.no_grad():
                zero=new_state(model)
                assert all(torch.equal(model(b),model(b,zero)) for b in tx.split(256))
            # Focused regression only on first2clients, all4target/oldconditions; others reused.
            for old_context,new_context in mapping.items():
                r=next(r for r in old_rows if r['client']==cid and r['target']==target and r['context']==old_context)
                if cid<2:
                    state=None
                    if new_context!='none':
                        transform={'correct_pair':target,'wrong_alt':WRONG_ALT[target],'noise_source':'noise_image'}[new_context]
                        state,_=paired_clean_oracle(model,sx,corrupt(sx,transform,ids,cid))
                        assert state_hash(model)==initial
                        regressions+=1
                    result,pred=evaluate(model,tx,qy,state)
                    assert abs(result['accuracy']-r['accuracy'])<1e-4
                    assert np.array_equal(pred.numpy(),old_pred[r['prediction_key']])
                key=f'r{len(rows)}';predictions[key]=old_pred[r['prediction_key']]
                rows.append(dict(r,context=new_context,prediction_key=key,reused_t005=True))
            ds.extend(dict(d,context='correct_pair',reused_t005=True) for d in old_ds
                      if d['client']==cid and d['target']==target and d['context']=='oracle_correct_pair')
            source=corrupt(sx,target,ids,cid)
            for choice in choices:
                perm=torch.tensor(choice['permutation'],device=sx.device)
                state,diag=paired_clean_oracle(model,sx[perm],source)
                assert state_hash(model)==initial
                episodes+=1
                result,pred=evaluate(model,tx,qy,state)
                key=f'r{len(rows)}';predictions[key]=pred.numpy()
                rows.append(dict(client=cid,target=target,context=choice['context'],n_query=len(qy),
                    prediction_key=key,reused_t005=False,**result))
                for d in diag:
                    if choice['context'].startswith('random_'):
                        d={k:v for k,v in d.items() if k not in ('a','b')}
                    ds.append(dict(client=cid,target=target,context=choice['context'],reused_t005=False,**d))
        print('T006_CLIENT_DONE',cid,flush=True)
    assert state_hash(model)==initial and sha(base/'global_state.pt')==CHECKPOINT_SHA
    error=max(abs(100*np.mean(predictions[r['prediction_key']]==predictions[f"c{r['client']}_labels"])-r['accuracy']) for r in rows)
    assert error<1e-4
    metadata=dict(code_commit=a.commit,clients=a.clients,smoke=a.clients!=100,checkpoint_sha256=CHECKPOINT_SHA,
        writer_sha256=writer_sha,corruption_sha256=sha(ROOT/'src/data/covariate.py'),
        t005_summary_sha256=sha(old/'summary.json'),clean_acc=old_summary['clean_identity']['none'],
        eps=1e-5,coefficient_cap=[-8,8],gpu=torch.cuda.get_device_name(),seconds=time.time()-start,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    verification=dict(records=len(rows),new_writer_episodes=episodes,focused_old_writer_regressions=regressions,
        old_prediction_regression_exact=True,old_condition_accuracy_max_error_pp=0.,
        all_derangements=True,target_multisets_exact_ids_images_features=True,labels_only_control_construction=True,
        labels_query_excluded_from_writer=True,high_ge_low_each_client=True,deterministic_random_offsets=True,
        support_query_overlap=0,model_unchanged_each_episode=True,zero_state_exact=True,nonfinite_count=0,
        prediction_max_error_pp=error,cap_gt10pct_layer_episodes=sum(d['cap_fraction']>.1 for d in ds))
    for name,value in [('metadata.json',metadata),('verification.json',verification),('raw_records.json',rows),
                       ('raw_restoration.json',ds),('pairings.json',pairings)]:
        (out/name).write_text(json.dumps(value,indent=2))
    write_csv(out/'pairing_semantics.csv',semantics)
    np.savez_compressed(out/'predictions.npz',**predictions)
    print('T006_EVALUATION_DONE',json.dumps(verification),flush=True)


if __name__=='__main__':
    main()
