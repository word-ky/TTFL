"""T002 evaluation only: matched support IDs/labels under fixed pixel shifts."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import adapt, new_state, state_norms
from src.eval.common import evaluate, state_hash
from src.data.covariate import corrupt, TARGETS, WRONG_ALT, sample_seed
from utils.data_utils import read_client_data
from scripts.eval_pfllib_context import tensors

CHECKPOINT_SHA = '260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def paired(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                fraction_positive=float(np.mean(np.array(values)>0)),
                **{f'p{p}':float(np.percentile(values,p)) for p in (10,25,75,90)})


def summarize(rows):
    summaries, pairs = [], []
    for target in TARGETS:
        groups = {ctx:[r for r in rows if r['target']==target and r['context']==ctx]
                  for ctx in ('none','correct','wrong_clean','wrong_alt','shuffled','noise_image')}
        weighted = {ctx:sum(r['accuracy']*r['n_query'] for r in group)/sum(r['n_query'] for r in group)
                    for ctx,group in groups.items()}
        for ctx, group in groups.items():
            summaries.append(dict(target=target,context=ctx,accuracy=weighted[ctx],
                                  macro_client_accuracy=float(np.mean([r['accuracy'] for r in group])),
                                  gain_pp=weighted[ctx]-weighted['none'],
                                  support_loss_before=float(np.mean([r['support_loss_before'] for r in group])),
                                  support_loss_after=float(np.mean([r['support_loss_after'] for r in group])),
                                  gamma_norm=float(np.mean([r['gamma_norm'] for r in group])),
                                  beta_norm=float(np.mean([r['beta_norm'] for r in group]))))
        clean = [c['accuracy']-w['accuracy'] for c,w in zip(groups['correct'],groups['wrong_clean'])]
        alt = [c['accuracy']-w['accuracy'] for c,w in zip(groups['correct'],groups['wrong_alt'])]
        checks = dict(gain_ge_2=weighted['correct']-weighted['none']>=2,
                      clean_gap_ge_2=weighted['correct']-weighted['wrong_clean']>=2,
                      alt_gap_ge_2=weighted['correct']-weighted['wrong_alt']>=2,
                      beats_noise=weighted['correct']>weighted['noise_image'])
        pairs.append(dict(target=target,correct_gain_pp=weighted['correct']-weighted['none'],
                          correct_minus_wrong_clean_pp=weighted['correct']-weighted['wrong_clean'],
                          correct_minus_wrong_alt_pp=weighted['correct']-weighted['wrong_alt'],
                          correct_minus_noise_pp=weighted['correct']-weighted['noise_image'],
                          wrong_clean_paired=paired(clean),wrong_alt_paired=paired(alt),
                          checks=checks,passed=all(checks.values())))
    return summaries, pairs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--clients',type=int,default=100)
    p.add_argument('--commit',required=True)
    a=p.parse_args()
    base, data, out = Path(a.baseline).resolve(), Path(a.data_root).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((base/'config.json').read_text())
    supports=json.loads((base/'support_indices.json').read_text())
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text())
    original_rows=json.loads((base/'context_records.json').read_text())
    assert cfg['dataset']=='Cifar10' and cfg['affine_lr']==.1 and cfg['affine_steps']==10
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    os.chdir(base/'system')  # Preserve the verified upstream relative dataset loader.
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',weights_only=True,map_location='cuda'))
    initial=state_hash(model)
    assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    started=time.time()
    rows, predictions, identities, client_pairs = [], {}, {}, []
    observed_min, observed_max, neutral_max, clean_error = 1., -1., 0., 0.
    checked_episodes=0
    for cid in range(a.clients):
        train=read_client_data('Cifar10',cid,is_train=True)
        query=read_client_data('Cifar10',cid,is_train=False)
        qx,qy=tensors(query)
        si=supports[str(cid)]['correct_train_indices']
        sx,sy=tensors([train[i] for i in si])
        ids=[manifest['clients'][str(cid)]['train'][i] for i in si]
        qids=manifest['clients'][str(cid)]['test']
        assert len(qids)==len(qy) and not set(ids).intersection(qids)
        assert len(set(ids))==len(ids)
        observed_min=min(observed_min,float(sx.min()),float(qx.min()))
        observed_max=max(observed_max,float(sx.max()),float(qx.max()))
        assert observed_min>=-1 and observed_max<=1
        clean_acc=evaluate(model,qx,qy)[0]['accuracy']
        old=next(r for r in original_rows if r['client']==cid and r['method']=='none')
        clean_error=max(clean_error,abs(clean_acc-old['accuracy']))
        assert abs(clean_acc-old['accuracy'])<1e-4
        perm=torch.randperm(len(sy),generator=torch.Generator().manual_seed(
            sample_seed(7,cid,0,'shuffle'))).cuda()
        identities[str(cid)]={'saved_train_indices':si,'support_original_ids':ids,
                              'query_original_ids':qids,'support_labels':sy.cpu().tolist(),
                              'label_permutation':perm.cpu().tolist()}
        predictions[f'c{cid}_labels']=qy.cpu().numpy()
        for target in TARGETS:
            tx=corrupt(qx,target,qids,cid)
            assert torch.equal(tx,corrupt(qx,target,qids,cid))
            with torch.no_grad():
                zero=new_state(model)
                diff=max((model(b)-model(b,zero)).abs().max().item() for b in tx.split(256))
            assert diff==0
            neutral_max=max(neutral_max,diff)
            correct=corrupt(sx,target,ids,cid)
            assert torch.equal(correct,corrupt(sx,target,ids,cid))
            contexts=[('none',target,correct,sy,ids),('correct',target,correct,sy,ids),
                      ('wrong_clean','clean',corrupt(sx,'clean',ids,cid),sy,ids),
                      ('wrong_alt',WRONG_ALT[target],corrupt(sx,WRONG_ALT[target],ids,cid),sy,ids),
                      ('shuffled',target,correct,sy[perm],ids),
                      ('noise_image','noise_image',corrupt(sx,'noise_image',ids,cid),sy,ids)]
            target_rows={}
            for context,transform,x,y,used_ids in contexts:
                assert used_ids==ids and len(x)==len(sx)
                assert torch.equal(torch.bincount(y,minlength=10),torch.bincount(sy,minlength=10))
                if context!='shuffled':
                    assert torch.equal(y,sy)
                before=evaluate(model,x,y)[0]['loss']
                if context=='none':
                    m,state,losses=model,None,[]
                else:
                    m,state,losses=adapt(model,x,y,10,.1)
                    assert state_hash(m)==initial and state_hash(model)==initial
                    checked_episodes+=1
                result,pred=evaluate(m,tx,qy,state)
                after=evaluate(m,x,y,state)[0]['loss']
                key=f'r{len(rows)}'
                predictions[key]=pred.numpy()
                row=dict(client=cid,target=target,context=context,support_transform=transform,
                         n_query=len(qy),n_support=len(sy),prediction_key=key,
                         support_loss_before=before,support_loss_after=after,
                         loss_trajectory=losses,**result,**state_norms(state))
                assert np.isfinite([before,after,result['loss'],row['gamma_norm'],row['beta_norm']]).all()
                rows.append(row)
                target_rows[context]=row
            client_pairs.append(dict(client=cid,target=target,n_query=len(qy),
                **{ctx+'_accuracy':r['accuracy'] for ctx,r in target_rows.items()},
                delta_wrong_clean=target_rows['correct']['accuracy']-target_rows['wrong_clean']['accuracy'],
                delta_wrong_alt=target_rows['correct']['accuracy']-target_rows['wrong_alt']['accuracy']))
        print('T002_CLIENT_DONE',cid,flush=True)
    assert state_hash(model)==initial and sha(base/'global_state.pt')==CHECKPOINT_SHA
    errors=[abs(float((predictions[r['prediction_key']]==predictions[f"c{r['client']}_labels"]).mean()*100)
                -r['accuracy']) for r in rows]
    assert max(errors)<1e-4
    summary,pairs=summarize(rows)
    passes=sum(r['passed'] for r in pairs)
    case='A' if passes>=2 else ('B' if any(r['correct_gain_pp']>0 for r in pairs) else 'C')
    metadata=dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,baseline=str(base),
                  support_file_sha256=sha(base/'support_indices.json'),
                  split_file_sha256=sha(data/'Cifar10/split_manifest.json'),clients=a.clients,
                  seed=7,affine_optimizer='SGD',affine_lr=.1,affine_steps=10,
                  input_observed_range=[observed_min,observed_max],
                  formulas='z=(x+1)/2; dark=.45*z; contrast=.35*(z-spatial_channel_mean)+mean; noise=clamp(z+.15*N(0,1),0,1); blur=Gaussian5 sigma1.5 reflect; output=2*z-1; noise_image=uniform[0,1]',
                  random_seed='SHA256(7:client:original_sample_id:transform), first8 bytes little-endian modulo2^63-1; CPU torch.Generator',
                  gpu=torch.cuda.get_device_name(),seconds=time.time()-started,
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  smoke=a.clients!=100)
    verification=dict(records=len(rows),adaptation_episodes=checked_episodes,
        matched_support_ids_and_labels=True,support_query_overlap=0,
        shared_hash_unchanged_each_episode=True,deterministic_corruptions=True,
        neutral_max_abs_diff=neutral_max,clean_baseline_max_error_pp=clean_error,
        prediction_max_error_pp=max(errors),checkpoint_sha256=CHECKPOINT_SHA)
    payload=dict(metadata=metadata,rows=summary,specificity=pairs,passed_corruptions=passes,case=case)
    for name,value in [('summary.json',payload),('verification.json',verification),
                       ('raw_records.json',rows),('identities.json',identities)]:
        (out/name).write_text(json.dumps(value,indent=2))
    write_csv(out/'summary.csv',summary)
    write_csv(out/'per_client.csv',client_pairs)
    np.savez_compressed(out/'predictions.npz',**predictions)
    lines=['# T002 Prior-Controlled Covariate Context Specificity', '',
           f"Case {case}; {passes}/4 corruptions pass. {'SMOKE ONLY' if a.clients!=100 else '100 clients, fixed existing checkpoint/support/query.'}", '',
           '| Corruption | None | Correct | Wrong clean | Wrong alt | Shuffled | Noise image | Pass |',
           '|---|---:|---:|---:|---:|---:|---:|:---:|']
    for target,pa in zip(TARGETS,pairs):
        group=[r for r in summary if r['target']==target]
        lines.append('| '+target+' | '+' | '.join(f"{r['accuracy']:.3f}" for r in group)+f" | {pa['passed']} |")
    lines += ['', 'Accuracies are sample-weighted percentages. Macro-client rows and mean support losses/norms are in summary.csv. Paired client differences and percentile summaries are in per_client.csv and summary.json.',
              '', 'PASS requires correct gain >=2pp, both matched-label wrong gaps >=2pp, and correct > noise-image. At least2/4 required overall. No severity/optimizer selection on these query results.',
              '', 'Existing PFLlib static label-skew partitions, merged original labeled CIFAR10 splits with per-client75/25 split; not an official CIFAR10 test benchmark. All conditions within each target share identical corrupted queries. Wrong supports share exact image IDs and labels; only pixel transform differs.',
              '', json.dumps(metadata,indent=2), '', json.dumps(pairs,indent=2)]
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print('T002_DONE',json.dumps({'case':case,'passed':passes,'specificity':pairs}),flush=True)


if __name__=='__main__':
    main()
