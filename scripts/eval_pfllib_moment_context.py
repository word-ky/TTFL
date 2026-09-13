"""T003 training-only reference collection and paired inference, no optimization."""
import argparse
import inspect
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
from src.adaptation.moments import estimate_global_reference_moments,write_affine_from_support_moments
from src.eval.common import evaluate,state_hash
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,paired,write_csv
from utils.data_utils import read_data,read_client_data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--t002',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--reference-only',action='store_true')
    p.add_argument('--clients',type=int,default=100)
    a=p.parse_args()
    base,data,out,refpath,oldpath=map(lambda s:Path(s).resolve(),
        (a.baseline,a.data_root,a.output,a.reference,a.t002))
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text())
    train_ids=[i for c in range(100) for i in manifest['clients'][str(c)]['train']]
    query_ids=[i for c in range(100) for i in manifest['clients'][str(c)]['test']]
    assert not set(train_ids).intersection(query_ids)
    assert len(set(train_ids))==len(train_ids)
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    os.chdir(base/'system')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model)
    assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    if a.reference_only:
        started=time.time()
        def images():
            for cid in range(100):
                # Select only pixels from unchanged PFLlib train loader; no labels forwarded.
                record=read_data('Cifar10',cid,is_train=True)
                pixels=torch.as_tensor(record['x'],dtype=torch.float32)
                assert len(pixels)==len(manifest['clients'][str(cid)]['train'])
                for batch in pixels.split(256):
                    yield batch.cuda()
                print('REFERENCE_CLIENT_DONE',cid,flush=True)
        reference=estimate_global_reference_moments(model,images())
        assert reference['samples']==len(train_ids)
        assert state_hash(model)==initial
        refpath.parent.mkdir(parents=True,exist_ok=True)
        torch.save(reference,refpath)
        provenance=dict(checkpoint_sha256=CHECKPOINT_SHA,code_commit=a.commit,
            split_manifest_sha256=sha(data/'Cifar10/split_manifest.json'),
            reference_sha256=sha(refpath),training_only=True,samples=len(train_ids),
            cap=None,order='client0..99, stored train index order, batch256',
            query_overlap=0,training_original_ids=train_ids,seconds=time.time()-started,
            layers=[dict(mu=l['mu'].tolist(),sigma=l['sigma'].tolist(),
                         activation_count=l['activation_count']) for l in reference['layers']])
        refpath.with_suffix('.json').write_text(json.dumps(provenance))
        print('REFERENCE_DONE',reference['samples'],sha(refpath),flush=True)
        return
    reference=torch.load(refpath,weights_only=True,map_location='cpu')
    provenance=json.loads(refpath.with_suffix('.json').read_text())
    assert provenance['training_original_ids']==train_ids and provenance['query_overlap']==0
    assert provenance['checkpoint_sha256']==CHECKPOINT_SHA and provenance['reference_sha256']==sha(refpath)
    assert list(inspect.signature(write_affine_from_support_moments).parameters)==['model','images','reference']
    supports=json.loads((base/'support_indices.json').read_text())
    old_ids=json.loads((oldpath/'identities.json').read_text())
    old_rows=json.loads((oldpath/'raw_records.json').read_text())
    old_summary=json.loads((oldpath/'summary.json').read_text())
    rows,diagnostics,sanity,predictions,identities=[],[],[],{},{}
    started=time.time(); zero_diff=0.; baseline_error=0.; episode_count=0
    for cid in range(a.clients):
        raw=read_data('Cifar10',cid,is_train=True)
        si=supports[str(cid)]['correct_train_indices']
        sx=torch.as_tensor(raw['x'][si],dtype=torch.float32).cuda()
        query=read_client_data('Cifar10',cid,is_train=False)
        qx=torch.stack([x for x,y in query]).cuda()
        qy=torch.stack([y for x,y in query]).cuda()  # Ground truth used only in reporting.
        ids=[manifest['clients'][str(cid)]['train'][i] for i in si]
        qids=manifest['clients'][str(cid)]['test']
        assert ids==old_ids[str(cid)]['support_original_ids'] and qids==old_ids[str(cid)]['query_original_ids']
        assert not set(ids).intersection(qids)
        identities[str(cid)]=dict(support_original_ids=ids,query_original_ids=qids)
        predictions[f'c{cid}_labels']=qy.cpu().numpy()
        clean_state,clean_diag=write_affine_from_support_moments(model,sx,reference)
        assert state_hash(model)==initial
        episode_count+=1
        clean_before,p0=evaluate(model,qx,qy)
        clean_after,p1=evaluate(model,qx,qy,clean_state)
        with torch.no_grad():
            differences=torch.cat([(model(b,clean_state)-model(b)).abs() for b in qx.split(256)])
        predictions[f'c{cid}_clean_none']=p0.numpy();predictions[f'c{cid}_clean_moment']=p1.numpy()
        sanity.append(dict(client=cid,n_query=len(qy),before=clean_before['accuracy'],
            after=clean_after['accuracy'],delta=clean_after['accuracy']-clean_before['accuracy'],
            logit_abs_difference_mean=differences.mean().item(),logit_abs_difference_max=differences.max().item()))
        for d in clean_diag:
            diagnostics.append(dict(client=cid,target='clean_sanity',context='moment_clean',**d))
        for target in TARGETS:
            tx=corrupt(qx,target,qids,cid)
            assert torch.equal(tx,corrupt(qx,target,qids,cid))
            with torch.no_grad():
                zero=new_state(model)
                diff=max((model(b)-model(b,zero)).abs().max().item() for b in tx.split(256))
            assert diff==0.;zero_diff=max(zero_diff,diff)
            contexts=[('none',None),('moment_correct',target),('moment_wrong_clean','clean'),
                      ('moment_wrong_alt',WRONG_ALT[target]),('moment_noise_image','noise_image')]
            for context,transform in contexts:
                state=None
                if transform is not None:
                    x=corrupt(sx,transform,ids,cid)
                    assert torch.equal(x,corrupt(sx,transform,ids,cid))
                    state,ds=write_affine_from_support_moments(model,x,reference)
                    assert state_hash(model)==initial
                    episode_count+=1
                    for d in ds:
                        diagnostics.append(dict(client=cid,target=target,context=context,**d))
                result,pred=evaluate(model,tx,qy,state)
                if context=='none':
                    old=next(r for r in old_rows if r['client']==cid and r['target']==target and r['context']=='none')
                    error=abs(result['accuracy']-old['accuracy'])
                    assert error<1e-4;baseline_error=max(baseline_error,error)
                key=f'r{len(rows)}';predictions[key]=pred.numpy()
                rows.append(dict(client=cid,target=target,context=context,n_query=len(qy),n_support=len(si),
                                 prediction_key=key,**result))
        print('T003_CLIENT_DONE',cid,flush=True)
    assert state_hash(model)==initial and sha(base/'global_state.pt')==CHECKPOINT_SHA
    summary,specificity,per_client=[],[],[]
    for target in TARGETS:
        groups={ctx:[r for r in rows if r['target']==target and r['context']==ctx]
                for ctx in ('none','moment_correct','moment_wrong_clean','moment_wrong_alt','moment_noise_image')}
        scores={ctx:sum(r['accuracy']*r['n_query'] for r in group)/sum(r['n_query'] for r in group)
                for ctx,group in groups.items()}
        for ctx,group in groups.items():
            summary.append(dict(target=target,context=ctx,accuracy=scores[ctx],
                macro_client_accuracy=float(np.mean([r['accuracy'] for r in group])),copied_t002=False))
        for ctx in ('correct','noise_image'):
            row=next(r for r in old_summary['rows'] if r['target']==target and r['context']==ctx)
            summary.append(dict(target=target,context='T002_ce_'+('correct' if ctx=='correct' else 'noise'),
                accuracy=row['accuracy'],macro_client_accuracy=row['macro_client_accuracy'],copied_t002=True))
        deltas={name:scores['moment_correct']-scores[ctx] for name,ctx in
            [('correct_gain_pp','none'),('correct_minus_wrong_clean_pp','moment_wrong_clean'),
             ('correct_minus_wrong_alt_pp','moment_wrong_alt'),('correct_minus_noise_pp','moment_noise_image')]}
        clean_delta=[c['accuracy']-w['accuracy'] for c,w in zip(groups['moment_correct'],groups['moment_wrong_clean'])]
        alt_delta=[c['accuracy']-w['accuracy'] for c,w in zip(groups['moment_correct'],groups['moment_wrong_alt'])]
        specificity.append(dict(target=target,**deltas,passed=all(v>=2 for v in deltas.values()),
                                wrong_clean_paired=paired(clean_delta),wrong_alt_paired=paired(alt_delta)))
        for cid in range(a.clients):
            per_client.append(dict(client=cid,target=target,n_query=groups['none'][cid]['n_query'],
                **{ctx:group[cid]['accuracy'] for ctx,group in groups.items()},
                delta_wrong_clean=clean_delta[cid],delta_wrong_alt=alt_delta[cid]))
    diagnostic_summary=[]
    for target,ctx,layer in dict.fromkeys((d['target'],d['context'],d['layer']) for d in diagnostics):
        ds=[d for d in diagnostics if (d['target'],d['context'],d['layer'])==(target,ctx,layer)]
        item=dict(target=target,context=ctx,layer=layer)
        for field in ('gamma_abs_mean','beta_abs_mean','clamp_fraction','clamp_low_fraction',
                      'clamp_high_fraction','moment_mismatch_before','moment_mismatch_after'):
            item[field]=float(np.mean([d[field] for d in ds]))
        for field in ('gamma_abs_max','beta_abs_max','clamp_fraction'):
            item[field+'_max']=max(d[field] for d in ds)
        item['clients_clamp_gt10pct']=sum(d['clamp_fraction']>.1 for d in ds)
        diagnostic_summary.append(item)
    errors=[abs(float(np.mean(predictions[r['prediction_key']]==predictions[f"c{r['client']}_labels"])*100)-r['accuracy']) for r in rows]
    assert max(errors)<1e-4
    passes=sum(r['passed'] for r in specificity)
    # M-B/M-C are descriptive research categories, unlike the exact frozen M-A gate.
    case='M-A' if passes>=2 else 'REVIEW_M_B_OR_M_C'
    total=sum(r['n_query'] for r in sanity)
    sanity_summary={k:sum(r[k]*r['n_query'] for r in sanity)/total for k in ('before','after','delta')}
    metadata=dict(code_commit=a.commit,reference_sha256=sha(refpath),checkpoint_sha256=CHECKPOINT_SHA,
        clients=a.clients,smoke=a.clients!=100,seed=7,eps=1e-5,scale_bounds=[.25,4.],
        sequential=True,labels_to_writer=False,optimizer=False,gpu=torch.cuda.get_device_name(),
        seconds=time.time()-started,reference_samples=reference['samples'],
        corruption_source_sha256=sha(ROOT/'src/data/covariate.py'),
        t002_summary_sha256=sha(oldpath/'summary.json'),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    verification=dict(records=len(rows),writer_episodes=episode_count,reference_training_only=True,
        reference_query_overlap=0,support_query_overlap=0,matched_support_ids=True,
        label_free_writer_api=True,shared_model_unchanged_each_episode=True,neutral_max_abs_diff=zero_diff,
        t002_baseline_max_error_pp=baseline_error,prediction_max_error_pp=max(errors),
        moment_mismatch_increases_gt1e_minus5=sum(d['moment_mismatch_after']>d['moment_mismatch_before']+1e-5 for d in diagnostics),
        moment_mismatch_max_increase=max(d['moment_mismatch_after']-d['moment_mismatch_before'] for d in diagnostics),
        layer_episodes_clamp_gt10pct=sum(d['clamp_fraction']>.1 for d in diagnostics))
    payload=dict(metadata=metadata,rows=summary,specificity=specificity,passed_corruptions=passes,
                 case=case,clean_sanity=sanity_summary,diagnostics=diagnostic_summary)
    for name,value in [('summary.json',payload),('verification.json',verification),('raw_records.json',rows),
                       ('writer_diagnostics.json',diagnostics),('identities.json',identities)]:
        (out/name).write_text(json.dumps(value,indent=2))
    for name,value in [('summary.csv',summary),('per_client.csv',per_client),
                       ('clean_sanity.csv',sanity),('moment_diagnostics.csv',diagnostic_summary)]:
        write_csv(out/name,value)
    np.savez_compressed(out/'predictions.npz',**predictions)
    lines=['# T003 Moment-Written Context Diagnostic','',f'{passes}/4 PASS; case {case}.',
        '', '| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for target in TARGETS:
        lines.append('| '+target+' | '+' | '.join(f"{r['accuracy']:.3f}" for r in summary if r['target']==target)+' |')
    lines+=['','Weighted accuracies (%); macro-client values in summary.csv. T002 CE rows are copied100-client references, not recomputed. Smoke primary rows, if present, use2clients only.',
        '',f"Clean sanity: {sanity_summary['before']:.3f}% -> {sanity_summary['after']:.3f}%.",
        '',f"Clamp warning: {verification['layer_episodes_clamp_gt10pct']} layer episodes have >10% channels at a clamp boundary. See moment_diagnostics.csv for every layer/context.",
        '', 'Frozen gate: all four gain/gap values>=2pp per corruption;>=2/4 forM-A. No query tuning. Reference uses all train data only; labels are used only to evaluate queries.',
        '', '```json',json.dumps(specificity,indent=2),'```']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print('T003_DONE',json.dumps(dict(passed=passes,specificity=specificity,clean_sanity=sanity_summary)),flush=True)


if __name__=='__main__':
    main()
