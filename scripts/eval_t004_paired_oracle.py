"""Evaluation-only paired-clean oracle; mandatory identity precedes corruption."""
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
from src.eval.common import evaluate,state_hash
from src.data.covariate import corrupt,TARGETS,WRONG_ALT
from src.data.pairing import target_derangement
from scripts.eval_pfllib_covariate_context import CHECKPOINT_SHA,sha,paired,write_csv
from utils.data_utils import read_data,read_client_data


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','data-root','output','t002','t003','identity','commit'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--identity-only',action='store_true')
    p.add_argument('--clients',type=int,default=100)
    p.add_argument('--pairing-audit',action='store_true')
    p.add_argument('--t004')
    a=p.parse_args()
    base,data,out,old2,old3,identity=map(lambda s:Path(s).resolve(),
        (a.baseline,a.data_root,a.output,a.t002,a.t003,a.identity))
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'Cifar10/split_manifest.json').read_text())
    supports=json.loads((base/'support_indices.json').read_text())
    old_ids=json.loads((old2/'identities.json').read_text())
    old_rows=json.loads((old2/'raw_records.json').read_text())
    old4=json.loads((Path(a.t004)/'raw_records.json').read_text()) if a.pairing_audit else None
    os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    assert sha(base/'global_state.pt')==CHECKPOINT_SHA
    model=ContextFedAvgCNN(3,10,1600).cuda().eval()
    model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model)
    assert initial==json.loads((base/'baseline.json').read_text())['final_hash']
    def client(cid):
        raw=read_data('Cifar10',cid,is_train=True)
        si=supports[str(cid)]['correct_train_indices']
        sx=torch.as_tensor(raw['x'][si],dtype=torch.float32).cuda()
        query=read_client_data('Cifar10',cid,is_train=False)
        qx=torch.stack([x for x,y in query]).cuda();qy=torch.stack([y for x,y in query]).cuda()
        ids=[manifest['clients'][str(cid)]['train'][i] for i in si]
        qids=manifest['clients'][str(cid)]['test']
        assert ids==old_ids[str(cid)]['support_original_ids'] and qids==old_ids[str(cid)]['query_original_ids']
        assert not set(ids).intersection(qids)
        return sx,qx,qy,ids,qids
    rows,ds,predictions=[],[],{}
    started=time.time()
    if a.identity_only:
        for cid in range(100):
            sx,qx,qy,ids,qids=client(cid)
            state,diag=paired_clean_oracle(model,sx,sx)
            assert state_hash(model)==initial
            none,p0=evaluate(model,qx,qy);result,p1=evaluate(model,qx,qy,state)
            with torch.no_grad():
                diff=torch.cat([(model(b,state)-model(b)).abs() for b in qx.split(256)])
            rows.append(dict(client=cid,n_query=len(qy),none=none['accuracy'],oracle=result['accuracy'],
                delta=result['accuracy']-none['accuracy'],logit_abs_diff_mean=diff.mean().item(),
                logit_abs_diff_max=diff.max().item()))
            predictions[f'c{cid}_labels']=qy.cpu().numpy();predictions[f'c{cid}_none']=p0.numpy();predictions[f'c{cid}_oracle']=p1.numpy()
            ds.extend(dict(client=cid,**d) for d in diag)
        total=sum(r['n_query'] for r in rows)
        scores={k:sum(r[k]*r['n_query'] for r in rows)/total for k in ('none','oracle','delta')}
        result=dict(**scores,passed=abs(scores['delta'])<=.1,clients=100,checkpoint_sha256=CHECKPOINT_SHA,
                    code_commit=a.commit,seconds=time.time()-started,
                    per_client_max_abs_delta=max(abs(r['delta']) for r in rows))
        identity.write_text(json.dumps(result,indent=2))
        write_csv(out/'clean_identity.csv',rows)
        (out/'identity_diagnostics.json').write_text(json.dumps(ds,indent=2))
        np.savez_compressed(out/'identity_predictions.npz',**predictions)
        print('IDENTITY_DONE',json.dumps(result),flush=True)
        assert result['passed'],'Mandatory clean identity failed; no formal corruption evaluation.'
        return
    sanity=json.loads(identity.read_text())
    assert sanity['passed'] and sanity['checkpoint_sha256']==CHECKPOINT_SHA
    baseline_error=0.;episodes=0
    pairings={};clean_scores=[];old4_error=0.;identity_regression=None
    correct_name='oracle_correct_pair' if a.pairing_audit else 'oracle_correct'
    second_name='oracle_target_permuted' if a.pairing_audit else 'oracle_wrong_clean'
    for cid in range(a.clients):
        sx,qx,qy,ids,qids=client(cid)
        if a.pairing_audit:
            permutation,offset=target_derangement(len(sx),cid)
            assert sorted(permutation.tolist())==list(range(len(sx)))
            assert len(sx)<=1 or (permutation!=torch.arange(len(sx))).all()
            permutation=permutation.cuda();inverse=permutation.argsort()
            assert torch.equal(sx[permutation][inverse],sx)
            with torch.no_grad():
                h,hp=sx,sx[permutation]
                for block in (model.conv1,model.conv2):
                    h,hp=block(h),block(hp)
                    assert torch.equal(hp[inverse],h)
            pairings[str(cid)]=dict(source_ids=ids,target_ids=[ids[i] for i in permutation.cpu().tolist()],
                permutation=permutation.cpu().tolist(),offset=offset,query_ids=qids)
            clean_result,_=evaluate(model,qx,qy)
            clean_scores.append((len(qy),clean_result['accuracy']))
            if cid==0:
                identity_state,_=paired_clean_oracle(model,sx,sx)
                identity_result,_=evaluate(model,qx,qy,identity_state)
                identity_regression=identity_result['accuracy']-clean_result['accuracy']
                assert abs(identity_regression)<=.1 and state_hash(model)==initial
        predictions[f'c{cid}_labels']=qy.cpu().numpy()
        for target in TARGETS:
            tx=corrupt(qx,target,qids,cid)
            assert torch.equal(tx,corrupt(qx,target,qids,cid))
            with torch.no_grad():
                zero=new_state(model)
                assert all(torch.equal(model(b),model(b,zero)) for b in tx.split(256))
            for ctx,transform in [('none',None),(correct_name,target),(second_name,target if a.pairing_audit else 'clean'),
                                   ('oracle_wrong_alt',WRONG_ALT[target]),('oracle_noise_source','noise_image')]:
                state=None
                if transform:
                    source=corrupt(sx,transform,ids,cid)
                    assert torch.equal(source,corrupt(sx,transform,ids,cid))
                    clean_target=sx[permutation] if a.pairing_audit and ctx==second_name else sx
                    state,diag=paired_clean_oracle(model,clean_target,source)
                    assert state_hash(model)==initial
                    episodes+=1
                    ds.extend(dict(client=cid,target=target,context=ctx,**d) for d in diag)
                result,pred=evaluate(model,tx,qy,state)
                if a.pairing_audit and ctx!=second_name:
                    old_context='oracle_correct' if ctx==correct_name else ctx
                    old=next(r for r in old4 if r['client']==cid and r['target']==target and r['context']==old_context)
                    error=abs(result['accuracy']-old['accuracy']);assert error<1e-4
                    old4_error=max(old4_error,error)
                if ctx=='none':
                    old=next(r for r in old_rows if r['client']==cid and r['target']==target and r['context']=='none')
                    error=abs(result['accuracy']-old['accuracy']);assert error<1e-4
                    baseline_error=max(baseline_error,error)
                key=f'r{len(rows)}';predictions[key]=pred.numpy()
                rows.append(dict(client=cid,target=target,context=ctx,n_query=len(qy),
                                 prediction_key=key,**result))
        print('T005_CLIENT_DONE' if a.pairing_audit else 'T004_CLIENT_DONE',cid,flush=True)
    assert state_hash(model)==initial and sha(base/'global_state.pt')==CHECKPOINT_SHA
    summaries,specificity,per_client,restoration=[],[],[],[]
    for target in TARGETS:
        groups={ctx:[r for r in rows if r['target']==target and r['context']==ctx] for ctx in
            ('none',correct_name,second_name,'oracle_wrong_alt','oracle_noise_source')}
        acc={ctx:sum(r['accuracy']*r['n_query'] for r in group)/sum(r['n_query'] for r in group)
             for ctx,group in groups.items()}
        for ctx,group in groups.items():
            summaries.append(dict(target=target,context=ctx,accuracy=acc[ctx],
                macro_client_accuracy=float(np.mean([r['accuracy'] for r in group])),copied=False))
        for path,ctx,name in [(old2,'correct','T002_ce_correct'),(old3,'moment_correct','T003_moment_correct')]:
            old=next(r for r in json.loads((path/'summary.json').read_text())['rows'] if r['target']==target and r['context']==ctx)
            summaries.append(dict(target=target,context=name,accuracy=old['accuracy'],
                                 macro_client_accuracy=old['macro_client_accuracy'],copied=True))
        gains=dict(correct_gain_pp=acc[correct_name]-acc['none'],
                   correct_minus_wrong_alt_pp=acc[correct_name]-acc['oracle_wrong_alt'],
                   correct_minus_noise_pp=acc[correct_name]-acc['oracle_noise_source'])
        deltas={ctx:[c['accuracy']-w['accuracy'] for c,w in zip(groups[correct_name],groups[ctx])]
                for ctx in (second_name,'oracle_wrong_alt','oracle_noise_source')}
        if a.pairing_audit:
            clean_acc=sum(n*acc for n,acc in clean_scores)/sum(n for n,acc in clean_scores)
            if a.clients==100:
                assert abs(clean_acc-sanity['none'])<1e-4
            headroom=clean_acc-acc['none']
            recovery=gains['correct_gain_pp']/headroom if headroom>0 else None
            advantage=acc[correct_name]-acc[second_name]
            specificity.append(dict(target=target,**gains,clean_acc=clean_acc,headroom_pp=headroom,
                recovery_fraction=recovery,pairing_advantage_pp=advantage,
                passed=recovery is not None and recovery>=.5 and advantage>=2 and
                    gains['correct_minus_wrong_alt_pp']>=2 and gains['correct_minus_noise_pp']>=2,
                paired={ctx:paired(delta) for ctx,delta in deltas.items()}))
        else:
            specificity.append(dict(target=target,**gains,wrong_clean_minus_none_pp=acc['oracle_wrong_clean']-acc['none'],
                passed=gains['correct_gain_pp']>=5 and gains['correct_minus_wrong_alt_pp']>=2 and gains['correct_minus_noise_pp']>=2,
                paired={ctx:paired(delta) for ctx,delta in deltas.items()}))
        for cid in range(a.clients):
            per_client.append(dict(client=cid,target=target,n_query=groups['none'][cid]['n_query'],
                **{ctx:group[cid]['accuracy'] for ctx,group in groups.items()},
                **{ctx+'_delta':delta[cid] for ctx,delta in deltas.items()}))
    for target,ctx,layer in dict.fromkeys((d['target'],d['context'],d['layer']) for d in ds):
        group=[d for d in ds if (d['target'],d['context'],d['layer'])==(target,ctx,layer)]
        ratios=[d['restoration_ratio'] for d in group if d['restoration_ratio'] is not None]
        item=dict(target=target,context=ctx,layer=layer,ratio_defined_clients=len(ratios),
                  restoration_ratio=float(np.mean(ratios)) if ratios else None)
        for k in ('mse_before','mse_after','gamma_abs_mean','beta_abs_mean','cap_fraction','negative_scale_fraction'):
            item[k]=float(np.mean([d[k] for d in group]))
        for k in ('gamma_abs_max','beta_abs_max','cap_fraction'):
            item[k+'_max']=max(d[k] for d in group)
        item['clients_cap_gt10pct']=sum(d['cap_fraction']>.1 for d in group)
        restoration.append(item)
    errors=[abs(100*np.mean(predictions[r['prediction_key']]==predictions[f"c{r['client']}_labels"])-r['accuracy']) for r in rows]
    assert max(errors)<1e-4
    passes=sum(r['passed'] for r in specificity)
    case='O-A' if passes>=2 else ('O-B' if passes==1 else 'REVIEW_O_B_OR_O_C')
    if a.pairing_audit:
        recovery_count=sum(r['recovery_fraction'] is not None and r['recovery_fraction']>=.5 for r in specificity)
        specific_count=sum(r['pairing_advantage_pp']>=2 and r['correct_minus_wrong_alt_pp']>=2 and r['correct_minus_noise_pp']>=2 for r in specificity)
        case='P-A' if passes>=2 else ('P-B' if recovery_count>=2 else ('P-C' if specific_count>=2 else 'P-D'))
    metadata=dict(code_commit=a.commit,checkpoint_sha256=CHECKPOINT_SHA,clients=a.clients,
        smoke=a.clients!=100,writer='paired_clean_oracle',deployable=False,labels_to_writer=False,
        query_to_writer=False,eps=1e-5,coefficient_cap=[-8,8],sequential=True,
        corruption_sha256=sha(ROOT/'src/data/covariate.py'),gpu=torch.cuda.get_device_name(),
        seconds=time.time()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    metadata['task']='T005' if a.pairing_audit else 'T004'
    verify=dict(records=len(rows),writer_episodes=episodes,zero_state_exact=True,
        global_hash_unchanged_each_episode=True,support_query_overlap=0,matched_ids=True,
        query_and_labels_excluded_from_writer=True,nonfinite_count=0,
        t002_none_max_error_pp=baseline_error,prediction_max_error_pp=max(errors),
        cap_gt10pct_layer_episodes=sum(d['cap_fraction']>.1 for d in ds),clean_identity=sanity)
    if a.pairing_audit:
        verify.update(target_derangements=True,target_image_and_feature_multisets_exact=True,
            clean_identity_focused_delta_pp=identity_regression,t004_reused_conditions_max_error_pp=old4_error,
            writer_sha256=sha(ROOT/'src/adaptation/paired_affine_oracle.py'))
        (out/'pairings.json').write_text(json.dumps(pairings,indent=2))
    payload=dict(metadata=metadata,rows=summaries,specificity=specificity,passed_corruptions=passes,case=case,
                 clean_identity=sanity,restoration=restoration)
    for name,value in [('summary.json',payload),('verification.json',verify),('raw_records.json',rows),('raw_restoration.json',ds)]:
        (out/name).write_text(json.dumps(value,indent=2))
    for name,value in [('summary.csv',summaries),('per_client.csv',per_client),('restoration_diagnostics.csv',restoration)]:
        write_csv(out/name,value)
    np.savez_compressed(out/'predictions.npz',**predictions)
    print('T005_DONE' if a.pairing_audit else 'T004_DONE',json.dumps(dict(case=case,passed=passes,specificity=specificity)),flush=True)


if __name__=='__main__':
    main()
