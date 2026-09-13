"""Supervised static-client context diagnostic after a fixed PFLlib checkpoint."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import adapt,new_state,state_norms
from src.eval.common import evaluate,state_hash
from utils.data_utils import read_client_data


def tensors(data):
    return torch.stack([x for x,y in data]).cuda(),torch.stack([y for x,y in data]).cuda()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    a=p.parse_args()
    out=Path(a.baseline).resolve()
    cfg=json.loads((out/'config.json').read_text())
    os.chdir(out/'system')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    model=ContextFedAvgCNN(cfg['in_features'],cfg['num_classes'],cfg['dim']).cuda().eval()
    checkpoint=out/'global_state.pt'
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    sha=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    initial=state_hash(model)
    assert initial==json.loads((out/'baseline.json').read_text())['final_hash']
    rows,predictions,support_ids=[],{},{}
    for cid in range(2 if cfg['smoke'] else 100):
        query=read_client_data(cfg['dataset'],cid,is_train=False)
        qx,qy=tensors(query)
        train=read_client_data(cfg['dataset'],cid,is_train=True)
        wrong_id=(cid+1)%100
        wrong=read_client_data(cfg['dataset'],wrong_id,is_train=True)
        gen=torch.Generator().manual_seed(cfg['seed']+cid)
        si=torch.randperm(len(train),generator=gen)[:cfg['support_count']]
        wi=torch.randperm(len(wrong),generator=gen)[:min(len(si),len(wrong))]
        sx,sy=tensors([train[i] for i in si.tolist()])
        wx,wy=tensors([wrong[i] for i in wi.tolist()])
        perm=torch.randperm(len(sy),generator=gen)
        noise=torch.rand(sx.shape,generator=gen).cuda()*2-1
        support_ids[str(cid)]={'correct_train_indices':si.tolist(),'wrong_client':wrong_id,
                              'wrong_train_indices':wi.tolist(),'label_permutation':perm.tolist()}
        neutral=(model(qx[:8])-model(qx[:8],new_state(model))).abs().max().item()
        assert neutral<1e-6
        base,_=evaluate(model,qx,qy,num_classes=cfg['num_classes'])
        predictions[f'c{cid}_labels']=qy.cpu().numpy()
        def record(m,state,method,context,x,y,bias=None,losses=None):
            result,pred=evaluate(m,qx,qy,state,bias,num_classes=cfg['num_classes'])
            key=f'r{len(rows)}'
            predictions[key]=pred.numpy()
            rows.append(dict(client=cid,method=method,context=context,n_query=len(qy),n_support=len(y),
                             baseline_accuracy=base['accuracy'],gain_pp=result['accuracy']-base['accuracy'],
                             neutral_max_abs_diff=neutral,checkpoint_sha256=sha,prediction_key=key,
                             support_loss_before=evaluate(model,x,y,num_classes=cfg['num_classes'])[0]['loss'],
                             support_loss_after=evaluate(m,x,y,state,bias,num_classes=cfg['num_classes'])[0]['loss'],
                             loss_trajectory=losses,**result,**state_norms(state)))
        record(model,None,'none','none',sx,sy)
        record(model,new_state(model,.01,cfg['seed']+cid),'random_no_write','random',sx,sy)
        for context,x,y in [('correct',sx,sy),('wrong',wx,wy),('shuffled',sx,sy[perm.cuda()]),('noise',noise,sy)]:
            m,state,losses=adapt(model,x,y,cfg['affine_steps'],cfg['affine_lr'])
            assert state_hash(m)==initial and state_hash(model)==initial
            record(m,state,'affine',context,x,y,losses=losses)
            prior=(torch.bincount(y,minlength=cfg['num_classes']).float()+1)/(len(y)+cfg['num_classes'])
            record(model,None,'prior',context,x,y,bias=(prior*cfg['num_classes']).log())
        print('CLIENT_CONTEXT_DONE',cid,flush=True)
    summary=[]
    for method,context in dict.fromkeys((r['method'],r['context']) for r in rows):
        group=[r for r in rows if r['method']==method and r['context']==context]
        total=sum(r['n_query'] for r in group)
        summary.append(dict(method=method,context=context,
                            accuracy=sum(r['accuracy']*r['n_query'] for r in group)/total,
                            gain_pp=sum(r['gain_pp']*r['n_query'] for r in group)/total,
                            macro_client_accuracy=float(np.mean([r['accuracy'] for r in group]))))
    (out/'context_records.json').write_text(json.dumps(rows,indent=2))
    (out/'context_summary.json').write_text(json.dumps({'rows':summary,'bn':'N/A: upstream FedAvgCNN has no BN',
                    'setting':'Static client label-skew, supervised support. Not dynamic context proof.',
                    'prior_reference':'uniform classes with Laplace1 support histogram',
                    'evaluation_source_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest()},indent=2))
    (out/'support_indices.json').write_text(json.dumps(support_ids))
    np.savez_compressed(out/'context_predictions.npz',**predictions)
    print('CONTEXT_DONE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
