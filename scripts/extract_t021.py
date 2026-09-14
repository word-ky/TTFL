"""T021 GPU equivalence preflight followed by frozen support representation pools."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t021_features import extract
from src.models.pfllib import ContextFedAvgCNN
from src.adaptation.affine import new_state
from src.data.covariate import corrupt,TARGETS
from src.eval.common import state_hash
from eval_t015_semantic_mixture import state_digest
from preflight_t018 import load,save,sha,C,B
from utils.data_utils import read_data


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','baseline','data-root','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);base=Path(a.baseline);data=Path(a.data_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p7=p/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
    p9=p/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    inputs={str(path):sha(path) for path in [base/'global_state.pt',p7/'states.pt',p9/'natural_support_manifest.json',p9/'source_decisions.json',p15/'support_logits.npz',p/'results/t014_class_conditional_factorization/class_templates.json.gz',data/'Cifar10/split_manifest.json',p7/'calibration_pools.json',ROOT/'src/models/pfllib.py',ROOT/'src/data/covariate.py']}
    assert inputs[str(base/'global_state.pt')]=='260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb'
    assert inputs[str(p7/'states.pt')]=='0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a'
    assert inputs[str(p9/'natural_support_manifest.json')]=='ba139b530bc231b13dc5f0fe9a370a25050bb74f1264eadc938472f56216afac'
    assert inputs[str(p15/'support_logits.npz')]==load(p15/'phaseA_freeze.json')['logits_sha256']
    assert inputs[str(p9/'source_decisions.json')]==load(p9/'selection_freeze.json')['decisions_sha256']
    save(out/'input_hashes.json',inputs)
    support=load(p9/'natural_support_manifest.json');split=load(data/'Cifar10/split_manifest.json');pools=load(p7/'calibration_pools.json')
    calibration_ids={r['original_id'] for pool in pools.values() for r in pool['samples']};query_ids={x for r in split['clients'].values() for x in r['test']}
    for i in range(100):
        ids={r['original_id'] for r in support['clients'][str(i)]['selected']};assert len(ids)==20 and not ids&calibration_ids and not ids&query_ids
    source={(r['client'],r['bank'],r['true_context']):r['selected'] for r in load(p9/'source_decisions.json')}
    old=dict(np.load(p15/'support_logits.npz'));os.chdir(base/'system');torch.set_num_threads(2);torch.manual_seed(7)
    model=ContextFedAvgCNN(3,10,1600).cuda().eval();model.load_state_dict(torch.load(base/'global_state.pt',map_location='cuda',weights_only=True))
    initial=state_hash(model);assert initial==load(base/'baseline.json')['final_hash']
    saved=torch.load(p7/'states.pt',map_location='cuda',weights_only=True)
    states={b:{'clean':new_state(model),**{c:saved[f'{b}|{c}|correct_pair'] for c in TARGETS}} for b in B}
    hashes={b:{c:state_digest(st) for c,st in bank.items()} for b,bank in states.items()}
    features={};logits={};comparisons=[];calls=0
    clients=sorted(set(np.linspace(0,99,20,dtype=int).tolist()+[64,87]))
    def client_calls(i,controls=False):
        nonlocal calls
        selected=support['clients'][str(i)]['selected'];ids=[r['original_id'] for r in selected];indices=[r['train_index'] for r in selected]
        x=torch.as_tensor(read_data('Cifar10',i,is_train=True)['x'][indices],dtype=torch.float32).cuda()
        for t in C:
            tx=corrupt(x,t,ids,i)
            for b in B:
                paths=C if controls and t=='clean' else list(dict.fromkeys([t,source[i,b,t]]))
                for c in paths:
                    key=f'{i}|{b}|{t}|{c}'
                    if key in features:continue
                    with torch.no_grad():h,l=extract(model,tx,states[b][c])
                    assert h.shape==(20,512) and l.shape==(20,10) and torch.isfinite(h).all() and torch.isfinite(l).all()
                    assert state_hash(model)==initial and state_digest(states[b][c])==hashes[b][c]
                    features[key]=h.cpu().numpy();logits[key]=l.cpu().numpy();calls+=1
                    if key in old:
                        diff=float(np.max(np.abs(logits[key]-old[key])));same=bool(np.array_equal(logits[key].argmax(1),old[key].argmax(1)))
                        comparisons.append(dict(client=i,bank=b,context=t,state=c,max_abs=diff,argmax_identical=same,samples=20))
                        if diff>1e-6 or not same:
                            save(out/'forward_equivalence.json',dict(status='T021-I',failed=comparisons[-1],checkpoint_unchanged=True,state_unchanged=True))
                            raise AssertionError('T021-I historical logits not reproduced')
    for i in clients:client_calls(i)
    assert sum(r['samples'] for r in comparisons)>=2000
    save(out/'forward_equivalence.json',dict(status='PASS',preflight_clients=clients,historical_samples=sum(r['samples'] for r in comparisons),
        max_abs=max(r['max_abs'] for r in comparisons),all_argmax_identical=True,model_hash=initial,state_digests=hashes,dimensions=dict(H=512,L=10),
        eval_no_grad=True,model_state_unchanged=True,support_calibration_query_disjoint=True,comparisons=comparisons))
    print('T021_FORWARD_EQUIVALENCE_PASS',sum(r['samples'] for r in comparisons),flush=True)
    for i in range(100):
        client_calls(i,controls=True)
        if i%20==19:print('T021_FEATURE_CLIENT',i+1,flush=True)
    np.savez_compressed(out/'support_H.npz',**features);np.savez_compressed(out/'support_L.npz',**logits)
    save(out/'historical_logit_replays.json',comparisons)
    assert state_hash(model)==initial and all(state_digest(st)==hashes[b][c] for b,bank in states.items() for c,st in bank.items())
    assert all(sha(Path(path))==digest for path,digest in inputs.items())
    save(out/'extraction_freeze.json',dict(status='PASS',runtime=a.commit,gpu=torch.cuda.get_device_name(),seconds=time.time()-start,forward_calls=calls,
        historical_samples=sum(r['samples'] for r in comparisons),max_logit_abs=max(r['max_abs'] for r in comparisons),model_hash=initial,state_digests=hashes,
        representation_hashes={name:sha(out/name) for name in ('support_H.npz','support_L.npz')},target_labels_used=False,query_outcomes_scored=False))
    print('T021_EXTRACTION_COMPLETE',calls,flush=True)


if __name__=='__main__':main()
