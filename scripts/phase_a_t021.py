"""Frozen raw L/H target-excluded observation models and three actual policies."""
import argparse,gzip,hashlib,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,save,sha,gzload,C,B,S
from src.context.representation_prototypes import excluded_prototype
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.matched_channel import exact_utilities
from report_t009 import write_csv
POLICIES=['oracle','source','clean_same_state']
def gzsave(p,x):p.write_bytes(gzip.compress(json.dumps(x,allow_nan=False).encode(),mtime=0))


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','extraction','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);ext=Path(a.extraction);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    freeze=load(ext/'extraction_freeze.json');assert freeze['status']=='PASS'
    assert all(sha(ext/name)==digest for name,digest in freeze['representation_hashes'].items())
    inputs=load(ext/'input_hashes.json');assert all(sha(Path(path))==digest for path,digest in inputs.items())
    p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p9=p/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    # Labels are loaded solely to calibrate other-client roles. No counts/pi fields are used here.
    calibration=load(p16/'support_truth.json');labels=np.array([r['labels_in_frozen_support_order'] for r in sorted(calibration,key=lambda r:r['client'])])
    membership={str(i):[j for j in range(100) if j!=i] for i in range(100)}
    source={(r['client'],r['bank'],r['true_context']):r['selected'] for r in load(p9/'source_decisions.json')}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p/'results/t014_class_conditional_factorization/class_templates.json.gz')['rows']}
    inputs[str(p16/'support_truth.json')]=sha(p16/'support_truth.json')
    inputs.update({str(ext/name):digest for name,digest in freeze['representation_hashes'].items()})
    save(out/'input_hashes.json',inputs);save(out/'prototype_membership.json',dict(other_clients=membership,samples_per_other_client=20,labels_role='offline_other_client_calibration_only',target_i_labels_used_by_target_i_estimator=False))
    diagnostics=[];receipts=[];fallbacks=[];outputs=[]
    for rep,dim in [('L',10),('H',512)]:
        data=dict(np.load(ext/f'support_{rep}.npz'))
        prototypes=np.zeros((2,100,2,5,dim,10));counts=np.zeros((2,100,2,5,10),dtype=np.int32)
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for kind in (0,1):
                    context=t if kind==0 else 'clean';pool=np.stack([data[f'{i}|{b}|{context}|{t}'] for i in range(100)])
                    for i in range(100):
                        matrix,n,owners=excluded_prototype(pool,labels,i);assert owners==membership[str(i)]
                        prototypes[kind,i,bi,ti]=matrix;counts[kind,i,bi,ti]=n
                        gram=matrix.T@matrix;eig=np.linalg.eigvalsh(gram)
                        diagnostics.append(dict(representation=rep,dimension=dim,prototype_kind=['context','clean_same_state'][kind],client=i,bank=b,context=t,
                            matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),counts=n.tolist(),gram_eigenvalues=eig.tolist(),rank=int(np.linalg.matrix_rank(matrix)),
                            condition=float(np.linalg.cond(matrix)),target_excluded=True))
        np.savez_compressed(out/f'{rep}_prototypes.npz',prototypes=prototypes,class_counts=counts)
        means=np.zeros((3,100,2,5,dim));pis=np.zeros((3,100,2,5,10));choices=np.zeros((3,100,2,5,8),dtype=np.uint8);masks=np.zeros_like(choices)
        for i in range(100):
            for bi,b in enumerate(B):
                for ti,t in enumerate(C):
                    for policy_idx,policy in enumerate(POLICIES):
                        context=source[i,b,t] if policy=='source' else t;ci=C.index(context);kind=int(policy=='clean_same_state')
                        matrix=prototypes[kind,i,bi,ci];m=data[f'{i}|{b}|{t}|{context}'].astype(np.float64).mean(0)
                        try:
                            pi,r=ActiveSetCLS(matrix).solve(m)
                            assert r['objective']<=r['initial_objective']+1e-12
                        except Exception as error:
                            save(out/'implementation_blocker.json',dict(stage='actual_CLS',representation=rep,policy=policy,client=i,bank=b,context=t,error=repr(error),
                                matrix=matrix.tolist(),mean=m.tolist(),target_i_labels_used_by_target_i_estimator=False,query_outcomes_scored=False))
                            raise
                        loc=policy_idx,i,bi,ti;means[loc]=m;pis[loc]=pi
                        record=dict(representation=rep,policy=policy,client=i,bank=b,context=t,prototype_context=context,**r);receipts.append(record)
                        if r['solver_path']=='cycle_face_fallback':fallbacks.append(record)
                        for si,s in enumerate(S):
                            for h in (0,1):
                                values,args=exact_utilities(pi,templates[s,b,h,i][ci]);slot=2*si+h;choices[loc+(slot,)]=args[0];masks[loc+(slot,)]=sum(1<<j for j in args)
            if i%20==19:print('T021_PHASE_A_CLIENT',rep,i+1,flush=True)
        np.savez_compressed(out/f'{rep}_actual.npz',means=means,pi=pis,choices=choices,argmax_masks=masks)
        outputs.extend([f'{rep}_prototypes.npz',f'{rep}_actual.npz'])
    gzsave(out/'prototype_diagnostics.json.gz',diagnostics);gzsave(out/'actual_solver_receipts.json.gz',receipts);save(out/'solver_fallbacks.json',fallbacks)
    outputs+=['prototype_diagnostics.json.gz','actual_solver_receipts.json.gz','solver_fallbacks.json','prototype_membership.json']
    assert len(receipts)==6000 and all(sha(Path(path))==digest for path,digest in inputs.items())
    save(out/'phaseA_choices_freeze.json',dict(status='PASS',lead='7b5bb34',runtime=a.commit,extraction=str(ext),extraction_freeze_sha256=sha(ext/'extraction_freeze.json'),
        hashes={name:sha(out/name) for name in outputs},real_observation_episodes=6000,template_choices=48000,prototype_cells=4000,
        target_i_labels_used_by_target_i_estimator=False,target_composition_used_for_scoring=False,query_outcomes_scored=False,
        labels_loaded_only_for_other_client_calibration=True,solver_fallbacks=len(fallbacks),seconds=time.time()-start))
    print('T021_PHASE_A_FROZEN',len(receipts),flush=True)


if __name__=='__main__':main()
