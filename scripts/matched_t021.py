"""Fixed exact-count L/H matched null after the complete Phase-A freeze."""
import argparse,hashlib,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,save,sha,gzload,C,B,S
from phase_a_t021 import gzsave
from src.context.representation_prototypes import class_pools,sample_class_pools
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.exact_template_lookup import compile_template
from src.context.bootstrap_regret import shared_integer_weights,exact_utilities_from_weights,canonical_and_mask


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--response',action='store_true')
    for k in ('project','phase-a','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);phase=Path(a.phase_a);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    f=load(phase/'phaseA_choices_freeze.json');assert f['status']=='PASS' and all(sha(phase/name)==digest for name,digest in f['hashes'].items())
    ext=Path(f['extraction']);inputs=load(phase/'input_hashes.json');assert all(sha(Path(path))==digest for path,digest in inputs.items())
    truth=load(p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics/support_truth.json')
    truth=sorted(truth,key=lambda r:r['client']);labels=np.array([r['labels_in_frozen_support_order'] for r in truth])
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p/'results/t014_class_conditional_factorization/class_templates.json.gz')['rows']}
    task='T022' if a.response else 'T021'
    save(out/'protocol.json',dict(lead='06647a0' if a.response else '7b5bb34',task=task,runtime=a.commit,phase_a=str(phase),phase_a_sha256=sha(phase/'phaseA_choices_freeze.json'),replicas=128,K=20,
        seed=f'full big-endian SHA256({task}|matched|representation|client|bank|context|replica) -> PCG64',exact_target_counts=True,query_outcomes_scored=False))
    cells=[];fallbacks=[]
    for rep,dim in ([('L',40),('H',2048)] if a.response else [('L',10),('H',512)]):
        data=dict(np.load(ext/f'{"signature" if a.response else "support"}_{rep}.npz'));matrices=np.load(phase/f'{rep}_prototypes.npz')['prototypes'][0]
        shape=(100,2,5,128);pi=np.zeros(shape+(10,));choices=np.zeros(shape+(8,),dtype=np.uint8);masks=np.zeros_like(choices)
        positions=np.zeros(shape+(20,),dtype=np.int32);qsample=np.zeros((100,2,5,2,dim));kkt=np.zeros(shape);sumerr=np.zeros(shape);updates=np.zeros(shape,dtype=np.uint8)
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                features=np.stack([data[f'{i}|{b}|{t}' if a.response else f'{i}|{b}|{t}|{t}'] for i in range(100)])
                for i in range(100):
                    pools,ids=class_pools(features,labels,i);matrix=matrices[i,bi,ti]
                    np.testing.assert_array_equal(np.stack([x.mean(0) for x in pools],axis=1),matrix)
                    solver=ActiveSetCLS(matrix);counts=truth[i]['counts'];compiled=[compile_template(templates[s,b,h,i][ti])[0] for s in S for h in (0,1)]
                    for r in range(128):
                        q,pos=sample_class_pools(pools,ids,i,counts,rep,b,t,r,task=task)
                        assert np.all(pos//20!=i);np.testing.assert_array_equal(np.bincount(labels.reshape(-1)[pos],minlength=10),counts)
                        try:
                            pp,receipt=solver.solve(q);assert receipt['objective']<=receipt['initial_objective']+1e-12
                        except Exception as error:
                            save(out/'implementation_blocker.json',dict(stage='matched_CLS',representation=rep,client=i,bank=b,context=t,replica=r,error=repr(error),matrix=matrix.tolist(),mean=q.tolist(),query_outcomes_scored=False));raise
                        loc=i,bi,ti,r;pi[loc]=pp;positions[loc]=pos;kkt[loc]=receipt['direct_KKT'];sumerr[loc]=receipt['sum_error'];updates[loc]=receipt['updates']
                        if r in (0,127):qsample[i,bi,ti,int(r==127)]=q
                        if receipt['solver_path']=='cycle_face_fallback':fallbacks.append(dict(representation=rep,client=i,bank=b,context=t,replica=r,pi=pp.tolist(),**receipt))
                        w,D=shared_integer_weights([pp])
                        for slot,coeff in enumerate(compiled):
                            values=exact_utilities_from_weights(w,coeff)[0];state,mask=canonical_and_mask(values);choices[loc+(slot,)]=state;masks[loc+(slot,)]=mask
                    cells.append(dict(representation=rep,client=i,bank=b,context=t,replicas=128,exact_class_counts=counts,source_ids_sha256=hashlib.sha256(positions[i,bi,ti].tobytes()).hexdigest(),
                        max_KKT=float(kkt[i,bi,ti].max()),max_sum_error=float(sumerr[i,bi,ti].max()),max_updates=int(updates[i,bi,ti].max())))
                print('T021_MATCHED_CONTEXT',rep,b,t,flush=True)
        np.savez_compressed(out/f'{rep}_matched.npz',pi=pi,choices=choices,argmax_masks=masks,source_ids=positions,qsample=qsample,KKT=kkt,sum_error=sumerr,updates=updates)
    gzsave(out/'matched_cells.json.gz',cells);save(out/'solver_fallbacks.json',fallbacks)
    assert all(sha(Path(path))==digest for path,digest in inputs.items())
    save(out/'matched_freeze.json',dict(status='PASS',runtime=a.commit,phase_a=str(phase),replicas=128,matched_solutions=256000,template_choices=2048000,
        solver_fallbacks=len(fallbacks),query_outcomes_scored=False,seconds=time.time()-start,hashes={name:sha(out/name) for name in ('L_matched.npz','H_matched.npz','matched_cells.json.gz','solver_fallbacks.json')}))
    print('T021_MATCHED_FROZEN',256000,flush=True)


if __name__=='__main__':main()
