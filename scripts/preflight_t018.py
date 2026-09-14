"""T018 numerical sanity before any T018 query evaluation; fixed 200-case reference."""
import argparse,csv,gzip,hashlib,json,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.constrained_prevalence import cls_s,direct_kkt,face_reference
from src.context.confusion_prevalence import project_simplex
from src.context.matched_channel import exact_utilities
from src.context.noise_free_ties import noise_free_sanity
from report_t009 import write_csv
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=['T013-S0','T013-S1','T013-S2','T013-S3']
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p14=project/'results/t014_class_conditional_factorization';p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics';p17=project/'runs/20260914-083609-ttfl-t017r-cached/artifacts/t017_channel_noise_decomposition'
    protocol=dict(lead='1744440',status_confirmation='e3e9c70',runtime=a.commit,timestamp=datetime.now(timezone.utc).isoformat(),objective='0.5*||C*pi-q||^2 on simplex',step='1/||C||_2^2',warm_start='unchanged ProjectSimplex(pinv(C)@q)',
        max_step_tolerance=1e-12,projected_gradient_residual_tolerance=1e-10,residual_definition='L*maxabs(pi-ProjectSimplex(pi-grad(pi)/L)), evaluated at returned pi',hard_cap=20000,
        independent_subset='clients0..19, bothbanks, all5oraclecontexts =200actual observations',independent_reference='exhaustive nonempty-simplex-face equality constrained solve; minimum feasible objective',reference_objective_atol=1e-10,unique_prevalence_atol=1e-7,
        query_metrics_opened=False,new_model_forwards=0)
    save(out/'protocol_freeze.json',protocol)
    inputs={}
    f16=load(p16/'phaseA_freeze.json');f17=load(p17/'bootstrap_receipt.json')
    for name,key in [('source_mixtures.json.gz','mixtures_sha256'),('calibration_counts.npz','calibration_sha256')]:assert sha(p16/name)==f16[key];inputs[str(p16/name)]=f16[key]
    for name,digest in f17['hashes'].items():assert sha(p17/name)==digest;inputs[str(p17/name)]=digest
    for path,expected in [(p14/'class_templates.json.gz','cd272ab8de6c9bb582972b0d700fe412af9b6bdc6287c4bc955723d311fd7289'),(p15/'support_logits.npz',load(p15/'phaseA_freeze.json')['logits_sha256']),
        (p15/'privileged_policy_choices.json.gz',load(p15/'phaseB_freeze.json')['choices_sha256']),
        (project/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer/states.pt','0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a'),
        (Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10/global_state.pt'),'260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb')]:
        assert sha(path)==expected;inputs[str(path)]=expected
    for p in (p16/'support_truth.json',ROOT/'src/context/confusion_prevalence.py'):inputs[str(p)]=sha(p)
    save(out/'input_hashes.json',inputs)
    cal=np.load(p16/'calibration_counts.npz');truth={r['client']:r for r in load(p16/'support_truth.json')};observations={(r['client'],r['bank'],r['target']):r for r in gzload(p16/'source_mixtures.json.gz') if r['policy']=='BBSE-S-01'}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    p00={(r['client'],r['bank'],r['target'],r['salt'],r['train_half']):r for r in gzload(p15/'privileged_policy_choices.json.gz') if r['policy']=='P00'}
    noise=[];singleton=tied=0
    for i in range(100):
        true=np.array(truth[i]['pi'])
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];q=matrix@true;pi,r=cls_s(matrix,q)
                assert r['converged'] and np.max(np.abs(pi-true))<=1e-8 and r['objective']<=1e-16
                for salt in S:
                    for train in (0,1):
                        template=templates[salt,b,train,i][ti];values,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],template);saved=p00[i,b,t,salt,train]
                        assert values==[Fraction(x) for x in saved['utility']] and [C[s] for s in arg]==saved['argmax']
                        _,selected=exact_utilities(pi,template);check=noise_free_sanity(pi,true,selected[0],values,arg[0]);tied+=check['tied'];singleton+=not check['tied']
                noise.append(dict(client=i,bank=b,context=t,max_error=float(np.abs(pi-true).max()),objective=r['objective'],iterations=r['iterations']))
    write_csv(out/'noise_free.csv',noise);save(out/'noise_free_verification.json',dict(passed=True,episodes=1000,singleton_checks=singleton,tied_checks=tied,max_pi_error=max(r['max_error'] for r in noise),max_objective=max(r['objective'] for r in noise)))
    print('T018_NOISE_FREE_PASS',flush=True)
    rows=[];vectors=[]
    for i in range(20):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];q=np.array(observations[i,b,t]['raw_soft'])
                pi,r=cls_s(matrix,q);ref,obj=face_reference(matrix,q);reference_kkt=direct_kkt(matrix,q,ref);objective_error=abs(obj-r['objective']);prevalence_error=float(np.abs(ref-pi).max())
                dominant=2*r['objective']<=2*r['initial_objective']+1e-12;unique=np.linalg.matrix_rank(matrix)==10
                rows.append(dict(client=i,bank=b,context=t,**r,direct_KKT=direct_kkt(matrix,q,pi),reference_KKT=reference_kkt,reference_objective=obj,objective_error=objective_error,unique=bool(unique),prevalence_error=prevalence_error,
                    reference_agreement=objective_error<=1e-10 and (not unique or prevalence_error<=1e-7),objective_dominance=dominant))
                vectors.append(dict(client=i,bank=b,context=t,q=q.tolist(),pi=pi.tolist(),reference_pi=ref.tolist()))
        print('T018_REFERENCE_CLIENT_DONE',i,'caps',sum(r['cap_hit'] for r in rows),'seconds',round(time.time()-start,1),flush=True)
    write_csv(out/'solver_subset.csv',rows);save(out/'solver_vectors.json',vectors)
    cap=[r for r in rows if r['cap_hit']];agreement=[r for r in rows if not r['reference_agreement']];dominance=[r for r in rows if not r['objective_dominance']]
    passed=not cap and not agreement and not dominance
    save(out/'solver_verification.json',dict(status='PASS' if passed else 'SOLVER_PREFLIGHT_BLOCKER',cases=len(rows),cap_hits=len(cap),reference_disagreements=len(agreement),objective_dominance_violations=len(dominance),
        max_objective_error=max(r['objective_error'] for r in rows),max_prevalence_error=max(r['prevalence_error'] for r in rows),max_reference_KKT=max(r['reference_KKT'] for r in rows),max_iterations=max(r['iterations'] for r in rows),
        query_metrics_opened=False,bootstrap_q_solved=0,new_model_forwards=0,noise_free_pass=True,full_tests_passed=95))
    assert all(sha(Path(p))==h for p,h in inputs.items())
    save(out/'metadata.json',dict(runtime=a.commit,seconds=time.time()-start,numpy_version=np.__version__,status='PASS' if passed else 'SOLVER_PREFLIGHT_BLOCKER',source_hashes_unchanged=True))
    print('T018_PREFLIGHT_RESULT',json.dumps(dict(cap_hits=len(cap),reference_disagreements=len(agreement),dominance_violations=len(dominance))),flush=True)
    assert passed,'Fixed solver cap/reference/dominance condition failed; stop before query evaluation'


if __name__=='__main__':main()
