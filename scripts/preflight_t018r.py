"""Solver-only T018R repair. No query count/accuracy outcomes are opened."""
import argparse, csv, json, sys, time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,gzload,sha,save,C,B,S
from src.context.constrained_prevalence import ActiveSetCLS,face_reference,direct_kkt
from src.context.matched_channel import exact_utilities
from src.context.noise_free_ties import noise_free_sanity
from report_t009 import write_csv


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    old=project/'runs/20260914-113842-ttfl-t018-preflight/artifacts/t018_constrained_prevalence'
    p14=project/'results/t014_class_conditional_factorization'
    p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    inputs=load(old/'input_hashes.json')
    # Previous runtime source path is preserved; source artifact bytes must match.
    for p,h in inputs.items():assert sha(Path(p))==h
    for name in ('solver_subset.csv','solver_vectors.json','solver_verification.json'):
        inputs[str(old/name)]=sha(old/name)
    save(out/'input_hashes.json',inputs)
    save(out/'protocol_freeze.json',dict(lead='c8aea5b',scientific_lead='1744440',runtime=a.commit,
        objective='unchanged 0.5*||C*pi-q||^2 on simplex',repair='T018R2 direct RHS application only; same active set',
        warm_start='unchanged ProjectSimplex(pinv(C)@q)',negative_face_tolerance=1e-12,dual_KKT_tolerance=1e-10,
        max_updates=100,cache='KKT matrix only; numpy.linalg.solve(K,actual_rhs) on each visit',
        subset='same clients0..19 x banksA/B x five contexts',new_model_forwards=0,query_metrics_opened=False))
    cal=np.load(p16/'calibration_counts.npz');truth={r['client']:r for r in load(p16/'support_truth.json')}
    obs={(r['client'],r['bank'],r['target']):r for r in gzload(p16/'source_mixtures.json.gz') if r['policy']=='BBSE-S-01'}
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    p00={(r['client'],r['bank'],r['target'],r['salt'],r['train_half']):r for r in gzload(p15/'privileged_policy_choices.json.gz') if r['policy']=='P00'}
    oldrows={(int(r['client']),r['bank'],r['context']):r for r in csv.DictReader((old/'solver_subset.csv').open())}
    oldvec={(r['client'],r['bank'],r['context']):r for r in load(old/'solver_vectors.json')}
    noise=[];rows=[];vectors=[];audit=[];singleton=tied=0
    stage='frozen_blocker_regression';cell=None
    try:
        blocked=project/'runs/20260914-124709-ttfl-t018r-science/artifacts/t018r_constrained_prevalence'
        blocker=load(blocked/'blocker.json');diagnosis=load(blocked/'blocker_diagnosis.json')
        i,bi,ti,replica=blocker['cell'];cell=(i,B[bi],C[ti],replica)
        matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];q=np.array(blocker['q'])
        fixture=load(ROOT/'tests/fixtures/t018r2_blocker.json')
        np.testing.assert_array_equal(matrix,fixture['C']);np.testing.assert_array_equal(q,fixture['q'])
        pi,r=ActiveSetCLS(matrix).solve(q);ref,obj=face_reference(matrix,q)
        assert r['clipped_coordinates']==0 and r['sum_error']<=1e-12 and np.max(np.abs(pi-ref))<=1e-7 and abs(r['objective']-obj)<=1e-10
        save(out/'frozen_blocker_regression.json',dict(cell=blocker['cell'],old_cached_pi=diagnosis['cached_pi'],old_sum_error=diagnosis['cached_sum_error'],
            new_pi=pi.tolist(),reference_pi=ref.tolist(),reference_error=float(np.abs(pi-ref).max()),reference_objective_error=abs(r['objective']-obj),**r))
        stage='noise_free'
        for i in range(100):
            true=np.array(truth[i]['pi'])
            for bi,b in enumerate(B):
                for ti,t in enumerate(C):
                    cell=(i,b,t)
                    matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:]
                    pi,r=ActiveSetCLS(matrix).solve(matrix@true)
                    assert np.max(np.abs(pi-true))<=1e-8 and r['objective']<=1e-16
                    for salt in S:
                        for train in (0,1):
                            template=templates[salt,b,train,i][ti]
                            values,arg=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],template)
                            saved=p00[i,b,t,salt,train]
                            assert values==[Fraction(x) for x in saved['utility']] and [C[s] for s in arg]==saved['argmax']
                            _,selected=exact_utilities(pi,template)
                            check=noise_free_sanity(pi,true,selected[0],values,arg[0]);tied+=check['tied'];singleton+=not check['tied']
                    noise.append(dict(client=i,bank=b,context=t,max_error=float(np.abs(pi-true).max()),**r))
        write_csv(out/'noise_free.csv',noise)
        save(out/'noise_free_verification.json',dict(passed=True,episodes=1000,singleton_checks=singleton,tied_checks=tied,
            max_pi_error=max(r['max_error'] for r in noise),max_objective=max(r['objective'] for r in noise)))
        print('T018R_NOISE_FREE_PASS',flush=True)
        stage='same_200_reference'
        for i in range(20):
            for bi,b in enumerate(B):
                for ti,t in enumerate(C):
                    cell=(i,b,t);matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:]
                    q=np.array(obs[cell]['raw_soft']);np.testing.assert_array_equal(q,oldvec[cell]['q'])
                    pi,r=ActiveSetCLS(matrix).solve(q);ref,obj=face_reference(matrix,q)
                    oldr=oldrows[cell];err=float(np.abs(pi-ref).max());objerr=abs(r['objective']-obj)
                    rows.append(dict(client=i,bank=b,context=t,**r,reference_KKT=direct_kkt(matrix,q,ref),reference_objective=obj,
                        prevalence_error=err,objective_error=objerr,old_PGD_cap=oldr['cap_hit']=='True',old_PGD_iterations=int(oldr['iterations']),
                        old_PGD_prevalence_error=float(oldr['prevalence_error']),reference_agreement=err<=1e-7 and objerr<=1e-10))
                    vectors.append(dict(client=i,bank=b,context=t,q=q.tolist(),pi=pi.tolist(),reference_pi=ref.tolist()))
                    assert err<=1e-7 and objerr<=1e-10 and 2*r['objective']<=2*r['initial_objective']+1e-12
            print('T018R_REFERENCE_CLIENT',i,flush=True)
        write_csv(out/'solver_subset.csv',rows);save(out/'solver_vectors.json',vectors)
        stage='all_1000_actual'
        for i in range(100):
            for bi,b in enumerate(B):
                for ti,t in enumerate(C):
                    cell=(i,b,t);matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:]
                    q=np.array(obs[cell]['raw_soft']);pi,r=ActiveSetCLS(matrix).solve(q)
                    assert 2*r['objective']<=2*r['initial_objective']+1e-12
                    audit.append(dict(client=i,bank=b,context=t,pi=pi.tolist(),**r))
        write_csv(out/'all_actual_audit.csv',audit);save(out/'all_actual_solutions.json',audit)
        assert all(sha(Path(p))==h for p,h in inputs.items())
    except Exception as error:
        save(out/'blocker.json',dict(stage=stage,cell=cell,error=repr(error),noise_free_completed=len(noise),subset_completed=len(rows),actual_completed=len(audit)))
        if rows:write_csv(out/'solver_subset_partial.csv',rows)
        if noise:write_csv(out/'noise_free_partial.csv',noise)
        save(out/'metadata.json',dict(status='SOLVER_PREFLIGHT_BLOCKER',runtime=a.commit,seconds=time.time()-start,new_model_forwards=0,query_metrics_opened=False))
        raise
    save(out/'solver_verification.json',dict(status='PASS',subset_cases=len(rows),historical_cap_cases_repaired=sum(r['old_PGD_cap'] for r in rows),
        historical_converged_cases_equivalent=sum(not r['old_PGD_cap'] for r in rows),all_actual_cases=len(audit),
        max_prevalence_error=max(r['prevalence_error'] for r in rows),max_objective_error=max(r['objective_error'] for r in rows),
        max_KKT=max(r['direct_KKT'] for r in audit),max_sum_error=max(r['sum_error'] for r in audit),max_updates=max(r['updates'] for r in audit),
        distributions={k:dict(zip(('p95','p99','max'),[float(x) for x in np.quantile([r[k] for r in audit],[.95,.99,1.])])) for k in ('updates','active_classes','direct_KKT','sum_error')},
        new_model_forwards=0,query_metrics_opened=False))
    save(out/'metadata.json',dict(status='PASS',runtime=a.commit,seconds=time.time()-start,numpy_version=np.__version__,source_hashes_unchanged=True))
    print('T018R_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
