"""T020R one fixed stream extension, parallel only across independent clients."""
import argparse,gzip,hashlib,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,gzload,sha,save,C,B,S
from report_t009 import write_csv
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.exact_template_lookup import compile_template
from src.context.bootstrap_regret import bootstrap_positions,shared_integer_weights,exact_utilities_from_weights,canonical_and_mask,expected_regret_objects
from src.context.bootstrap_convergence import BLOCKS,aggregate_blocks,convergence_gate
R=4096;SAMPLE=(256,1024,2048,4095)


def initialize(project,output):
    global PROB,MAT,OLD,LEGACY,TEMPLATES,OUT
    p=Path(project);OUT=Path(output)
    p19=p/'runs/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity'
    p20=p/'runs/20260914-154818-ttfl-t020/artifacts/t020_bootstrap_expected_regret'
    p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    PROB=np.load(p19/'real_decomposition.npz')['probabilities'];OLD=dict(np.load(p20/'bootstrap_arrays.npz'))
    cal=np.load(p16/'calibration_counts.npz');MAT=cal['soft_numerator']/cal['soft_denominator'][...,None,:]
    LEGACY={(r['client'],r['bank'],r['context'],r['salt'],r['train_half']):r for r in gzload(p20/'utility_means_exact.json.gz')}
    TEMPLATES={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p/'results/t014_class_conditional_factorization/class_templates.json.gz')['rows']}


def run_client(i):
    records=[];cellstats=[];samples_pi=np.zeros((2,5,4,10));samples_q=np.zeros_like(samples_pi);samples_pos=np.zeros((2,5,4,20),dtype=np.uint8)
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            loc=(i,bi,ti);solver=ActiveSetCLS(MAT[loc]);point,pr=solver.solve(PROB[loc].mean(0));pw,pD=shared_integer_weights([point])
            pi=np.empty((R,10));q=np.empty_like(pi);positions=np.empty((R,20),dtype=np.uint8)
            pi[:256]=OLD['pi'][loc];q[:256]=OLD['q'][loc];positions[:256]=OLD['positions'][loc]
            maxk=float(OLD['KKT'][loc].max());maxsum=float(OLD['sum_error'][loc].max());maxupdates=int(OLD['updates'][loc].max());minimp=float(OLD['objective_improvement'][loc].min())
            for replica in range(256,R):
                pos=bootstrap_positions(i,b,t,replica);qq=PROB[loc][pos].mean(0);pp,receipt=solver.solve(qq)
                assert 2*receipt['objective']<=2*receipt['initial_objective']+1e-12
                pi[replica]=pp;q[replica]=qq;positions[replica]=pos
                maxk=max(maxk,receipt['direct_KKT']);maxsum=max(maxsum,receipt['sum_error']);maxupdates=max(maxupdates,receipt['updates']);minimp=min(minimp,receipt['initial_objective']-receipt['objective'])
            np.testing.assert_array_equal(pi[:256],OLD['pi'][loc]);np.testing.assert_array_equal(q[:256],OLD['q'][loc]);np.testing.assert_array_equal(positions[:256],OLD['positions'][loc])
            for j,replica in enumerate(SAMPLE):samples_pi[bi,ti,j]=pi[replica];samples_q[bi,ti,j]=q[replica];samples_pos[bi,ti,j]=positions[replica]
            weights,D=shared_integer_weights(pi)
            for si,s in enumerate(S):
                for h in (0,1):
                    slot=2*si+h;coeff,td=compile_template(TEMPLATES[s,b,h,i][ti]);ps=canonical_and_mask(exact_utilities_from_weights(pw,coeff)[0])[0]
                    assert ps==OLD['point'][loc][slot]
                    nums=exact_utilities_from_weights(weights,coeff);oldstates,oldmasks,legacy=expected_regret_objects(nums[:256],D*td,ps)
                    np.testing.assert_array_equal(oldstates,OLD['replica_choices'][loc][:,slot]);np.testing.assert_array_equal(oldmasks,OLD['replica_argmax_masks'][loc][:,slot])
                    saved=LEGACY[i,b,t,s,h]
                    assert all(legacy[k]==saved[k] for k in legacy),'Legacy first128/all256 aggregate changed'
                    objects,diagnostics,states,masks=aggregate_blocks(nums,D*td,ps,BLOCKS)
                    h1=[Fraction(x) for x in objects['H1']['means_exact']];h2=[Fraction(x) for x in objects['H2']['means_exact']]
                    records.append(dict(client=i,bank=b,context=t,salt=s,train_half=h,point=ps,BER256=saved['BER256'],BER4096=objects['ALL']['state'],blocks=objects,
                        H1_H2_mean_abs_difference=[float(abs(x-y)) for x,y in zip(h1,h2)],H1_H2_vote_abs_difference=[abs(x-y) for x,y in zip(objects['H1']['votes'],objects['H2']['votes'])],
                        all_replica_choices_sha256=hashlib.sha256(states.tobytes()).hexdigest(),all_replica_masks_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),**diagnostics))
            cellstats.append(dict(client=i,bank=b,context=t,new_CLS_solves=R-256,immutable_prefix=256,max_KKT=maxk,max_sum_error=maxsum,max_updates=maxupdates,min_objective_improvement=minimp,
                full_pi_sha256=hashlib.sha256(pi.tobytes()).hexdigest(),full_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),full_positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
                prefix_pi_sha256=hashlib.sha256(pi[:256].tobytes()).hexdigest(),prefix_q_sha256=hashlib.sha256(q[:256].tobytes()).hexdigest(),prefix_positions_sha256=hashlib.sha256(positions[:256].tobytes()).hexdigest()))
    np.savez_compressed(OUT/f'client_{i:03d}_verification.npz',replica_ids=np.array(SAMPLE),pi=samples_pi,q=samples_q,positions=samples_pos)
    path=OUT/f'client_{i:03d}_accumulators.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(records=records,cells=cellstats),allow_nan=False).encode(),mtime=0))
    return i


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p20=p/'runs/20260914-154818-ttfl-t020/artifacts/t020_bootstrap_expected_regret'
    inputs=load(p20/'input_hashes.json');assert all(sha(Path(path))==h for path,h in inputs.items())
    for path in p20.iterdir():
        if path.is_file():inputs[str(path)]=sha(path)
    save(out/'input_hashes.json',inputs)
    (out/'PROTOCOL_FREEZE.md').write_bytes((p/'results/t020r_mc_convergence/PROTOCOL_FREEZE.md').read_bytes())
    save(out/'runtime.json',dict(lead='f0807dd',runtime=a.commit,R_FINAL=R,seed_namespace='T020',workers=4,blas_threads=2,labels_parsed=False,query_outcomes_parsed=False,new_model_forwards=0))
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(p),str(out))) as executor:
        futures=[executor.submit(run_client,i) for i in range(100)]
        for done,future in enumerate(as_completed(futures),1):print('T020R_CLIENT_DONE',future.result(),'completed',done,'seconds',round(time.time()-start,1),flush=True)
    records=[];cells=[]
    for i in range(100):
        data=gzload(out/f'client_{i:03d}_accumulators.json.gz');records+=data['records'];cells+=data['cells']
    choices={name:[r['blocks'][name]['state'] for r in records] for name in BLOCKS};banks=[r['bank'] for r in records];contexts=[r['context'] for r in records]
    gate=convergence_gate(choices,banks,contexts);pairs=[('H1','H2'),('H1','ALL'),('H2','ALL')]+[(a,b) for ai,a in enumerate(('Q0','Q1','Q2','Q3')) for b in ('Q0','Q1','Q2','Q3')[ai+1:]]
    convergence=[]
    for aa,bb in pairs:
        for b in ['ALL']+B:
            for t in ['ALL']+C:
                g=[r for r in records if (b=='ALL' or r['bank']==b) and (t=='ALL' or r['context']==t)]
                differences=[float(abs(Fraction(x)-Fraction(y))) for r in g for x,y in zip(r['blocks'][aa]['means_exact'],r['blocks'][bb]['means_exact'])]
                vd=[abs(x-y) for r in g for x,y in zip(r['blocks'][aa]['votes'],r['blocks'][bb]['votes'])]
                convergence.append(dict(block1=aa,block2=bb,bank=b,context=t,cases=len(g),agreements=sum(r['blocks'][aa]['state']==r['blocks'][bb]['state'] for r in g),
                    agreement=sum(r['blocks'][aa]['state']==r['blocks'][bb]['state'] for r in g)/len(g),mean_utility_abs_difference_max=max(differences),mean_utility_abs_difference_p95=float(np.quantile(differences,.95)),
                    vote_difference_max=max(vd),vote_difference_p95=float(np.quantile(vd,.95))))
    write_csv(out/'mc_convergence.csv',convergence)
    disagreements=[r for r in records if r['blocks']['H1']['state']!=r['blocks']['H2']['state']]
    fields=('client','bank','context','salt','train_half','point','BER256','BER4096','top_mean_utility_margin','p_mode')
    write_csv(out/'mc_disagreements.csv',[dict(**{k:r[k] for k in fields},H1=r['blocks']['H1']['state'],H2=r['blocks']['H2']['state']) for r in disagreements])
    finalrows=[]
    for r in records:
        row={k:v for k,v in r.items() if k not in ('blocks','all_replica_choices_sha256','all_replica_masks_sha256')}
        row.update(final_mean_utilities_exact=json.dumps(r['blocks']['ALL']['means_exact']),final_expected_regrets_exact=json.dumps(r['blocks']['ALL']['regrets_exact']))
        finalrows.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    write_csv(out/'final_ber_uncertainty.csv',finalrows)
    write_csv(out/'transition_256_to_4096.csv',[dict(BER256=i,BER4096=j,slots=sum(r['BER256']==i and r['BER4096']==j for r in records)) for i in range(5) for j in range(5)])
    stable=[r['top_mean_utility_margin'] for r in records if r not in disagreements];unstable=[r['top_mean_utility_margin'] for r in disagreements]
    gate.update(H1_H2_disagreements=len(disagreements),unique_clients=len(set(r['client'] for r in disagreements)),unique_cells=len(set((r['client'],r['bank'],r['context']) for r in disagreements)),
        stable_margin_quantiles=[float(x) for x in np.quantile(stable,[.05,.5,.95])],unstable_margin_quantiles=[float(x) for x in np.quantile(unstable,[.05,.5,.95])] if unstable else [])
    save(out/'mc_gate.json',gate);save(out/'solver_verification.json',dict(passed=True,logical_replicas=4096000,new_CLS_solves=sum(r['new_CLS_solves'] for r in cells),immutable_prefix_replicas=256000,
        max_KKT=max(r['max_KKT'] for r in cells),max_sum_error=max(r['max_sum_error'] for r in cells),max_updates=max(r['max_updates'] for r in cells),min_objective_improvement=min(r['min_objective_improvement'] for r in cells),
        legacy_aggregate_replays=8000,verification_samples=4000,labels_parsed=False,query_outcomes_parsed=False,new_model_forwards=0))
    save(out/'cell_stream_receipts.json',cells)
    assert all(sha(Path(path))==h for path,h in inputs.items())
    names=[p.name for p in out.iterdir() if p.name.endswith('_accumulators.json.gz') or p.name.endswith('_verification.npz')]+['mc_convergence.csv','mc_disagreements.csv','final_ber_uncertainty.csv','transition_256_to_4096.csv','mc_gate.json','cell_stream_receipts.json']
    save(out/'phaseB_choices_freeze.json',dict(runtime=a.commit,R_FINAL=R,seed_namespace='T020',convergence_pass=gate['passed'],labels_parsed=False,query_outcomes_parsed=False,hashes={name:sha(out/name) for name in names}))
    summary=dict(status='PREPARATION_PASS' if gate['passed'] else 'T020R-MC2',runtime=a.commit,seconds=time.time()-start,numpy_version=np.__version__,R_FINAL=R,**gate,
        BER_REGRET_A='NOT_EXECUTED',BER_CAP_A='NOT_EXECUTED',BER_SRC_A='NOT_EXECUTED',scientific_diagnosis='NOT_ASSIGNED',labels_parsed=False,query_outcomes_parsed=False,new_model_forwards=0,source_hashes_unchanged=True)
    save(out/'summary.json',summary);print('T020R_RESULT',json.dumps(summary),flush=True)
    assert gate['passed'],'T020R-MC2: fixed R4096 convergence failed; remain sealed'


if __name__=='__main__':main()
