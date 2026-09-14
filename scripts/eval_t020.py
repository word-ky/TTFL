"""Label-blind T020 preparation and mandatory 128/256 stability stop."""
import argparse,gzip,hashlib,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,gzload,sha,save,C,B,S
from report_t009 import write_csv
from src.context.confusion_prevalence import probabilities
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.exact_template_lookup import compile_template
from src.context.bootstrap_regret import bootstrap_positions,shared_integer_weights,exact_utilities_from_weights,canonical_and_mask,expected_regret_objects


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p19=p/'runs/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity'
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics';p14=p/'results/t014_class_conditional_factorization'
    (out/'PROTOCOL_FREEZE.md').write_bytes((p/'results/t020_bootstrap_expected_regret/PROTOCOL_FREEZE.md').read_bytes())
    inputs=load(p19/'input_hashes.json');assert all(sha(Path(path))==h for path,h in inputs.items())
    for path in (p19/'real_decomposition.npz',p19/'utility_choices.npz',p14/'class_templates.json.gz',p16/'calibration_counts.npz',ROOT/'src/context/constrained_prevalence.py'):
        inputs[str(path)]=sha(path)
    for name in ('real_decomposition.npz','utility_choices.npz'):
        assert sha(p19/name)==load(p19/'phaseB_choices_freeze.json')['hashes'][name]
    save(out/'input_hashes.json',inputs)
    # Lazy NPZ member access: no other members of either T019 archive are read.
    prob=np.load(p19/'real_decomposition.npz')['probabilities']
    historical_point=np.load(p19/'utility_choices.npz')['actual_choices']
    logits=np.load(p15/'support_logits.npz')
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):np.testing.assert_array_equal(prob[i,bi,ti],probabilities(logits[f'{i}|{b}|{t}|{t}']))
    cal=np.load(p16/'calibration_counts.npz');num=cal['soft_numerator'];den=cal['soft_denominator'];mat=num/den[...,None,:]
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in context] for context in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    shape=(100,2,5,256);indices=np.zeros(shape+(20,),dtype=np.uint8);qraw=np.zeros(shape+(10,));piraw=np.zeros_like(qraw)
    kkt=np.zeros(shape);sumerror=np.zeros(shape);updates=np.zeros(shape,dtype=np.uint8);improvement=np.zeros(shape)
    bootchoices=np.zeros(shape+(8,),dtype=np.uint8);bootmasks=np.zeros_like(bootchoices)
    points=np.zeros((100,2,5,8),dtype=np.uint8);ber128=np.zeros_like(points);ber256=np.zeros_like(points)
    exactrows=[];csvrows=[];seed_replays=[];point_checks=0
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=(i,bi,ti);solver=ActiveSetCLS(mat[loc]);point,r=solver.solve(prob[loc].mean(0));assert 2*r['objective']<=2*r['initial_objective']+1e-12
                for replica in range(256):
                    rr=loc+(replica,);positions=bootstrap_positions(i,b,t,replica);q=prob[loc][positions].mean(0);pi,v=solver.solve(q)
                    assert 2*v['objective']<=2*v['initial_objective']+1e-12
                    indices[rr]=positions;qraw[rr]=q;piraw[rr]=pi;kkt[rr]=v['direct_KKT'];sumerror[rr]=v['sum_error'];updates[rr]=v['updates'];improvement[rr]=v['initial_objective']-v['objective']
                    if replica in (0,255):
                        seed=int(hashlib.sha256(f'T020|{i}|{b}|{t}|{replica}'.encode()).hexdigest(),16)
                        alternate=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=20);np.testing.assert_array_equal(positions,alternate)
                        seed_replays.append(dict(client=i,bank=b,context=t,replica=replica,positions=positions.tolist()))
                weights,D=shared_integer_weights(piraw[loc]);pw,pD=shared_integer_weights([point])
                for si,s in enumerate(S):
                    for h in (0,1):
                        slot=2*si+h;coeff,td=compile_template(templates[s,b,h,i][ti]);pointnums=exact_utilities_from_weights(pw,coeff)[0]
                        ps=canonical_and_mask(pointnums)[0];assert ps==historical_point[loc][slot];point_checks+=1
                        nums=exact_utilities_from_weights(weights,coeff);states,masks,diagnostics=expected_regret_objects(nums,D*td,ps)
                        bootchoices[loc][:,slot]=states;bootmasks[loc][:,slot]=masks;points[loc][slot]=ps;ber128[loc][slot]=diagnostics['BER128'];ber256[loc][slot]=diagnostics['BER256']
                        row=dict(client=i,bank=b,context=t,salt=s,train_half=h,**diagnostics);exactrows.append(row)
                        csvrows.append({key:(json.dumps(value) if isinstance(value,list) else value) for key,value in row.items() if not key.endswith('_exact')})
        print('T020_CLIENT',i,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(out/'bootstrap_arrays.npz',positions=indices,q=qraw,pi=piraw,KKT=kkt,sum_error=sumerror,updates=updates,objective_improvement=improvement,replica_choices=bootchoices,replica_argmax_masks=bootmasks,point=points,BER128=ber128,BER256=ber256)
    (out/'utility_means_exact.json.gz').write_bytes(gzip.compress(json.dumps(exactrows,allow_nan=False).encode(),mtime=0))
    write_csv(out/'bootstrap_uncertainty.csv',csvrows)
    write_csv(out/'ber_choices.csv',[{key:r[key] for key in ('client','bank','context','salt','train_half','point_state','BER128','BER256','agreement')} for r in exactrows])
    stability=[]
    for b in B:
        for t in C:
            group=[r for r in exactrows if r['bank']==b and r['context']==t]
            stability.append(dict(bank=b,context=t,cases=len(group),agreements=sum(r['agreement'] for r in group),agreement_fraction=sum(r['agreement'] for r in group)/len(group),
                max_mean_utility_difference=max(r['max_mean_utility_difference'] for r in group),max_vote_frequency_difference=max(r['max_vote_frequency_difference'] for r in group)))
    write_csv(out/'bootstrap_stability.csv',stability)
    global_fraction=float(np.mean(ber128==ber256));global_pass=global_fraction>=.99;cluster_pass=all(r['agreement_fraction']>=.99 for r in stability);passed=global_pass and cluster_pass
    save(out/'seed_receipt.json',dict(replays=seed_replays,cases=len(seed_replays),label_blind_positions=True,with_replacement=True,R=256))
    save(out/'phaseA_verification.json',dict(bootstrap_cells=256000,solver_invariants_pass=True,max_KKT=float(kkt.max()),max_sum_error=float(sumerror.max()),max_updates=int(updates.max()),
        min_objective_improvement=float(improvement.min()),probability_replay_bitwise=True,point_choice_replays=point_checks,independent_position_replays=len(seed_replays),labels_opened=False,query_outcomes_opened=False,new_model_forwards=0))
    assert all(sha(Path(path))==h for path,h in inputs.items())
    save(out/'phaseB_choices_freeze.json',dict(runtime=a.commit,labels_opened=False,query_outcomes_opened=False,stability_pass=passed,
        hashes={name:sha(out/name) for name in ('bootstrap_arrays.npz','utility_means_exact.json.gz','bootstrap_uncertainty.csv','bootstrap_stability.csv','ber_choices.csv','seed_receipt.json')}))
    summary=dict(status='PREPARATION_PASS' if passed else 'MONTE_CARLO_INSTABILITY',runtime=a.commit,seconds=time.time()-start,numpy_version=np.__version__,
        global_128_256_agreement=global_fraction,global_pass=global_pass,all_bank_context_groups_pass=cluster_pass,disagreements=int(np.sum(ber128!=ber256)),total_choices=int(points.size),
        max_mean_utility_difference=max(r['max_mean_utility_difference'] for r in exactrows),max_vote_frequency_difference=max(r['max_vote_frequency_difference'] for r in exactrows),
        BER_REGRET_A='NOT_EXECUTED',BER_CAP_A='NOT_EXECUTED',BER_SRC_A='NOT_EXECUTED',scientific_diagnosis='NOT_ASSIGNED',labels_opened=False,query_outcomes_opened=False,new_model_forwards=0,source_hashes_unchanged=True)
    save(out/'summary.json',summary);print('T020_PREPARATION_RESULT',json.dumps(summary),flush=True)
    assert passed,'Mandatory fixed-R Monte Carlo stability failed; stop before labels/query outcomes'


if __name__=='__main__':main()
