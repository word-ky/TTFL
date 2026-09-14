"""Required observational regression before resuming the two missing clients."""
import gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.matched_channel import exact_utilities
from fractions import Fraction
OUT=ROOT/'results/t020r2_mc_convergence'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start=time.time();old={};exec(compile((OUT/'frozen_solver.py').read_text(),'frozen_solver.py','exec'),old)
    p19=ROOT/'research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity'
    prob=np.load(p19/'real_decomposition.npz')['probabilities'];choices=np.load(p19/'utility_choices.npz')['actual_choices']
    p16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    cal=np.load(p16/'calibration_counts.npz');mat=cal['soft_numerator']/cal['soft_denominator'][...,None,:]
    p20=ROOT/'research_log/t020_receipts/20260914-154818-ttfl-t020/artifacts/t020_bootstrap_expected_regret';prefix=dict(np.load(p20/'bootstrap_arrays.npz'))
    p20r=ROOT/'research_log/t020r_receipts/20260914-163857-ttfl-t020r/artifacts/t020r_mc_convergence'
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in json.loads(gzip.decompress((ROOT/'results/t014_class_conditional_factorization/class_templates.json.gz').read_bytes()))['rows']}
    real=replicas=decisions=0;maxsaved=0.;preserved={}
    C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
    for i in range(100):
        samplepath=p20r/f'client_{i:03d}_verification.npz';sample=dict(np.load(samplepath)) if samplepath.exists() else None
        if sample is not None:
            for name in [samplepath.name,f'client_{i:03d}_accumulators.json.gz']:preserved[name]=sha(p20r/name)
        for bi,b in enumerate(['A','B']):
            for ti,t in enumerate(C):
                loc=i,bi,ti;frozen=old['ActiveSetCLS'](mat[loc]);new=ActiveSetCLS(mat[loc])
                q=prob[loc].mean(0);pp,rr=frozen.solve(q);pi,r=new.solve(q)
                np.testing.assert_array_equal(pp,pi);assert r['solver_path']=='active_set' and {k:v for k,v in r.items() if k!='solver_path'}==rr;real+=1
                for si in range(4):
                    for h in (0,1):
                        table=templates[f'T013-S{si}',b,h,i][ti];v,args=exact_utilities(pi,table);ov,oa=exact_utilities(pp,table)
                        assert v==ov and args==oa and args[0]==choices[loc][2*si+h];decisions+=1
                samples=[] if sample is None else [(sample['q'][bi,ti,j],sample['pi'][bi,ti,j]) for j in range(4)]
                if i<10:samples.append((prefix['q'][loc][0],prefix['pi'][loc][0]))
                for q,saved in samples:
                    pi,r=new.solve(q);pp,rr=frozen.solve(q);np.testing.assert_array_equal(pi,pp)
                    assert r['solver_path']=='active_set' and {k:v for k,v in r.items() if k!='solver_path'}==rr
                    error=float(np.max(np.abs(pi-saved)));assert error<=1e-12;maxsaved=max(maxsaved,error);replicas+=1
        if i%20==19:print('T020R2_REGRESSION_CLIENTS',i+1,flush=True)
    checked=[]
    for name in ('t015','t016r','t017r','t018r2','t019','t020','t020r'):
        for row in load(ROOT/f'research_log/{name}_artifact_manifest.json'):
            assert sha(ROOT/row['path'])==row['sha256'];checked.append(row)
    expected=load(ROOT/'results/t020r_mc_convergence/partial_receipts_hashes.json')['hashes']
    assert len(preserved)==196 and all(h==expected[name] for name,h in preserved.items())
    result=dict(status='PASS',real_observations=real,noncycle_bootstrap_samples=replicas,normal_fallback_activations=0,
        repaired_vs_frozen_pi_bitwise_identical=True,normal_receipt_fields_identical=True,max_saved_pi_error=maxsaved,point_template_decisions_and_argmax_masks=decisions,
        source_manifest_entries_unchanged=len(checked),source_files=checked,preserved_98_hashes=preserved,old_solver_sha256=sha(OUT/'frozen_solver.py'),
        repaired_solver_sha256=sha(ROOT/'src/context/constrained_prevalence.py'),seconds=time.time()-start,labels_parsed=False,query_outcomes_parsed=False)
    (OUT/'noncycle_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','preserved_98_hashes')},indent=2))


if __name__=='__main__':main()
