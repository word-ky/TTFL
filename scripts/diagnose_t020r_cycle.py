"""Reproduce only missing T020R clients; reference solve is diagnostic, never a BER fallback."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.constrained_prevalence import ActiveSetCLS,face_reference,direct_kkt
RUN='20260914-163857-ttfl-t020r'
RAW=ROOT/f'research_log/t020r_receipts/{RUN}/artifacts/t020r_mc_convergence'
OUT=ROOT/'results/t020r_mc_convergence'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B']


def main():
    prob=np.load(ROOT/'research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity/real_decomposition.npz')['probabilities']
    cal=np.load(ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics/calibration_counts.npz')
    matrices=cal['soft_numerator']/cal['soft_denominator'][...,None,:]
    missing=[i for i in range(100) if not (RAW/f'client_{i:03d}_accumulators.json.gz').exists()];failures=[]
    for i in missing:
        failed=False;preceding=0
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                matrix=matrices[i,bi,ti];solver=ActiveSetCLS(matrix)
                for k in range(256,4096):
                    seed=int(hashlib.sha256(f'T020|{i}|{b}|{t}|{k}'.encode()).hexdigest(),16)
                    positions=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=20);q=prob[i,bi,ti][positions].mean(0)
                    try:solver.solve(q)
                    except RuntimeError as exc:
                        fresh_error=None
                        try:ActiveSetCLS(matrix).solve(q)
                        except RuntimeError as fresh:fresh_error=str(fresh)
                        assert fresh_error is not None
                        reference,obj=face_reference(matrix,q);kkt=direct_kkt(matrix,q,reference)
                        name=f'cycle_client_{i:03d}_{b}_{t}_replica_{k}.npz'
                        np.savez_compressed(OUT/name,C=matrix,q=q,positions=positions,reference_pi=reference)
                        failures.append(dict(client=i,bank=b,context=t,replica=k,preceding_new_solves_in_client=preceding,error=str(exc),fresh_solver_error=fresh_error,
                            reference_objective=float(obj),reference_KKT=float(kkt),reference_sum_error=float(abs(reference.sum()-1)),reference_min_pi=float(reference.min()),
                            reference_active_classes=np.flatnonzero(reference>0).tolist(),diagnostic_npz=name,npz_sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
                        failed=True;break
                    preceding+=1
                if failed:break
            if failed:break
        assert failed,'Missing client did not reproduce a solver failure'
    receipt=dict(status='REPRODUCED_SOLVER_CYCLE',missing_clients=missing,failures=failures,numpy_version=np.__version__,labels_parsed=False,query_outcomes_parsed=False,
        reference_used_for_policy=False,scientific_evaluation='NOT_EXECUTED')
    (OUT/'solver_cycle_diagnosis.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
