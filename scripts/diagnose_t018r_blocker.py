"""Read-only numerical diagnosis of the stopped case; never a policy repair."""
import json,sys
from pathlib import Path
import numpy as np
from src.context.constrained_prevalence import direct_kkt,face_reference,ActiveSetCLS
project=Path('/home/wenchang/asdasdsad/wjq/TTFL')
out=project/'runs/20260914-124709-ttfl-t018r-science/artifacts/t018r_constrained_prevalence'
block=json.loads((out/'blocker.json').read_text());i,bi,ti,r=block['cell']
cal=np.load(project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics/calibration_counts.npz')
C=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];q=np.array(block['q'])
saved=np.load(project/'runs/20260914-083609-ttfl-t017r-cached/artifacts/t017_channel_noise_decomposition/bootstrap_arrays.npz')
np.testing.assert_array_equal(q,saved['q'][0,i,bi,ti,r])
try:ActiveSetCLS(C).solve(q)
except RuntimeError as error:replay=str(error)
else:raise AssertionError('same-runtime blocker did not reproduce')
A=[0,1,2,3,6,8,9];H=C.T@C;f=C.T@q;m=len(A)
K=np.zeros((m+1,m+1));K[:m,:m]=H[np.ix_(A,A)];K[:m,m]=1.;K[m,:m]=1.;rhs=np.append(f[A],1.)
cached=(np.linalg.solve(K,np.eye(m+1))@rhs)[:m];direct=np.linalg.solve(K,rhs)[:m]
pc=np.zeros(10);pc[A]=cached;pd=np.zeros(10);pd[A]=direct
ref,obj=face_reference(C,q)
receipt=dict(cell=block['cell'],reproduced_error=replay,numpy_version=np.__version__,failing_face=A,
    cached_pi=pc.tolist(),cached_sum=float(pc.sum()),cached_sum_error=float(abs(pc.sum()-1)),cached_min=float(pc.min()),cached_direct_KKT=direct_kkt(C,q,pc),
    cached_objective=float(.5*np.sum((C@pc-q)**2)),reference_objective=obj,reference_pi=ref.tolist(),reference_KKT=direct_kkt(C,q,ref),
    reference_max_pi_error=float(np.abs(pc-ref).max()),direct_face_sum_error=float(abs(pd.sum()-1)),direct_face_KKT=direct_kkt(C,q,pd),
    direct_face_pi=pd.tolist(),interpretation='cached inverse-map application exceeds fixed simplex sum tolerance; direct solve and exhaustive reference are diagnostics only, not accepted policy replacements',
    matched_successful_cases_before_stop=int(np.ravel_multi_index((i,bi,ti,r),(100,2,5,128))),query_metrics_opened=False,new_model_forwards=0)
(out/'blocker_diagnosis.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
print(json.dumps(receipt))
