"""Target-local rotation probes, scramble diagnostics and sealed Phase-A freeze."""
import argparse,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from t023_common import *
from src.context.rotation_probe import folds,prepare,evaluate,independent_score,rotation_labels,scramble_labels
from report_t009 import write_csv

def main():
    ap=argparse.ArgumentParser()
    for k in ('extraction','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();ext=Path(a.extraction);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    ef=load(ext/'extraction_freeze.json');assert ef['status']=='PASS'
    assert all(sha(ext/name)==digest for name,digest in ef['hashes'].items())
    inputs=load(ext/'input_hashes.json');inputs.update({str(ext/name):digest for name,digest in ef['hashes'].items()});inputs[str(ext/'extraction_freeze.json')]=sha(ext/'extraction_freeze.json')
    fs=load(ext/'fold_membership.json');scores=np.empty((100,2,5,5));errors=np.empty((100,2,5,5,2));ranks=np.empty((100,2,5,5,2),dtype=np.int16);scrambles=np.empty((100,2,5,16))
    y=rotation_labels();audit=[];maxerror=0.
    for i in range(100):
        f=fs[str(i)];ids=f['support_ids'];split=folds(i,ids)
        assert split[0].tolist()==f['A_positions'] and split[1].tolist()==f['B_positions'] and len(set(split[0])&set(split[1]))==0
        labels=[scramble_labels(i,ids,r) for r in range(16)]
        for lab in labels:np.testing.assert_array_equal(lab.sum(1),np.ones((20,4)))
        data=np.load(ext/f'H_{i:03d}.npz')
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                features=data[f'{b}|{t}'];assert features.shape==(5,20,4,512)
                for si,s in enumerate(C):
                    prep=prepare(features[si],split);score,err,rank=evaluate(prep,y);loc=i,bi,ti,si;scores[loc]=score;errors[loc]=err;ranks[loc]=rank
                    if i<40:
                        ref,referr,refrank=independent_score(features[si],split,y);delta=max(abs(ref-score),float(np.max(np.abs(np.array(referr)-err))))
                        assert delta<=1e-10 and rank==refrank;maxerror=max(maxerror,delta);audit.append(dict(client=i,bank=b,context=t,state=s,max_error=delta,ranks=rank))
                    if si==0:
                        for r,lab in enumerate(labels):scrambles[i,bi,ti,r]=evaluate(prep,lab)[0]
        if i%10==9:print('T023_PROBE_CLIENT',i+1,flush=True)
    choices=scores.argmin(-1).astype(np.uint8);masks=((scores==scores.min(-1,keepdims=True))*np.array([1,2,4,8,16])).sum(-1).astype(np.uint8)
    clean=np.repeat(choices[:,:,0:1],5,axis=2);cleanmask=np.repeat(masks[:,:,0:1],5,axis=2)
    np.savez_compressed(out/'phaseA_scores.npz',scores=scores,errors=errors,ranks=ranks,current_choices=choices,current_masks=masks,clean_choices=clean,clean_masks=cleanmask)
    np.savez_compressed(out/'scramble_scores.npz',scores=scrambles)
    rows=[];cells=[];aggregate=[]
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=i,bi,ti;v=scrambles[loc];med=float(np.median(v));true=float(scores[loc+(0,)])
                rows.append(dict(client=i,bank=b,context=t,ROT_SCORE=scores[loc].tolist(),direction_errors=errors[loc].tolist(),direction_ranks=ranks[loc].tolist(),current_state=int(choices[loc]),current_tie_mask=int(masks[loc]),clean_surrogate_state=int(clean[loc]),clean_surrogate_tie_mask=int(cleanmask[loc])))
                cells.append(dict(client=i,bank=b,context=t,true_score=true,scramble_p05=float(np.quantile(v,.05)),scramble_median=med,scramble_p95=float(np.quantile(v,.95)),true_beats_scramble_median=true<med))
    for bi,b in enumerate(B):
        for ti,t in enumerate(C):
            means=scrambles[:,bi,ti].mean(0);v=scrambles[:,bi,ti]
            aggregate.append(dict(bank=b,context=t,true_score_mean=float(scores[:,bi,ti,0].mean()),scramble_client_mean_p05=float(np.quantile(means,.05)),scramble_client_mean_median=float(np.median(means)),scramble_client_mean_p95=float(np.quantile(means,.95)),fraction_clients_true_beats_own_scramble_median=float(np.mean(scores[:,bi,ti,0]<np.median(v,axis=1)))))
    write_csv(out/'phaseA_choices.csv',rows);write_csv(out/'scramble_cells.csv',cells);write_csv(out/'scramble_sanity.csv',aggregate);gzsave(out/'independent_probe_audit.json.gz',audit)
    save(out/'probe_numerical_audit.json',dict(status='PASS',cross_fitted_scores=len(audit),individual_fits=2*len(audit),reference='numpy.linalg.lstsq raw H+bias,rcond1e-12',main='explicit thin SVD minimum-norm',max_score_difference=maxerror,rank_min=int(ranks.min()),rank_max=int(ranks.max()),finite=True,target_class_labels_used=False))
    assert len(audit)>=2000 and np.isfinite(scores).all() and all(sha(path)==digest for path,digest in inputs.items());save(out/'input_hashes.json',inputs)
    files=['phaseA_scores.npz','scramble_scores.npz','phaseA_choices.csv','scramble_cells.csv','scramble_sanity.csv','independent_probe_audit.json.gz','probe_numerical_audit.json']
    save(out/'phaseA_choices_freeze.json',dict(status='PASS',task='T023',lead='89466b3',runtime=a.commit,extraction=str(ext),extraction_freeze_sha256=sha(ext/'extraction_freeze.json'),hashes={name:sha(out/name) for name in files},cells=1000,state_scores=5000,main_probe_fits=10000,scramble_replicates=16,scramble_probe_fits=32000,target_class_labels_used=False,privileged_utility_loaded=False,query_outcomes_scored=False,model_parameters_updated=False,new_model_forwards=0,seconds=time.time()-start))
    print('T023_PHASE_A_FROZEN',5000,flush=True)

if __name__=='__main__':main()
