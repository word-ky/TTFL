"""Reuse the immutable98 and compute only64/87 before the unchanged sealed gate."""
import argparse,json,shutil,time
from pathlib import Path
import eval_t020r as base


def main():
    ap=argparse.ArgumentParser()
    for key in ('project','output','commit'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    old=p/'runs/20260914-163857-ttfl-t020r/artifacts/t020r_mc_convergence';report=p/'results/t020r2_mc_convergence'
    regression=base.load(report/'noncycle_regression.json');assert regression['status']=='PASS'
    assert base.sha(base.ROOT/'src/context/constrained_prevalence.py')==regression['repaired_solver_sha256']
    inputs=base.load(old/'input_hashes.json');assert all(base.sha(Path(path))==h for path,h in inputs.items())
    for name,digest in regression['preserved_98_hashes'].items():
        assert base.sha(old/name)==digest;inputs[str(old/name)]=digest;shutil.copyfile(old/name,out/name)
    base.save(out/'preserved_98_verification.json',dict(status='PASS',files=regression['preserved_98_hashes'],source=str(old)))
    shutil.copyfile(report/'SOLVER_REPAIR.md',out/'PROTOCOL_FREEZE.md');shutil.copyfile(report/'noncycle_regression.json',out/'noncycle_regression.json')
    base.save(out/'input_hashes.json',inputs)
    base.save(out/'runtime.json',dict(lead='43d2f85',runtime=a.commit,R_FINAL=4096,resumed_clients=[64,87],reused_clients=98,labels_parsed=False,query_outcomes_parsed=False,new_model_forwards=0))
    base.initialize(str(p),str(out))
    for i in (64,87):
        base.run_client(i);print('T020R2_RESUMED_CLIENT',i,flush=True)
    records=[];cells=[]
    for i in range(100):
        data=base.gzload(out/f'client_{i:03d}_accumulators.json.gz');assert {r['client'] for r in data['records']}=={i}
        records+=data['records'];cells+=data['cells']
    keys={(r['client'],r['bank'],r['context'],r['salt'],r['train_half']) for r in records}
    expected={(i,b,t,s,h) for i in range(100) for b in base.B for t in base.C for s in base.S for h in (0,1)}
    assert len(records)==8000 and keys==expected and len(cells)==1000
    assert len({(r['client'],r['bank'],r['context']) for r in cells})==1000
    fallbacks=[f for c in cells for f in c.get('fallbacks',[])]
    base.save(out/'solver_fallbacks.json',fallbacks)
    base.write_csv(out/'solver_fallbacks.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in fallbacks])
    base.save(out/'resume_receipt.json',dict(status='COMPLETE_100_CLIENTS',reused_clients=98,resumed_clients=[64,87],resumed_new_CLS_solves=2*10*(4096-256),
        preserved_new_CLS_solves=98*10*(4096-256),logical_replicas=4096000,template_slots=8000,cells=1000,fallback_activations=len(fallbacks),
        hashes={f'client_{i:03d}_{suffix}':base.sha(out/f'client_{i:03d}_{suffix}') for i in (64,87) for suffix in ('accumulators.json.gz','verification.npz')}))
    base.finish(out,records,cells,inputs,a.commit,start)


if __name__=='__main__':main()
