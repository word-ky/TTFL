"""T019 privileged observation/utility audit; no query-count file is opened here."""
import argparse,gzip,hashlib,json,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from preflight_t018 import load,gzload,sha,save,C,B,S
from report_t009 import write_csv
from src.context.confusion_prevalence import probabilities,channel_from_other_clients
from src.context.constrained_prevalence import ActiveSetCLS
from src.context.observation_audit import draw_source_ids,decompose,single_class_repair
from src.context.exact_template_lookup import compile_template,exact_lookup
from src.context.matched_channel import exact_utilities
from eval_t017_bootstrap import entropy


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();p=Path(a.project);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p15=p/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=p/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p18=p/'runs/20260914-132634-ttfl-t018r2-science/artifacts/t018r2_constrained_prevalence'
    p18pre=p/'runs/20260914-132551-ttfl-t018r2-preflight/artifacts/t018r2_constrained_prevalence'
    p14=p/'results/t014_class_conditional_factorization'
    inputs=load(p18/'input_hashes.json')
    for path,h in inputs.items():assert sha(Path(path))==h
    for directory in (p18,p18pre):
        for path in directory.iterdir():
            if path.is_file():inputs[str(path)]=sha(path)
    for path in (p16/'calibration_membership.json',p16/'support_truth.json',p14/'class_templates.json.gz',ROOT/'src/context/constrained_prevalence.py'):
        inputs[str(path)]=sha(path)
    assert sha(ROOT/'src/context/constrained_prevalence.py')==sha(p/'releases/20260914-132535-ttfl-t018r2-preflight/src/context/constrained_prevalence.py')
    save(out/'input_hashes.json',inputs)
    protocol=(p/'results/t019_real_channel_heterogeneity/PROTOCOL_FREEZE.md').read_bytes()
    (out/'PROTOCOL_FREEZE.md').write_bytes(protocol)
    save(out/'runtime.json',dict(lead='d37bbc4',runtime=a.commit,numpy_version=np.__version__,new_model_forwards=0,query_metrics_opened=False))
    truth={r['client']:r for r in load(p16/'support_truth.json')};labels=np.array([truth[i]['labels_in_frozen_support_order'] for i in range(100)])
    counts=np.array([truth[i]['counts'] for i in range(100)]);true=counts/20
    assert np.array_equal(counts,np.array([np.bincount(y,minlength=10) for y in labels]))
    mapping=[dict(flat_id=i*20+k,client=i,position=k,original_id=truth[i]['original_ids'][k],label=int(labels[i,k])) for i in range(100) for k in range(20)]
    save(out/'source_id_mapping.json',mapping)
    logits=np.load(p15/'support_logits.npz');probs=np.zeros((100,2,5,20,10))
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                z=logits[f'{i}|{b}|{t}|{t}'];probs[i,bi,ti]=probabilities(z)
                assert probs[i,bi,ti].tobytes()==probabilities(z).tobytes()
    cal=np.load(p16/'calibration_counts.npz');sn=cal['soft_numerator'];sd=cal['soft_denominator'];matrices=sn/sd[...,None,:]
    members=load(p16/'calibration_membership.json')
    obs={(r['client'],r['bank'],r['target']):r for r in gzload(p16/'source_mixtures.json.gz') if r['policy']=='BBSE-S-01'}
    shape=(100,2,5);aq=np.zeros(shape+(10,));residual=np.zeros_like(aq);mu=np.full(shape+(10,10),np.nan);difference=mu.copy();weighted=np.zeros_like(mu);norm=np.zeros(shape+(3,))
    perclass=[];max_q_error=max_identity=0.
    for i in range(100):
        assert members[i]['target_client']==i and members[i]['calibration_clients']==[j for j in range(100) if j!=i]
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                _,num,den,used=channel_from_other_clients(i,range(100),lambda j:(probs[j,bi,ti],labels[j]))
                np.testing.assert_array_equal(num,sn[i,bi,ti]);np.testing.assert_array_equal(den,sd[i,bi,ti]);assert used==members[i]['calibration_clients']
                n,m,d,w,q,r=decompose(probs[i,bi,ti],labels[i],matrices[i,bi,ti]);loc=(i,bi,ti)
                error=float(np.max(np.abs(q-obs[i,b,t]['raw_soft'])));identity=float(np.max(np.abs(r-w.sum(0))));assert error<=1e-12 and identity<=1e-12
                max_q_error=max(max_q_error,error);max_identity=max(max_identity,identity)
                aq[loc]=q;residual[loc]=r;mu[loc]=m;difference[loc]=d;weighted[loc]=w
                norm[loc]=[np.abs(r).sum(),np.linalg.norm(r),sum(true[i,y]*np.abs(d[y]).sum() for y in np.flatnonzero(n))]
                for y in np.flatnonzero(n):
                    perclass.append(dict(client=i,bank=b,context=t,label=int(y),count=int(n[y]),mu_real=json.dumps(m[y].tolist()),mu_cross=json.dumps(matrices[loc][:,y].tolist()),
                        d=json.dumps(d[y].tolist()),d_L1=float(np.abs(d[y]).sum()),d_L2=float(np.linalg.norm(d[y])),weighted_contribution=json.dumps(w[y].tolist()),
                        residual=json.dumps(r.tolist()),residual_L1=float(norm[loc][0]),residual_L2=float(norm[loc][1])))
    write_csv(out/'per_class_real_residuals.csv',perclass)
    np.savez_compressed(out/'real_decomposition.npz',probabilities=probs,counts=counts,q=aq,residual=residual,mu=mu,d=difference,weighted=weighted,norm=norm)
    save(out/'phaseA_verification.json',dict(cells=1000,channel_numerators_bitwise_equal=True,channel_denominators_exact=True,target_excluded=True,
        max_q_reconstruction_error=max_q_error,max_residual_identity_error=max_identity,probability_replay_bitwise=True,absent_classes_not_imputed=True,paired_support_coverage=1.))
    print('T019_PHASE_A_PASS',flush=True)
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    compiled={};values={};args={};margins=np.zeros(shape+(8,));true_regrets={}
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for h in (0,1):
                        key=(i,bi,ti,si*2+h);table=templates[s,b,h,i][ti];compiled[key]=compile_template(table)
                        vv,aa=exact_utilities([Fraction(int(n),20) for n in counts[i]],table);values[key]=vv;args[key]=aa
                        ordered=sorted(vv,reverse=True);margins[key]=float(ordered[0]-ordered[1]);true_regrets['|'.join(map(str,key))]=[str(max(vv)-v) for v in vv]
    save(out/'true_template_regrets.json',true_regrets)
    def score(pi,i,bi,ti):
        choice=np.zeros(8,dtype=np.uint8);metric=np.zeros((8,4))
        for slot in range(8):
            key=(i,bi,ti,slot);s,du=exact_lookup(pi,*compiled[key],counts[i]);vv=values[key];aa=args[key]
            choice[slot]=s;metric[slot]=[du,float(max(vv)-vv[s]),s in aa,s==aa[0]]
        return choice,metric
    def sem(pi,i):
        tr=true[i];return [float(np.abs(pi-tr).sum()),entropy((pi+tr)/2)-(entropy(pi)+entropy(tr))/2,float(pi.argmax()==tr.argmax())]
    solve_checks={};max_kkt=max_sum=0.;max_updates=0
    def solve(solver,q,kind):
        nonlocal max_kkt,max_sum,max_updates
        pi,r=solver.solve(q);assert 2*r['objective']<=2*r['initial_objective']+1e-12
        max_kkt=max(max_kkt,r['direct_KKT']);max_sum=max(max_sum,r['sum_error']);max_updates=max(max_updates,r['updates'])
        solve_checks[kind]=solve_checks.get(kind,0)+1
        return pi
    actualpi=np.zeros_like(aq);actualchoice=np.zeros(shape+(8,),dtype=np.uint8);actualscore=np.zeros(shape+(8,4));actualsem=np.zeros(shape+(3,))
    repairpi=np.full(shape+(10,10),np.nan);repairchoice=np.full(shape+(10,8),255,dtype=np.uint8);repairscore=np.full(shape+(10,8,4),np.nan);repairsem=np.full(shape+(10,3),np.nan)
    allchoice=np.zeros_like(actualchoice);all_sanity=[]
    previous={(r['client'],r['bank'],r['context']):r for r in load(p18pre/'all_actual_solutions.json')}
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=(i,bi,ti);solver=ActiveSetCLS(matrices[loc]);pi=solve(solver,aq[loc],'actual');np.testing.assert_array_equal(pi,previous[i,b,t]['pi'])
                actualpi[loc]=pi;actualchoice[loc],actualscore[loc]=score(pi,*loc);actualsem[loc]=sem(pi,i)
                qall=matrices[loc]@true[i];assert np.max(np.abs(aq[loc]-weighted[loc].sum(0)-qall))<=1e-12
                pa=solve(solver,qall,'all_repaired');ch,sc=score(pa,*loc);allchoice[loc]=ch
                assert np.max(np.abs(pa-true[i]))<=1e-8 and .5*np.sum((matrices[loc]@pa-qall)**2)<=1e-16
                assert np.all(sc[:,1]==0) and np.all(sc[:,2]==1)
                all_sanity.append(dict(client=i,bank=b,context=t,max_pi_error=float(np.abs(pa-true[i]).max()),all_optimal=True,zero_regret=True))
                for y in np.flatnonzero(counts[i]):
                    rp=solve(solver,single_class_repair(aq[loc],weighted[loc],y),'single_repaired');repairpi[loc+(y,)]=rp
                    repairchoice[loc+(y,)],repairscore[loc+(y,)]=score(rp,*loc);repairsem[loc+(y,)]=sem(rp,i)
    save(out/'all_repaired_sanity.json',all_sanity)
    nshape=shape+(128,);nullq=np.zeros(nshape+(10,));nullpi=np.zeros_like(nullq);nullnorm=np.zeros(nshape+(3,));classnorm=np.full(nshape+(10,),np.nan)
    nullchoice=np.zeros(nshape+(8,),dtype=np.uint8);nullscore=np.zeros(nshape+(8,4));nullsem=np.zeros(nshape+(3,));source_ids=np.zeros(nshape+(20,),dtype=np.uint16)
    pairednorm=np.full(nshape+(2,),np.nan);pairedchange=np.full(nshape+(8,2),np.nan);paired_clean_pi=np.full_like(nullpi,np.nan)
    flatlabels=labels.ravel();flatprob=probs.transpose(1,2,0,3,4).reshape(2,5,2000,10);replays=[]
    for i in range(100):
        for bi,b in enumerate(B):
            clean_solver=ActiveSetCLS(matrices[i,bi,0])
            for ti,t in enumerate(C):
                solver=ActiveSetCLS(matrices[i,bi,ti]);expected=matrices[i,bi,ti]@true[i];expected_clean=matrices[i,bi,0]@true[i]
                for replica in range(128):
                    loc=(i,bi,ti,replica);ids=draw_source_ids(i,b,t,replica,counts[i],flatlabels)
                    assert not np.any(ids//20==i);np.testing.assert_array_equal(np.bincount(flatlabels[ids],minlength=10),counts[i]);source_ids[loc]=ids
                    if replica==0:
                        independent=[]
                        for y,n in enumerate(counts[i]):
                            if n:
                                seed=int(hashlib.sha256(f'T019|{i}|{b}|{t}|{replica}|{y}'.encode()).hexdigest(),16)
                                pool=[j*20+k for j in range(100) if j!=i for k in range(20) if labels[j,k]==y]
                                rng=np.random.Generator(np.random.PCG64(seed));independent.extend(np.array(pool)[rng.integers(0,len(pool),size=int(n))].tolist())
                        np.testing.assert_array_equal(ids,independent);replays.append(dict(client=i,bank=b,context=t,replica=0,ids=independent))
                    pp=flatprob[bi,ti,ids];n,m,d,w,q,r=decompose(pp,flatlabels[ids],matrices[i,bi,ti]);pi=solve(solver,q,'null')
                    nullq[loc]=q;nullpi[loc]=pi;nullchoice[loc],nullscore[loc]=score(pi,i,bi,ti);nullsem[loc]=sem(pi,i)
                    nullnorm[loc]=[np.abs(r).sum(),np.linalg.norm(r),sum(true[i,y]*np.abs(d[y]).sum() for y in np.flatnonzero(n))]
                    for y in np.flatnonzero(n):classnorm[loc+(y,)]=np.abs(d[y]).sum()
                    if ti:
                        qc=flatprob[bi,0,ids].mean(0);pc=solve(clean_solver,qc,'paired_clean');paired_clean_pi[loc]=pc
                        _,sc=score(pc,i,bi,0);dr=(q-expected)-(qc-expected_clean)
                        pairednorm[loc]=[np.abs(dr).sum(),np.linalg.norm(dr)];pairedchange[loc]=nullscore[loc][:,:2]-sc[:,:2]
        print('T019_NULL_CLIENT',i,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(out/'null_observations.npz',q=nullq,pi=nullpi,norm=nullnorm,class_norm_L1=classnorm,source_ids=source_ids,paired_norm=pairednorm,paired_task_change=pairedchange,paired_clean_pi=paired_clean_pi)
    np.savez_compressed(out/'utility_choices.npz',actual_pi=actualpi,actual_choices=actualchoice,actual_scores=actualscore,actual_semantic=actualsem,
        null_choices=nullchoice,null_scores=nullscore,null_semantic=nullsem,repair_pi=repairpi,repair_choices=repairchoice,repair_scores=repairscore,repair_semantic=repairsem,all_repaired_choices=allchoice,margin=margins)
    save(out/'seed_replay.json',dict(cases=len(replays),policy='with replacement; SHA256(T019|client|bank|context|replica|class) full big endian integer; PCG64',replays=replays,paired_source_ids=True))
    assert all(sha(Path(path))==h for path,h in inputs.items())
    save(out/'solver_verification.json',dict(passed=True,counts=solve_checks,max_KKT=max_kkt,max_sum_error=max_sum,max_updates=max_updates,all_repaired_sanity=True,source_exclusion=True,seed_replays=len(replays),new_model_forwards=0,query_metrics_opened=False))
    save(out/'phaseB_choices_freeze.json',dict(runtime=a.commit,seconds=time.time()-start,query_metrics_opened=False,hashes={name:sha(out/name) for name in ('real_decomposition.npz','null_observations.npz','utility_choices.npz','true_template_regrets.json','seed_replay.json')}))
    print('T019_CHOICES_FROZEN',time.time()-start,flush=True)


if __name__=='__main__':main()
