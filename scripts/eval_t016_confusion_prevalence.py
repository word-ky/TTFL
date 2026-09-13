"""T016 fixed cross-client calibration; source freeze precedes target evaluation.

Labels have episode-specific roles: j!=i labels are authorized offline calibration
for i. They are never target-i estimator inputs. Historical T015 replay is isolated.
"""
import argparse,csv,gzip,hashlib,json,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.confusion_prevalence import probabilities,channel_from_other_clients,normalized_channel,estimate,source_estimate
from report_t009 import write_csv
import report_t015 as old_report

C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];BANKS=['A','B'];SALTS=['T013-S0','T013-S1','T013-S2','T013-S3']
POLICIES=['BBSE-H-01','BBSE-H-11','BBSE-S-01','BBSE-S-11']
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def gzsave(p,x):p.write_bytes(gzip.compress(json.dumps(x,separators=(',',':'),allow_nan=False).encode(),mtime=0))
def csvsave(p,rows):write_csv(p,[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
def choose(pi,template):
    w=[Fraction(float(x)) for x in pi]
    values=[sum((w[k]*template[k][s] for k in range(10)),Fraction()) for s in range(5)]
    arg=[C[s] for s,v in enumerate(values) if v==max(values)]
    return arg[0],arg,[str(v) for v in values]


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','data','output','commit'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();project=Path(a.project);data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    def artifact(task,run,name):return project/f'runs/{run}/artifacts/{name}'
    p7=artifact('t007r','20260913-203200-ttfl-t007r-gpu1','t007r_transfer')
    p9=artifact('t009','20260913-224708-ttfl-t009-gpu1','t009_natural_context')
    p11=artifact('t011','20260914-003521-ttfl-t011-gpu1-lf','t011_task_proximal')
    p13=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    p14=project/'results/t014_class_conditional_factorization'
    p15=artifact('t015','20260914-043941-ttfl-t015-gpu1','t015_unlabeled_semantic_mixture')
    baseline=Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10')
    hashes={str(baseline/'global_state.pt'):'260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb',
        str(p7/'states.pt'):'0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a',
        str(p9/'natural_support_manifest.json'):'ba139b530bc231b13dc5f0fe9a370a25050bb74f1264eadc938472f56216afac',
        str(p14/'class_templates.json.gz'):'cd272ab8de6c9bb582972b0d700fe412af9b6bdc6287c4bc955723d311fd7289',
        str(p11/'candidate_predictions.npz'):load(p11/'query_freeze.json')['predictions_sha256'],
        str(p13/'half_integer_counts.npz'):'41938b48c2eb7bb5dae846306e0952c5b2253680862c8f0ba26722c7699a9763',
        str(ROOT/'src/models/pfllib.py'):'22acbc42950652b62dd01fba765ddc7204e2dba665643e47bb9dc2edf79162d3'}
    assert all(sha(Path(p))==h for p,h in hashes.items())
    # Explicitly required historical replay, isolated from new estimator inputs.
    old_report.RAW=p15;old_report.P13=p13;old_report.P14=p14;old_report.OUT=out/'historical_preflight'
    replay=old_report.main(audit_only=True)
    historical_summary=load(project/'results/t015_unlabeled_semantic_mixture/summary.json')
    assert all(historical_summary[k]==v for k,v in replay['summary'].items())
    for name in ('mixture_quality.csv','mixture_quality_episodes.csv','scientific_gates.csv'):
        with (old_report.OUT/name).open() as f:rows=list(csv.DictReader(f))
        with (project/'results/t015_unlabeled_semantic_mixture'/name).open() as f:oldrows=list(csv.DictReader(f))
        assert rows==oldrows,name
    support=load(p9/'natural_support_manifest.json');split=load(data/'Cifar10/split_manifest.json');pools=load(p7/'calibration_pools.json')
    calids={r['original_id'] for p in pools.values() for r in p['samples']};queryids={x for r in split['clients'].values() for x in r['test']}
    for i in range(100):
        ids={r['original_id'] for r in support['clients'][str(i)]['selected']}
        assert len(ids)==20 and not ids&calids and not ids&queryids
    assert sha(p9/'source_decisions.json')==load(p9/'selection_freeze.json')['decisions_sha256']
    decisions={(r['client'],r['bank'],r['true_context']):r['selected'] for r in load(p9/'source_decisions.json')}
    mixtures=gzload(p15/'source_mixtures.json.gz');mi={(r['client'],r['bank'],r['true_context']):r for r in mixtures}
    logits=np.load(p15/'support_logits.npz')
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    assert len(templates)==1600
    save(out/'preflight.json',dict(historical_replay=replay,historical_quality_csv_exact=True,hashes=hashes,all_disjointness_verified=True,
        ordering='Isolated historical T015 replay first. T016 Phase A may use other-client labels for offline calibration only; target outcomes are not estimator inputs. Final target evaluation follows global source freeze.'))
    print('T016_PREFLIGHT_PASS',flush=True)

    # Hard calibration uses authorized other-client query observations. The loader
    # receives only j!=i; caching does not change which clients enter each matrix.
    query_archive=np.load(p11/'candidate_predictions.npz')
    @lru_cache(None)
    def hard_data(j,b,c):
        return query_archive[f'{b}|{c}|{c}|c{j}'],query_archive[f'c{j}_labels']
    @lru_cache(None)
    def support_labels(j):
        d=np.load(data/f'Cifar10/train/{j}.npz',allow_pickle=True)['data'].item()
        return np.asarray(d['y'])[[r['train_index'] for r in support['clients'][str(j)]['selected']]]
    @lru_cache(None)
    def soft_data(j,b,c):
        return probabilities(logits[f'{j}|{b}|{c}|{c}']),support_labels(j)
    shape=(100,2,5,10,10);hn=np.zeros(shape,dtype=np.int64);sn=np.zeros(shape);hd=np.zeros((100,2,5,10),dtype=np.int64);sd=np.zeros_like(hd)
    matrices={};stats=[];members=[]
    for i in range(100):
        for bi,b in enumerate(BANKS):
            for ti,c in enumerate(C):
                hard,_,den,used=channel_from_other_clients(i,range(100),lambda j:hard_data(j,b,c))
                _,soft,sden,sused=channel_from_other_clients(i,range(100),lambda j:soft_data(j,b,c))
                assert used==sused==[j for j in range(100) if j!=i]
                assert np.array_equal(hard.sum(0),den)
                hn[i,bi,ti]=hard;sn[i,bi,ti]=soft;hd[i,bi,ti]=den;sd[i,bi,ti]=sden
                for family,num,d in [('H',hard,den),('S',soft,sden)]:
                    mat=normalized_channel(num,d);matrices[i,b,c,family]=mat
                    sv=np.linalg.svd(mat,compute_uv=False);cond=float(np.linalg.cond(mat))
                    stats.append(dict(client=i,bank=b,context=c,family=family,counts=d.tolist(),count_min=int(d.min()),count_median=float(np.median(d)),singular_values=sv.tolist(),rank=int(np.linalg.matrix_rank(mat)),condition=cond if np.isfinite(cond) else 'inf',matrix_sha256=hashlib.sha256(mat.tobytes()).hexdigest()))
        members.append(dict(target_client=i,calibration_clients=[j for j in range(100) if j!=i],target_excluded=True))
    np.savez_compressed(out/'calibration_counts.npz',hard_numerator=hn,hard_denominator=hd,soft_numerator=sn,soft_denominator=sd)
    gzsave(out/'calibration_matrices.json.gz',dict(order=dict(clients=100,banks=BANKS,contexts=C,orientation='predicted_dimension,true_class'),hard_numerator=hn.tolist(),hard_denominator=hd.tolist(),soft_numerator=sn.tolist(),soft_denominator=sd.tolist()))
    csvsave(out/'calibration_matrix_stats.csv',stats);save(out/'calibration_membership.json',members)
    print('T016_CALIBRATION_READY',len(stats),flush=True)
    estimated=[];choices=[]
    for i in range(100):
        for b in BANKS:
            for t in C:
                source=decisions[i,b,t];assert source==mi[i,b,t]['source_context']
                for mode,ctx in [('01',t),('11',source)]:
                    pp=probabilities(logits[f'{i}|{b}|{t}|{ctx}']);qs=pp.mean(0);qh=np.bincount(pp.argmax(-1),minlength=10)/20
                    raw=mi[i,b,t]['pi_oracle_state_soft' if mode=='01' else 'pi_source_state_soft'];assert np.max(np.abs(qs-raw))<1e-12
                    for family,q in [('H',qh),('S',qs)]:
                        bankmat={c:matrices[i,b,c,family] for c in C}
                        pi,z=source_estimate(bankmat,source,q) if mode=='11' else estimate(bankmat[t],q)
                        assert np.all(pi>=0) and abs(pi.sum()-1)<1e-10
                        policy=f'BBSE-{family}-{mode}'
                        estimated.append(dict(client=i,bank=b,target=t,source_context=source,mode=mode,family=family,policy=policy,matrix_context=ctx,
                            raw_soft=qs.tolist(),raw_hard=qh.tolist(),pi=pi.tolist(),unprojected=z.tolist(),negative_entries=int((z<0).sum()),projection_L1=float(np.abs(pi-z).sum())))
                        for salt in SALTS:
                            for train in (0,1):
                                selected,arg,values=choose(pi,templates[salt,b,train,i][C.index(ctx)])
                                choices.append(dict(client=i,bank=b,target=t,source_context=source,salt=salt,train_half=train,eval_half=1-train,policy=policy,template_context=ctx,selected=selected,argmax=arg,utility=values))
    gzsave(out/'source_mixtures.json.gz',estimated);gzsave(out/'source_choices.json.gz',choices)
    freeze=dict(timestamp=datetime.now(timezone.utc).isoformat(),choices_sha256=sha(out/'source_choices.json.gz'),mixtures_sha256=sha(out/'source_mixtures.json.gz'),calibration_sha256=sha(out/'calibration_counts.npz'),calibration_json_sha256=sha(out/'calibration_matrices.json.gz'),choices=len(choices),primary='soft_emission_pinv',pinv='numpy.linalg.pinv(C), library default rcond',numpy_version=np.__version__,target_labels_used_by_own_estimator=False,target_query_evaluation_opened=False,
        calibration_label_scope='Other-client T011 query labels for hard; other-client K20 support labels for soft. All clients rotate as targets, so global process has read labeled calibration examples, but every target-specific channel excludes that target before loading/summing. Historical T015 replay is isolated and privileged. No assertion that all labels are globally unopened.')
    save(out/'phaseA_freeze.json',freeze);print('T016_PHASE_A_FROZEN',json.dumps(freeze),flush=True)

    # Target prevalence audit / candidate-outcome evaluation starts here.
    assert sha(out/'source_choices.json.gz')==load(out/'phaseA_freeze.json')['choices_sha256']
    truth=[]
    for i in range(100):
        labels=support_labels(i);counts=np.bincount(labels,minlength=10)
        assert counts.tolist()==next(r['counts'] for r in load(p15/'support_true_composition.json') if r['client']==i)
        truth.append(dict(client=i,counts=counts.tolist(),pi=(counts/20).tolist(),labels_in_frozen_support_order=labels.tolist(),original_ids=[r['original_id'] for r in support['clients'][str(i)]['selected']]))
    save(out/'support_truth.json',truth)
    halves=load(p13/'query_halves.json');hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];hits=cc.sum(-1);n=ct.sum(-1);totals=ct[0].sum((0,1))
    # Verify the offline T014 target-excluded templates against exact half counts.
    from src.context.class_factorization import class_templates
    for (salt,b,train,i),template in templates.items():
        assert class_templates(cc[SALTS.index(salt),BANKS.index(b),:,:,train,:,:],ct[SALTS.index(salt),:,train,:],i)==template
    ci={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):C.index(r['selected']) for r in choices}
    metrics=[];receipts=[];regretrows=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                for p in POLICIES:
                    correct=np.zeros(10,dtype=np.int64);client_acc=[];regrets=[];freq={c:0 for c in C};hit=gt2=gt5=0
                    for i in range(100):
                        y=query_archive[f'c{i}_labels'];nc=0
                        for half in (0,1):
                            s=ci[salt,b,i,t,half,p];idx=halves[salt][str(i)][half]['indices'];pr=query_archive[f'{b}|{t}|{C[s]}|c{i}'][idx];yy=y[idx]
                            count=np.bincount(yy[pr==yy],minlength=10);assert np.array_equal(count,cc[si,bi,i,ti,half,s])
                            correct+=count;nc+=int(count.sum());freq[C[s]]+=1
                            reg=Fraction(100*(int(hits[si,bi,i,ti,half].max())-int(count.sum())),int(n[si,i,half]));regrets.append(float(reg));hit+=reg==0;gt2+=reg>2;gt5+=reg>5
                            regretrows.append(dict(salt=salt,bank=b,target=t,client=i,half=half,policy=p,selected=C[s],regret=float(reg)))
                        client_acc.append(100*nc/len(y))
                    macro=sum((Fraction(100*int(x),int(y)) for x,y in zip(correct,totals)),Fraction())/10
                    metrics.append(dict(salt=salt,bank=b,target=t,policy=p,macro_class=float(macro),sample_weighted=100*int(correct.sum())/int(totals.sum()),macro_client=float(np.mean(client_acc)),state_choice_frequency=freq,oracle_hit_rate=hit/200,regret_median=float(np.median(regrets)),regret_p75=float(np.quantile(regrets,.75)),regret_p90=float(np.quantile(regrets,.9)),regret_gt2_fraction=gt2/200,regret_gt5_fraction=gt5/200))
                    receipts.append(dict(salt=salt,bank=b,target=t,policy=p,class_correct=correct.tolist(),class_total=totals.tolist()))
    save(out/'policy_metrics.json',metrics);csvsave(out/'policy_metrics.csv',metrics);save(out/'integer_count_receipts.json',receipts);gzsave(out/'half_regrets.json.gz',regretrows)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    assert sha(out/'source_choices.json.gz')==freeze['choices_sha256'] and sha(out/'source_mixtures.json.gz')==freeze['mixtures_sha256']
    save(out/'verification.json',dict(historical_T015_reproduction=replay,exact_historical_quality_and_gates=True,frozen_hashes_unchanged=True,source_query_original_calibration_disjoint=True,
        all2000_channels_target_excluded=True,hard_exact_columns=True,soft_columns_normalized=True,target_support_labels_no_path_to_own_estimate=True,source_matrix_uses_T009_context_only=True,phaseA_frozen_before_target_evaluation=True,
        offline_other_client_labels_explicitly_supervised=True,all1600_template_exclusions_reconstructed=True,all160_metrics_per_example_count_reconstructed=True,new_support_forwards=0,new_query_forwards=0,model_or_state_updates=0))
    save(out/'metadata.json',dict(code_commit=a.commit,seconds=time.time()-start,numpy_version=np.__version__,primary='soft_emission_pinv',new_forwards=0,tests_passed=63,
        calibration_hard='other99 T011 query predictions under matching modeled context/state',calibration_soft='other99 T015 K20 support posterior under matching modeled context/state',fixed_solver='np.linalg.pinv default rcond; standard Euclidean simplex projection'))
    print('T016_DONE',time.time()-start,flush=True)


if __name__=='__main__':main()
