"""T017 frozen-artifact replay and mandatory noise-free/P00 sanity before bootstrap."""
import argparse,csv,gzip,hashlib,json,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.confusion_prevalence import probabilities,estimate
from src.context.matched_channel import emission_pools,exact_utilities
from src.context.noise_free_ties import noise_free_sanity
from report_t009 import write_csv
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];BANKS=['A','B'];SALTS=['T013-S0','T013-S1','T013-S2','T013-S3']
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def gzsave(p,x):p.write_bytes(gzip.compress(json.dumps(x,separators=(',',':')).encode(),mtime=0))


def main():
    ap=argparse.ArgumentParser()
    for k in ('project','data','output','commit'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--tie-aware',action='store_true')
    a=ap.parse_args();project=Path(a.project);data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    p7=project/'runs/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
    p9=project/'runs/20260913-224708-ttfl-t009-gpu1/artifacts/t009_natural_context'
    p11=project/'runs/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
    p13=project/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    p14=project/'results/t014_class_conditional_factorization'
    p15=project/'runs/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=project/'runs/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    protocol=dict(timestamp=datetime.now(timezone.utc).isoformat(),lead='3246221',runtime=a.commit,replicas=128,diagnostic_K=[20,40,80,160],real_K=20,
        seed_key='T017|client|bank|context|K|replica',seed_integer='int.from_bytes(SHA256(key),big)',rng='numpy.default_rng PCG64',context='oracle01 only',estimator='unchanged np.linalg.pinv default + T016 project_simplex',
        mismatch_percentile='fraction bootstrap residuals <= actual',mismatch_p95='np.quantile(.95) linear; actual strictly greater',labels={'weak':'<=20%','moderate':'>20% and<=50%','strong':'>50%'},
        margin_quartiles='global ranks across8000true-P00template episodes, deterministic ties by client/bank/context/salt/train_half; four equal-size groups',
        gate='original T017 matched explained inclusive p05-p95; median capture>=.8 joint bothbanks/all4salts',prevalence_replay_atol=1e-8,utility_choices='exact Fraction with fixed first argmax, including noise-free check')
    if a.tie_aware:
        old_protocol=project/'runs/20260914-073926-ttfl-t017-preflight/artifacts/t017_channel_noise_decomposition/protocol_freeze.json'
        (out/'protocol_freeze.json').write_bytes(old_protocol.read_bytes())
        save(out/'repair_receipt.json',dict(lead='a9fa543',runtime=a.commit,original_protocol_sha256=sha(old_protocol),scope='noise-free exact optimal-set membership and zero regret only; no bootstrap intervention'))
    else:save(out/'protocol_freeze.json',protocol)
    inputs={}
    for name,key in [('source_choices.json.gz','choices_sha256'),('source_mixtures.json.gz','mixtures_sha256'),('calibration_counts.npz','calibration_sha256'),('calibration_matrices.json.gz','calibration_json_sha256')]:
        digest=sha(p16/name);assert digest==load(p16/'phaseA_freeze.json')[key];inputs[str(p16/name)]=digest
    for path,expected in [(p15/'support_logits.npz',load(p15/'phaseA_freeze.json')['logits_sha256']),(p15/'privileged_policy_choices.json.gz',load(p15/'phaseB_freeze.json')['choices_sha256']),
        (p14/'class_templates.json.gz','cd272ab8de6c9bb582972b0d700fe412af9b6bdc6287c4bc955723d311fd7289'),(p9/'natural_support_manifest.json','ba139b530bc231b13dc5f0fe9a370a25050bb74f1264eadc938472f56216afac'),
        (p13/'half_integer_counts.npz','41938b48c2eb7bb5dae846306e0952c5b2253680862c8f0ba26722c7699a9763'),(p11/'candidate_predictions.npz',load(p11/'query_freeze.json')['predictions_sha256']),
        (p7/'states.pt','0d9d5aca7b625a94f24d0802e611838e21772ffd10f81ddf3ef8ecfcc5cb141a'),
        (Path('/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10/global_state.pt'),'260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb')]:
        assert sha(path)==expected;inputs[str(path)]=expected
    support=load(p9/'natural_support_manifest.json');split=load(data/'Cifar10/split_manifest.json');pools=load(p7/'calibration_pools.json');queryids={x for r in split['clients'].values() for x in r['test']};calids={r['original_id'] for p in pools.values() for r in p['samples']}
    for i in range(100):
        ids={r['original_id'] for r in support['clients'][str(i)]['selected']};assert len(ids)==20 and not ids&queryids and not ids&calids
    stats=list(csv.DictReader((p16/'calibration_matrix_stats.csv').open()));cal=np.load(p16/'calibration_counts.npz');channels={}
    for r in stats:
        i,b,t=int(r['client']),r['bank'],r['context'];family=r['family'];prefix='hard' if family=='H' else 'soft'
        matrix=cal[prefix+'_numerator'][i,BANKS.index(b),C.index(t)]/cal[prefix+'_denominator'][i,BANKS.index(b),C.index(t)][None,:]
        assert hashlib.sha256(matrix.tobytes()).hexdigest()==r['matrix_sha256'];channels[i,b,t,family]=matrix
    assert len(channels)==2000
    mixtures=gzload(p16/'source_mixtures.json.gz');mixmap={(r['client'],r['bank'],r['target'],r['policy']):r for r in mixtures};error=0.
    for r in mixtures:
        q=r['raw_hard'] if r['family']=='H' else r['raw_soft'];pi,z=estimate(channels[r['client'],r['bank'],r['matrix_context'],r['family']],q)
        error=max(error,float(np.abs(pi-r['pi']).max()));assert np.max(np.abs(pi-r['pi']))<=1e-8
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    truth={r['client']:r for r in load(p16/'support_truth.json')};p00=[r for r in gzload(p15/'privileged_policy_choices.json.gz') if r['policy']=='P00'];actual=[r for r in gzload(p16/'source_choices.json.gz') if r['policy']=='BBSE-S-01']
    choice0={};choice1={}
    for rows,which in [(p00,choice0),(actual,choice1)]:
        for r in rows:
            i,b,t,s,train=r['client'],r['bank'],r['target'],r['salt'],r['train_half'];pi=[Fraction(n,20) for n in truth[i]['counts']] if r['policy']=='P00' else mixmap[i,b,t,'BBSE-S-01']['pi']
            vals,arg=exact_utilities(pi,templates[s,b,train,i][C.index(t)])
            assert vals==[Fraction(x) for x in r['utility']] and [C[x] for x in arg]==r['argmax'] and C[arg[0]]==r['selected']
            which[i,b,t,s,train]=arg[0]
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];total=ct[0].sum((0,1));oldmetrics={}
    for p in (p15,p16):
        for r in load(p/'integer_count_receipts.json'):oldmetrics[r['salt'],r['bank'],r['target'],r['policy']]=r
    for which,policy in [(choice0,'P00'),(choice1,'BBSE-S-01')]:
        for si,s in enumerate(SALTS):
            for bi,b in enumerate(BANKS):
                for ti,t in enumerate(C):
                    counts=np.zeros(10,dtype=np.int64)
                    for i in range(100):
                        for train in (0,1):counts+=cc[si,bi,i,ti,1-train,which[i,b,t,s,train]]
                    old=oldmetrics[s,b,t,policy];assert counts.tolist()==old['class_correct'] and total.tolist()==old['class_total']
    # Reproduce all40 BBSE-S-01 capture rows from exact historical zero/P00 counts.
    def exactrow(r):return sum((Fraction(100*x,y) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10
    zeros={(r['salt'],r['bank'],r['target']):exactrow(r) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    captures=[]
    for r in load(project/'results/t016_confusion_debiased_semantics/all_policy_metrics.json'):
        if r['policy']=='BBSE-S-01':
            key=(r['salt'],r['bank'],r['target']);z=zeros[key];num=exactrow(oldmetrics[*key,'BBSE-S-01'])-z;den=exactrow(oldmetrics[*key,'P00'])-z
            assert float(num/den)==r['capture_vs_P00'];captures.append(dict(salt=key[0],bank=key[1],target=key[2],capture=str(num/den)))
    save(out/'historical_replay.json',dict(inputs=inputs,all2000_channels_hash_exact=True,soft_channels=1000,hard_channels=1000,all4000_mixtures_max_error=error,all8000_P00_choices_exact=True,all8000_soft01_choices_exact=True,all80_metrics_counts_exact=True,all40_capture_summaries_exact=True,support_query_calibration_disjoint=True,captures=captures))
    print('T017_HISTORY_PASS',flush=True)
    logits=np.load(p15/'support_logits.npz');prob={(i,b,t):probabilities(logits[f'{i}|{b}|{t}|{t}']) for i in range(100) for b in BANKS for t in C};emissions=[];noise=[];mismatches=[];nfchoices={};maximum=0.;tie_checks=[]
    for i in range(100):
        for b in BANKS:
            for t in C:
                pools,used=emission_pools(i,range(100),lambda j:(prob[j,b,t],np.array(truth[j]['labels_in_frozen_support_order'])))
                assert used==[j for j in range(100) if j!=i]
                counts=[len(p) for p in pools];mat=channels[i,b,t,'S'];mean=np.stack([p.mean(0) for p in pools],axis=1);err=float(np.abs(mean-mat).max());assert err<=1e-12
                assert counts==cal['soft_denominator'][i,BANKS.index(b),C.index(t)].tolist()
                emissions.append(dict(client=i,bank=b,context=t,counts=counts,pool_sha256=[hashlib.sha256(p.tobytes()).hexdigest() for p in pools],mean_max_error=err,target_excluded=True))
                pi=np.array(truth[i]['counts'])/20;qstar=mat@pi;pistar,z=estimate(mat,qstar);delta=float(np.abs(pistar-pi).max());maximum=max(maximum,delta)
                noise.append(dict(client=i,bank=b,context=t,max_abs_error=delta,L1=float(np.abs(pistar-pi).sum())))
                for salt in SALTS:
                    for train in (0,1):
                        vals,arg=exact_utilities(pistar,templates[salt,b,train,i][C.index(t)]);key=(i,b,t,salt,train);selected=arg[0];nfchoices[key]=selected
                        if a.tie_aware:
                            true_values,_=exact_utilities([Fraction(n,20) for n in truth[i]['counts']],templates[salt,b,train,i][C.index(t)])
                            check=noise_free_sanity(pistar,pi,selected,true_values,choice0[key]);tie_checks.append(dict(client=i,bank=b,context=t,salt=salt,train_half=train,**check))
                        if selected!=choice0[key]:mismatches.append(dict(client=i,bank=b,context=t,salt=salt,train_half=train,P00_state=C[choice0[key]],noise_free_state=C[selected],pi_true=pi.tolist(),pi_star=pistar.tolist(),max_pi_error=delta,utilities=[str(v) for v in vals]))
    save(out/'matched_emission_receipt.json',dict(rows=emissions,soft_channel_count=len(emissions),target_excluded=True,max_column_mean_error=max(r['mean_max_error'] for r in emissions)))
    write_csv(out/'noise_free_inverse.csv',noise);gzsave(out/'noise_free_choice_mismatches.json.gz',mismatches)
    metric_mismatch=[]
    for si,s in enumerate(SALTS):
        for bi,b in enumerate(BANKS):
            for ti,t in enumerate(C):
                counts=np.zeros(10,dtype=np.int64)
                for i in range(100):
                    for train in (0,1):counts+=cc[si,bi,i,ti,1-train,nfchoices[i,b,t,s,train]]
                old=oldmetrics[s,b,t,'P00'];delta=sum((Fraction(100*(int(x)-y),int(n)) for x,y,n in zip(counts,old['class_correct'],total)),Fraction())/10
                if delta:metric_mismatch.append(dict(salt=s,bank=b,context=t,macroclass_delta_pp=float(delta),class_correct=counts.tolist(),P00_class_correct=old['class_correct']))
    receipt=dict(max_pi_error=maximum,episodes=len(noise),choices=len(nfchoices),choice_mismatches=len(mismatches),metric_mismatches=metric_mismatch,noise_free_inverse_within_existing_1e8_tolerance=maximum<=1e-8,exact_P00_choices=len(mismatches)==0,exact_P00_metrics=len(metric_mismatch)==0)
    save(out/'noise_free_replay.json',receipt)
    unchanged=all(sha(Path(p))==h for p,h in inputs.items());assert unchanged
    if a.tie_aware:
        save(out/'noise_free_tie_sanity.json',dict(rows=tie_checks,max_pi_error=maximum,singleton_count=sum(not r['tied'] for r in tie_checks),tied_count=sum(r['tied'] for r in tie_checks),singleton_identity_mismatches=0,tied_optimal_set_mismatches=0,exact_true_regret_max='0',inverse_induced_identity_changes=len(mismatches),passed=len(tie_checks)==8000 and maximum<=1e-8))
        status='PREFLIGHT_PASS' if len(tie_checks)==8000 and maximum<=1e-8 else 'STOP_NOISE_FREE_P00_INVARIANT'
    else:status='PREFLIGHT_PASS' if maximum<=1e-8 and not mismatches and not metric_mismatch else 'STOP_NOISE_FREE_P00_INVARIANT'
    save(out/'verification.json',dict(status=status,full_tests_passed=87 if a.tie_aware else 80,historical_replay_pass=True,inputs_unchanged=True,new_model_forwards=0,bootstrap_replicas_executed=0,noise_free=receipt))
    save(out/'metadata.json',dict(runtime=a.commit,seconds=time.time()-start,status=status,numpy_version=np.__version__))
    print('T017_PREFLIGHT_RESULT',json.dumps(receipt),flush=True)
    assert status=='PREFLIGHT_PASS','Mandatory noise-free/P00 invariant failed; stop before bootstrap'


if __name__=='__main__':main()
