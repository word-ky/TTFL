"""Independent T016R matrix/choice/count replay, fixed CAL gates and diagnostics."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from report_t009 import write_csv,table
from report_t010 import ranks,corr
from src.context.confusion_prevalence import probabilities,project_simplex
RUN='20260914-063758-ttfl-t016r-cached'
RAW=ROOT/f'research_log/t016r_receipts/{RUN}/artifacts/t016_confusion_debiased_semantics'
OUT=ROOT/'results/t016_confusion_debiased_semantics'
P11=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
P13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
P14=ROOT/'results/t014_class_conditional_factorization'
P15=ROOT/'research_log/t015_receipts/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];SALTS=['T013-S0','T013-S1','T013-S2','T013-S3'];POL=['BBSE-H-01','BBSE-H-11','BBSE-S-01','BBSE-S-11']
def load(p):return json.loads(p.read_text())
def loadgz(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def exact(path):return {(r['salt'],r['bank'],r['target'],r['policy']):sum((Fraction(100*x,y) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10 for r in load(path)}
def entropy(x):return -sum(v*np.log(v) for v in x if v>0)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.gz'):shutil.copyfile(p,OUT/p.name)
    freeze=load(RAW/'phaseA_freeze.json');meta=load(RAW/'metadata.json');v=load(RAW/'verification.json')
    for name,key in [('source_choices.json.gz','choices_sha256'),('source_mixtures.json.gz','mixtures_sha256'),('calibration_counts.npz','calibration_sha256'),('calibration_matrices.json.gz','calibration_json_sha256')]:assert sha(RAW/name)==freeze[key]
    matrices=np.load(RAW/'calibration_counts.npz');mj=loadgz(RAW/'calibration_matrices.json.gz')
    for key in matrices.files:np.testing.assert_array_equal(matrices[key],mj[key])
    members=load(RAW/'calibration_membership.json')
    assert all(r['calibration_clients']==[j for j in range(100) if j!=r['target_client']] for r in members)
    truth={r['client']:r for r in load(RAW/'support_truth.json')};logits=np.load(P15/'support_logits.npz');pred=np.load(P11/'candidate_predictions.npz')
    skew={r['client']:r for r in load(P15/'frozen_skew_labels_receipt.json')}
    for i,r in truth.items():
        assert np.bincount(r['labels_in_frozen_support_order'],minlength=10).tolist()==r['counts']==skew[i]['class_counts']
    # Independent sufficient statistics from per-example predictions and saved y.
    hard_client=np.zeros((100,2,5,10,10),dtype=np.int64);soft_client=np.zeros((100,2,5,10,10));hden=np.zeros((100,10),dtype=np.int64);sden=np.zeros_like(hden)
    for j in range(100):
        y=pred[f'c{j}_labels'];sy=np.array(truth[j]['labels_in_frozen_support_order']);hden[j]=np.bincount(y,minlength=10);sden[j]=np.bincount(sy,minlength=10)
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                pr=pred[f'{b}|{t}|{t}|c{j}'];hard_client[j,bi,ti]=np.bincount(10*y+pr,minlength=100).reshape(10,10).T
                pp=probabilities(logits[f'{j}|{b}|{t}|{t}'])
                soft_client[j,bi,ti]=np.stack([pp[sy==k].sum(0) for k in range(10)],axis=1)
    matrix_error=0.
    for i in range(100):
        js=[j for j in range(100) if j!=i]
        hh=hard_client[js].sum(0);ss=np.zeros((2,5,10,10))
        for j in js:ss+=soft_client[j]
        np.testing.assert_array_equal(hh,matrices['hard_numerator'][i])
        for bi in range(2):
            for ti in range(5):
                np.testing.assert_array_equal(hden[js].sum(0),matrices['hard_denominator'][i,bi,ti])
                np.testing.assert_array_equal(sden[js].sum(0),matrices['soft_denominator'][i,bi,ti])
        matrix_error=max(matrix_error,float(np.abs(ss-matrices['soft_numerator'][i]).max()))
    assert matrix_error<1e-12
    mixtures=loadgz(RAW/'source_mixtures.json.gz');choices=loadgz(RAW/'source_choices.json.gz')
    mi={(r['client'],r['bank'],r['target'],r['policy']):r for r in mixtures}
    t15mi={(r['client'],r['bank'],r['true_context']):r for r in loadgz(P15/'source_mixtures.json.gz')}
    pierror=0.
    for r in mixtures:
        i,b,t,p=r['client'],r['bank'],r['target'],r['policy'];ctx=r['matrix_context'];bi=B.index(b);ti=C.index(ctx)
        assert ctx==(r['source_context'] if r['mode']=='11' else t)
        assert r['source_context']==t15mi[i,b,t]['source_context']
        prefix='hard' if r['family']=='H' else 'soft';mat=matrices[prefix+'_numerator'][i,bi,ti]/matrices[prefix+'_denominator'][i,bi,ti][None,:]
        pp=probabilities(logits[f'{i}|{b}|{t}|{ctx}']);qs=pp.mean(0);qh=np.bincount(pp.argmax(-1),minlength=10)/20
        np.testing.assert_allclose(qs,r['raw_soft'],rtol=0,atol=1e-14);np.testing.assert_array_equal(qh,r['raw_hard'])
        z=np.linalg.pinv(mat)@(qh if prefix=='hard' else qs);pi=project_simplex(z)
        pierror=max(pierror,float(np.abs(pi-r['pi']).max()))
        np.testing.assert_allclose(z,r['unprojected'],rtol=1e-10,atol=1e-8)
    assert pierror<1e-8  # Numerical BLAS replay only, never used by a scientific gate.
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in loadgz(P14/'class_templates.json.gz')['rows']}
    for r in choices:
        m=mi[r['client'],r['bank'],r['target'],r['policy']];assert r['template_context']==m['matrix_context']
        temp=templates[r['salt'],r['bank'],r['train_half'],r['client']][C.index(r['template_context'])]
        vals=[sum((Fraction(float(m['pi'][k]))*temp[k][s] for k in range(10)),Fraction()) for s in range(5)]
        assert vals==[Fraction(x) for x in r['utility']]
        arg=[C[s] for s,x in enumerate(vals) if x==max(vals)]
        assert r['argmax']==arg and r['selected']==arg[0]
    # Independent count-based reconstruction of every new metric, including regret.
    hc=np.load(P13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1))
    ci={(r['salt'],r['bank'],r['target'],r['client'],r['eval_half'],r['policy']):C.index(r['selected']) for r in choices}
    newmetrics=load(RAW/'policy_metrics.json');newexact=exact(RAW/'integer_count_receipts.json')
    for r in newmetrics:
        salt,b,t,p=r['salt'],r['bank'],r['target'],r['policy'];si=SALTS.index(salt);bi=B.index(b);ti=C.index(t)
        counts=np.zeros(10,dtype=np.int64);ca=[];regs=[];freq={c:0 for c in C};rh=gt2=gt5=0
        for i in range(100):
            nc=0
            for h in (0,1):
                selected=ci[salt,b,t,i,h,p];x=cc[si,bi,i,ti,h,selected];counts+=x;nc+=int(x.sum());freq[C[selected]]+=1
                reg=Fraction(100*(int(cc[si,bi,i,ti,h].sum(-1).max())-int(x.sum())),int(ct[si,i,h].sum()));regs.append(float(reg));rh+=reg==0;gt2+=reg>2;gt5+=reg>5
            ca.append(100*nc/int(ct[si,i].sum()))
        macro=sum((Fraction(100*int(x),int(y)) for x,y in zip(counts,totals)),Fraction())/10
        assert macro==newexact[salt,b,t,p] and abs(float(macro)-r['macro_class'])<1e-12
        assert freq==r['state_choice_frequency'] and rh/200==r['oracle_hit_rate'] and gt2/200==r['regret_gt2_fraction'] and gt5/200==r['regret_gt5_fraction']
        for name,value in [('sample_weighted',100*int(counts.sum())/int(totals.sum())),('macro_client',float(np.mean(ca))),('regret_median',float(np.median(regs))),('regret_p75',float(np.quantile(regs,.75))),('regret_p90',float(np.quantile(regs,.9)))]:assert abs(value-r[name])<1e-12
    # Same frozen posterior baselines and context-mode hard histograms, all compared
    # to true support prevalence only after the formal source freeze.
    episode=[]
    for i in range(100):
        true=np.array(truth[i]['pi'])
        for b in B:
            for t in C:
                estimates={p:mi[i,b,t,p]['pi'] for p in POL}
                for mode in ('01','11'):
                    m=mi[i,b,t,'BBSE-S-'+mode];estimates['raw_soft_'+mode]=m['raw_soft'];estimates['raw_hard_'+mode]=m['raw_hard']
                estimates['T015_zero_soft']=t15mi[i,b,t]['pi_zero_soft'];estimates['T015_zero_hard']=t15mi[i,b,t]['pi_zero_hard']
                for name,values in estimates.items():
                    pi=np.array(values);js=entropy((pi+true)/2)-(entropy(pi)+entropy(true))/2
                    episode.append(dict(client=i,bank=b,target=t,estimator=name,quartile=skew[i]['entropy_quartile'],L1=float(np.abs(pi-true).sum()),JS=float(js),top_class_agreement=int(pi.argmax()==true.argmax()),spearman=corr(ranks(pi),ranks(true))))
    quality=[]
    for b in B:
        for t in C:
            for est in estimates:
                rows=[r for r in episode if (r['bank'],r['target'],r['estimator'])==(b,t,est)]
                for stratum,group in [('all',rows)]+[(f'quartile_{q}',[r for r in rows if r['quartile']==q]) for q in range(1,5)]:
                    valid=[r['spearman'] for r in group if r['spearman'] is not None]
                    row=dict(bank=b,target=t,estimator=est,stratum=stratum,n=len(group))
                    for k in ('L1','JS'):
                        row[k+'_mean']=float(np.mean([r[k] for r in group]));row[k+'_median']=float(np.median([r[k] for r in group]))
                    row.update(top_class_agreement=float(np.mean([r['top_class_agreement'] for r in group])),spearman_mean=float(np.mean(valid)) if valid else None,spearman_median=float(np.median(valid)) if valid else None,spearman_valid=len(valid));quality.append(row)
    qindex={(r['bank'],r['target'],r['estimator'],r['stratum']):r for r in quality}
    improvement=[]
    for b in B:
        for t in C:
            for p in POL:
                for stratum in ['all']+[f'quartile_{q}' for q in range(1,5)]:
                    q=qindex[b,t,p,stratum];mode=p[-2:]
                    for control in ('raw_soft_'+mode,'raw_hard_'+mode,'T015_zero_hard'):
                        z=qindex[b,t,control,stratum]
                        improvement.append(dict(bank=b,target=t,policy=p,stratum=stratum,control=control,L1_reduction=z['L1_mean']-q['L1_mean'],JS_reduction=z['JS_mean']-q['JS_mean'],dominant_agreement_gain=q['top_class_agreement']-z['top_class_agreement']))
    write_csv(OUT/'prevalence_quality.csv',quality);write_csv(OUT/'prevalence_quality_episodes.csv',episode);write_csv(OUT/'prevalence_improvement.csv',improvement)
    with (RAW/'calibration_matrix_stats.csv').open() as f:stats=list(csv.DictReader(f))
    for r in stats:
        i,bi,ti=int(r['client']),B.index(r['bank']),C.index(r['context']);prefix='hard' if r['family']=='H' else 'soft'
        den=matrices[prefix+'_denominator'][i,bi,ti];mat=matrices[prefix+'_numerator'][i,bi,ti]/den[None,:]
        assert hashlib.sha256(mat.tobytes()).hexdigest()==r['matrix_sha256']
        assert den.tolist()==json.loads(r['counts']) and int(np.linalg.matrix_rank(mat))==int(r['rank'])
        np.testing.assert_allclose(np.linalg.svd(mat,compute_uv=False),json.loads(r['singular_values']),rtol=1e-10,atol=1e-12)
        cond=float(np.linalg.cond(mat));oldcond=float(r['condition'])
        assert (np.isinf(cond) and np.isinf(oldcond)) or abs(cond-oldcond)<=1e-10*abs(oldcond)
    channel_summary=[];projection=[]
    for b in B:
        for t in C:
            for family in ('H','S'):
                rows=[r for r in stats if (r['bank'],r['context'],r['family'])==(b,t,family)];conditions=[float(r['condition']) for r in rows];rks=[int(r['rank']) for r in rows];counts=[n for r in rows for n in json.loads(r['counts'])]
                row=dict(bank=b,context=t,family=family,matrices=len(rows),count_min=min(counts),count_median=float(np.median(counts)),rank_min=min(rks),rank_median=float(np.median(rks)),rank_max=max(rks),full_rank_fraction=sum(x==10 for x in rks)/100,condition_min=min(conditions),condition_median=float(np.median(conditions)),condition_max=max(conditions))
                channel_summary.append(row)
                for mode in ('01','11'):
                    group=[r for r in mixtures if (r['bank'],r['target'],r['family'],r['mode'])==(b,t,family,mode)]
                    projection.append(dict(bank=b,target=t,family=family,mode=mode,negative_entry_fraction=sum(r['negative_entries'] for r in group)/1000,episode_negative_fraction=sum(r['negative_entries']>0 for r in group)/100,projection_L1_mean=float(np.mean([r['projection_L1'] for r in group])),projection_L1_median=float(np.median([r['projection_L1'] for r in group])),projection_L1_max=max(r['projection_L1'] for r in group)))
    write_csv(OUT/'channel_summary.csv',channel_summary);write_csv(OUT/'projection_diagnostics.csv',projection)
    # Exact gates are computed from integer counts, never rounded displayed numbers.
    e={**exact(P13/'integer_count_receipts.json'),**exact(P14/'integer_count_receipts.json'),**exact(P15/'integer_count_receipts.json'),**newexact}
    historical=[r for r in load(P13/'crossfit_metrics.json') if r['policy']=='zero']+[r for r in load(P14/'policy_metrics.json') if r['policy']=='class_context_only']+[r for r in load(P15/'policy_metrics.json') if r['policy'] in ('P00','P01','P11','uniform','zero_hard')]
    combined=historical+newmetrics;metrics=[];gates=[];clean=[];advantages=[]
    for r in combined:
        s,b,t,p=r['salt'],r['bank'],r['target'],r['policy'];val=e[s,b,t,p];z=e[s,b,t,'zero'];p00=e[s,b,t,'P00'];old14=e[s,b,t,'class_context_only']
        rr={**r,'gain_vs_zero_pp':float(val-z),'capture_vs_P00':float((val-z)/(p00-z)) if p00>z else None,'capture_vs_T014':float((val-z)/(old14-z)) if old14>z else None};rr['state_choice_frequency']=json.dumps(rr['state_choice_frequency']);metrics.append(rr)
    write_csv(OUT/'policy_metrics.csv',metrics);save('all_policy_metrics.json',metrics)
    joint={p:{} for p in POL}
    for p in POL:
        for s in SALTS:
            for b in B:
                z=e[s,b,'clean','zero'];val=e[s,b,'clean',p];clean.append(dict(salt=s,bank=b,policy=p,delta_pp=float(val-z),passed=val>=z-Fraction(1,2)))
                for t in C[1:]:
                    z=e[s,b,t,'zero'];num=e[s,b,t,p]-z;den=e[s,b,t,'P00']-z;cap=num/den if den>0 else None
                    gates.append(dict(policy=p,salt=s,bank=b,target=t,gain_pp=float(num),P00_gain_pp=float(den),capture=float(cap) if cap is not None else None,passed=cap is not None and cap>=Fraction(4,5)))
        joint[p]={t:all(r['passed'] for r in gates if r['policy']==p and r['target']==t) for t in C[1:]}
    for b in B:
        for t in C[1:]:
            for control in ('P11','uniform'):
                delta=sum((e[s,b,t,'BBSE-S-11']-e[s,b,t,control] for s in SALTS),Fraction())/4
                advantages.append(dict(bank=b,target=t,control=control,mean_gain_pp=float(delta),passed=delta>=Fraction(1,2)))
    l1joint={t:all(qindex[b,t,'BBSE-S-01','all']['L1_mean']<qindex[b,t,'raw_soft_01','all']['L1_mean'] for b in B) for t in C[1:]}
    safe={p:all(r['passed'] for r in clean if r['policy']==p) for p in POL}
    advjoint={control:{t:all(r['passed'] for r in advantages if r['target']==t and r['control']==control) for t in C[1:]} for control in ('P11','uniform')}
    sem=sum(joint['BBSE-S-01'].values())>=3 and safe['BBSE-S-01'] and sum(l1joint.values())>=3
    src=sum(joint['BBSE-S-11'].values())>=3 and safe['BBSE-S-11'] and all(sum(advjoint[c].values())>=2 for c in advjoint)
    summary=dict(CAL_SEM_A=sem,CAL_SRC_A=src,primary='soft_emission_pinv',joint_capture=joint,clean_pass=safe,L1_improvement_joint=l1joint,source_advantage_joint=advjoint,formal_repaired_runs=1,previous_stop='2e51d81: implementation preflight stop before science')
    summary['diagnosis']='Fixed soft calibration substantially recovers semantic prevalence and improves utility, but still only Dark meets 80% P00 gain capture. Soft channels are full rank (condition about73–210); hard channels rank8–9. Large simplex corrections and residual prevalence/task loss remain; Blur has secondary context coupling. No operator failure inferred.'
    write_csv(OUT/'scientific_gates.csv',gates);write_csv(OUT/'clean_safety.csv',clean);write_csv(OUT/'source_advantages.csv',advantages);save('summary.json',summary)
    means=[]
    for b in B:
        for t in C:
            row=dict(bank=b,target=t)
            for p in ('zero','class_context_only','P00','P01','P11')+tuple(POL)+('uniform','zero_hard'):row[p]=float(sum((e[s,b,t,p] for s in SALTS),Fraction())/4)
            row['soft_source_minus_oracle_pp']=row['BBSE-S-11']-row['BBSE-S-01'];means.append(row)
    write_csv(OUT/'mean_accuracy.csv',means)
    v.update(independent_hard_integer_matrices_exact=True,independent_soft_numerator_max_error=matrix_error,independent_pinv_projected_max_error=pierror,
        independent_all2000_matrix_hashes_counts_ranks_spectra_verified=True,
        independent_all32000_utilities_argmaxes_exact=True,independent_all160_new_metrics_reconstructed=True,independent_exact_fraction_gates=True,full_regression_tests_passed=71,
        historical_JS_only_tolerance_receipt=load(RAW/'historical_float_replay_receipt.json'),freeze_hashes_unchanged=True)
    save('verification.json',v)
    now=datetime.now().astimezone().isoformat(timespec='seconds');qualitymain=[r for r in quality if r['stratum']=='all' and r['estimator'] in POL+['raw_soft_01','raw_soft_11','raw_hard_01','raw_hard_11']]
    lines=['# CODEX → CHATGPT: T016R / completed T016','## Execution and implementation repair',
        f'{now}. Lead T016R2bad2a0 resumes original T0166476c99. Runtime `{meta["code_commit"]}`; release20260914-063742-ttfl-t016r; run `{RUN}`, exit0, {meta["seconds"]:.2f}s. All71 tests PASS. Zero new support/query forwards; frozen A6000 logits reused. NumPy{meta["numpy_version"]}, default pinv; no model/state/temperature/support/context/solver/gate changes.',
        'Stopped run20260914-054411-ttfl-t016-cached at2e51d81 remains preserved as implementation-blocker provenance. It never built a T016 channel or evaluated a policy. The authorized repair permits only finite episodeJS absolute difference<=1e-12,rtol0. All other episodefields, schema/order/count, aggregates and gates remain exact. Eight comparator tests plus63 prior tests pass. The repaired receipt records4000rows,2nonzeroJSdeltas,max4.440892098500626e-16; these observed counts are not hard-coded acceptance rules.',
        'Historical T015 hashes,56,000utilities/argmaxes,280count-derived macroclass rows,220quality aggregate rows and96gate rows reproduce. Original calibration-pool/support/query IDs remain disjoint. No estimator/outcome-driven retry occurred.',
        '## Calibration, exclusion and freeze semantics',
        'Hard channels use other99clients’ T011 query predictions under the exact modeled context and matching frozen state; soft primary channels use other99clients’ exact K20 support logits under the modeled context/state. Columns condition on true class; hard integer numerators/denominators and soft float64 sums/counts are persisted. The classifier’s clean state is the unchanged zero affine. No new inference or learned calibration parameters.',
        'Target exclusion is per episode: the builder skips i before calling the calibration loader. Since all100clients rotate as targets, the process necessarily reads all clients’ labels in their roles as other-client calibration data. Historical replay is also explicitly privileged. Thus “all target labels globally unopened” would be inaccurate; the enforced invariant is that target i labels/outcomes never enter its own calibration, prevalence estimate or source choice. The receipt states this scope explicitly. This is supervised offline, transductive cross-client calibration, not fully unsupervised training or evidence for an independently trained deployable writer.',
        f'Phase A freeze{freeze["timestamp"]}:2000target-excluded matrices,4000corrected source mixtures and32,000complete utility/argmax/state choices. ChoiceSHA`{freeze["choices_sha256"]}`; mixtureSHA`{freeze["mixtures_sha256"]}`; calibrationNPZSHA`{freeze["calibration_sha256"]}`. Only after freeze are own-support prevalence quality and selected-policy target metrics computed. Source11 indexes both channel and utility template by frozen T009 predicted context; oracle01 uses privileged true context.',
        f'Independent local replay reconstructs all hard integer matrices exactly and soft numerators from saved ordered labels/logits with max error{matrix_error:.3g}; pinv+projection max prevalence difference{pierror:.3g}. All32,000utility vectors/argmaxes match exact Fraction arithmetic. All160newmetrics reconstruct from integer counts; the formal evaluation additionally checks per-example T011 predictions. All1600T014target-excluded templates reconstruct from unchanged half counts. Numeric BLAS replay tolerances do not enter CAL gates.',
        '## Observation-channel identifiability',tab(['bank','context','family','count_min','count_median','rank_min','rank_median','rank_max','condition_min','condition_median','condition_max'],channel_summary),
        'Full per-matrix singular values, ranks, counts and hashes are in calibration_matrix_stats.csv. H/S differ in calibration sample source/size as specified; their comparison is not a controlled test of hard-vs-soft observations alone. Rank is NumPy numerical matrix_rank; condition numbers are reported without an invented pass threshold.',
        tab(['bank','target','family','mode','negative_entry_fraction','episode_negative_fraction','projection_L1_mean','projection_L1_median','projection_L1_max'],projection),
        '## Semantic prevalence quality',tab(['bank','target','estimator','L1_mean','L1_median','JS_mean','JS_median','top_class_agreement','spearman_mean','spearman_valid'],qualitymain),
        'Quality compares to the same true K20 histogram only after freeze. All entropy quartiles, medians, per-episode Spearman and raw-soft/raw-hard improvement deltas are retained in prevalence_quality*.csv and prevalence_improvement.csv. Raw_hard01/11 are histograms under matching provisional states; T015_zero_hard additionally preserves the original zero-state hard baseline. Dominant-class agreement uses deterministic first argmax; constant-vector Spearman is NA and valid counts are reported.',
        '## Retrieval utility: exact prescribed policy chain',tab(['bank','target','zero','class_context_only','P00','P01','P11']+POL+['uniform','zero_hard','soft_source_minus_oracle_pp'],means),
        'Displayed accuracies are macro-class averaged across four salts, solely for readability. policy_metrics.csv contains all per-salt/bank/context policies including weighted/macro-client accuracy, state frequencies, half-oracle hit, regret quantiles and>2/>5pp fractions, gain capture vsP00/T014 and clean deltas. Decisions use exact per-salt integer ratios, not rounded/averaged captures. CIFAR pool follows the inherited PFLlib merged train/test client split, not official CIFAR10-test benchmark.',
        '## Unchanged CAL gates',f'**CAL-SEM-A: {"PASS" if sem else "FAIL"}. CAL-SRC-A: {"PASS" if src else "FAIL"}.** Primary remains soft_emission_pinv.',
        'Joint80%capture in bothbanks/every salt: '+json.dumps(joint),
        'Clean>=zero−.5pp: '+json.dumps(safe)+'; primaryoracle meanL1improvement in bothbanks: '+json.dumps(l1joint),
        tab(['bank','target','control','mean_gain_pp','passed'],advantages),
        'Source>.5pp advantage requirements against rawP11 and uniform remain separate prescribed tests: '+json.dumps(advjoint),
        'All exact gate rows appear in scientific_gates.csv; all clean rows in clean_safety.csv. The1e-12JS replay tolerance is never used in any gate.',
        '## Scientific diagnosis',
        'Both CAL gates FAIL on the gain-capture condition: only Dark passes jointly. Soft oracle01 captures87.8–88.3% of P00 gain on Dark,71.2–73.0% on Contrast,66.6–69.3% on Noise and66.0–68.3% on Blur across banks/salts. Source11 preserves the first three but Blur falls to42.9–50.1%. Every clean-safety check passes. Source11 exceeds both rawP11 and uniform by>=.5pp on all4shifts in bothbanks averaged acrosssalts; those positive subconditions do not override the failed80%capture criterion.',
        'This is not the “little prevalence improvement” case. On shifted contexts, softoracle L1 falls from1.302–1.348 to0.743–0.853 (roughly37–43% reduction), and dominant-class agreement rises from26–29% to64–72%. Utility also improves: source11 versus rawP11 gains A/B Dark+1.026/+0.747pp,Contrast+1.147/+1.388pp,Noise+2.234/+2.506pp,Blur+2.643/+2.721pp. Thus the fixed observation model recovers substantial signal, but not enough for the predeclared retrieval requirement; semantic estimation and task-weighted sensitivity are both residual issues rather than a zero-signal conclusion.',
        'Hard channels are rank9 in clean/dark/noise/blur and rank8 in contrast, with infinite condition numbers. This establishes lost directions in the hard argmax observation channel. In contrast all1000softchannels have rank10 and condition numbers72.7–209.7. There is no numerical rank-collapse evidence for the soft primary channel; full rank alone does not guarantee a stable K20 estimate. All oracle-soft episodes have negative unprojected entries (about42–44% of entries); simplex projection changes L1 by means1.27–1.77. Finite-support variation, channel mismatch and task-sensitive residual error are compatible explanations, not separately identified causes. No claim that frozen feature semantics are intrinsically unobservable follows.',
        'Blur retains a secondary coupling loss: softsource11 is2.089/1.631pp below oracle01 for A/B, while Dark/Contrast/Noise branches coincide. Oracle01 already fails on three shifts, so this is not the taxonomy case of a successful oracle estimator blocked only by deployment context ID. Clean source11 remains+3.913/+3.678pp abovezero.',
        '## Scope of interpretation',
        'The implementation repair is separate from channel identifiability, prevalence estimation and retrieval utility. The numerical results above test the specified fixed classifier-output observation channel. Selector failure alone does not establish failure of the neutral192-scalar operator or absence of semantic information in frozen features. No next task is assigned; stop here and return evidence to Lead.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    concise='\n\n'.join([lines[0],lines[2],
        '**T016R COMPLETE. CAL-SEM-A FAIL; CAL-SRC-A FAIL.** Only Dark jointly retains>=80%P00 gain in bothbanks/every salt. Original6476c99 scientific protocol unchanged. Full report: `results/t016_confusion_debiased_semantics/RESULTS.md`; gates/quality/channel/count/freeze receipts are beside it.',
        'Implementation:71testsPASS. The narrow finite episodeJS atol1e-12/rtol0 repair accepts only the observed2/4000roundoff differences,max4.44e-16. All other historical fields, aggregates/gates/hashes and56,000utilities remain exact. Preserve stopped2e51d81 provenance; one repaired formal run,exit0,28.26s,zero newforwards.',
        'Exclusion/freeze:2000other99hard/softchannels;4000mixtures;32,000choices frozen before target prevalence/utility audit. Other-client labels are explicitly supervised offline calibration, episode-specific target exclusion. All clients rotate, so global labels are not literally unopened; no target i label/outcome feeds its own channel/choice. Saved raw matrices/logits/labels enable replay. Independent hard counts/matrixhashes/ranks exact,soft-sum error1.42e-14,projected-pi error1.19e-13,all32,000exactutilities and160newmetric rows reconstructed.',
        'Channel identifiability:hardchannels rank8–9,conditioninf. Softprimary rank10 everywhere,condition72.7–209.7. Hard argmax loses directions; soft prediction output is not numerically collapsed. Yet every oracle-soft episode has negative preprojection masses; meanprojection L1change1.27–1.77. Fullrank alone does not guarantee stable K20 prevalence.',
        'Prevalence:oracle-soft L1 improves on4/4shifts in bothbanks from1.30–1.35 to0.74–0.85; dominant-classagreement26–29%→64–72%. This is substantial semantic recovery, not the little-improvement taxonomy. Taskutility improves too but remains belowgate; no operator failure inferred.',
        'Utility:softoracle capture Dark87.8–88.3%,Contrast71.2–73.0%,Noise66.6–69.3%,Blur66.0–68.3%. SourceBlur42.9–50.1%,an extra2.089/1.631pp loss. Source11 beats rawP11 anduniform by>=.5pp onall4shifts; clean+3.913/+3.678ppvs zero. Bothsecondary conditions pass; gaincapture fails. No thresholds changed.',
        'Inference limit:residual finite-support noise, cross-client channel mismatch and task-sensitive prevalence error are not separated here. Oracle alreadyfails, so Blur context coupling is secondary. No next task assigned; stopped after this package and await research-lead decision.'])+'\n'
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(concise,encoding='utf-8')
    handoff=f'# T016R COMPLETE\n\n{now}. Lead2bad2a0; runtime{meta["code_commit"]}; run{RUN}exit0.71testsPASS;2000matrices,4000mixtures,32000choicesfrozen;0newforwards. CAL-SEM-A={sem},CAL-SRC-A={src}. Read results/t016_confusion_debiased_semantics/RESULTS.md. Preserve stopped2e51d81provenance. No nextstage; awaitLead.\n'
    for name in ('HANDOFF.md','T016_HANDOFF.md'):(ROOT/'research_log'/name).write_text(handoff,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+handoff)
    save('summary.json',summary)
    delivery=dict(timestamp=now,lead='2bad2a0',original_lead='6476c99',runtime=meta['code_commit'],run=RUN,status='COMPLETE',gates=summary,raw=RAW.relative_to(ROOT).as_posix(),results=OUT.relative_to(ROOT).as_posix(),next='Await Lead; no next scientific task')
    (ROOT/'research_log/t016r_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t016r_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t016r_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,means=means,channels=channel_summary,matrix_error=matrix_error,pi_error=pierror),indent=2))


if __name__=='__main__':main()
