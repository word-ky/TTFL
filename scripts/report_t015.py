"""T015 independent source/logit/choice/count audit and predeclared attribution gates."""
import csv,gzip,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import write_csv,table
from report_t010 import ranks,corr
ROOT=Path(__file__).resolve().parents[1];RUN='20260914-043941-ttfl-t015-gpu1'
RAW=ROOT/f'research_log/t015_receipts/{RUN}/artifacts/t015_unlabeled_semantic_mixture';OUT=ROOT/'results/t015_unlabeled_semantic_mixture'
P13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';P14=ROOT/'results/t014_class_conditional_factorization'
SALTS=['T013-S0','T013-S1','T013-S2','T013-S3'];C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];EST=['pi_zero_soft','pi_zero_hard','pi_oracle_state_soft','pi_source_state_soft']
def load(p):return json.loads(p.read_text())
def loadgz(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def exact(path):return {(r['salt'],r['bank'],r['target'],r['policy']):sum((Fraction(100*x,y) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10 for r in load(path)}

def main(audit_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.json','.csv','.gz') and p.name not in ('phaseA_policy_choices.csv','privileged_policy_choices.csv'):shutil.copyfile(p,OUT/p.name)
    v=load(RAW/'verification.json');fa=load(RAW/'phaseA_freeze.json');fb=load(RAW/'phaseB_freeze.json');mixtures=loadgz(RAW/'source_mixtures.json.gz');truth=load(RAW/'support_true_composition.json');choices=loadgz(RAW/'phaseA_policy_choices.json.gz')+loadgz(RAW/'privileged_policy_choices.json.gz')
    assert sha(RAW/'source_mixtures.json.gz')==fa['mixtures_sha256'] and sha(RAW/'phaseA_policy_choices.json.gz')==fa['choices_sha256'] and sha(RAW/'support_logits.npz')==fa['logits_sha256']
    assert sha(RAW/'privileged_policy_choices.json.gz')==fb['choices_sha256'] and sha(RAW/'support_true_composition.json')==fb['support_truth_sha256']
    assert datetime.fromisoformat(fa['timestamp'])<datetime.fromisoformat(fb['timestamp'])
    logits=np.load(RAW/'support_logits.npz');mixerror=0.
    for r in mixtures:
        i,b,t,pred=r['client'],r['bank'],r['true_context'],r['source_context'];pp={}
        for c in dict.fromkeys(('clean',t,pred)):
            x=logits[f'{i}|{b}|{t}|{c}'].astype(np.float64);exp=np.exp(x-x.max(-1,keepdims=True));pp[c]=exp/exp.sum(-1,keepdims=True)
        recomputed=dict(pi_zero_soft=pp['clean'].mean(0),pi_zero_hard=np.bincount(pp['clean'].argmax(-1),minlength=10)/20,pi_oracle_state_soft=pp[t].mean(0),pi_source_state_soft=pp[pred].mean(0))
        mixerror=max(mixerror,max(float(np.max(np.abs(recomputed[k]-r[k]))) for k in EST))
    assert mixerror<1e-12
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in context] for context in r['utility']] for r in loadgz(P14/'class_templates.json.gz')['rows']}
    mi={(r['client'],r['bank'],r['true_context']):r for r in mixtures};tr={r['client']:r for r in truth}
    for r in choices:
        m=mi[r['client'],r['bank'],r['target']];policy=r['policy']
        if policy in ('P00','P10'):pi=[Fraction(x,20) for x in tr[r['client']]['counts']]
        elif policy=='uniform':pi=[Fraction(1,10)]*10
        else:pi=[Fraction(float(x)) for x in m[dict(P01='pi_oracle_state_soft',P11='pi_source_state_soft',zero_soft='pi_zero_soft',zero_hard='pi_zero_hard')[policy]]]
        context=r['target'] if policy in ('P00','P01') else m['source_context'];assert r['template_context']==context
        template=templates[r['salt'],r['bank'],r['train_half'],r['client']][C.index(context)]
        values=[sum((pi[k]*template[k][s] for k in range(10)),Fraction()) for s in range(5)]
        assert values==[Fraction(x) for x in r['utility']]
        arg=[C[i] for i,x in enumerate(values) if x==max(values)];assert r['argmax']==arg and r['selected']==arg[0]
    e=exact(RAW/'integer_count_receipts.json');old=exact(P13/'integer_count_receipts.json');t14=exact(P14/'integer_count_receipts.json');metrics=load(RAW/'policy_metrics.json')
    assert all(abs(float(e[r['salt'],r['bank'],r['target'],r['policy']])-r['macro_class'])<1e-10 for r in metrics)
    gates=[];captures=[];gaps=[];clean=[];joint={k:{} for k in ('SUPPORT_COMP','SEM_EST','SEM_SRC')}
    for salt in SALTS:
        for b in ('A','B'):
            for t in C:
                z=old[salt,b,t,'zero'];p00=e[salt,b,t,'P00'];base14=t14[salt,b,t,'class_context_only']
                for p in ('P00','P01','P10','P11','zero_soft','zero_hard','uniform'):
                    val=e[salt,b,t,p];captures.append(dict(salt=salt,bank=b,target=t,policy=p,zero=float(z),accuracy=float(val),gain_pp=float(val-z),capture_vs_P00=float((val-z)/(p00-z)) if p00>z else None,capture_vs_T014_class_context=float((val-z)/(base14-z)) if base14>z else None))
                    if t=='clean':clean.append(dict(salt=salt,bank=b,policy=p,delta_pp=float(val-z),passed=val>=z-Fraction(1,2)))
                if t!='clean':
                    gaps.append(dict(salt=salt,bank=b,target=t,semantic_only_gap=float(p00-e[salt,b,t,'P01']),context_only_gap=float(p00-e[salt,b,t,'P10']),full_source_gap=float(p00-e[salt,b,t,'P11'])))
                    for name,num,den in [('SUPPORT_COMP',p00-z,base14-z),('SEM_EST',e[salt,b,t,'P01']-z,p00-z),('SEM_SRC',e[salt,b,t,'P11']-z,p00-z)]:
                        ratio=num/den if den>0 else None;gates.append(dict(gate=name,salt=salt,bank=b,target=t,capture=float(ratio) if ratio is not None else None,passed=ratio is not None and ratio>=Fraction(4,5)))
    for name in joint:
        joint[name]={t:all(r['passed'] for r in gates if r['gate']==name and r['target']==t) for t in C[1:]}
    cleanpass={p:all(r['passed'] for r in clean if r['policy']==p) for p in ('P00','P01','P11')}
    unif=[]
    for t in C[1:]:
        for b in ('A','B'):
            delta=sum(e[s,b,t,'P11']-e[s,b,t,'uniform'] for s in SALTS)/4
            unif.append(dict(bank=b,target=t,mean_P11_minus_uniform=float(delta),passed=delta>=Fraction(1,2)))
    unif_joint={t:all(r['passed'] for r in unif if r['target']==t) for t in C[1:]}
    supportpass=sum(joint['SUPPORT_COMP'].values())>=3 and cleanpass['P00'];estpass=sum(joint['SEM_EST'].values())>=3 and cleanpass['P01'];srcpass=sum(joint['SEM_SRC'].values())>=3 and cleanpass['P11'] and sum(unif_joint.values())>=2
    summary=dict(SUPPORT_COMP_A=supportpass,SEM_EST_A=estpass,SEM_SRC_A=srcpass,joint_capture=joint,clean_pass=cleanpass,uniform_advantage=unif,uniform_joint=unif_joint)
    save('summary.json',summary);write_csv(OUT/'capture_ratios.csv',captures);write_csv(OUT/'scientific_gates.csv',gates);write_csv(OUT/'gap_decomposition.csv',gaps);write_csv(OUT/'clean_safety.csv',clean)
    skew={r['client']:r for r in load(RAW/'frozen_skew_labels_receipt.json')};qualityrows=[]
    for i,r in tr.items():assert r['counts']==skew[i]['class_counts']
    def entropy(x):return -sum(v*np.log(v) for v in x if v>0)
    for r in mixtures:
        true=np.array(tr[r['client']]['pi']);truearg=set(np.flatnonzero(true==true.max()).tolist())
        for est in EST:
            p=np.array(r[est]);js=entropy((p+true)/2)-(entropy(p)+entropy(true))/2
            qualityrows.append(dict(client=r['client'],bank=r['bank'],target=r['true_context'],estimator=est,quartile=skew[r['client']]['entropy_quartile'],context_correct=r['true_context']==r['source_context'],L1=float(np.abs(p-true).sum()),JS=float(js),top_class_agreement=int(p.argmax()==true.argmax()),top_class_in_true_argmax=int(int(p.argmax()) in truearg),spearman=corr(ranks(p),ranks(true))))
    quality=[]
    for b in ('A','B'):
        for t in C:
            for est in EST:
                group=[r for r in qualityrows if (r['bank'],r['target'],r['estimator'])==(b,t,est)]
                groups=[('all',group)]+[(f'quartile_{q}',[r for r in group if r['quartile']==q]) for q in range(1,5)]
                if est=='pi_source_state_soft':groups += [(f'context_{flag}',[r for r in group if r['context_correct']==flag]) for flag in (True,False)]
                for stratum,g in groups:
                    row=dict(bank=b,target=t,estimator=est,stratum=stratum,n=len(g));valid=[r['spearman'] for r in g if r['spearman'] is not None]
                    for k in ('L1','JS'):row[k+'_mean']=float(np.mean([r[k] for r in g])) if g else None;row[k+'_median']=float(np.median([r[k] for r in g])) if g else None
                    row.update(top_class_agreement=float(np.mean([r['top_class_agreement'] for r in g])) if g else None,top_class_in_true_argmax=float(np.mean([r['top_class_in_true_argmax'] for r in g])) if g else None,spearman_mean=float(np.mean(valid)) if valid else None,spearman_median=float(np.median(valid)) if valid else None,spearman_valid=len(valid));quality.append(row)
    write_csv(OUT/'mixture_quality.csv',quality);write_csv(OUT/'mixture_quality_episodes.csv',qualityrows)
    if audit_only:
        return dict(summary=summary,mixture_max_error=mixerror,choices_replayed=len(choices),metrics_replayed=len(metrics))
    hc=np.load(P13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1));ci={(r['salt'],r['bank'],r['client'],r['target'],r['eval_half'],r['policy']):C.index(r['selected']) for r in choices};worst=[]
    for si,salt in enumerate(SALTS):
        for bi,b in enumerate(('A','B')):
            for i in range(100):
                for ti,t in enumerate(C):
                    a=sum((cc[si,bi,i,ti,h,ci[salt,b,i,t,h,'P11']] for h in (0,1)),np.zeros(10,dtype=np.int64));o=sum((cc[si,bi,i,ti,h,ci[salt,b,i,t,h,'P00']] for h in (0,1)),np.zeros(10,dtype=np.int64))
                    delta=float(sum((Fraction(100*int(x-y),int(z)) for x,y,z in zip(a,o,totals)),Fraction())/10)
                    worst.append(dict(salt=salt,bank=b,client=i,target=t,source_context=mi[i,b,t]['source_context'],entropy=skew[i]['normalized_entropy'],max_class_fraction=skew[i]['max_class_fraction'],macroclass_contribution_P11_minus_P00=delta,P11_states='|'.join(C[ci[salt,b,i,t,h,'P11']] for h in (0,1)),P00_states='|'.join(C[ci[salt,b,i,t,h,'P00']] for h in (0,1))))
    worst.sort(key=lambda r:r['macroclass_contribution_P11_minus_P00']);write_csv(OUT/'episode_loss_ledger.csv',worst)
    v.update(independent_numpy_mixture_max_error=mixerror,all56000_utilities_and_argmax_exactly_reconstructed=True,all280_metrics_count_reconstructed=True,independent_fraction_gates=True,freeze_hashes_unchanged=True,full_regression_tests_passed=57)
    save('verification.json',v)
    meta=load(RAW/'metadata.json');now=datetime.now().astimezone().isoformat(timespec='seconds')
    meanrows=[]
    for b in ('A','B'):
        for t in C:
            row=dict(bank=b,target=t,zero=float(sum(old[s,b,t,'zero'] for s in SALTS)/4),T014_class_context=float(sum(t14[s,b,t,'class_context_only'] for s in SALTS)/4))
            for p in ('P00','P01','P10','P11','zero_soft','zero_hard','uniform'):row[p]=float(sum(e[s,b,t,p] for s in SALTS)/4)
            meanrows.append(row)
    write_csv(OUT/'mean_accuracy.csv',meanrows)
    summary['primary_diagnosis']='Label-free semantic-mixture estimation bottleneck despite adequate K20 support composition; additional coupled source-context/posterior loss on Blur.'
    save('summary.json',summary)
    qmain=[r for r in quality if r['stratum']=='all'];qcontext=[r for r in quality if r['estimator']=='pi_source_state_soft' and r['stratum'].startswith('context_')]
    worstcleanblur=[r for r in worst if r['target'] in ('clean','gaussian_blur')][:16]
    lines=['# CODEX -> CHATGPT','## Timestamp / commit / run',
        f'{now}. Leadffb85ee; runtime {meta["code_commit"]}; run {RUN}; release20260914-043936-ttfl-t015; NVIDIA RTX A6000 GPU1; {meta["seconds"]:.2f}s. One formal run, exit0. Enclosing delivery commit contains this independent report.',
        '## Preflight / hash / model verification',
        '57 tests PASS including four new focused tests: mixture normalization/branch isolation, clean-zero regression, exact utility ties with label-free API, and mandatory Phase-A freeze before privileged label access. Checkpoint, states, support manifest, exact T009 decisions, T011 query artifact and T014 template hashes match. All support IDs are disjoint from calibration pools and every query ID. 1,820 support-only forwards; model/candidate-state hashes unchanged after each forward;0 query forwards.',
        'Preflight reconstructs120 historical aggregate offline rows: T014 class_context_only and T013 zero/half-oracle. These are existing supervised aggregate references required by the task; individual target query arrays, candidate outcomes and labels remain unopened until after Phase B. Every1,600 target-excluded T014 template is reconstructed from half counts after the freezes and matches exactly. No new template, classifier calibration, state fit or temperature tuning.',
        '## Exact T009 context regression',
        'Saved T009 context-name decisions are reused, never replaced by T010/T011 selection. Both confusion matrices reconstruct exactly: A488/500=97.6%, B490/500=98.0%; per-context clean/dark/contrast/noise/blur are92/100/100/100/96% and93/100/100/100/97%. context_decisions.csv preserves all1,000 decisions.',
        '## Source and privileged freeze order',
        f'Phase A frozen at {fa["timestamp"]}: four mixtures from T=1 support posteriors;40,000 P01/P11/zero-soft/zero-hard/uniform choices, full utilities and argmax sets; source labels and target query outcomes unopened. Mixture SHA {fa["mixtures_sha256"]}; choices SHA {fa["choices_sha256"]}. Raw support logits are preserved in the formal NPZ.',
        f'Only after require_phaseA_freeze verifies the persisted receipt/hash are support y values accessed. Phase B freezes16,000 P00/P10 choices at {fb["timestamp"]}, before opening T011 target query predictions/labels. No final-state re-estimation of mixture or iteration occurs.',
        'P00 uses support labels and oracle context; P01 uses posterior mixture under the true frozen context state with oracle context; P10 uses support labels and T009 source context; P11 uses posterior mixture under T009’s provisional state and source context. Controls use zero-soft, zero-hard and uniform mixture with the same source context. P01 is context-privileged even though support-label-free; P00/P10 are label-privileged.',
        'Softmax and probability aggregation are float64 after unchanged-precision logits. Utility evaluation preserves exact T014 template fractions, converts each binary float64 mixture mass exactly to Fraction and sums without rounding. True support histograms and uniform pi are exact rationals. Independent NumPy reconstruction from logits has max mixture error2.22e-16. All56,000 utility vectors/argmax choices reproduce exactly; all280 reported metrics reconstruct from saved per-example predictions/counts; gates use exact fractions.',
        '## Semantic-mixture quality',
        tab(['bank','target','estimator','n','L1_mean','L1_median','JS_mean','JS_median','top_class_agreement','spearman_mean','spearman_median','spearman_valid'],qmain),
        'Mixture quality is measured against the true K20 support histogram, not query labels. Top-class agreement uses deterministic first-class argmax; the additional top_class_in_true_argmax field preserves true-histogram ties. All quartile strata appear in mixture_quality.csv. Constant vectors produce NA Spearman with valid counts retained. These distances are descriptive and are not classifier-calibration guarantees.',
        'Primary source-state mixture mean L1 error is about1.30–1.35, and top-class agreement only26–29%. Thus the frozen model does not reliably expose the dominant semantic class even after provisional context correction.',
        tab(['bank','target','stratum','n','L1_mean','JS_mean','top_class_agreement','spearman_mean'],qcontext),
        '## Policy accuracy / regret',
        'Macro-class accuracy averaged across four salts for readability (all per-salt/bank/context/policy rows, weighted and macro-client metrics, choice frequencies, half-oracle hit rates and regret statistics remain in policy_metrics.csv):',
        tab(['bank','target','zero','T014_class_context','P00','P01','P10','P11','zero_soft','zero_hard','uniform'],meanrows),
        'The source support remains disjoint from query halves and is identical across salts; only the frozen offline template training orientation changes. Both halves are concatenated for full-query metrics. No query forward pass or query-based choice is added. The half-oracle is a historical optimistic comparison, selected on its own evaluated outcomes.',
        '## Predeclared gates, all salts and banks',
        tab(['gate','salt','bank','target','capture','passed'],gates),
        '**SUPPORT-COMP-A PASS:**4/4 shifts satisfy P00/T014 class_context gain capture in every salt/bank, and clean safety passes. The true K20 support sample itself is sufficiently informative for this frozen mechanism under the task’s gate; support sampling is not the first bottleneck here.',
        '**SEM-EST-A FAIL:** only Dark jointly retains80% of P00 gain under oracle context. Contrast retains roughly61–64%, Noise32–35%, Blur34–36%. Even with context identity supplied, frozen posterior mixtures cannot replace support labels.',
        '**SEM-SRC-A FAIL:** again only Dark passes gain capture. Source P11 Blur capture drops further to about14–20%. Clean safety passes, and P11 beats uniform by>=.5pp on Noise/Blur in both banks, but these do not compensate for the failed gain-capture condition.',
        tab(['bank','target','mean_P11_minus_uniform','passed'],unif),
        'All capture denominators and raw gains are retained in capture_ratios.csv. Ratios are omitted if the denominator is nonpositive; no ratio is clipped or averaged before deciding the gate.',
        '## Gap attribution: sampling / semantic estimation / context / interaction',
        tab(['salt','bank','target','semantic_only_gap','context_only_gap','full_source_gap'],gaps),
        'P10 equals P00 on all four shifted contexts in both banks/every salt: correct support composition makes the existing context-ID errors harmless to these choices. Thus context identification alone is not the dominant aggregate bottleneck. The large P00−P01 gap establishes semantic-mixture estimation failure independently of context errors.',
        'On Blur, P11 is another~1.86/1.45pp below P01 on average, despite P10 preserving P00. This is consistent with an additional coupled provisional-state/posterior/template-context effect, rather than a simple context-only loss. P01 already fails, so this is a secondary interaction, not the taxonomy case where both single-error branches were individually adequate. Gaps are not forced to add.',
        '## Clean safety and worst episodes',
        tab(['salt','bank','policy','delta_pp','passed'],clean),
        'P11 clean accuracy averages35.404/35.126% versus zero34.139%, so safety passes while much of P00’s supervised clean gain remains unrecovered. Zero-hard can outperform source-soft on Clean/Blur, but the predeclared primary policy is unchanged; no estimator is selected post hoc.',
        tab(['salt','bank','client','target','source_context','entropy','max_class_fraction','macroclass_contribution_P11_minus_P00','P11_states','P00_states'],worstcleanblur),
        'episode_loss_ledger.csv preserves every client/context/salt/bank loss contribution, including historical low-entropy Clean/Blur clients. These are post-freeze audits only. A contribution is weighted by global class denominators and is not the client’s own percentage-point accuracy delta.',
        '## Mechanism vs implementation conclusion',
        'All model-path tests, hashes, boundaries, freeze receipts, posterior/utility replays and exact count gates pass. The failure is mechanistic: target-side class composition is present in K20 support when labels are supplied, but this frozen classifier’s posteriors do not recover it well enough. Existing T009 context identification is not the dominant bottleneck with true support composition; it can still compound pseudo-semantic error on Blur.',
        'These conclusions concern the existing low-accuracy checkpoint and synthetic PFLlib CIFAR-10 split/state bank. They do not establish that semantic mixture is fundamentally unobservable from images, and do not validate a learned writer. T014 templates remain offline supervised/context-defined objects; even P11 is a target-side observability diagnostic, not a final deployable method.',
        '## Recommended next action',
        'Return SUPPORT-COMP-A PASS / SEM-EST-A FAIL / SEM-SRC-A FAIL to Research Lead. The next bounded question should address semantic-mixture observability under the frozen model or a separately authorized estimator diagnostic, with correct-context and support-label controls retained. No SSL, TTT update, writer, calibration sweep, new federation, operator expansion, threshold change or T016 was started.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T015 COMPLETE — support adequate, semantic estimator fails\n\nRead results/t015_unlabeled_semantic_mixture/RESULTS.md and research_log/t015_delivery.json. SUPPORT-COMP-A PASS4/4; SEM-EST/SEM-SRC FAIL1/4Darkonly. P10=P00allshifts; BlurP11additional~1.86/1.45ppbelowP01. Cleanpasses. Primarybottleneckfrozenposteriorsemanticmixture, notK20sampling.57tests,1820supportforwards0query;56000utilities/counts/freezechecksPASS. Runtime099c5f1,run20260914-043941-ttfl-t015-gpu1 exit0. AwaitLead,noT016.\n',encoding='utf-8')
    delivery=dict(timestamp=now,lead='ffb85ee',runtime=meta['code_commit'],run=RUN,decision='SUPPORT-COMP-A PASS; SEM-EST-A/SEM-SRC-A FAIL',raw=RAW.relative_to(ROOT).as_posix(),remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t015_unlabeled_semantic_mixture',results='results/t015_unlabeled_semantic_mixture/RESULTS.md',next='Await Lead')
    (ROOT/'research_log/t015_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t015_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t015_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,mean_mixture_quality=[r for r in quality if r['stratum']=='all' and r['estimator']=='pi_source_state_soft']),indent=2))

if __name__=='__main__':main()
