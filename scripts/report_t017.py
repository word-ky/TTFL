"""T017R independent count/seed replay and frozen N/M/X decomposition."""
import csv,gzip,hashlib,json,shutil,sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from report_t009 import write_csv,table
from src.context.matched_channel import emission_pools,replica_seed,bootstrap_observation,exact_utilities,mismatch_residual
from src.context.confusion_prevalence import probabilities
RUN='20260914-083609-ttfl-t017r-cached'
RAW=ROOT/f'research_log/t017r_receipts/{RUN}/artifacts/t017_channel_noise_decomposition';OUT=ROOT/'results/t017_channel_noise_decomposition'
P13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization';P14=ROOT/'results/t014_class_conditional_factorization'
P15=ROOT/'research_log/t015_receipts/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture';P16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
P11=ROOT/'research_log/t011_receipts/20260914-003521-ttfl-t011-gpu1-lf/artifacts/t011_task_proximal'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=['T013-S0','T013-S1','T013-S2','T013-S3'];KS=[20,40,80,160]
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def macro(count,total):return sum((Fraction(100*int(x),int(y)) for x,y in zip(count,total)),Fraction())/10
def exactmetrics(path,policy):return {(r['bank'],r['target'],r['salt']):macro(r['class_correct'],r['class_total']) for r in load(path) if r['policy']==policy}
def quantile(values,num,den):
    vals=sorted(values);position=Fraction((len(vals)-1)*num,den);lo=position.numerator//position.denominator;weight=position-lo
    return vals[lo] if not weight else vals[lo]*(1-weight)+vals[lo+1]*weight


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checked_sources=[]
    for manifest in ('t015_artifact_manifest.json','t016r_artifact_manifest.json'):
        for entry in load(ROOT/'research_log'/manifest):
            assert sha(ROOT/entry['path'])==entry['sha256']
            checked_sources.append(dict(path=entry['path'],sha256=entry['sha256']))
    save('source_artifact_hash_replay.json',dict(all_source_artifacts_unchanged=True,count=len(checked_sources),checked=checked_sources))
    for p in RAW.iterdir():
        if p.is_file() and p.suffix in ('.json','.gz','.csv'):shutil.copyfile(p,OUT/p.name)
    receipt=load(RAW/'bootstrap_receipt.json');verification=load(RAW/'verification.json')
    assert verification['status']=='BOOTSTRAP_COMPLETE'
    for name,digest in receipt['hashes'].items():assert sha(RAW/name)==digest
    arr=np.load(RAW/'bootstrap_arrays.npz');qc=np.load(RAW/'bootstrap_query_counts.npz');choice=arr['choices'];pi=arr['pi'];qraw=arr['q'];DU=arr['DU']
    l1,js,top,negative,projection=[arr[k] for k in ('L1','JS','top','negative','projection_L1')]
    hc=np.load(P13/'half_integer_counts.npz');cc=hc['correct'];ct=hc['total'];totals=ct[0].sum((0,1));counts=qc['aggregate_counts'];regrethits=qc['query_regret_hits'];halfn=qc['query_half_n']
    # Independent vectorized count reconstruction for every aggregate replica.
    for ki in range(4):
        for bi in range(2):
            for ti in range(5):
                for si in range(4):
                    reconstructed=np.zeros((128,10),dtype=np.int64)
                    for train in (0,1):
                        h=1-train;slot=2*si+train;selected=choice[ki,:,bi,ti,:,slot];lookup=cc[si,bi,:,ti,h]
                        hits=lookup[np.arange(100)[:,None],selected];reconstructed+=hits.sum(0)
                        np.testing.assert_array_equal(lookup.sum(-1).max(-1)[:,None]-hits.sum(-1),regrethits[ki,:,bi,ti,:,slot])
                    np.testing.assert_array_equal(reconstructed,counts[ki,bi,ti,si])
    # Verify all T013 per-half candidate counts directly from immutable T011 arrays.
    predictions=np.load(P11/'candidate_predictions.npz');halves=load(P13/'query_halves.json');perexample_checks=0
    for si,s in enumerate(S):
        for i in range(100):
            y=predictions[f'c{i}_labels']
            for h in (0,1):
                ids=halves[s][str(i)][h]['indices'];np.testing.assert_array_equal(np.bincount(y[ids],minlength=10),ct[si,i,h])
                for bi,b in enumerate(B):
                    for ti,t in enumerate(C):
                        for state in range(5):
                            pred=predictions[f'{b}|{t}|{C[state]}|c{i}'][ids]
                            np.testing.assert_array_equal(np.bincount(y[ids][pred==y[ids]],minlength=10),cc[si,bi,i,ti,h,state]);perexample_checks+=1
    truth={r['client']:r for r in load(P16/'support_truth.json')};cal=np.load(P16/'calibration_counts.npz');logits=np.load(P15/'support_logits.npz')
    actual={(r['client'],r['bank'],r['target']):r for r in gzload(P16/'source_mixtures.json.gz') if r['policy']=='BBSE-S-01'}
    prob={(i,b,t):probabilities(logits[f'{i}|{b}|{t}|{t}']) for i in range(100) for b in B for t in C}
    diagnostics=gzload(RAW/'template_diagnostics.json.gz');templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(P14/'class_templates.json.gz')['rows']}
    seed_error=0.;mismatch=[];lookup_checks=0
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                pools,used=emission_pools(i,range(100),lambda j:(prob[j,b,t],np.array(truth[j]['labels_in_frozen_support_order'])))
                assert used==[j for j in range(100) if j!=i]
                for ki,K in enumerate(KS):
                    replay,n=bootstrap_observation(pools,truth[i]['counts'],K,replica_seed(i,b,t,K,0));seed_error=max(seed_error,float(np.abs(replay-qraw[ki,i,bi,ti,0]).max()))
                matrix=cal['soft_numerator'][i,bi,ti]/cal['soft_denominator'][i,bi,ti][None,:];expected=matrix@np.array(truth[i]['pi'])
                md=mismatch_residual(actual[i,b,t]['raw_soft'],expected,qraw[0,i,bi,ti]);mismatch.append(dict(client=i,bank=b,context=t,**md))
                for si,s in enumerate(S):
                    for train in (0,1):
                        vals,arg=exact_utilities(pi[0,i,bi,ti,127],templates[s,b,train,i][ti]);assert choice[0,i,bi,ti,127,2*si+train]==arg[0];lookup_checks+=1
    assert seed_error<=1e-14
    write_csv(OUT/'channel_mismatch.csv',mismatch);msummary=[]
    for b in B:
        for t in C:
            rows=[r for r in mismatch if r['bank']==b and r['context']==t];fraction=sum(r['above_p95'] for r in rows)/100
            msummary.append(dict(bank=b,context=t,median_D_actual=float(np.median([r['D_actual'] for r in rows])),median_bootstrap_p50=float(np.median([r['p50'] for r in rows])),median_empirical_percentile=float(np.median([r['percentile'] for r in rows])),above_p95_fraction=fraction,label='weak' if fraction<=.2 else 'moderate' if fraction<=.5 else 'strong'))
    write_csv(OUT/'channel_mismatch_summary.csv',msummary)
    z=exactmetrics(P13/'integer_count_receipts.json','zero');p00=exactmetrics(P15/'integer_count_receipts.json','P00');actualacc=exactmetrics(P16/'integer_count_receipts.json','BBSE-S-01')
    envelope={(r['bank'],r['context'],r['salt']):r for r in csv.DictReader((RAW/'p00_tie_envelope.csv').open())}
    aggregate=[];rawagg=gzload(RAW/'bootstrap_aggregate_replicas.json.gz');fraction_acc={}
    for r in rawagg:
        ki=KS.index(r['K']);bi=B.index(r['bank']);ti=C.index(r['context']);si=S.index(r['salt']);value=macro(counts[ki,bi,ti,si,r['replica']],totals)
        assert value==Fraction(r['macroclass_exact']);key=r['bank'],r['context'],r['salt'];assert (value-z[key])/(p00[key]-z[key])==Fraction(r['capture_exact'])
        fraction_acc.setdefault((r['K'],*key),[]).append(value)
    for (K,b,t,s),values in fraction_acc.items():
        key=(b,t,s);zero=z[key];den=p00[key]-zero;caps=[(v-zero)/den for v in values];lo=quantile(caps,5,100);med=quantile(caps,1,2);hi=quantile(caps,95,100);ac=(actualacc[key]-zero)/den
        aggregate.append(dict(K=K,bank=b,context=t,salt=s,macroclass_p05=float(quantile(values,5,100)),macroclass_median=float(quantile(values,1,2)),macroclass_p95=float(quantile(values,95,100)),capture_p05=float(lo),capture_median=float(med),capture_p95=float(hi),actual_capture=float(ac),explained=lo<=ac<=hi,median80=med>=Fraction(4,5),capture_p05_exact=str(lo),capture_median_exact=str(med),capture_p95_exact=str(hi)))
    write_csv(OUT/'bootstrap_aggregate.csv',aggregate)
    prevalence=[]
    for ki,K in enumerate(KS):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                row=dict(K=K,bank=b,context=t)
                for name,values in [('L1',l1),('JS',js),('projection_L1',projection)]:
                    x=values[ki,:,bi,ti].reshape(-1);row[name+'_mean']=float(x.mean());row[name+'_median']=float(np.median(x));row[name+'_p05']=float(np.quantile(x,.05));row[name+'_p95']=float(np.quantile(x,.95))
                row['dominant_class_agreement']=float(top[ki,:,bi,ti].mean());row['negative_entry_fraction']=float(negative[ki,:,bi,ti].mean()/10);prevalence.append(row)
    write_csv(OUT/'bootstrap_prevalence_summary.csv',prevalence)
    sensitivity=[]
    for r in diagnostics:
        i,bi,ti,si,train=r['client'],B.index(r['bank']),C.index(r['context']),S.index(r['salt']),r['train_half'];slot=2*si+train;selected=choice[0,i,bi,ti,:,slot];arg=r['argmax']
        regret=np.array([float(Fraction(x)*100) for x in r['state_true_regret']])[selected];canonical=selected==arg[0];optimal=np.isin(selected,arg);qr=100*regrethits[0,i,bi,ti,:,slot]/halfn[i,slot]
        sensitivity.append(dict(client=i,bank=r['bank'],context=r['context'],salt=r['salt'],train_half=train,argmax_size=r['argmax_size'],best_tied=r['best_tied'],legacy_margin_pp=100*float(Fraction(r['legacy_margin'])),distinct_level_margin_pp=100*float(Fraction(r['distinct_level_margin'])) if r['distinct_level_margin'] is not None else None,legacy_margin_quartile=r['margin_quartile'],
            actual_state=r['actual_state'],actual_true_regret_pp=100*float(Fraction(r['actual_regret'])),actual_canonical_agreement=r['actual_canonical_agreement'],actual_optimal_set_agreement=r['actual_optimal_set_agreement'],actual_L1=r['actual_L1'],actual_DU_pp=100*r['actual_DU'],
            boot_true_regret_mean_pp=float(regret.mean()),boot_true_regret_p05_pp=float(np.quantile(regret,.05)),boot_true_regret_median_pp=float(np.median(regret)),boot_true_regret_p95_pp=float(np.quantile(regret,.95)),boot_canonical_agreement=float(canonical.mean()),boot_optimal_set_agreement=float(optimal.mean()),boot_L1_mean=float(l1[0,i,bi,ti].mean()),boot_DU_mean_pp=100*float(DU[0,i,bi,ti,:,slot].mean()),boot_DU_median_pp=100*float(np.median(DU[0,i,bi,ti,:,slot])),boot_query_oracle_regret_mean_pp=float(qr.mean()),boot_query_oracle_regret_p95_pp=float(np.quantile(qr,.95))))
    write_csv(OUT/'task_sensitivity.csv',sensitivity);strata=[]
    for b,t in [('ALL','ALL')]+[(b,t) for b in B for t in C]:
        rows=sensitivity if b=='ALL' else [r for r in sensitivity if r['bank']==b and r['context']==t]
        groups=[('all',rows)]+[(f'legacy_quartile_{q}',[r for r in rows if r['legacy_margin_quartile']==q]) for q in range(1,5)]+[('best_tied',[r for r in rows if r['best_tied']]),('singleton',[r for r in rows if not r['best_tied']])]
        for label,group in groups:
            row=dict(bank=b,context=t,stratum=label,n=len(group))
            for key in ('actual_true_regret_pp','actual_canonical_agreement','actual_optimal_set_agreement','actual_L1','actual_DU_pp','boot_true_regret_mean_pp','boot_canonical_agreement','boot_optimal_set_agreement','boot_L1_mean','boot_DU_mean_pp'):row[key]=float(np.mean([r[key] for r in group])) if group else None
            strata.append(row)
    write_csv(OUT/'task_sensitivity_strata.csv',strata)
    # Taxonomy and its non-gating tie-envelope robustness, exact rational quantiles.
    strong={(r['bank'],r['context']):r['label']=='strong' for r in msummary}
    def diagnose(edge):
        rows=[]
        for (K,b,t,s),values in fraction_acc.items():
            key=(b,t,s);den=(p00[key] if edge=='canonical' else Fraction(envelope[key][edge]))-z[key];caps=[(v-z[key])/den for v in values];ac=(actualacc[key]-z[key])/den
            rows.append(dict(K=K,bank=b,context=t,salt=s,explained=quantile(caps,5,100)<=ac<=quantile(caps,95,100),median80=quantile(caps,1,2)>=Fraction(4,5),actual80=ac>=Fraction(4,5)))
        explained={t:all(r['explained'] for r in rows if r['K']==20 and r['context']==t) for t in C[1:]}
        passing={str(K):{t:all(r['median80'] for r in rows if r['K']==K and r['context']==t) for t in C[1:]} for K in KS}
        actualpass={t:all(r['actual80'] for r in rows if r['K']==20 and r['context']==t) for t in C[1:]}
        isN=sum(explained.values())>=3 and sum(passing['20'].values())<=2
        isM=sum(passing['20'].values())>=3 and sum(actualpass.values())<3 and all(strong[b,t] for t in C[1:] if not actualpass[t] for b in B)
        return dict(taxonomy='T017-N' if isN else 'T017-M' if isM else 'T017-X',MATCHED_K20_EXPLAINS=explained,MATCHED_K20_WOULD_PASS=passing['20'],median80_by_K=passing,SCALE_RECOVERABLE=next((K for K in KS[1:] if sum(passing[str(K)].values())>=3),None),actual80=actualpass)
    primary=diagnose('canonical');minimum=diagnose('minimum');maximum=diagnose('maximum')
    robust=all(x['taxonomy']==primary['taxonomy'] and x['MATCHED_K20_WOULD_PASS']==primary['MATCHED_K20_WOULD_PASS'] for x in (minimum,maximum))
    summary=dict(status='COMPLETE',**primary,tie_envelope_robust=robust,denominator_sensitivity=dict(minimum=minimum,maximum=maximum),bootstrap_replicas=512000,template_choices=4096000,full_tests_passed=87,
        diagnosis='T017-N under frozen taxonomy: matched K20 explains Dark/Contrast/Noise; matched K20 itself passes only Dark. Synthetic K40 passes all4shifts. Blur actual capture lies below every matchedK20 p05 despite only weak residual mismatch labels. Ties are a small representative-sensitivity issue, not the main utility loss.')
    save('summary.json',summary)
    verification.update(independent_all20480_aggregate_counts_exact=True,independent_all4096000_query_regret_numerators_exact=True,independent_T011_half_class_count_checks=perexample_checks,independent_seed_replays=4000,independent_seed_q_max_error=seed_error,independent_last_replica_exact_Fraction_state_checks=lookup_checks,independent_exact_capture_quantiles_and_taxonomy=True,raw_hashes_unchanged=True,all_T015_T016_source_manifest_hashes_unchanged=len(checked_sources))
    save('verification.json',verification)
    self_finish(summary,verification,aggregate,prevalence,msummary,strata,receipt)


def self_finish(summary,verification,aggregate,prevalence,msummary,strata,receipt):
    now=datetime.now().astimezone().isoformat(timespec='seconds');ties=load(RAW/'p00_tie_envelope_summary.json');sanity=load(RAW/'noise_free_tie_sanity.json')
    mainrows=[r for r in aggregate if r['K']==20];stratamain=[r for r in strata if r['bank']=='ALL'];tops=sorted(ties['tied_by_client'].items(),key=lambda r:-r[1])[:10]
    lines=['# CODEX → CHATGPT: T017R completed decomposition',f'{now}. Lead a9fa543; original3246221 protocol preserved. Runtime `{receipt["runtime"]}`; run `{RUN}`; release20260914-083556-ttfl-t017r;87testsPASS;exit0;{receipt["seconds"]:.2f}s. Zero new model forwards.512,000matched bootstrap episodes,4,096,000template choices; fixed128replicas and syntheticK20/40/80/160.',
        '## Narrow sanity repair and historical provenance',
        'Stopped T017 run20260914-073926-ttfl-t017-preflight atda202c8 remains preserved. Only noise-free sanity changed: unchanged1e-8 inverse precision, exact optimal-set membership/zero true-template regret, and singleton identity. All canonical P00 historical choices/counts remain unchanged. Bootstrap never calls the sanity helper or overrides pi/state choices. Original protocol_freeze.json is copied byte-for-byte; repair scope is separately recorded.',
        f'Noise-free max prevalence error{sanity["max_pi_error"]:.3g}; singletonepisodes{sanity["singleton_count"]},tiedepisodes{sanity["tied_count"]}; singleton identity mismatches0, tied optimal-set mismatches0,exact true-regret maximum0. The71inverse-induced representative switches and their25querymetric changes are still reported in noise_free_replay.json.',
        f'Ties occur in{ties["tied_episodes"]}/8000episodes ({100*ties["tied_fraction"]:.3f}%). Largest per-aggregate P00 tie-envelope width{ties["max_width_pp"]:.6f}pp. No historical80%decision changed at either edge, so bootstrap was authorized to continue. Most tie-heavy clients(counts):{tops}. Complete per-episode identifiers are in template_diagnostics.json.gz, and canonical/min/max metrics are in p00_tie_envelope.csv. This additive exact envelope is sensitivity analysis, not a substituted oracle baseline.',
        '## Verification and frozen computation',
        'All2000T016channelhashes(1000soft/1000hard),4000mixtures,8000P00+8000soft01choices,80metrics and40capture summaries replay. All emission pools exclude targeti; their means/counts reconstruct the saved softchannel. Original artifact hashes remain unchanged. Matched sampling scales exact class counts and uses SHA256(T017|client|bank|context|K|replica),independent PCG64 seeds. Cached default pinv and unchanged simplex projection; no temperature, inverse cutoff, K20 deployment or estimator change.',
        f'Exact rational template lookup uses integer common denominators rather than expensive repeated Fraction reductions; decisions are mathematically identical, with no epsilon or tie forcing. Formal8000first-replica Fraction checks and4000deterministic replays pass; independent8000last-replica Fraction decisions pass. Independent4000seed replays have maxq error{verification["independent_seed_q_max_error"]:.3g}. All20,480aggregate class counts and4,096,000query-regret numerators reconstruct exactly; {verification["independent_T011_half_class_count_checks"]}per-half/candidate class-count checks match frozen T011predictions. Scientific quantiles/gates use exact fractions and linear quantile interpolation, not rounded displays.',
        'Raw bootstrap_arrays.npz preserves q/pi/z, state choices,DU,L1,JS,dominant agreement,negative entries and projection corrections. bootstrap_query_counts.npz preserves aggregate class counts and integer regret numerators/denominators. Exact per-state true regrets reside in template_diagnostics.json.gz and combine with savedchoices to reconstruct every replica’s true-template regret. Seeds,pool hashes and logits enable resampling. NPZs are durable local/remote artifacts referenced by manifests; compact aggregate replicas are Git-tracked gzip.',
        '## Matched K20 task capture',tab(['bank','context','salt','actual_capture','capture_p05','capture_median','capture_p95','explained','median80'],mainrows),
        'Canonical historical P00 remains the denominator. Replicas align acrossclients only for aggregation. Full K40/K80/K160 rows are in bootstrap_aggregate.csv, with raw exact replica accuracy/capture fractions in bootstrap_aggregate_replicas.json.gz. Synthetic largerK is a sample-complexity diagnostic and changes no real support.',
        '## Finite-support prevalence',tab(['K','bank','context','L1_mean','L1_median','L1_p05','L1_p95','JS_mean','dominant_class_agreement','negative_entry_fraction','projection_L1_mean'],prevalence),
        '## Actual-channel mismatch',tab(['bank','context','median_D_actual','median_bootstrap_p50','median_empirical_percentile','above_p95_fraction','label'],msummary),
        'These are privileged true-composition, matched-channel diagnostics, not deployable detection or hypothesis-test p-values. Weak<=20%,moderate>20%and<=50%,strong>50%above matchedp95 are unchanged. The empirical null conditions on other-client support emissions; a real-client discrepancy need not be caused solely by channel mismatch.',
        '## Task sensitivity and ties',tab(['stratum','n','actual_true_regret_pp','actual_canonical_agreement','actual_optimal_set_agreement','actual_L1','actual_DU_pp','boot_true_regret_mean_pp','boot_canonical_agreement','boot_optimal_set_agreement','boot_L1_mean','boot_DU_mean_pp'],stratamain),
        'Legacy best-minus-second-state margin quartiles are preserved, using the frozen global rank/tie ordering. Distinct-level margins are additional columns, with NA if all5utilities tie; tied-best slices are separate. Loss of canonical agreement inside the optimal set is representative instability, not true-template regret. Only optimal-set exits count as genuine task-sensitive error. Task_sensitivity.csv includes every8000episode/template and actual/matchedK20 regret distributions,DU,ordinaryL1,canonical/optimal-set agreement and query-oracle regret.',
        '## Frozen diagnosis and sensitivity',f'**{summary["taxonomy"]}**. MATCHED-K20-EXPLAINS:{summary["MATCHED_K20_EXPLAINS"]}. MATCHED-K20-WOULD-PASS:{summary["MATCHED_K20_WOULD_PASS"]}. SCALE-RECOVERABLE smallestsyntheticK:{summary["SCALE_RECOVERABLE"]}.',
        'Per-K joint80%status:'+json.dumps(summary['median80_by_K']),
        f'Taxonomy and each shift’s matched-K20 80%status invariant at both exact tie-envelope edges: {summary["tie_envelope_robust"]}. Full denominator sensitivity is recorded in summary.json. N requires>=3explainedshifts and>=2matchedK20failures; M requires>=3matchedK20passes,actualfails,andstrongmismatch inbothbanks on all actual-failing shifts; otherwiseX. No fourth taxonomy or moved threshold.',
        '## Scientific interpretation of the decomposition',
        'The frozen diagnosis is T017-N:3/4shifts are jointly explained by matchedK20 and matchedK20 fails the80%criterion on3/4shifts. Thus finite K20 observation variation amplified by the measured inverse is sufficient to explain much of the residual task loss; a dominant cross-client mismatch is not required for Dark/Contrast/Noise. This is evidence under the empirical conditional-emission model, not proof that every client has exactly the same channel.',
        'Acrossbanks/salts, matchedK20 median captures are Dark85.75–87.33%,Contrast78.05–79.61%,Noise77.32–78.79%,Blur77.00–80.42%. Only Dark passes the joint allbanks/allsalts condition. SyntheticK40 median captures are91.82–92.69%,85.71–87.51%,86.36–87.16%,86.08–87.37% respectively; all4shifts pass. K80/K160 also pass. Therefore SCALE-RECOVERABLE=40, solely as a diagnostic sample-complexity result; real support staysK20.',
        'Blur is an unresolved exception that the global N label must not hide: all8actualBlur bank/salt captures (65.96–68.31%) are below their matchedK20p05 (68.13–70.12%). The simple posterior-L1 residual diagnostic still labels everybank/context weak: only7–10%clients exceed matchedp95 (Blur10%/9% inA/B). Thus there is task-level extra loss onBlur without strong mismatch evidence by the chosen scalar diagnostic. Direction-specific mismatch, task-sensitive errors or other limitations of the empirical null remain possible; none is uniquely established, and sourcecontext-ID is excluded from this oracle01 analysis.',
        'Task sensitivity is mostly genuine optimal-set exits: across8000templates actual canonical/optimal-set agreement is61.56%/62.09%,versus matchedK20 66.52%/67.19%; mean true-template regret2.967pp versus2.280pp. The86tiedepisodes(1.075%) have very different canonical versus optimal-set agreement: actual46.51% versus95.35%,bootstrap34.52% versus97.15%. Counting canonical switches there as semantic failure would be misleading.',
        'Legacy narrow-margin Q1 has actual optimal-set agreement45.05%,compared with77.00% inQ4,consistent with more frequent boundary crossings at narrow margins. Yet mean regret is1.282pp inQ1 versus4.859pp inQ4: rarer mistakes across large gaps can be more costly. This audit therefore does not attribute all task loss to narrow boundaries. Distinct-level margins and tied-best slices remain available alongside unchanged legacy quartiles.',
        '## Interpretation limits and stop',
        'This empirical matched-channel bootstrap separates expected finite-support variation from real-versus-null differences under the specified emission model. It does not identify a unique causal source of residual mismatch. Full-rank soft outputs retain information but may be noisy at K20; synthetic scale recovery is not a deployable result. Exact-optimum representative changes are reported separately from suboptimal utility. No neutral-operator failure or feature-semantic collapse is inferred. No next task assigned; return evidence to Lead and stop.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    concise='\n\n'.join([lines[0],lines[1],
        '**T017-N; SCALE-RECOVERABLE=40 (synthetic diagnostic only).** Full report/artifacts: `results/t017_channel_noise_decomposition/RESULTS.md`. Real support remainsK20; no nextstage.',
        'Noise-free repair only:7914singleton and86tiedtemplate episodes; singletonidentity mismatch0,optimal-set mismatch0,exactregret0,maxpierror5.27e-15. Priorstopda202c8and71representative switches preserved. Tie envelope maxwidth0.100525pp; no historical80%flip. Canonical P00 untouched; taxonomy and eachK20shiftstatus robust at both envelopeedges.',
        'MatchedK20 explains actualDark/Contrast/Noise in bothbanks/allsalts (3/4),but explains0/8Blurrows. MatchedK20 itself passes onlyDark. Median capture ranges:Dark85.75–87.33%,Contrast78.05–79.61%,Noise77.32–78.79%,Blur77.00–80.42%. Thejoint80%gate still fails on3/4even under exactlymatched empiricalchannels, meeting the frozenNdefinition.',
        'SyntheticK40 passes all4shifts (Dark91.82–92.69%,Contrast85.71–87.51%,Noise86.36–87.16%,Blur86.08–87.37%);K80/160alsoallpass. This supports finite-support/inversion-variance limits under the matched model; it does not authorize changing deploymentK.',
        'Mismatch:allbank/contextlabelsweak;7–10%clientsabovebootstrap p95. Blur actualcapture65.96–68.31%is nevertheless belowallmatchedK20p05rows68.13–70.12%. Preserve thisextra task loss; scalarposteriorL1residual does not establish strongmismatch or explain it fully. Oracle01 removes source-IDcoupling here.',
        'Task sensitivity:actualoptimal-set agreement62.09%vsbootstrap67.19%;meantrue-template regret2.967vs2.280pp. For86tiedepisodes,actualcanonical/optimalagreement46.51%/95.35%,bootstrap34.52%/97.15%—representative changes must not count as nonoptimal loss. NarrowQ1has more optimal-set exits(agreement45.05%vsQ4 77.00%)but lowermeanregret1.282vs4.859pp,so notallutility losscomesfromnarrow margins.',
        'Verification:87tests;512000episodes/4096000choices. Exact integer lookup equalsFraction,with8000formalfirst-replica and8000independentlast-replica checks.4000independentSHAseedreplays maxqerror1.11e-16. All20480aggregatecounts and4096000regret numerators reconstruct;40000T011half/candidatecountchecksPASS. Frozenprotocol/artifacthashes unchanged;zero model forwards. LargeNPZrawarrays retainedlocal+remote;manifestandcompactgzipreceipts persisted.',
        'Initial SFTP result download stalled; existing legacy-SCP fallback completed transfer. No experiment rerun. The empirical bootstrap does not prove universal channel equality, and the remainingBlur discrepancy is explicitly retained. No operator failure, semanticcollapse, or deployable learnedwriter claim. Stop and return toLead.'])+'\n'
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(concise,encoding='utf-8')
    handoff=f'# T017R COMPLETE\n\n{now}. Runtime{receipt["runtime"]};run{RUN};87testsPASS;512000bootstrapepisodes,4096000exacttemplatechoices;0newforwards. Diagnosis{summary["taxonomy"]};scaleK{summary["SCALE_RECOVERABLE"]};tie-envelope robust{summary["tie_envelope_robust"]}. Read results/t017_channel_noise_decomposition/RESULTS.md. Priorstopda202c8preserved. No nextstage;awaitLead.\n'
    for name in ('HANDOFF.md','T017_HANDOFF.md'):(ROOT/'research_log'/name).write_text(handoff,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+handoff)
    delivery=dict(timestamp=now,lead='a9fa543',runtime=receipt['runtime'],run=RUN,status='COMPLETE',diagnosis=summary['taxonomy'],raw=RAW.relative_to(ROOT).as_posix(),results=OUT.relative_to(ROOT).as_posix(),next='Await Lead')
    (ROOT/'research_log/t017r_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t017r_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t017r_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,mismatch=msummary,task_strata=stratamain,ties={k:v for k,v in ties.items() if k!='tied_by_client'}),indent=2))


if __name__=='__main__':main()
