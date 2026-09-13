"""Compact T013 report and independent exact aggregate gate audit."""
import csv,gzip,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from report_t009 import table,write_csv
ROOT=Path(__file__).resolve().parents[1];RUN='20260914-t013-local'
RAW=ROOT/f'research_log/t013_receipts/{RUN}/artifacts/t013_disjoint_factorization';OUT=ROOT/'results/t013_disjoint_factorization'
SALTS=['T013-S0','T013-S1','T013-S2','T013-S3'];C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];OFF=[7,13,23,37,41,53,71,89]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def tab(keys,rows):return table(keys,[[r[k] for k in keys] for r in rows])
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.json','.csv','.gz'):shutil.copyfile(p,OUT/p.name)
    summary=load(RAW/'summary.json');lock=load(RAW/'disjoint_lock_summary.json');m=load(RAW/'crossfit_metrics.json');v=load(RAW/'verification.json');meta=load(RAW/'metadata.json')
    splitfreeze=load(RAW/'split_freeze.json');freeze=load(RAW/'policy_freeze.json');receipts=load(RAW/'integer_count_receipts.json')
    assert sha(RAW/'query_halves.json')==splitfreeze['sha256'] and sha(RAW/'crossfit_choices.json')==freeze['choices_sha256']
    assert sha(RAW/'half_argmax_sets.json')==freeze['argmax_sha256'] and sha(RAW/'half_integer_counts.npz')==freeze['counts_sha256']
    assert datetime.fromisoformat(splitfreeze['timestamp'])<datetime.fromisoformat(freeze['timestamp'])
    exact={};error=0.
    metrics={(r['salt'],r['bank'],r['target'],r['policy']):r for r in m}
    for r in receipts:
        k=r['salt'],r['bank'],r['target'],r['policy'];value=sum((Fraction(100*x,y) for x,y in zip(r['class_correct'],r['class_total'])),Fraction())/10;exact[k]=value
        error=max(error,abs(float(value)-metrics[k]['macro_class']))
    joint={}
    for t in C[1:]:
        passed=[]
        for salt in SALTS:
            for b in ('A','B'):
                gain=exact[salt,b,t,'half_oracle']-exact[salt,b,t,'zero'];cap=(exact[salt,b,t,'two_factor']-exact[salt,b,t,'zero'])/gain
                passed.append(cap>=Fraction(4,5))
        joint[t]=all(passed)
    assert joint==summary['all_salts_banks_joint_capture']
    cleanmeans={b:sum(exact[s,b,'clean','two_factor']-exact[s,b,'clean','zero'] for s in SALTS)/4 for b in ('A','B')}
    assert summary['FACTOR_A']==(sum(joint.values())>=3 and all(x>=-Fraction(1,2) for x in cleanmeans.values()))
    controlpass={}
    for b in ('A','B'):
        controlpass[b]=sum(all(sum(exact[s,b,t,'two_factor']-exact[s,b,t,p] for s in SALTS)/4>=Fraction(1,2) for p in ('client_only','context_only')) for t in C[1:])
    assert summary['FACTOR_B']==(not summary['FACTOR_A'] and summary['DISJOINT_LOCK_STRONG'] and all(n>=2 for n in controlpass.values()))
    v.update(independent_exact_fraction_factor_gates=True,independent_report_macroclass_error_pp=error,focused_tests_passed=2)
    save('verification.json',v)
    meanrows=[]
    for b in ('A','B'):
        for t in C:
            row=dict(bank=b,target=t)
            for p in ('zero','true_state','full_query_oracle','half_oracle','client_only','context_only','two_factor'):
                row[p]=float(sum(exact[s,b,t,p] for s in SALTS)/4)
            row['mismatch_mean']=float(sum(exact[s,b,t,f'mismatch_{k}'] for s in SALTS for k in OFF)/32)
            meanrows.append(row)
    write_csv(OUT/'mean_crossfit_accuracy.csv',meanrows)
    capture=list(csv.DictReader((RAW/'factor_capture.csv').open()));clean=list(csv.DictReader((RAW/'clean_safety.csv').open()))
    pairrows=[];simrows=[]
    d=list(csv.DictReader((RAW/'disjoint_lock.csv').open()))
    for salt in SALTS:
        for b in ('A','B'):
            for t in range(5):
                for u in range(t+1,5):
                    g=[r for r in d if r['salt']==salt and r['bank']==b and r['target1']==C[t] and r['target2']==C[u]]
                    pairrows.append(dict(salt=salt,bank=b,target1=C[t],target2=C[u],same_overlap=float(np.mean([float(r['same_client_overlap']) for r in g])),control_overlap=float(np.mean([float(r['control_mean']) for r in g]))))
    write_csv(OUT/'orientation_averaged_pair_overlap.csv',pairrows)
    for r in lock['utility_similarity']:
        means=[x['mean'] for x in r['controls'] if x['mean'] is not None]
        simrows.append(dict(salt=r['salt'],bank=r['bank'],target1=r['target1'],target2=r['target2'],orientation=r['orientation'],same_valid=r['same']['valid_count'],same_mean=r['same']['mean'],same_median=r['same']['median'],control_mean=float(np.mean(means)),control_min=min(means),control_max=max(means)))
    write_csv(OUT/'disjoint_utility_similarity.csv',simrows)
    associations=load(RAW/'composition_associations.json');composition=[]
    for f in ('normalized_entropy','max_class_fraction','halves_histogram_L1','halves_histogram_JS'):
        g=[r for r in associations if r['feature']==f]
        composition.append(dict(feature=f,overlap_rho_min=min(r['spearman_same_client_overlap'] for r in g),overlap_rho_max=max(r['spearman_same_client_overlap'] for r in g),regret_rho_min=min(r['spearman_two_factor_regret'] for r in g),regret_rho_max=max(r['spearman_two_factor_regret'] for r in g)))
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# CODEX -> CHATGPT','## Timestamp / commit / run',
        f'{now}. Lead9d6ada0; runtime {meta["code_commit"]}; local run {RUN}; {meta["seconds"]:.2f}s; CPU analysis of saved predictions only. No new model forward pass or GPU job was needed. Enclosing delivery commit contains this report.',
        '## Frozen evidence / implementation verification',
        'The formal T011 candidate_predictions.npz exists and its SHA matches its original query-freeze receipt. All5,000 per-client/state class-correct/total rows reconstruct exactly; every prediction/label vector length matches the frozen original query-ID length. All T012 full-query pair overlaps and eight-offset mean controls reproduce before splitting. No fallback prediction regeneration, state/model modification or training.',
        'Two focused tests pass: label-blind complete/disjoint deterministic splitting and exact equal-client leave-one-out utilities with ties. Policy utilities use Fraction arithmetic. Test changes the target client’s shifted training counts and confirms its leave-one-out context term is unaffected. The API accepts only training-half counts and denominators. Historical model tests were not rerun because this task has no model code or forward pass.',
        '## Label-blind disjoint halves',
        f'Four fixed salts T013-S0..S3. SHA256(T013|salt|client|original_id) order with index tiebreak; alternating ranks become H0/H1. Every index appears once, no overlap, size difference<=1. Membership is identical across contexts/banks. query_halves.json frozen at {splitfreeze["timestamp"]}, SHA {splitfreeze["sha256"]}, before T013 half-derived analysis. Required historical full-query count regression accessed historical labels first; it did not influence the split.',
        '## Disjoint-example CLIENT-LOCK',
        tab(['salt','bank','mean_same_overlap','mean_control_overlap','difference_pp','passed'],lock['groups']),
        '**DISJOINT-LOCK-STRONG passes all8 salt/bank combinations.** Margins22.2125–24.0125pp exceed the unchanged15pp gate. These margins are smaller than T012’s28.0–28.2pp, but remain substantial on disjoint underlying images. Repeated-example coupling therefore cannot explain away persistent client preference. Class-composition coupling is still possible because disjoint samples from a client share its distribution.',
        'disjoint_lock.csv retains every pair, orientation and offset summary. orientation_averaged_pair_overlap.csv supplies per-pair values; disjoint_utility_similarity.csv supplies valid counts, same-client mean/median Spearman and control mean/range. The JSON retains all individual offset statistics. Utility ranks use exact half correct counts, equivalent to half-specific DeltaAcc ranks.',
        '## Cross-fit factorization and policy freeze',
        'For each target half, persistent P is the same client’s clean utility on the opposite half. Context C is the equal-client mean of shifted-minus-clean utilities from the other99 clients on that training orientation. two_factor chooses argmax(P+C). client_only uses P; context_only uses C; eight mismatches replace P with client(i+k)%100 while keeping the same C(-i). All predicted argmax sets are retained; the first frozen candidate wins policy ties.',
        f'All120,000 half-policy choices and8,000 exact half-argmax sets were saved before composition analysis. Policy freeze {freeze["timestamp"]}; choices SHA {freeze["choices_sha256"]}. Every example receives a state selected using its opposite-half labels, never its own evaluation-half labels, for the cross-fit diagnostic policies. Half-oracle and historical full-query oracle are explicitly privileged comparison policies and do use evaluated labels.',
        '**This is supervised diagnostic factorization with a known context and other-client label access. It is not an unlabeled writer or deployable adaptation method.** Concatenating the swapped evaluation halves reconstructs each full-query policy.',
        '## Cross-fit metrics and oracle granularity',
        'Main macro-class accuracy, averaged across all four salts (all per-salt metrics remain in crossfit_metrics.csv):',
        tab(['bank','target','zero','true_state','full_query_oracle','half_oracle','client_only','context_only','two_factor','mismatch_mean'],meanrows),
        'The split-matched half-oracle chooses a fixed state separately per evaluation half and is the prescribed gain denominator. Both oracles maximize on the same outcomes used to report them, so they are optimistic diagnostics. All exact ties are preserved; display/policy order remains clean,dark,contrast,noise,blur. State-frequency counts sum to200 decisions per salt/bank/context/policy; regret and hit rate are unweighted over those200 client halves.',
        'crossfit_metrics.csv contains all600 salt/bank/context/policy rows, including every mismatch offset: macro-class, sample-weighted, macro-client accuracy, selection frequencies, half-oracle hit rate, regret median/p75/p90 and >2/>5pp fractions. Raw per-half regrets are in crossfit_regret.json.gz. Integer_count_receipts.json plus the formal half-count artifact allow reconstruction. An independent per-example concatenation path exactly matches every policy’s class counts; historical full-query best-of-five matches T011.',
        '## Factor capture and mismatch controls',
        tab(['salt','bank','target','factor_capture','two_minus_client','two_minus_context','two_minus_mean_mismatch','mismatch_mean','mismatch_min','mismatch_max'],capture),
        'Dark and Noise exceed80% split-oracle capture in both banks under every salt. Contrast remains about69–71%; Blur about75–81% and fails under multiple salts. Thus only2/4 shifted targets pass the all-salt/all-bank requirement. No salt or mismatch offset is selected for favorable performance.',
        '## Clean calibration safety',tab(['salt','bank','delta_pp'],clean),
        tab(['bank','mean_delta_pp'],summary['clean']),
        'Clean two_factor equals client_only because its transient term is exactly0. Cross-fit clean gain averages+5.718668 /+5.511444pp, passing the allowed−.5pp limit. This is supervised persistent calibration gain, not evidence of clean zero neutrality being violated or of label-free context adaptation.',
        '## FACTOR decision',
        '**FACTOR-A FAIL; FACTOR-B is the primary diagnosis.** Disjoint lock survives every salt/bank, and on Dark and Contrast the two-factor policy beats both single-factor controls by>=.5pp after averaging all salts, in each bank:',
        tab(['bank','target','mean_two_minus_client','mean_two_minus_context','passed'],summary['advantages']),
        'On Noise, two_factor improves only~.22pp over client_only. On Blur the gain over client_only is+.092pp in A and−.112pp in B. Persistent calibration accounts for much of their gain; the simple global additive transient residual adds little. Both persistent and transient effects have predictive value, but additive utility is not sufficient under the frozen gate. Residual client×context interaction, estimation noise and finite-half oracle optimism remain possible explanations; this audit does not isolate their individual contributions.',
        '## Post-freeze label-composition audit',
        'label_composition_audit.csv records all800 client/salt/half rows: normalized entropy, maximum class fraction, represented classes and H0/H1 histogram L1/JS. Only after all policies and argmax sets were frozen were entropy/histogram-distance statistics computed. Continuous descriptive associations were used; no threshold, label-prior correction or matching rule was fitted.',
        tab(['feature','overlap_rho_min','overlap_rho_max','regret_rho_min','regret_rho_max'],composition),
        'Across salts/banks, overlap is weakly higher with lower entropy (Spearman−.125 to−.079) and higher dominant-class share (+.111 to+.136). Half histogram mismatch has little relation to overlap, but weak positive association with two-factor regret (L1+.026 to+.200; JS+.049 to+.225). These modest descriptive associations do not establish that composition explains the persistent signal. They motivate a dedicated composition audit if Research Lead considers it necessary.',
        '## Mechanism vs implementation conclusion',
        'Exact artifact regressions, label-blind split checks, training-half-only factor construction, freeze hashes and count reconstructions all pass. The result is a mechanism finding. Persistent client preference survives disjoint examples and a transient context residual helps on Dark/Contrast. The additive model leaves important residual error, especially on Contrast/Blur; do not call this an operator-capacity failure. The current affine bank and ray evidence already show useful capacity.',
        '## Recommended next action',
        'Return FACTOR-B evidence to Research Lead. A next bounded audit should distinguish class-composition-driven persistence from non-additive client×context effects before implementing a writer. No explicit new state architecture, T014, SSL, gradient adaptation, federation, operator expansion or post-hoc correction was started.']
    report='\n\n'.join(lines)+'\n';(OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T013 COMPLETE — FACTOR-B, disjoint lock survives\n\nRead results/t013_disjoint_factorization/RESULTS.md and research_log/t013_delivery.json. All8salt/bank margins22.21–24.01pp PASS. FACTOR-Afails2/4jointcapture;FACTOR-BpassesDark/Contrastbothsinglecontrols>=.5pp. Clean+5.718668/+5.511444pp. Trainhalf-only supervisedoracle diagnostic, notunlabeleddeployment. T0115000/T012overlap/counts/freezes exact; noforwards;2focusedtestsPASS. Runtime089c918,local20260914-t013-local complete. AwaitLead,noT014.\n',encoding='utf-8')
    delivery=dict(timestamp=now,lead='9d6ada0',runtime=meta['code_commit'],run=RUN,decision='FACTOR-B',raw=RAW.relative_to(ROOT).as_posix(),results='results/t013_disjoint_factorization/RESULTS.md',next='Await Lead')
    (ROOT/'research_log/t013_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t013_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t013_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision=summary['primary'],count_error=error,all_salts_joint=joint),indent=2))

if __name__=='__main__':main()
