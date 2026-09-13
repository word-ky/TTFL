"""Independent diagnosis of the T017 noise-free tie stop; no bootstrap execution."""
import gzip,hashlib,json,shutil
from datetime import datetime
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='20260914-073926-ttfl-t017-preflight'
RAW=ROOT/f'research_log/t017_receipts/{RUN}/artifacts/t017_channel_noise_decomposition';OUT=ROOT/'results/t017_channel_noise_decomposition'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    noise=load(RAW/'noise_free_replay.json');emission=load(RAW/'matched_emission_receipt.json');meta=load(RAW/'metadata.json');history=load(RAW/'historical_replay.json')
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):r['utility'] for r in gzload(ROOT/'results/t014_class_conditional_factorization/class_templates.json.gz')['rows']}
    truth={r['client']:r for r in load(ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics/support_truth.json')}
    rows=[]
    for r in gzload(RAW/'noise_free_choice_mismatches.json.gz'):
        template=templates[r['salt'],r['bank'],r['train_half'],r['client']][C.index(r['context'])]
        pi=[Fraction(n,20) for n in truth[r['client']]['counts']]
        values=[sum((pi[k]*Fraction(template[k][s]) for k in range(10)),Fraction()) for s in range(5)]
        arg=[C[s] for s,v in enumerate(values) if v==max(values)];actual=r['noise_free_state'];perturbed=[Fraction(v) for v in r['utilities']]
        assert r['P00_state']==arg[0]
        rows.append(dict(client=r['client'],bank=r['bank'],context=r['context'],salt=r['salt'],train_half=r['train_half'],P00_state=r['P00_state'],noise_free_state=actual,
            true_argmax=arg,noise_free_in_true_argmax=actual in arg,true_utility_regret=str(max(values)-values[C.index(actual)]),perturbed_advantage=float(perturbed[C.index(actual)]-perturbed[C.index(r['P00_state'])]),true_utilities=[str(v) for v in values]))
    assert len(rows)==noise['choice_mismatches'] and all(r['noise_free_in_true_argmax'] and Fraction(r['true_utility_regret'])==0 for r in rows)
    log=(RAW.parent.parent/'train.log').read_text();assert 'Ran 80 tests' in log and 'exit_code=1' in log
    lo=min(r['macroclass_delta_pp'] for r in noise['metric_mismatches']);hi=max(r['macroclass_delta_pp'] for r in noise['metric_mismatches'])
    save(OUT/'tie_diagnosis.json',rows)
    summary=dict(status='STOP_NOISE_FREE_P00_INVARIANT',scientific_taxonomy='NOT_EXECUTED',MATCHED_K20_EXPLAINS='NOT_EXECUTED',MATCHED_K20_WOULD_PASS='NOT_EXECUTED',SCALE_RECOVERABLE='NOT_EXECUTED',
        full_tests_passed=80,historical_replay_pass=True,emission_mean_max_error=emission['max_column_mean_error'],noise_free_pi_max_error=noise['max_pi_error'],
        total_choices=8000,changed_choices=len(rows),all_changed_choices_in_exact_true_argmax=True,all_changed_choices_true_template_regret_zero=True,unique_affected_clients=sorted({r['client'] for r in rows}),
        changed_macroclass_rows=len(noise['metric_mismatches']),macroclass_delta_range_pp=[lo,hi],max_perturbed_utility_advantage=max(r['perturbed_advantage'] for r in rows),bootstrap_replicas_executed=0,new_model_forwards=0,
        diagnosis='Float64 inverse recovers true prevalence to numerical precision; tiny residual masses split exact P00 utility ties. Selected-state identity/query-metric equality fails despite zero true-template regret. No sampling/mismatch conclusion.')
    save(OUT/'summary.json',summary)
    verification=load(RAW/'verification.json');verification['independent_exact_tie_membership_checked']=True;verification['independent_exact_true_utility_regret_zero']=True;save(OUT/'verification.json',verification)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    report=f'''# CODEX → CHATGPT: T017 noise-free preflight stop

{now}. Lead `3246221`; runtime `{meta['runtime']}`; release `20260914-073913-ttfl-t017-preflight`; run `{RUN}`. One preflight execution, exit1 after {meta['seconds']:.2f}s. **80 tests PASS. No bootstrap or new model inference.**

## Outcome

T017 stopped at its mandatory noise-free/P00 invariant. The inverse recovers prevalence to float64 precision, but exact selected-state identity and query-count identity do not hold on true utility ties. **T017-N/M/X, MATCHED-K20-EXPLAINS, MATCHED-K20-WOULD-PASS and SCALE-RECOVERABLE are NOT EXECUTED**, not scientific FAIL.

## Checks that passed

- Protocol/128 replica seeds/K20,40,80,160 definitions were frozen before new diagnostics. No real support or estimator change.
- All 2,000 saved T016 channel hashes reproduce (1,000 soft plus 1,000 hard; T016 did not contain 2,000 distinct soft channels).
- All 4,000 corrected mixtures reproduce; maximum prevalence replay error `{history['all4000_mixtures_max_error']}` within the existing numerical tolerance.
- All 8,000 BBSE-S-01 choices and all 8,000 P00 choices, including exact utility vectors and argmax sets, reproduce. Their 80 metric rows reconstruct from exact class counts; all 40 soft01 capture summaries reproduce.
- Support remains disjoint from calibration pools and query IDs. Each empirical soft-emission pool excludes target i before loading/summing. Its class counts equal T016 counts. All 1,000 pool means reproduce channel columns, max absolute error `{emission['max_column_mean_error']}`.
- Checkpoint, states, logits, predictions, templates and original T015/T016 inputs remain unchanged.

## Noise-free result and exact tie diagnosis

For all 1,000 episodes, `q_star=C@pi_true` and the unchanged `ProjectSimplex(pinv(C)@q_star)` give max absolute prevalence error **{noise['max_pi_error']:.17g}**. Thus the numerical inverse sanity itself passes the existing 1e-8 precision check.

However, **{len(rows)}/8,000** selected-state identities differ from deterministic first-argmax P00. They affect clients **{summary['unique_affected_clients']}**. Independent exact Fraction reconstruction shows:

- Every changed state is in the **exact true-P00 argmax set**.
- Every changed state has **exactly zero true-template utility regret**.
- Tiny inverse residual masses split those ties; maximum perturbed advantage over the canonical P00 state is **{summary['max_perturbed_utility_advantage']:.17g}** in utility proportions.

For example client15 has true support prevalence entirely in class4. An inverse residual of order1e-15 assigns tiny masses to originally absent classes. Several states have exactly equal utility on class4; the tiny other-class masses break that equality. P00 chooses the first exact argmax, while the perturbed prevalence chooses another formerly tied state. No regularization or alternative tie policy was applied.

Because the offline other-client template is not the target-query outcome, equal true-template utility does not require identical held-out query predictions. **{len(noise['metric_mismatches'])}/40** macro-class metric rows change, ranging **{lo:.9f} to +{hi:.9f} pp**. These changes are real integer-count differences; they must not be silently treated as exact historical reproduction or hidden behind a metric tolerance.

`noise_free_inverse.csv` retains all episode errors. `noise_free_choice_mismatches.json.gz` retains perturbed prevalence/utility vectors and selected identities. `tie_diagnosis.json` independently records exact true utilities, argmax sets and zero regret for all71changes. `noise_free_replay.json` preserves all25changed count rows. No diagonal-affine capacity or channel-mismatch conclusion follows from this tie discontinuity.

## Why execution stopped

Lead section4 requires “exact agreement with P00 true-composition state choices and metrics,” and section2 says to stop if a decision-bearing replay fails. Those two equality conditions fail despite a correct-to-precision inverse and unchanged historical artifacts. Therefore no matched-channel bootstrap, actual-vs-bootstrap mismatch diagnostic, task-margin stratification or synthetic sample-size conclusion was run. Empty/fabricated bootstrap CSVs are not provided.

The issue is the discontinuity of exact argmax at a true tie, not evidence that a full-rank channel cannot invert its own expectation. This package cannot legitimately force the prescribed N/M/X taxonomy from a stopped sanity check.

## Narrow decision for Lead

The recorded evidence supports distinguishing noise-free numerical inverse recovery and membership in the true optimal-state set from equality to one canonical tied representative. If Lead authorizes a tie-aware noise-free sanity rule, it can preserve the analytic P00 baseline separately while retaining the raw inverse-induced choices/count deltas. Such a change must be explicit; no comparator/metric tolerance, prevalence snapping, tie rule, inverse setting or scientific criterion was changed in this run. Bootstrap sampling/helper tests are implemented, but the full bootstrap execution remains pending this mandatory sanity condition.

No next scientific task is assigned. Original T016R evidence and all frozen artifacts are preserved; return this implementation/numerical-check blocker to Lead.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    concise=f'''# CODEX → CHATGPT: T017 STOPPED at noise-free exact-P00 check

{now}. Lead3246221; runtime{meta['runtime']}; run{RUN};80testsPASS;exit1;no newforwards;bootstrap0. Report: `results/t017_channel_noise_decomposition/RESULTS.md`.

Historical replay PASS:2000channelhashes(1000soft+1000hard),4000mixtures,8000P00+8000soft01choices,80countmetrics and40capture summaries. All target pools exclude their client; softpool means match T016max{emission['max_column_mean_error']:.3g}. Frozen hashes/support disjointness unchanged.

Noise-free inverse maxpierror{noise['max_pi_error']:.3g}; precision passes. But71/8000state identities differ. Exact independent audit proves ALL71remain in trueP00argmax and ALLhave zero true-template regret. Tiny residual masses split true utility ties (maxperturbedadvantage{summary['max_perturbed_utility_advantage']:.3g}). Affectedclients15,21,38,40,81,93.

Different tied states alter25/40querymacroclass rows by {lo:.6f} to+{hi:.6f}pp. These are actual count differences, not serialized float discrepancies. Lead demanded exactP00choices/metrics, so stopbeforebootstrap; no N/M/X or SCALE-RECOVERABLE conclusion. Full mismatch vectors, exacttie audit and countdeltas retained.

Cause:argmax discontinuity at exactties despite accurate inverse; not channel/operator failure. Await explicit Lead treatment of the noise-free tie invariant. No snapping, newtolerance, inverse change, or automaticnextstage.
'''
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(concise,encoding='utf-8')
    handoff=f'# T017 STOP — NOISE-FREE EXACT-P00 TIES\n\n{now}. Lead3246221;runtime{meta["runtime"]};run{RUN};80testsPASS;historical replayPASS; inverseerror5.27e-15 but71/8000state changes all exactP00ties,zero trueutilityregret.25/40querycountrows change−.067to+.020pp. MandatoryexactP00invariantfails;bootstrap0. Read results/t017_channel_noise_decomposition/RESULTS.md. AwaitLead; no automaticretry or scientifictaxonomy.\n'
    for name in ('HANDOFF.md','T017_HANDOFF.md'):(ROOT/'research_log'/name).write_text(handoff,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+handoff)
    save(ROOT/'research_log/t017_delivery.json',dict(timestamp=now,lead='3246221',runtime=meta['runtime'],run=RUN,status=summary['status'],raw=RAW.relative_to(ROOT).as_posix(),results=OUT.relative_to(ROOT).as_posix(),next='Await explicit Lead treatment of noise-free ties; no bootstrap'))
    save(ROOT/'research_log/t017_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t017_receipts').rglob('*')) if p.is_file()])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
