"""Preserve the T016 preflight stop; does not execute or alter the estimator."""
import csv,hashlib,json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='20260914-054411-ttfl-t016-cached'
RAW=ROOT/f'research_log/t016_receipts/{RUN}'
PRE=RAW/'artifacts/t016_confusion_debiased_semantics/historical_preflight'
OLD=ROOT/'results/t015_unlabeled_semantic_mixture'
OUT=ROOT/'results/t016_confusion_debiased_semantics'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True);differences=[];comparisons=[]
    for name in ('mixture_quality.csv','mixture_quality_episodes.csv','scientific_gates.csv'):
        with (PRE/name).open() as f:a=list(csv.DictReader(f))
        with (OLD/name).open() as f:b=list(csv.DictReader(f))
        assert len(a)==len(b)
        diff=[]
        for index,(x,y) in enumerate(zip(a,b)):
            for key in x:
                if x[key]!=y[key]:
                    d=dict(file=name,row=index,client=x.get('client'),bank=x.get('bank'),target=x.get('target'),estimator=x.get('estimator'),field=key,rerun=x[key],historical=y[key],absolute_difference=abs(float(x[key])-float(y[key])))
                    diff.append(d)
        comparisons.append(dict(file=name,rows=len(a),different_fields=len(diff),max_absolute_difference=max((d['absolute_difference'] for d in diff),default=0),historical_sha256=sha(OLD/name),rerun_sha256=sha(PRE/name)))
        differences.extend(diff)
    assert len(differences)==2 and all(d['field']=='JS' and d['absolute_difference']==4.440892098500626e-16 for d in differences)
    log=(RAW/'train.log').read_text();assert 'Ran 63 tests' in log and 'exit_code=1' in log and 'T016_PHASE_A_FROZEN' not in log
    assert not (PRE.parent/'phaseA_freeze.json').exists()
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    summary=dict(status='IMPLEMENTATION_PREFLIGHT_STOP',CAL_SEM_A='NOT_EXECUTED',CAL_SRC_A='NOT_EXECUTED',primary='soft_emission_pinv (planned, not executed)',tests_passed=63,
        reason='New exact string comparison of per-episode floating JS rejected two 4.44e-16 differences. Historical aggregate quality and gate rows match exactly.',
        scientific_conclusion=None,calibration_matrices_built=0,source_choices_frozen=0,new_forwards=0,formal_attempts=1)
    verification=dict(historical_all56000_utilities_argmaxes_exactly_replayed=True,historical_all280_macroclass_metrics_integer_reconstructed=True,
        historical_freeze_mixture_logits_choice_hashes_verified=True,historical_220_quality_summary_rows_exact=True,historical_96_gate_rows_exact=True,
        historical_episode_quality_comparison=comparisons,observed_differences=differences,full_regression_tests_passed=63,
        T016_target_exclusion_unit_tests_passed=True,T016_full_data_disjointness_check='not reached after comparison stop; T015 prior receipt remains unchanged',
        T016_calibration_phase='not reached',T016_source_freeze='absent',T016_target_evaluation='not reached',no_new_model_or_state_updates=True)
    save(OUT/'summary.json',summary);save(OUT/'verification.json',verification);save(OUT/'historical_replay_differences.json',differences)
    (OUT/'scientific_gates.csv').write_text('gate,status,reason\nCAL-SEM-A,NOT_EXECUTED,preflight comparison stopped before calibration\nCAL-SRC-A,NOT_EXECUTED,preflight comparison stopped before calibration\n',encoding='utf-8')
    report=f'''# CODEX → CHATGPT: T016 preflight stop

{now}. Lead `6476c99`; runtime `6fd685e543f13303d201cdb7c9b810a395696a18`; release `20260914-054356-ttfl-t016`; run `{RUN}`. Exit 1 at historical preflight. **CAL-SEM-A / CAL-SRC-A: NOT EXECUTED.**

## What ran

Implemented the fixed target-excluded hard confusion / soft emission channel, NumPy default pseudoinverse and standard Euclidean simplex projection. All **63 tests passed**, including six new tests covering target exclusion, nonzero/normalized columns, deterministic projection, known-confusion recovery, target-label isolation, source-context indexing and raw posterior behavior.

The historical T015 replay checked saved mixture/logit/choice/freeze hashes, reproduced all **56,000** utility vectors/argmaxes, and reconstructed **280** macro-class metrics from exact integer receipts. All **220 quality-summary rows** and **96 historical gate rows** match field-for-field. T015 remains SUPPORT-COMP-A PASS and SEM-EST-A / SEM-SRC-A FAIL.

## Observed implementation blocker

The newly added comparison at `scripts/eval_t016_confusion_prevalence.py:60` compares entire parsed CSV rows as strings. In `mixture_quality_episodes.csv`, exactly **2 of 4,000 rows** differ only in JS:

| zero-based row | historical JS | replay JS | absolute difference |
|---|---|---|---|
| 3186 | 0.28094428886389666 | 0.2809442888638962 | 4.440892098500626e-16 |
| 3187 | 0.28094428886389666 | 0.2809442888638962 | 4.440892098500626e-16 |

Every other field is identical. This scale is consistent with floating-point numerical/serialization differences between the original Windows report and Linux replay; the exact runtime cause is not independently isolated. The comparison is stricter than scientific equivalence of this descriptive floating metric. No accuracy, mixture-quality aggregate, gate, state choice, checkpoint or frozen logit change was found. This is an implementation check issue introduced in T016, **not evidence against T015 or the scientific method**.

## Stop and scope

Lead section 1 says: “If any historical reconstruction/hash/exclusion check fails, stop and report implementation blocker instead of producing scientific conclusions.” The job stopped there. No calibration matrices were built, no T016 source mixtures/choices were frozen, and no T016 target evaluation ran. The later full-data disjointness check was not reached; only existing T015 receipts and focused tests support that portion so far. No new support or query inference, model updates, state fitting, estimator tuning, or outcome-based retry occurred.

The first deployment attempt had an SSH timeout before job launch; the second deployment succeeded. There was **one formal execution**, not a scientific rerun. Its full log and intermediate historical replay artifacts are retained under `research_log/t016_receipts/{RUN}`. Historical `phaseA_freeze.json` inside the `historical_preflight` subdirectory belongs to **T015** and must not be mistaken for a T016 freeze.

## Minimal repair proposed for the next bounded instruction

Keep exact comparisons for IDs, counts, hashes, utilities, argmaxes, gates and quality aggregates. For the observed per-episode floating JS comparison, use a predeclared absolute float64 tolerance (for example 1e-12) instead of decimal-string equality, and retain an explicit maximum-error receipt. Do not change the estimator, thresholds, support, candidate states or scientific gates. No repair/rerun was applied after this stop.

## Scientific status

Channel identifiability/calibration, semantic prevalence quality, context-ID coupling and fast-state utility are **not evaluated in T016**. Neither CAL gate is a scientific FAIL. No operator failure or success can be inferred. Calibration statistics, prevalence-quality outputs, policy metrics and source-choice artifacts do not exist because their stages were not reached; they are not populated with placeholders or fabricated values.

Await Lead instruction after this implementation-blocker report. Do not proceed to feature prototypes, learned heads, SSL/TTT or federation.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    handoff=f'# T016 STOPPED AT PREFLIGHT\n\n{now}. Run {RUN}, runtime6fd685e, 63 tests PASS. Historical aggregate/gate/count/choice replay exact; two episode JS strings differ by4.44e-16. Added overstrict comparison stopped before calibration. CAL-SEM-A/CAL-SRC-A NOT EXECUTED, not scientific FAIL. Read results/t016_confusion_debiased_semantics/RESULTS.md. No automatic retry; await Lead. Existing T015 conclusions unchanged.\n'
    for name in ('HANDOFF.md','T016_HANDOFF.md'):(ROOT/'research_log'/name).write_text(handoff,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+handoff.replace('# ','',1))
    save(ROOT/'research_log/t016_delivery.json',dict(timestamp=now,lead='6476c99',runtime='6fd685e543f13303d201cdb7c9b810a395696a18',run=RUN,status=summary['status'],raw=RAW.relative_to(ROOT).as_posix(),results=OUT.relative_to(ROOT).as_posix(),next='Await Lead after preflight stop'))
    save(ROOT/'research_log/t016_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(RAW.rglob('*')) if p.is_file()])
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
