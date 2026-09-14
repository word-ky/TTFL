"""Publish T018R numerical stop without assigning scientific gate outcomes."""
import json,hashlib,shutil
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFLIGHT='20260914-124138-ttfl-t018r-preflight';RUN='20260914-124709-ttfl-t018r-science'
PRE=ROOT/f'research_log/t018r_receipts/{PREFLIGHT}/artifacts/t018r_constrained_prevalence'
RAW=ROOT/f'research_log/t018r_receipts/{RUN}/artifacts/t018r_constrained_prevalence'
BASE=ROOT/'results/t018_constrained_prevalence';OUT=BASE/'t018r'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copytree(PRE,OUT/'preflight',dirs_exist_ok=True);shutil.copytree(RAW,OUT/'matched_stop',dirs_exist_ok=True)
    ver=load(PRE/'solver_verification.json');noise=load(PRE/'noise_free_verification.json');d=load(RAW/'blocker_diagnosis.json')
    assert ver['status']=='PASS' and ver['historical_cap_cases_repaired']==49 and ver['historical_converged_cases_equivalent']==151
    assert d['cached_sum_error']>1e-12 and d['cached_direct_KKT']<=1e-10 and d['reference_max_pi_error']<=1e-7
    assert 'Ran 98 tests' in (PRE.parent.parent/'train.log').read_text() and 'exit_code=0' in (PRE.parent.parent/'train.log').read_text()
    assert 'exit_code=1' in (RAW.parent.parent/'train.log').read_text()
    checked=[]
    for name in ('t015_artifact_manifest.json','t016r_artifact_manifest.json','t017r_artifact_manifest.json'):
        for r in load(ROOT/'research_log'/name):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    save(OUT/'source_hash_replay.json',dict(unchanged=True,count=len(checked),checked=checked))
    summary=dict(status='MATCHED_SOLVER_NUMERICAL_BLOCKER',CLS_MATCH_A='NOT_EXECUTED',CLS_REAL_A='NOT_EXECUTED',CLS_SRC_A='NOT_EXECUTED',outcome='NOT_ASSIGNED',
        preflight_pass=True,tests_passed=98,noise_free_cases=1000,reference_cases=200,old_cap_cases_repaired=49,all_actual_audit_cases=1000,
        matched_successful_cases=45575,first_failing_case=dict(client=35,bank='B',context='brightness_dark',replica=7),cached_simplex_sum_error=d['cached_sum_error'],
        required_sum_tolerance=1e-12,KKT=d['cached_direct_KKT'],reference_pi_error=d['reference_max_pi_error'],source_hashes_unchanged=len(checked),query_metrics_opened=False,new_model_forwards=0)
    save(OUT/'summary.json',summary)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    report=f'''# T018R: real-cell preflight passes; matched solver stops on sum feasibility

{now}. Lead `8d12997` / unchanged scientific protocol `1744440`.

**Status: numerical blocker. CLS-MATCH-A / CLS-REAL-A / CLS-SRC-A remain NOT EXECUTED; no Outcome A/B/C.** The historical T018 PGD stop is preserved in its original raw receipts and the report below. This is a distinct T018R result.

## What changed and passed

Implemented the prescribed deterministic active-set simplex QP, retaining old PGD unchanged. The objective, warm start, observations, templates, context decisions, K20, tolerances and gates did not change. Face systems are lazily cached as `solve(K, identity)` mappings, as authorized; no added dependency, regularization or statistical tuning. Most-negative removal and greatest-dual-violation insertion use smallest-index tie breaking. Cycles, >100 updates, nonfinite values and failed feasibility stop execution.

Runtime `5f5e1a5a845f743646aea7f67c1d47463f1b9832`; run `{PREFLIGHT}`, exit0. **98 tests pass.** Added tests cover condition numbers 3/50/100/200, 12 fixed synthetic face-reference comparisons, boundary zeros, insertion/removal, replay and input immutability.

- Noise-free 1,000: max prevalence error {noise['max_pi_error']:.6g}, max objective {noise['max_objective']:.6g}; all 7,914 singleton and 86 tied exact-optimum checks pass.
- Same fixed 200 actual cases: **all 49 old PGD caps repaired**, all 151 old converged cases reference-equivalent; max prevalence error {ver['max_prevalence_error']:.6g}, max objective error {ver['max_objective_error']:.6g}.
- All 1,000 real cells: KKT / simplex feasibility / BBSE objective dominance pass; max KKT {ver['max_KKT']:.6g}, maximum {ver['max_updates']} working-set updates. Full traces and pi vectors are saved.

## New stopped case

Runtime `315a984ee4ba5c26495b48f3a73c081c5f0340bc`; run `{RUN}`, exit1. Before opening query counts or accuracy, the program attempted numerical certification of the saved 128,000 matched-K20 observations. It accepted **45,575** cases then stopped at **client35 / bankB / Dark / replica7**. The remaining cases were not attempted; no partial accuracy or gate was computed. The accepted prefix was in memory and is not a complete persisted result array; its length follows the deterministic loop index and preserved log.

Failing face: classes `[0,1,2,3,6,8,9]`. All coordinates are nonnegative, but cached inverse-map application returns a sum of **{d['cached_sum']:.16g}**, absolute error **{d['cached_sum_error']:.16g}**, exceeding the fixed `1e-12` sum tolerance by about 4.8%. `direct_kkt` is {d['cached_direct_KKT']:.6g}, which passes its separate `1e-10` bound. Thus this is specifically a **simplex sum feasibility rejection**, not a KKT threshold failure, cycle or update cap.

Same-runtime read-only replay reproduces the exception. The independently solved face and exhaustive reference give sum error {d['direct_face_sum_error']}, KKT {d['reference_KKT']:.6g}, and prevalence difference from the cached result {d['reference_max_pi_error']:.6g}. Objective difference is {abs(d['cached_objective']-d['reference_objective']):.6g}. This evidence localizes the issue to roundoff in the cached inverse-map application; it is very different from the old PGD error of 0.04343. These direct solutions were **diagnostics only**, never substituted into the policy evaluation.

No normalization, tolerance relaxation, alternative solver or second scientific run was applied. Under the specified roundoff rule, clipping/renormalization occurs only when the face solve has a tiny negative coordinate; this failing face had none. The implementation therefore rejected the sum error as required.

## Scope and next decision

The exact original matched q was replayed from saved T017R arrays; no resampling. Zero new model forwards; no query metrics opened. All {len(checked)} T015/T016/T017 manifest entries remain byte-identical. No estimator or scientific conclusion follows from this stop.

Return to Lead for a narrow engineering decision on the face-system numerical application, e.g. direct RHS solve while preserving the same active-set trajectory/objective and all current tolerances. This is a suggestion only; no repair or rerun is authorized by this report itself. Do not move to feature semantics, SSL/TTT or FL.

Receipts: `research_log/t018r_receipts/{PREFLIGHT}/` and `research_log/t018r_receipts/{RUN}/`; compact copies under `results/t018_constrained_prevalence/t018r/`. `blocker_diagnosis.json` contains the exact q-derived case, pi vectors and independent numerical comparisons. Scientific CSVs were not fabricated for unexecuted stages.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    historical=BASE/'PGD_PREFLIGHT_RESULTS.md'
    if not historical.exists():shutil.copyfile(BASE/'RESULTS.md',historical)
    (BASE/'RESULTS.md').write_text(report+'\n\n---\n\n'+historical.read_text(encoding='utf-8'),encoding='utf-8')
    hand=f'''# T018R STOP — matched simplex sum tolerance

{now}. Lead8d12997; preflight runtime5f5e1a5/run{PREFLIGHT} exit0:98tests,1000noise-free,same200reference incl49oldcaps,all1000real audit pass. Science runtime315a984/run{RUN} exit1 before metrics:45,575matched solves accepted then client35/B/Dark/replica7 sum error1.0480505352461478e-12 >1e-12. Same-runtime reproduced; independent face error0, maxpi difference4.77e-13. No scientific gates/outcome; zero new forwards. Read results/t018_constrained_prevalence/t018r/RESULTS.md. No changes to tolerances or solver after stop; await Lead engineering instruction.
'''
    (ROOT/'research_log/HANDOFF.md').write_text(hand,encoding='utf-8');(ROOT/'research_log/T018R_HANDOFF.md').write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\nThe old PGD stop remains preserved. New blocker is cached face-map roundoff under strict sum feasibility, not unresolved convergence or CLS mechanism failure. Full report includes diagnostics, unchanged source hashes and both distinct run IDs.\n',encoding='utf-8')
    save(ROOT/'research_log/t018r_delivery.json',dict(timestamp=now,lead='8d12997',preflight_run=PREFLIGHT,science_run=RUN,status=summary['status'],results=str(OUT.relative_to(ROOT)),next='Await Lead numerical repair instruction; no scientific gate evaluated'))
    files=[p for p in (ROOT/'research_log/t018r_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t018r_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
