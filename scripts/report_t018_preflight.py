"""T018 fixed-PGD convergence blocker report; no query or bootstrap evaluation."""
import csv,json,hashlib,shutil
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='20260914-113842-ttfl-t018-preflight'
RAW=ROOT/f'research_log/t018_receipts/{RUN}/artifacts/t018_constrained_prevalence';OUT=ROOT/'results/t018_constrained_prevalence'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    v=load(RAW/'solver_verification.json');noise=load(RAW/'noise_free_verification.json');meta=load(RAW/'metadata.json');rows=list(csv.DictReader((RAW/'solver_subset.csv').open()))
    caps=[r for r in rows if r['cap_hit']=='True'];converged=[r for r in rows if r['cap_hit']=='False'];bad=[r for r in rows if r['reference_agreement']=='False']
    assert len(caps)==v['cap_hits'] and len(bad)==v['reference_disagreements'] and all(r['reference_agreement']=='True' for r in converged)
    assert all(r['objective_dominance']=='True' for r in rows) and all(float(r['reference_KKT'])<=1e-10 for r in rows)
    log=(RAW.parent.parent/'train.log').read_text();assert 'Ran 95 tests' in log and 'exit_code=1' in log
    checked=[]
    for manifest in ('t015_artifact_manifest.json','t016r_artifact_manifest.json','t017r_artifact_manifest.json'):
        for r in load(ROOT/'research_log'/manifest):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(dict(path=r['path'],sha256=r['sha256']))
    save(OUT/'source_hash_replay.json',dict(unchanged=True,count=len(checked),checked=checked))
    bycontext={t:sum(r['context']==t for r in caps) for t in ('clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur')}
    worst=max(rows,key=lambda r:float(r['prevalence_error']));now=datetime.now().astimezone().isoformat(timespec='seconds')
    summary=dict(status='SOLVER_PREFLIGHT_BLOCKER',CLS_MATCH_A='NOT_EXECUTED',CLS_REAL_A='NOT_EXECUTED',CLS_SRC_A='NOT_EXECUTED',outcome_A_B_C='NOT_ASSIGNED',
        tests_passed=95,noise_free_cases=1000,independent_actual_cases=200,converged_cases=len(converged),cap_hits=len(caps),reference_disagreements=len(bad),objective_dominance_violations=0,
        cap_hits_by_context=bycontext,max_prevalence_error=v['max_prevalence_error'],max_objective_error=v['max_objective_error'],max_reference_KKT=v['max_reference_KKT'],
        bootstrap_cases_solved=0,query_metrics_opened=False,new_model_forwards=0,diagnosis='Fixed step projected gradient does not reach declared convergence within20000iterations on49/200real observations. CLS scientific effect remains untested; independent face optimum is well certified.')
    save(OUT/'summary.json',summary);v.update(converged_cases_all_reference_agree=True,source_manifest_hashes_unchanged=len(checked),worst_case=worst);save(OUT/'solver_verification.json',v)
    report=f'''# CODEX → CHATGPT: T018 solver preflight blocker

{now}. Scientific Lead `1744440`, confirmed by `e3e9c70`; runtime `{meta['runtime']}`; release `20260914-113818-ttfl-t018-preflight`; run `{RUN}`. One preflight run, exit1 after {meta['seconds']:.2f}s. **95 tests PASS. No new model inference, query metrics or bootstrap evaluation.**

## Decision

The fixed CLS-S numerical path hits its declared iteration cap on **49/200 actual observations**. **CLS-MATCH-A / CLS-REAL-A / CLS-SRC-A are NOT EXECUTED.** No scientific Outcome A/B/C can be assigned. This is a convergence-budget implementation blocker, not evidence that measurement-space simplex estimation fails at K20.

## Exact implementation

The objective is unchanged `0.5*||C*pi-q||²` on the simplex. Start from historical `ProjectSimplex(pinv(C)@q)`. Every iteration uses `C.T@(C@pi-q)` and step `eta=1/||C||₂²`, followed by the unchanged T016 simplex projection. Stop requires both max coordinate step<=1e-12 and projected-gradient residual<=1e-10. The residual is `L*maxabs(pi-ProjectSimplex(pi-grad(pi)/L))`, evaluated at the returned iterate. Hard cap20,000 is unchanged. No acceleration, regularization, extra inverse cutoff, temperature, or step search was introduced.

The fixed independent subset is clients0–19 × banksA/B × all5oraclecontexts:200real observations. This numerical diagnostic completed even after cap events to record the predeclared reference comparison; no capped result was accepted for state-policy evaluation.

## Sanity that passed

- All1,000noise-free matched-channel cases pass: maximum prevalence error **{noise['max_pi_error']:.17g}**, maximum measurement objective **{noise['max_objective']:.17g}**.
- All7,914singleton and86tied exact-P00 template checks pass under the inherited tie-aware noise-free rule. Historical P00 values and argmax sets reproduce exactly; no state override occurs in CLS-S.
- All200actual cases satisfy measurement-objective dominance over warm-start BBSE within the required squared-residual tolerance1e-12.
- The independent reference enumerates all1,023nonempty simplex faces, solves each equality-constrained quadratic problem directly, and chooses the minimum feasible objective. Its maximum direct KKT residual across200cases is **{v['max_reference_KKT']:.17g}**. All channels in this subset are full rank, so the objective optimum is unique.
- All**151converged cases** satisfy objective agreement<=1e-10 and prevalence max-abs agreement<=1e-7 with this independent reference.
- All**{len(checked)}T015/T016/T017 source-manifest file hashes** remain unchanged. T017bootstrap arrays were hash-verified but not resampled or evaluated. The preflight does not open frozen query outcomes/count metrics.

## Cap failures

| context | cap hits / 40 observations |
|---|---:|
| clean | {bycontext['clean']} |
| dark | {bycontext['brightness_dark']} |
| low contrast | {bycontext['contrast_low']} |
| Gaussian noise | {bycontext['gaussian_noise']} |
| Gaussian blur | {bycontext['gaussian_blur']} |

Among49cap hits,41fail the prescribed independent objective/prevalence agreement; the other8still fail the hard convergence/cap rule and were not accepted. Across all200cases, maximum objective difference is **{v['max_objective_error']:.17g}**, maximum prevalence coordinate difference is **{v['max_prevalence_error']:.17g}**.

Worst case: client{worst['client']},bank{worst['bank']},{worst['context']}. At20,000iterations, maxstep={worst['max_step']}, projected-gradient residual={worst['projected_gradient_residual']}; objective={worst['objective']} versus reference={worst['reference_objective']}; prevalence maxerror={worst['prevalence_error']}. There are{worst['active_classes']}active classes. This is a material optimization gap, not a decimal serialization discrepancy.

The returned objective decreases substantially from the BBSE warm start, but the fixed plain projected-gradient iteration is too slow on some measured channels to meet the declared precision within the cap. The independent face solution is diagnostic only; it was **not substituted** as the primary estimator. An engineering revision would require Lead authorization and can preserve the same convex objective, but none was made in this run.

## Artifacts and scope

`solver_subset.csv` records all200iteration counts, step/residuals, direct/reference KKT, objectives, active counts, L1 changes and comparison flags. `solver_vectors.json` records q, returned pi and independent reference pi. `noise_free.csv` records all1,000sanity cases. `protocol_freeze.json`, `input_hashes.json` and `source_hash_replay.json` preserve definitions and source identity. The full process log is retained in `research_log/t018_receipts/{RUN}`.

Matched-K20, actual-oracle, full-source and Blur task CSVs are absent because their scientific stages were never reached; no placeholder accuracy is supplied. K20, state banks, templates, context decisions and historical results remain unchanged. Do not classify this as OutcomeC or start upstream feature observability from a numerical stop. Return this concrete convergence blocker to Lead; no next scientific task or automatic solver change.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    concise=f'''# CODEX → CHATGPT: T018 solver preflight STOP

{now}. Lead1744440/statusconfirmatione3e9c70; runtime{meta['runtime']}; run{RUN};95testsPASS;exit1,{meta['seconds']:.2f}s. Report: `results/t018_constrained_prevalence/RESULTS.md`.

**49/200fixedactual cases hit20,000iterations;41fail independent reference agreement.** All151convergedcases agree. No querymetrics/bootstrap ran; CLS-MATCH/REAL/SRC areNOT_EXECUTED, noOutcomeA/B/C.

Noise-free1000casesPASS,maxpierror5.27e-15,maxobjective4.76e-31;7914singleton/86tiedexact-optimum checksPASS. Everyactualcase improves measurementobjective versus BBSE withinrequiredtolerance. Independent exhaustive1023-face reference maxKKT2.22e-16; it was usedonlyforverification, neverasprimary.

Capcounts/40percontext:clean6,dark12,contrast14,noise8,blur9. Worstclient6/B/contrast:maxstep3.97e-6,PGresidual5.76e-6,objectivegap2.57e-7,maxpierror0.04343. This is unresolved optimization error underfixedbudget, notroundoff and notCLSscientificfailure.8othercapcasesmeetreferenceaccuracybutstillviolatedeclaredtermination and were notaccepted.

No change to step,cap,tolerances,objective,projection,warmstart,K20,statesorcontext. All{len(checked)}sourceartifacthashesunchanged;zero newmodelinference. Fullsolverrows/vectors/referenceKKT andlogsretained. AwaitLeadengineeringdecision; noautomaticalternative solver, output-space method proliferation orupstreamstage.
'''
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(concise,encoding='utf-8')
    handoff=f'# T018 STOP — FIXED PGD CAP\n\n{now}. Lead1744440;runtime{meta["runtime"]};run{RUN};95testsPASS;noise-free1000PASS;49/200actualcap20000,41referencefailures;151convergedallreferenceagree. Worstpierror.04343. Noquery/bootstrapevaluation;CLSgatesNOT_EXECUTED. Read results/t018_constrained_prevalence/RESULTS.md. No solverchange ornextstage;awaitLead.\n'
    for name in ('HANDOFF.md','T018_HANDOFF.md'):(ROOT/'research_log'/name).write_text(handoff,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+handoff)
    save(ROOT/'research_log/t018_delivery.json',dict(timestamp=now,lead='1744440',runtime=meta['runtime'],run=RUN,status='SOLVER_PREFLIGHT_BLOCKER',raw=RAW.relative_to(ROOT).as_posix(),results=OUT.relative_to(ROOT).as_posix(),next='Await Lead engineering instruction; no scientific evaluation'))
    save(ROOT/'research_log/t018_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'research_log/t018_receipts').rglob('*')) if p.is_file()])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
