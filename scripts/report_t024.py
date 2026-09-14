"""Publish verified clean-visual semantic-state oracle geometry, with bounded claims."""
import csv,json,hashlib,shutil
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];DIR='t024_semantic_state_geometry';RECEIPTS=ROOT/'research_log/t024_receipts';OUT=ROOT/'results'/DIR
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def number(x,scale=1):return f'{float(F(x))*scale:.3f}'

def main():
    analysis=RECEIPTS/'20260914-t024-local/artifacts'/DIR;verification=RECEIPTS/'20260914-t024-verify/artifacts'/DIR
    for folder in (analysis,verification):
        for p in folder.iterdir():shutil.copyfile(p,OUT/p.name)
    v=load(OUT/'independent_verification.json');s=load(OUT/'summary.json');g=load(OUT/'diagnostic_gates.json');pre=load(OUT/'preflight.json');assert v['status']=='PASS' and v['max_exact_discrepancy']==0
    assert 'T024_GEOMETRY_COMPLETE '+s['diagnosis'] in (RECEIPTS/'20260914-t024-local/run.log').read_text(encoding='utf-8')
    assert 'T024_INDEPENDENT_PASS '+s['diagnosis'] in (RECEIPTS/'20260914-t024-verify/run.log').read_text(encoding='utf-8')
    log=(RECEIPTS/'full_suite.log').read_text(encoding='utf-8');assert 'Ran 142 tests' in log and 'exit_code=0' in log
    assert sha(OUT/'semantic_shift_freeze.json')==s['semantic_freeze_sha256'] and all(sha(Path(p))==h for p,h in load(OUT/'input_hashes.json').items())
    aggregate=rows(OUT/'aggregate_by_lambda.csv');mismatch=rows(OUT/'mismatch_by_salt.csv');trans=load(OUT/'state_transitions.json');correlation=load(OUT/'distance_correlations.json')
    gate='| Gate | Bank A | Bank B | Overall |\n|---|---|---|---|\n'
    for name,key in [('SEM-HEADROOM-A','headroom'),('SEM-SPEC-A','specificity'),('SEM-TREND-A','trend')]:gate+=f"| {name} | {g['banks']['A'][key]} | {g['banks']['B'][key]} | {all(g['banks'][b][key] for b in ('A','B'))} |\n"
    severe='| Bank | Stale mean / median / p90 pp | Stale >=1pp % | Switch % | Mismatch mean pp | Mismatch >=1pp % | Current−zero / historical−zero pp |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for b in ('A','B'):
        r=next(x for x in aggregate if x['bank']==b and x['lambda_']=='1');bg=g['banks'][b]
        severe+=f"| {b} | {float(r['stale_regret_mean_pp']):.3f} / {float(r['stale_regret_median_pp']):.3f} / {float(r['stale_regret_p90_pp']):.3f} | {number(bg['stale_ge_1pp_fraction'],100)} | {number(bg['switch_rate'],100)} | {float(r['mismatch_regret_mean_pp']):.3f} | {number(bg['mismatch_ge_1pp_fraction'],100)} | {float(r['oracle_vs_zero_mean_pp']):.3f} / {float(r['historical_minus_zero_mean_pp']):.3f} |\n"
    trend='| Bank / lambda | Mean stale pp | Median / p75 / p90 pp | Switch % | Mismatch mean pp | Oracle−zero / historical−zero pp |\n|---|---:|---:|---:|---:|---:|\n'
    for r in aggregate:trend+=f"| {r['bank']} / {r['lambda_']} | {float(r['stale_regret_mean_pp']):.3f} | {float(r['stale_regret_median_pp']):.3f} / {float(r['stale_regret_p75_pp']):.3f} / {float(r['stale_regret_p90_pp']):.3f} | {100*float(r['switch_rate']):.3f} | {float(r['mismatch_regret_mean_pp']):.3f} | {float(r['oracle_vs_zero_mean_pp']):.3f} / {float(r['historical_minus_zero_mean_pp']):.3f} |\n"
    salts='| Bank / salt | Severe mean mismatch pp | >=0.5pp |\n|---|---:|---|\n'
    for r in mismatch:
        if r['lambda_']=='1':salts+=f"| {r['bank']} / {r['salt']} | {float(r['mean_pp']):.3f} | {F(r['mean_exact'])>=F(1,200)} |\n"
    matrices=''
    for r in trans:
        if r['lambda_']=='1':
            matrices+=f"\nBank {r['bank']}, lambda1: rows historical, columns current. Total800 source cells.\n\n| Historical / current | s0 | s1 | s2 | s3 | s4 |\n|---|---:|---:|---:|---:|---:|\n"
            for i,row in enumerate(r['counts']):matrices+='| s'+str(i)+' | '+' | '.join(map(str,row))+' |\n'
    rho='| Bank | Distance | Spearman vs stale regret | Cell×lambda rows |\n|---|---|---:|---:|\n'
    for r in correlation:rho+=f"| {r['bank']} | {r['distance']} | {r['spearman']:.6f} | {r['cells']} |\n"
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    report=f'''# T024 — {s['diagnosis']}: semantic-state oracle geometry

{now}. Lead d55a0f9 (unchanged heartbeat96682c0). **Clean visual context + semantic label-composition shift**, with anonymous fixed neutral operators `s0..s4`; `s0` is the strict no-op. Target composition is privileged. This is an oracle-template diagnostic, not a deployable adaptation method or a new sampled benchmark.

## Verified decision

{gate}

All three frozen gates pass in both banks: **T024-S**. Severe semantic change makes the historical oracle action materially stale, and a mismatched current composition also selects materially worse actions under the same target template. This supports the specified semantic-state abstraction within the existing fixed bank and the predeclared cyclic-shift construction. It is a mechanism/geometry result, not an implementation blocker.

{severe}

Both banks switch canonical state in693/800source cells (86.625%). Stale regret exceeds1pp in78.25%/76.5% of cells; mismatched-current regret exceeds1pp in65.375%/64.25%. Under severe shift the current oracle has mean positive gain versus no-op of5.835/5.671pp, while the historical choice has mean gain−3.809/−3.865pp. Thus headroom is not just an oracle beating no-op: retaining an otherwise historically optimal action is itself harmful on average after the semantic change.

## Frozen construction and units

The800historical compositions are100clients×4salts×2train-half orientations. Each normalizes the exact T014 saved class counts, verified against T013 counts. For every source composition, the nine nonzero right-cyclic class permutations are compared by exact TV; maximum TV wins and smallest offset breaks ties. No utility is used to construct the counter-state. There are zero uniform degenerate compositions. All four fixed lambdas, resulting3200current compositions and800SHA256 next-in-sorted-order mismatch pairings were saved before class-template utility access. Pairing is bijective/self-excluding within salt/half and shared across banks.

The mismatch has the same lambda as the target, but its realized TV magnitude can differ because its source composition differs. Both current and mismatched compositions are evaluated using the *target's same* leave-one-client-out class template. All scoring uses clean visual context; these operator columns are anonymous actions, not visual client-state labels.

T014 class templates store candidate class accuracy **minus no-op accuracy**. Consequently stored U values are expected template gains relative to no-op, not absolute accuracies. This common baseline cancels exactly in choices and stale/mismatch regret. Reported pp are expected accuracy differences under the frozen other99-client template, not newly measured current-client test accuracy. All vectors, choices, full argmax sets, regrets and gate comparisons use Fraction arithmetic. Top-two margins preserve tied-zero margins. Floats are used only for descriptive quantiles, JS and correlations.

Semantic freeze SHA256: `{s['semantic_freeze_sha256']}`. JS denotes square root of base2 Jensen-Shannon divergence. The protocol and generator were unchanged after scoring.

## Shift-strength trend and zero-state safety

{trend}

Mean stale regret increases from0.548→9.644pp in bankA and0.576→9.536pp in bankB. Severe-minus-mild gaps are9.096/8.960pp, exceeding the0.5pp gate. Nondecreasing stale regret is partly an algebraic consequence of this fixed-bank linear mixture with a historically optimal action; the substantive empirical requirements are the magnitude, affected-cell fraction and mismatch specificity. Positive oracle−zero is also structurally nonnegative because no-op belongs to the bank; the measured magnitude and historical−zero remain informative.

{rho}

The3200rows per bank reuse800compositions across4lambdas. Likewise800source cells are repeated views of100clients, not800independent clients. These correlations are descriptive; they never replace a gate.

## Current-state specificity and transitions

{salts}

Both banks pass all4salts (required>=3) and their pooled >=1pp affected-cell fractions exceed25%. Complete per-lambda/salt controls are in mismatch_by_salt.csv. Full8transition matrices and all cell-level current/historical/mismatch argmax sets are retained.

{matrices}

## Independent verification and execution

**142 tests PASS**, including focused cyclic target, exact TV scaling, pairing, tie, oracle and gate-boundary tests. The separate verifier does not import the main geometry/gate functions. It reconstructs all800pi0/q vectors,3200pi_lambda vectors,800pairings,1600clean target-excluded templates directly from the other99clients' integer counts,19200utility vectors,6400choice/regret rows,8lambda aggregates,32mismatch aggregates and8transition matrices. Every choice/full argmax set, stale/mismatch/zero regret, switch/overlap bit and all three gates agree exactly. Maximum exact discrepancy0; maximum JS difference{v['max_JS_distance_discrepancy']}; maximum descriptive Spearman difference{v['max_Spearman_discrepancy']}.

All {pre['manifest_entries']} source manifest entries and all analysis input hashes remain unchanged. Expected source/template/row cardinalities, state order and no-op index pass. Analysis and verification are pure local CPU saved-evidence calculations: no self-supervised loss, writer, gradient update, scientific model forward, new training or federation was executed. The remote job ran only the regression suite.

- Analysis runtime `{s['runtime']}`, local receipt `research_log/t024_receipts/20260914-t024-local`, seconds{s['seconds']:.3f}.
- Independent verifier runtime `{v['runtime']}`, local receipt `research_log/t024_receipts/20260914-t024-verify`, seconds{v['seconds']:.3f}.
- Full suite remote run `20260914-232500-ttfl-t024-tests`, source release `20260914-232433-ttfl-t024-tests` from analysis runtime; exit0,142tests.

## Scope of the positive result and next-state request

This intentionally severe, deterministic maximum-TV cyclic construction establishes oracle geometry within the tested bank. It does not establish unlabeled semantic-state estimation, a real sampled test-time distribution, or writer/federation performance. The inherited data remains the existing PFLlib CIFAR10 client partition; no new benchmark/data/checkpoint was created.

Return **T024-S** and request Lead review of a bounded *real sampled semantic-shift benchmark* task. Do not start that benchmark, an estimator, another SSL objective, a continuous writer or federated sharing without a later explicit Lead task.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f"# T024 complete — {s['diagnosis']}\n\n{now}. Leadd55a0f9;142testsPASS/analysis+independentverificationPASS. Clean visual input, privileged semantic composition, anonymous fixeds0..s4(s0no-op). Exactdiscrepancy0.\n\n"+gate+'\n'+severe+f"\nAnalysisruntime{s['runtime']};verifier{v['runtime']}. Report results/{DIR}/RESULTS.md. Oraclegeometrymechanism result, notdeployableadaptation. AwaitLead; requestbounded sampledsemantic-shiftbenchmark next; do not start estimator/SSL/writer/FL.\n"
    for name in ('HANDOFF.md','T024_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+trend+'\n'+salts+'\n'+matrices+'\nVerification:all800source/3200currentcompositions,800pairings,1600templates,19200utilityvectors,6400rows/allgatesexact. No new scientific model forwards or training. Full protocol, distributions, correlations and scope limitations in RESULTS.md.\n',encoding='utf-8')
    save(ROOT/'research_log/t024_delivery.json',dict(timestamp=now,lead='d55a0f9',runtime=s['runtime'],verification_runtime=v['runtime'],analysis_run='20260914-t024-local',verification_run='20260914-t024-verify',full_test_run='20260914-232500-ttfl-t024-tests',diagnosis=s['diagnosis'],next='Await Lead; request bounded sampled semantic-shift benchmark; no autonomous next stage'))
    files=[p for p in RECEIPTS.rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t024_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]);print(hand)

if __name__=='__main__':main()
