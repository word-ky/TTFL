"""Compact verified T023 result handoff; large frozen arrays stay in remote runs."""
import argparse,csv,gzip,hashlib,json,shutil
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DIR='t023_rotation_ssl_alignment'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open()))
def span(v,scale=1):return f'{min(v)*scale:.3f} to {max(v)*scale:.3f}'

def main():
    ap=argparse.ArgumentParser()
    for stage in ('extraction','phase_a','scoring','verification'):ap.add_argument('--'+stage,required=True)
    runs=vars(ap.parse_args());out=ROOT/'results'/DIR;folders={k:ROOT/f'research_log/t023_receipts/{v}/artifacts/{DIR}' for k,v in runs.items()}
    for stage,folder in folders.items():
        target=out if stage in ('scoring','verification') else out/stage;target.mkdir(parents=True,exist_ok=True)
        for p in folder.iterdir():
            if p.is_file():shutil.copyfile(p,target/p.name)
        assert 'exit_code=0' in (folder.parent.parent/'train.log').read_text()
    assert 'Ran 138 tests' in (folders['scoring'].parent.parent/'train.log').read_text()
    v=load(out/'independent_verification.json');s=load(out/'summary.json');g=load(out/'diagnostic_gates.json');assert v['status']=='PASS'
    for r in load(out/'remote_artifact_manifest.json'):
        local=ROOT/'research_log/t023_receipts'/r['path'].split('/runs/',1)[1]
        if local.exists():assert sha(local)==r['sha256']
        else:assert local.suffix=='.npz'
    checked=[]
    for task in ('t015','t016r','t017r','t018r2','t019','t020','t020r','t020r2','t021','t022'):
        for r in load(ROOT/f'research_log/{task}_artifact_manifest.json'):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    save(out/'upstream_local_manifest_verification.json',dict(status='PASS',entries=len(checked),files=checked))
    first=ROOT/f'research_log/t023_receipts/20260914-212711-ttfl-t023-score/artifacts/{DIR}';identical=[]
    for path in folders['scoring'].iterdir():
        if path.name in ('summary.json','scoring_start.json'):continue
        assert sha(path)==sha(first/path.name);identical.append(dict(name=path.name,sha256=sha(path)))
    save(out/'completion_scoring_replay.json',dict(first_run='20260914-212711-ttfl-t023-score',canonical_run=runs['scoring'],reason='Lead38b01e8 required additional sealed-input assertions and Phase-A runtime receipt; scoring-only repetition, all scientific outputs unchanged.',identical_scientific_outputs=identical))
    aa=rows(out/'actual_aggregate.csv');pa=rows(out/'paired_regret_context.csv');rank=rows(out/'rank_summary.csv');scramble=rows(out/'phase_a/scramble_sanity.csv');hist=load(out/'state_choice_histograms.json');contexts=['brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
    gate=f"| Gate | Shift contexts passing | Overall |\n|---|---:|---|\n| ROT-TASK-A | {sum(g['task_context_pass'].values())}/4 | {g['ROT_TASK_A']} |\n| ROT-REGRET-A | {sum(g['ROT_REGRET_A']['context_pass'].values())}/4 | {g['ROT_REGRET_A']['passed']} |\n| ROT-CTX-A | {sum(g['ROT_CTX_A']['context_pass'].values())}/4 | {g['ROT_CTX_A']['passed']} |\n"
    capture='| Shift/policy | Capture %, min to max | Median % | Delta vs BBSE-S-01 pp |\n|---|---:|---:|---:|\n'
    for t in contexts:
        for policy in ('ROT-CURRENT','ROT-CLEAN-SURROGATE','P'):
            r=[z for z in aa if z['context']==t and z['policy']==policy];cap=[float(z['capture']) for z in r]
            capture+=f"| {t}/{policy} | {span(cap,100)} | {np.median(cap)*100:.3f} | {span([float(z['delta_vs_BBSE_pp']) for z in r])} |\n"
    effect='| Shift/comparison (CURRENT) | Mean regret reduction % | New p90 | Baseline p90 | Clients better /100 |\n|---|---:|---:|---:|---:|\n'
    for t in contexts:
        for comp in ('vs_P','current_vs_clean'):
            r=[z for z in pa if z['context']==t and z['policy']=='ROT-CURRENT' and z['comparison']==comp]
            effect+=f"| {t}/{comp} | {span([float(z['relative_mean_reduction']) for z in r],100)} | {span([float(z['new_p90']) for z in r])} | {span([float(z['baseline_p90']) for z in r])} | {span([float(z['clients_better']) for z in r])} |\n"
    ranktable='| Policy/bank/context | Median Spearman | Q25/Q75 | Positive % | Undefined /800 |\n|---|---:|---:|---:|---:|\n'
    for r in rank:ranktable+=f"| {r['policy']}/{r['bank']}/{r['context']} | {r['median']} | {r['p25']}/{r['p75']} | {100*float(r['fraction_positive']):.1f} | {r['undefined']} |\n"
    sanity='| Bank/context | True MSE mean | Scramble mean MSE p05/median/p95 | Clients true < own scramble median % |\n|---|---:|---:|---:|\n'
    for r in scramble:sanity+=f"| {r['bank']}/{r['context']} | {float(r['true_score_mean']):.5f} | {float(r['scramble_client_mean_p05']):.5f}/{float(r['scramble_client_mean_median']):.5f}/{float(r['scramble_client_mean_p95']):.5f} | {100*float(r['fraction_clients_true_beats_own_scramble_median']):.1f} |\n"
    histogram='| Policy/bank/context | clean/Dark/Contrast/Noise/Blur counts | Exact tie cells /100 |\n|---|---|---:|\n'
    for r in hist:histogram+=f"| {r['policy']}/{r['bank']}/{r['context']} | {r['client_state_counts']} | {r['exact_tie_cells']} |\n"
    now=datetime.now().astimezone().isoformat(timespec='seconds');ef=load(out/'extraction/extraction_freeze.json');pf=load(out/'phase_a/phaseA_choices_freeze.json');audit=load(out/'phase_a/probe_numerical_audit.json')
    report=f'''# T023 — {s['diagnosis']} / {s['CTX']}

{now}. Scientific Lead89466b3; completion Lead38b01e8. All four canonical stages exit0; final full suite138tests PASS; independent verification PASS. This is a verified mechanism outcome, not an implementation blocker.

## Decision

{gate}

Shifted regression safety: {g['shifted_regression_safety']}. The fixed rotation auxiliary objective does not meet the predeclared task-proximity criteria. No alternate objective, normalization, regularization, angle, fold or state subset was tried. No actual writer/model update or federation was run. T023's context flag concerns this rotation selector only; it does not invalidate the earlier T021/T022 CTX+ evidence.

## Capture and regret

All ranges span both banks/all4salts. Capture uses the identical P00 denominator and opposite-half integer counts. It is a percentage of available oracle gain, not accuracy percentage. TASK requires>=80% in every bank/salt for>=3/4shifts and all shifted rows>=BBSE-S-01−0.5pp.

{capture}

{effect}

Positive reduction means less regret. REGRET vsP and CTX vsCLEAN each additionally require p90<=1.05baseline and>=15% mean reduction in every bank/salt for>=3/4shifts. Exact-zero baseline means remain noneligible for relative gain. The two halves are paired within each client before better/equal/worse counting. Historical P integer numerators reproduce T018 exactly. Both CURRENT and CLEAN have complete comparisons against P in paired_regret_context.csv.

## Rotation sanity and state ranking

{sanity}

Each client has16 independent-per-image label permutations with balanced4-way counts. Table scramble quantiles are across16 replica means over100clients; the final column compares each client to its own16-replica median. Per-client p05/median/p95 are in phase_a/scramble_cells.csv. This diagnostic was frozen before task scoring and was not a task-selection gate.

{ranktable}

Spearman uses average ranks, with the exact rational utility order and negative rotation MSE. Each bank/context summary contains100clients×4salts×2halves; these800utility comparisons reuse the same support and are not independent trials. Constant vectors are undefined and excluded from the defined-correlation denominator with explicit counts.

{histogram}

## Fixed implementation and independent audit

H is the unchanged512-D fc1/ReLU representation. Rotations are exact quarter turns after deterministic corruption. The same SHA256-sorted10/10support split is reused across states, banks, contexts and rotations. Raw float64 H plus bias is fitted to generated4-way rotation one-hot labels by minimum-norm least squares with fixed rcond1e-12. Score is the mean over40held-out views×4output coordinates, averaged over both directions. Disposable coefficients never touch query data and are not persisted or backpropagated. Only x is accessed from the PFLlib support storage; class-label fields are not accessed during extraction/probe fit/choice.

Extraction reused5000unrotated paths, ran15000rotated batches and250unrotated replay batches on A6000. Probe stage froze5000state scores,1000CURRENT choices and1000CLEAN-SURROGATE choices, all tie masks/ranks and16scrambles. Before privileged scoring, {audit['cross_fitted_scores']} cross-fitted scores ({audit['individual_fits']} individual fits) were independently reproduced by numpy.linalg.lstsq with max error{audit['max_score_difference']}; observed rank range{audit['rank_min']}–{audit['rank_max']}.

The separate verifier uses original model forward with a final-classifier input hook, not the main extractor; SciPy LAPACK least-squares, not the main SVD helper; independent fold/scramble SHA code; and historical integer/Fraction arithmetic, not the main scorer. It verifies{v['unrotated_original_model_samples']} unrotated and{v['rotated_original_model_samples']} rotated original-model H samples, {v['all_unrotated_cache_samples']} reused cache sample vectors, {v['exact_quarter_turn_sample_tensors']} quarter-turn sample tensors; maximum H difference{v['max_H_difference']}. It reconstructs{v['cross_fitted_probe_scores']} cross-fitted probe scores ({v['individual_probe_fits']} fits), max score error{v['max_probe_score_difference']}; all16replicas in{v['scramble_cells']} cells ({v['scramble_cell_replicas']} cell-replicas), max error{v['max_scramble_score_difference']}; all1000choice/tie cells and all scramble summaries.

Explicit cardinalities precede gate reconstruction:120 aggregate rows,24000regret episodes with exactly200unique client-halves per policy/bank/context/salt,120paired rows,16000rank correlations,20rank summaries and20histogram cells. All integer numerators/P00 capture, exact mean/p90 regret, paired counts, histograms, ranks and three gates are independently reconstructed. Model/state hashes and all Phase-A/privileged input hashes remain unchanged. Scoring performs zero new model/query forwards. {len(checked)} local upstream manifest entries pass.

## Runtime and recovery

- Extraction `{ef['runtime']}` / `{runs['extraction']}`.
- Phase-A `{pf['runtime']}` / `{runs['phase_a']}`.
- Scoring `{s['runtime']}` / `{runs['scoring']}`.
- Verification `{v['runtime']}` / `{runs['verification']}`.

Initial scorer20260914-212711-ttfl-t023-score completed atd9490a3. Lead38b01e8 then required explicit sealed-input assertions and Phase-A runtime metadata; the scoring-only repeat is byte-identical on all scientific outputs (completion_scoring_replay.json). Frozen extraction/probe choices were not regenerated. One verifier deployment upload lost SSH connectivity before any verifier job started; the same scientific inputs were retained on recovery.

Compact receipts and logs are under research_log/t023_receipts; extraction/ and phase_a/ retain stage receipts. Large H/pixel/score NPZ arrays remain in canonical remote project run directories with exact byte sizes, paths and hashes in remote_artifact_manifest.json. The existing PFLlib CIFAR10 split is44961train/15039query after merged-source client partitioning,100clients and the historical10% participation checkpoint; this is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining occurred.

Return {s['diagnosis']}/{s['CTX']} for Lead review. Request the next bounded research decision; do not implement a writer, alternate SSL objective or T024 autonomously.
'''
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f"# T023 complete — {s['diagnosis']} / {s['CTX']}\n\n{now}. Lead89466b3+completion38b01e8.138testsPASS/all4canonicalstagesexit0/independentPASS. Mechanism outcome, no implementation blocker.\n\n"+gate+f"\nScoring runtime{s['runtime']}; verifier{v['runtime']}. Read results/{DIR}/RESULTS.md. Raw NPZ remote with manifest; compact receipts local. Stop for Lead; no alternateSSL/writer/FL/T024.\n"
    for name in ('HANDOFF.md','T023_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+capture+'\n'+effect+'\n'+sanity+'\nRank/IQR, histograms, all maximum discrepancies and input/runtime hashes are in RESULTS.md. Request next bounded Lead decision; no method rescue.\n',encoding='utf-8')
    save(ROOT/'research_log/t023_delivery.json',dict(timestamp=now,lead='89466b3',completion_lead='38b01e8',runs=runs,runtime=s['runtime'],verification_runtime=v['runtime'],diagnosis=s['diagnosis'],CTX=s['CTX'],next='Await Lead; no alternate SSL/writer/FL/T024'))
    paths=[p for p in (ROOT/'research_log/t023_receipts').rglob('*') if p.is_file()]+[p for p in out.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t023_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]);print(hand)

if __name__=='__main__':main()
