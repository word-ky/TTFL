"""Compact local T021 evidence bundle; large verified arrays stay in remote runs."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/t021_frozen_representation_observability'
RUNS=dict(extraction='20260914-183139-ttfl-t021-extract-gpu1',phase_a='20260914-183528-ttfl-t021-phase-a',matched='20260914-183911-ttfl-t021-matched',scoring='20260914-184639-ttfl-t021-score',verification='20260914-185130-ttfl-t021-verify-gpu1')
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open()))
def rr(v,scale=1):return f'{min(v)*scale:.3f}–{max(v)*scale:.3f}'


def main():
    folders={stage:ROOT/f'research_log/t021_receipts/{run}/artifacts/t021_frozen_representation_observability' for stage,run in RUNS.items()}
    for stage,folder in folders.items():
        target=OUT if stage in ('scoring','verification') else OUT/stage;target.mkdir(parents=True,exist_ok=True)
        for path in folder.iterdir():
            if path.is_file():shutil.copyfile(path,target/path.name)
    verified=load(OUT/'independent_verification.json');assert verified['status']=='PASS'
    summary=load(OUT/'summary.json');gates=load(OUT/'diagnostic_gates.json');assert summary['diagnosis']==gates['diagnosis']=='T021-M'
    for stage,n in [('extraction',125),('phase_a',127),('matched',127),('scoring',130)]:assert f'Ran {n} tests' in (folders[stage].parent.parent/'train.log').read_text()
    for stage in RUNS:assert 'exit_code=0' in (folders[stage].parent.parent/'train.log').read_text()
    for r in load(OUT/'remote_artifact_manifest.json'):
        relative=r['path'].split('/runs/',1)[1];local=ROOT/'research_log/t021_receipts'/relative
        if local.exists():assert sha(local)==r['sha256']
        else:assert local.suffix=='.npz'
    checked=[]
    for task in ('t015','t016r','t017r','t018r2','t019','t020','t020r','t020r2'):
        for r in load(ROOT/f'research_log/{task}_artifact_manifest.json'):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    save(OUT/'upstream_local_manifest_verification.json',dict(status='PASS',entries=len(checked),files=checked))
    matched=rows(OUT/'matched_aggregate.csv');actual=rows(OUT/'actual_aggregate.csv');paired=rows(OUT/'paired_regret_context.csv')
    contexts=['brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
    table='| Context | P matched % | L matched % | H matched % | P real % | L real % | H real % |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for t in contexts:
        values=[rr([float(r['capture_median']) for r in matched if r['representation']==rep and r['context']==t],100) for rep in ('P','L','H')]
        values+=[rr([float(r['capture']) for r in actual if r['representation']==rep and r['policy']=='oracle' and r['context']==t],100) for rep in ('P','L','H')]
        table+='| '+t+' | '+' | '.join(values)+' |\n'
    effects='| Representation/context | Mean regret reduction vs P, % | Context vs clean@same-state reduction, % | Context paired better clients /100 |\n|---|---:|---:|---:|\n'
    for rep in ('L','H'):
        for t in contexts:
            baseline=[r for r in paired if r['representation']==rep and r['context']==t and r['comparison']=='vs_P']
            control=[r for r in paired if r['representation']==rep and r['context']==t and r['comparison']=='context_vs_clean_same_state']
            effects+=f"| {rep}/{t} | {rr([float(r['relative_mean_reduction']) for r in baseline],100)} | {rr([float(r['relative_mean_reduction']) for r in control],100)} | {rr([float(r['clients_better']) for r in control])} |\n"
    proto=gzload(OUT/'phase_a/prototype_diagnostics.json.gz');solver=gzload(OUT/'phase_a/actual_solver_receipts.json.gz');nullcells=gzload(OUT/'matched/matched_cells.json.gz')
    diagnostic={rep:dict(rank_range=[min(r['rank'] for r in proto if r['representation']==rep),max(r['rank'] for r in proto if r['representation']==rep)],
        condition_range=[min(r['condition'] for r in proto if r['representation']==rep),max(r['condition'] for r in proto if r['representation']==rep)],
        actual_max_KKT=max(r['direct_KKT'] for r in solver if r['representation']==rep),matched_max_KKT=max(r['max_KKT'] for r in nullcells if r['representation']==rep)) for rep in ('L','H')}
    save(OUT/'representation_diagnostic_summary.json',diagnostic)
    now=datetime.now().astimezone().isoformat(timespec='seconds');ef=load(OUT/'extraction/extraction_freeze.json');af=load(OUT/'phase_a/phaseA_choices_freeze.json');mf=load(OUT/'matched/matched_freeze.json')
    report=f'''# T021 — T021-M with CTX+ for both frozen representations

{now}. Lead `7b5bb34`. Scoring runtime `{summary['runtime']}`, run `{RUNS['scoring']}`; independent verification runtime `{verified['runtime']}`, run `{RUNS['verification']}`. All five stages exit0. Final full suite **130 tests PASS**.

## Decision

**T021-M. L and H pass REP-MATCH-A but both fail REP-REAL-A.** L matched passes3/4shifts (Blur fails); H matched passes4/4. Both real paths pass onlyDark1/4. Both inherited real regression-safety checks pass. Neither representation passes REP-REGRET-A versus historical posterior CLS:0/4 shifts satisfy all bank/salt conditions.

**CTX+: L passes3/4 (Noise fails), H passes4/4.** Because both representations share the T021-M diagnosis and both context gates pass, the CTX+ flag is unambiguous without choosing a preferred representation. **REP-SRC-A is NOT_EXECUTED** for both, since neither real path qualifies. Source choices were frozen in Phase-A but not scored; no SRC success/failure is assigned.

T021 therefore does not establish that raw logits or512-D classifier-input features alone close the real K20 gap. The matched result shows that the fixed first-moment observer can be useful under its own class-conditional sample model. Context-specific prototypes outperform clean-image prototypes evaluated under the same frozen state, establishing a need for context-matched observation geometry within this tested model. That control improvement does not imply the80% real-capture target has been met. These results do not show that all task information is absent from H; only the specified raw Euclidean class-prototype observer was tested.

No architecture/operator change, learned projection/head, whitening, ridge, temperature, SSL/TTT, gradient step or federation was used. The BER branch remains closed.

## Inherited gain-capture evidence

Every range spans both banks/all four salts; each context needs every one of its8rows to reach80%. Matched columns use the median across the128 exact-count replicas. P is the frozen T018R2 posterior-CLS baseline; its observations/models were not recomputed or tuned.

{table}

Gate implementation preserves T018's exact matched-median aggregation and real safety: every shifted oracle row must stay within0.5pp of BBSE-S-01, not a newly invented tolerance against P. The historical source clean rule would require no worse than zero−0.5pp, but source scoring was not authorized here. The same frozen P00 denominator, exact canonical tie order and opposite-half query counts are used throughout. Baseline P real integer numerators are independently reproduced exactly.

## Regret and context specificity

Positive reductions mean lower mean true-template regret. The gate additionally requires p90 regret no more than5% worse and the15% mean improvement in both banks/all salts for at least3/4shifts. Ranges below are effects across the8bank/salt rows; complete mean/p90, optimal-set agreement, prevalence L1 and client better/equal/worse counts are in the CSVs.

{effects}

Clean@same-state control uses clean other-client images passed through state-t while the target remains shift-t passed through state-t. Only prototype calibration context changes. It is a diagnostic fixed policy, never a per-client prototype selector. Client comparisons aggregate each client's two held-out halves before counting better/equal/worse. Salts/halves reuse the same support and should not be interpreted as independent resampling trials. A zero baseline regret is not counted as a relative15% reduction.

## Execution and numerical validity

Frozen extraction: `{RUNS['extraction']}`, runtime `{ef['runtime']}`, {ef['forward_calls']}GPU support-batch forwards, max historical logit difference {ef['max_logit_abs']}. Canonical H is512-D fc1/ReLU output; L is the exact10-D final linear output. No preprocessing/normalization changes. Model and fast-state hashes and support/calibration/query disjointness pass.

Phase-A: `{RUNS['phase_a']}`, runtime `{af['runtime']}`,4000 prototype cells,6000 actual CLS solutions and48000 exact state/argmax choices. The other99clients' labels calibrate each target's prototypes; target-i labels contribute zero to its own M or mean. Complete actual choices are frozen before target composition scoring. Matched stage: `{RUNS['matched']}`, runtime `{mf['runtime']}`,256000 certified solutions/2048000 choices, exactly128 draws preserving each target's20-label counts. Namespace is full big-endian SHA256(`T021|matched|representation|client|bank|context|replica`) into PCG64.

Both L/H prototype matrices have rank ranges {diagnostic['L']['rank_range']}/{diagnostic['H']['rank_range']}; condition ranges L={diagnostic['L']['condition_range']}, H={diagnostic['H']['condition_range']} are diagnostic only. The existing solver forms a10×10 Gram matrix; no512×512 covariance is built. Maximum actual KKT: L={diagnostic['L']['actual_max_KKT']:.6g}, H={diagnostic['H']['actual_max_KKT']:.6g}; maximum matched KKT: L={diagnostic['L']['matched_max_KKT']:.6g}, H={diagnostic['H']['matched_max_KKT']:.6g}. Original acceptance tolerances hold.

Actual solves used zero fallbacks. Matched solves used9 cycle-only exhaustive-face fallbacks, all in L, listed in `matched/solver_fallbacks.json`. Each was independently re-enumerated with zero reference pi/objective difference; no fallback was added for any other error.

## Independent verification and artifacts

The separate verifier calls the original model forward with a pre-hook on the final classifier rather than the extractor helper: **2000 historical samples**, max logit and H difference0, same argmax, unchanged model/states. It reconstructs280 target-excluded prototype cells spanning both representations, all6000 target means and48000 Phase-A states/argmax masks; replays4000 matched CLS solutions with pi difference0 and32000 sampled template choices; verifies every matched draw's exact class counts/target exclusion and all9 fallbacks.

It reconstructs40000 exact true-template regrets,200 actual integer-count rows,10240 matched aggregate count rows, every reported gate, and all client paired counts. All runtime input hashes remain unchanged. Local cross-task manifests add **{len(checked)} unchanged upstream entries**. Independent verification adds100 GPU audit forwards; these are replay only, with no new query inference. Scoring adds zero model forwards.

The inherited dataset is the existing PFLlib CIFAR10 client split (44961 train /15039 query after merged-source partitioning),100clients and historical10% participation checkpoint. It is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining or new federation run occurred.

`extraction/`, `phase_a/`, `matched/` retain compact stage receipts; root contains final aggregate/paired/count tables, gates and independent verification. Large H/L, prototype, mean/pi/choice and matched NPZ arrays remain in the five canonical remote project run directories, with exact paths/sizes/hashes in `remote_artifact_manifest.json`; they are not claimed to have been copied into the local compact bundle. Logs and compact raw receipts are under `research_log/t021_receipts`. This keeps durable project evidence while avoiding another local D: space failure.

Return T021-M/CTX+ to Lead for the next bounded research decision. Do not infer authorization for whitening, learned features/heads, SSL writing or federation.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f'''# T021 complete — T021-M / CTX+; source not executed

{now}. Lead7b5bb34;scoring{summary['runtime']}/run{RUNS['scoring']};verification{verified['runtime']}/run{RUNS['verification']}.130testsPASS/all5stagesexit0. L MATCH3/4,H4/4;bothREAL1/4DarkFAIL;bothREGRET0/4;CTX L3/4,H4/4PASS. SourceNOT_EXECUTED because neitherREALpasses. Independent2000historicalsamples/280prototypes/6000means/48000choices/4000CLS/9fallbacks/40000regrets/allintegercounts/gates and {len(checked)}upstreamentriesPASS. MaxhistoricalH/Ldiff0. Read results/t021_frozen_representation_observability/RESULTS.md. RawlargeNPZremoteonly, manifestpaths/hashesrecorded; compactreceiptslocal. AwaitLead; no whitening/learnedhead/SSL/operator/FL or BER revival.
'''
    for name in ('HANDOFF.md','T021_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\nInterpretation: raw upstream dimensionality alone does not close real-to-other-client observation mismatch; context-specific geometry is useful under the fixed same-state control. Source was frozen but not scored. No next stage started.\n',encoding='utf-8')
    save(ROOT/'research_log/t021_delivery.json',dict(timestamp=now,lead='7b5bb34',runtime=summary['runtime'],verification_runtime=verified['runtime'],runs=RUNS,diagnosis='T021-M',CTX=dict(L=True,H=True),SRC='NOT_EXECUTED',
        results=OUT.relative_to(ROOT).as_posix(),large_arrays='remote canonical run paths in remote_artifact_manifest.json',next='Await Lead bounded instruction'))
    files=[p for p in (ROOT/'research_log/t021_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t021_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
