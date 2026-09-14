"""Persist compact T022 receipts, verified comparisons and delivery manifests."""
import argparse,csv,gzip,hashlib,json,shutil
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DIR='t022_state_response_semantics'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open()))
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def span(v,scale=1):return f'{min(v)*scale:.3f} to {max(v)*scale:.3f}'

def main():
    ap=argparse.ArgumentParser()
    for stage in ('extraction','phase_a','matched','scoring','verification'):ap.add_argument('--'+stage,required=True)
    runs=vars(ap.parse_args());out=ROOT/'results'/DIR
    folders={stage:ROOT/f'research_log/t022_receipts/{run}/artifacts/{DIR}' for stage,run in runs.items()}
    for stage,folder in folders.items():
        dest=out if stage in ('scoring','verification') else out/stage;dest.mkdir(parents=True,exist_ok=True)
        for path in folder.iterdir():
            if path.is_file():shutil.copyfile(path,dest/path.name)
        assert 'exit_code=0' in (folder.parent.parent/'train.log').read_text()
    assert 'Ran 133 tests' in (folders['scoring'].parent.parent/'train.log').read_text()
    v=load(out/'independent_verification.json');s=load(out/'summary.json');g=load(out/'diagnostic_gates.json');assert v['status']=='PASS'
    for r in load(out/'remote_artifact_manifest.json'):
        local=ROOT/'research_log/t022_receipts'/r['path'].split('/runs/',1)[1]
        if local.exists():assert sha(local)==r['sha256']
        else:assert local.suffix=='.npz'
    checked=[]
    for task in ('t015','t016r','t017r','t018r2','t019','t020','t020r','t020r2','t021'):
        for r in load(ROOT/f'research_log/{task}_artifact_manifest.json'):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    save(out/'upstream_local_manifest_verification.json',dict(status='PASS',entries=len(checked),files=checked))
    ma=rows(out/'matched_aggregate.csv');aa=rows(out/'actual_aggregate.csv');pa=rows(out/'paired_regret_context.csv');contexts=['brightness_dark','contrast_low','gaussian_noise','gaussian_blur']
    decision='| Response | MATCH contexts | REAL contexts | Regression safety | REGRET vs P | CTX | SRC |\n|---|---:|---:|---|---:|---:|---|\n'
    for rep in ('L','H'):
        q=g['representations'][rep]
        decision+=f"| Phi_{rep} | {sum(q['matched_pass'].values())}/4 | {sum(q['real_pass'].values())}/4 | {q['real_regression_safety']} | {sum(q['RESP_REGRET_A']['context_pass'].values())}/4 | {sum(q['RESP_CTX_A']['context_pass'].values())}/4 | {q['RESP_SRC_A']} |\n"
    capture='| Context/observer | Matched capture %, min to max | Real capture %, min to max |\n|---|---:|---:|\n'
    for t in contexts:
        for rep in ('P','T021_L','T021_H','L','H'):
            capture+=f"| {t}/{rep} | {span([float(r['capture_median']) for r in ma if r['representation']==rep and r['context']==t],100)} | {span([float(r['capture']) for r in aa if r['representation']==rep and r['policy']=='oracle' and r['context']==t],100)} |\n"
    effect='| Response/context/comparison | Mean regret reduction % | New p90 regret | Baseline p90 regret | Clients better /100 |\n|---|---:|---:|---:|---:|\n'
    for rep in ('L','H'):
        for t in contexts:
            for comp in ('vs_P','vs_T021_absolute','context_vs_clean_prototype'):
                group=[r for r in pa if r['representation']==rep and r['context']==t and r['comparison']==comp]
                effect+=f"| {rep}/{t}/{comp} | {span([float(r['relative_mean_reduction']) for r in group],100)} | {span([float(r['new_p90']) for r in group])} | {span([float(r['baseline_p90']) for r in group])} | {span([float(r['clients_better']) for r in group])} |\n"
    proto=gzload(out/'phase_a/prototype_diagnostics.json.gz');solver=gzload(out/'phase_a/actual_solver_receipts.json.gz');null=gzload(out/'matched/matched_cells.json.gz')
    diag={rep:dict(rank_range=[min(r['rank'] for r in proto if r['representation']==rep),max(r['rank'] for r in proto if r['representation']==rep)],condition_range=[min(r['condition'] for r in proto if r['representation']==rep),max(r['condition'] for r in proto if r['representation']==rep)],actual_max_KKT=max(r['direct_KKT'] for r in solver if r['representation']==rep),matched_max_KKT=max(r['max_KKT'] for r in null if r['representation']==rep)) for rep in ('L','H')}
    save(out/'representation_diagnostic_summary.json',diag)
    now=datetime.now().astimezone().isoformat(timespec='seconds');ef=load(out/'extraction/extraction_freeze.json');af=load(out/'phase_a/phaseA_choices_freeze.json');mf=load(out/'matched/matched_freeze.json')
    stop='The frozen first-moment observer branch stops here under the predeclared T022-M/N rule. No further observer trick, rescue or next task was started.' if s['diagnosis'] in ('T022-M','T022-N') else 'Return the verified diagnosis to Lead for the next bounded decision; no next stage was started.'
    report=f'''# T022 — {s['diagnosis']}

{now}. Lead `06647a0`. All five stages exit0; final full suite **133 tests PASS**. Independent verification **PASS**.

## Frozen decision

{decision}

Flags per response: {json.dumps(g['flags'])}. The diagnosis uses the prespecified preference for response logits before response H. For M/N there is no arbitrary winning representation: per-representation CTX flags remain explicit. Source choices were frozen for both representations, but source scoring is authorized only when RESP-REAL-A passes.

{stop}

## Observation and controls

Phi_L is40-D and Phi_H is2048-D. Each concatenates raw float32 differences in fixed order Dark−clean, Contrast−clean, Noise−clean, Blur−clean for the same sample and bank. There is no normalization, scaling, projection, whitening, ridge, learned metric, gradient update, new state/operator or federation. The target's exact same signature mean is used for oracle, source and clean-prototype policies; only the prototype context and predeclared utility context differ. Source uses the frozen T009 decision. Clean-prototype control uses clean other-client signatures and target-context utility. Offline other-client labels calibrate M; target-i contributes zero samples or labels to its own M/estimator.

The checkpoint, banks, K20 support IDs, deterministic corruptions, target-excluded T014 utility templates, solver tolerances and canonical exact tie order are unchanged. Phase-A froze6000 estimates and48000 choices before composition/query scoring. Matched draws preserve exact target class counts, use only the other99clients, and use full big-endian SHA256(`T022|matched|L-or-H|client|bank|context|replica`) into PCG64; actual seed tokens are `L` and `H`. R=128 is fixed, with256000 solutions and2048000 matched choices.

## Capture against frozen baselines

Ranges span both banks/all four salts. Matched values are medians over128 exact-count replicas. P and T021 absolute L/H are historical baselines, not retrained or refitted. Their real integer numerators are reproduced exactly before comparison.

{capture}

RESP-MATCH-A requires matched median capture>=80% in every bank/salt for at least3/4shifts. RESP-REAL-A requires real capture>=80% under the same aggregation and every shifted row>=BBSE-S-01−0.5pp. The source clean safety rule remains>=zero−0.5pp. These are the original T018/T021 expressions. Opposite-half integer query counts and P00 denominators are unchanged. No query forward is added.

## Paired regret and context specificity

{effect}

Positive reduction means less true-template regret. RESP-REGRET-A vs P and RESP-CTX-A vs clean-prototype require>=15% mean reduction with p90 no more than5% worse in both banks/all salts for>=3/4shifts. A zero baseline mean is noneligible for relative improvement. The T021 absolute comparison is diagnostic only, never an additional gate. Client better/equal/worse counts aggregate the two held-out halves per client. Complete mean/p90, exact regret, L1, optimal/canonical agreement and paired counts are retained in CSV/gzip receipts. Salts and halves reuse support and are not independent sample replications.

## Numerical and independent verification

Extraction reused{ef['reused_paths']} historical paths and computed{ef['new_forward_calls']} missing A6000 paths; all5000 logical state paths/100000 sample outputs are available through old+new artifacts. Model/state hashes are unchanged and support/calibration/query IDs are disjoint. Independent reconstruction checks{v['independent_signature_vectors']} complete response vectors directly from original cached and newly frozen state outputs, with exact arithmetic equality; checks{v['historical_cached_samples']} cached sample arrays by their frozen hashes. This is a cache/state-output audit, not a new model-forward replay. T021's independent original-model replay remains unchanged upstream evidence.

Independent checks: {v['target_excluded_prototype_checks']} prototype cells, {v['actual_CLS_replays']} actual CLS replays, {v['actual_mean_reconstructions']} target means, {v['all_phase_a_state_and_argmax_checks']} actual choice/mask checks, {v['matched_CLS_replays']} matched CLS replays, {v['matched_sample_choice_checks']} sampled matched choice checks, all256000 draw count/exclusion checks, {len(v['all_fallback_reenumerations'])} exhaustive fallback re-enumerations, {v['exact_true_regret_reconstructions']} exact regrets, {v['actual_integer_count_rows']} actual integer rows and {v['matched_integer_count_rows']} matched integer rows. Every gate and paired count is reconstructed. Maximum replay pi difference={v['max_pi_difference']}. All input hashes and{len(checked)} local upstream manifest entries pass.

Prototype numerical diagnostics (rank/condition are descriptive only): `{json.dumps(diag)}`. The solver forms only a10×10 Gram matrix. No2048×2048 covariance is constructed. Existing cycle-only exhaustive-face handling is unchanged; no new tolerance or fallback rule was introduced.

## Runtime and artifacts

- Extraction `{ef['runtime']}` / `{runs['extraction']}`.
- Phase-A `{af['runtime']}` / `{runs['phase_a']}`.
- Matched `{mf['runtime']}` / `{runs['matched']}`.
- Scoring `{s['runtime']}` / `{runs['scoring']}`.
- Verification `{v['runtime']}` / `{runs['verification']}`.

Compact stage receipts are under extraction/, phase_a/, matched/ and the result root; train logs and raw compact receipts are in research_log/t022_receipts. Large NPZ arrays remain in canonical remote project runs with full paths, byte sizes and SHA256 in remote_artifact_manifest.json. This report does not claim local copies of those arrays. The inherited PFLlib CIFAR10 split has44961train/15039query samples after merged-source client partitioning,100clients and the historical10% participation checkpoint. It is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining occurred.
'''
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f"# T022 complete — {s['diagnosis']}\n\n{now}. Lead06647a0. 133testsPASS/all5stagesexit0/independentPASS. Flags {json.dumps(g['flags'])}. Read results/{DIR}/RESULTS.md.\n\n"+decision+'\n'+stop+f"\nScoring runtime{s['runtime']}, verifier{v['runtime']}. Large NPZ remote only with path/hash manifest; compact receipts local.\n"
    for name in ('HANDOFF.md','T022_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+capture+'\nComplete effect sizes, limitations and runtime hashes are in RESULTS.md.\n',encoding='utf-8')
    save(ROOT/'research_log/t022_delivery.json',dict(timestamp=now,lead='06647a0',runtime=s['runtime'],verification_runtime=v['runtime'],runs=runs,diagnosis=s['diagnosis'],flags=g['flags'],results=out.relative_to(ROOT).as_posix(),next=stop))
    files=[p for p in (ROOT/'research_log/t022_receipts').rglob('*') if p.is_file()]+[p for p in out.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t022_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(hand)

if __name__=='__main__':main()
