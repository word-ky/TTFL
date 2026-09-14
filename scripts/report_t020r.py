"""Independent label-free T020R convergence replay and sealed-stop delivery."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.constrained_prevalence import ActiveSetCLS,direct_kkt
from src.context.exact_template_lookup import compile_template
from src.context.bootstrap_regret import shared_integer_weights,exact_utilities_from_weights,expected_regret_objects
from src.context.matched_channel import exact_utilities
RUN='20260914-163857-ttfl-t020r'
RAW=ROOT/f'research_log/t020r_receipts/{RUN}/artifacts/t020r_mc_convergence'
OUT=ROOT/'results/t020r_mc_convergence'
TEST_COUNT=121;VERIFY_ONLY=False
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=[f'T013-S{i}' for i in range(4)]
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    incomplete=not (RAW/'summary.json').exists()
    summary=load(RAW/'runtime.json' if incomplete else RAW/'summary.json')
    assert f'Ran {TEST_COUNT} tests' in (RAW.parent.parent/'train.log').read_text()
    if not incomplete:assert all(sha(RAW/name)==h for name,h in load(RAW/'phaseB_choices_freeze.json')['hashes'].items())
    for p in RAW.iterdir():
        if p.is_file() and p.suffix!='.npz':shutil.copyfile(p,OUT/p.name)
    p20=ROOT/'research_log/t020_receipts/20260914-154818-ttfl-t020/artifacts/t020_bootstrap_expected_regret'
    old=dict(np.load(p20/'bootstrap_arrays.npz'))
    legacy={(r['client'],r['bank'],r['context'],r['salt'],r['train_half']):r for r in gzload(p20/'utility_means_exact.json.gz')}
    p19=ROOT/'research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity'
    prob=np.load(p19/'real_decomposition.npz')['probabilities']
    p16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    cal=np.load(p16/'calibration_counts.npz');mat=cal['soft_numerator']/cal['soft_denominator'][...,None,:]
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(ROOT/'results/t014_class_conditional_factorization/class_templates.json.gz')['rows']}
    allrows=[];solved=legacychecks=blockchecks=0;maxpi=maxobj=0.
    clients=[i for i in range(100) if (RAW/f'client_{i:03d}_accumulators.json.gz').exists()]
    for i in clients:
        sample=dict(np.load(RAW/f'client_{i:03d}_verification.npz'))
        np.testing.assert_array_equal(sample['replica_ids'],[256,1024,2048,4095])
        data=gzload(RAW/f'client_{i:03d}_accumulators.json.gz');allrows+=data['records']
        bykey={(r['bank'],r['context'],r['salt'],r['train_half']):r for r in data['records']}
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=i,bi,ti;solver=ActiveSetCLS(mat[loc]);pv=old['pi'][loc]
                for j,k in enumerate(sample['replica_ids']):
                    seed=int(hashlib.sha256(f'T020|{i}|{b}|{t}|{k}'.encode()).hexdigest(),16)
                    pos=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=20)
                    np.testing.assert_array_equal(pos,sample['positions'][bi,ti,j])
                    q=prob[loc][pos].mean(0);np.testing.assert_array_equal(q,sample['q'][bi,ti,j])
                    pi,v=solver.solve(q);saved=sample['pi'][bi,ti,j]
                    pe=float(np.max(np.abs(pi-saved)));oe=float(abs(v['objective']-.5*np.sum((mat[loc]@saved-q)**2)))
                    assert pe<=1e-7 and oe<=1e-10 and direct_kkt(mat[loc],q,saved)<=1e-10
                    maxpi=max(maxpi,pe);maxobj=max(maxobj,oe);solved+=1
                w,D=shared_integer_weights(pv)
                means={n:[sum((Fraction(float(x)) for x in pv[:n,y]),Fraction())/n for y in range(10)] for n in (128,256)}
                for si,s in enumerate(S):
                    for h in (0,1):
                        slot=2*si+h;table=templates[s,b,h,i][ti];coeff,td=compile_template(table)
                        nums=exact_utilities_from_weights(w,coeff);states,masks,replay=expected_regret_objects(nums,D*td,int(old['point'][loc][slot]))
                        prev=legacy[i,b,t,s,h]
                        # NumPy/libm log differs by one ULP across Windows/Linux; entropy is diagnostic only.
                        assert abs(replay['vote_entropy']-prev['vote_entropy'])<=2e-15
                        assert all(replay[k]==prev[k] for k in replay if k!='vote_entropy'),{k:(replay[k],prev[k]) for k in replay if k!='vote_entropy' and replay[k]!=prev[k]}
                        np.testing.assert_array_equal(states,old['replica_choices'][loc][:,slot]);np.testing.assert_array_equal(masks,old['replica_argmax_masks'][loc][:,slot])
                        for n in (128,256):
                            values,args=exact_utilities(means[n],table)
                            assert values==[Fraction(x) for x in prev[f'mean_utility{n}_exact']] and args[0]==prev[f'BER{n}']
                        legacychecks+=1
                        row=bykey[b,t,s,h];assert row['BER256']==prev['BER256'] and row['point']==prev['point_state']
                        for name,block in row['blocks'].items():
                            values=[Fraction(x) for x in block['means_exact']];mx=max(values);state=values.index(mx)
                            assert state==block['state']
                            den=int(block['mean_denominator']);sums=[int(x) for x in block['utility_numerator_sums']]
                            assert [Fraction(x,den) for x in sums]==values
                            regrets=[Fraction(int(block['max_utility_numerator_sum'])-x,den) for x in sums]
                            assert regrets==[Fraction(x) for x in block['regrets_exact']] and regrets[state]==min(regrets)
                            blockchecks+=1
                        for name,parts in [('H1',['Q0','Q1']),('H2',['Q2','Q3']),('ALL',['H1','H2'])]:
                            for j in range(5):
                                assert Fraction(row['blocks'][name]['means_exact'][j])==sum(Fraction(row['blocks'][p]['means_exact'][j]) for p in parts)/2
                                assert row['blocks'][name]['votes'][j]==sum(row['blocks'][p]['votes'][j] for p in parts)/2
                        assert row['BER4096']==row['blocks']['ALL']['state']
        if i%20==19:print('T020R_VERIFIED_CLIENTS',i+1,flush=True)
    if incomplete:
        report_solver_stop(summary,clients,allrows,solved,legacychecks,blockchecks,maxpi,maxobj)
        return
    def agree(rows,a,b):return sum(r['blocks'][a]['state']==r['blocks'][b]['state'] for r in rows)/len(rows)
    global_values={f'{a}_{b}':agree(allrows,a,b) for a,b in [('H1','H2'),('H1','ALL'),('H2','ALL')]}
    assert global_values==summary['global_agreement']
    for row in csv.DictReader((RAW/'mc_convergence.csv').open()):
        group=[r for r in allrows if (row['bank']=='ALL' or r['bank']==row['bank']) and (row['context']=='ALL' or r['context']==row['context'])]
        assert agree(group,row['block1'],row['block2'])==float(row['agreement'])
    groups=[agree([r for r in allrows if r['bank']==b and r['context']==t],'H1','H2') for b in B for t in C]
    pooled=[agree([r for r in allrows if r['context']==t],'H1','H2') for t in C]
    passed=global_values['H1_H2']>=.99 and global_values['H1_ALL']>=.995 and global_values['H2_ALL']>=.995 and min(groups)>=.985 and min(pooled)>=.99
    assert passed==summary['passed']
    checked=[]
    for name in ('t015','t016r','t017r','t018r2','t019','t020'):
        for row in load(ROOT/f'research_log/{name}_artifact_manifest.json'):
            assert sha(ROOT/row['path'])==row['sha256'];checked.append(row)
    verify=dict(status='PASS',independent_position_and_CLS_replays=solved,replicas=[256,1024,2048,4095],max_pi_error=maxpi,max_objective_error=maxobj,
        legacy_all_aggregate_replays=legacychecks,independent_legacy_fraction_mean_replays=legacychecks,block_exact_choices=blockchecks,final_BER4096_choices=len(allrows),
        all_quarter_half_global_agreements_replayed=True,convergence_pass=passed,source_manifest_entries_unchanged=len(checked),source_files=checked,
        target_composition_or_true_utilities_parsed=False,query_counts_or_metrics_parsed=False)
    save(OUT/'independent_verification.json',verify)
    if VERIFY_ONLY:print('T020R_INDEPENDENT_REPLAY_PASS');return
    if passed:print('T020R_INDEPENDENT_PREPARATION_PASS');return
    assert summary['status']=='T020R-MC2'
    phase=load(RAW/'solver_verification.json');now=datetime.now().astimezone().isoformat(timespec='seconds')
    table='| Context | A H1/H2 % | B H1/H2 % | Pooled % |\n|---|---:|---:|---:|\n'
    for t in C:
        vals=[next(r['agreement'] for r in summary['bank_context'] if r['bank']==b and r['context']==t) for b in B]
        pool=next(r['agreement'] for r in summary['pooled_context'] if r['context']==t)
        table+=f'| {t} | {100*vals[0]:.3f} | {100*vals[1]:.3f} | {100*pool:.3f} |\n'
    transition=sum(r['BER256']!=r['BER4096'] for r in allrows)
    report=f'''# T020R-MC2 — fixed R4096 remains insufficiently stable

{now}. Lead `f0807dd`; runtime `{summary['runtime']}`; run `{RUN}`. 121 tests PASS. Preparation ended with the prescribed sealed stop after {summary['seconds']:.2f}s.

## Frozen convergence decision

| Comparison | Agreement | Required |
|---|---:|---:|
| H1 vs H2 | {100*global_values['H1_H2']:.4f}% | 99% |
| H1 vs ALL | {100*global_values['H1_ALL']:.4f}% | 99.5% |
| H2 vs ALL | {100*global_values['H2_ALL']:.4f}% | 99.5% |

Each bank/context also requires 98.5%, and each context pooled across banks requires 99%:

{table}

H1/H2 differ in {summary['H1_H2_disagreements']}/8000 slots, spanning {summary['unique_clients']} clients and {summary['unique_cells']} client/bank/context cells. These eight template slots reuse each support stream and are correlated. BER256→BER4096 changes {transition}/8000 slots; this is diagnostic only. Stable ALL top-two margin p05/median/p95: {summary['stable_margin_quantiles']}; unstable: {summary['unstable_margin_quantiles']} (fractional utility units). Quarter-pair comparisons and utility/vote max/p95 differences are in `mc_convergence.csv`.

**T020R-MC2. BER-REGRET-A, BER-CAP-A, BER-SRC-A remain NOT_EXECUTED. No T020-R/C/F/X scientific diagnosis is assigned.** True support composition, true utilities and query outcomes remain unopened. This result concerns finite-MC policy convergence, and cannot establish whether BER helps true regret, accuracy, or the neutral operator. No further R extension, seed retry, alternate policy or feature/SSL/FL task was started.

## Execution and verification

Same deterministic T020 SHA256→PCG64 namespace; immutable original replicas0..255 plus replicas256..4095. No labels, gradients or new model forwards. Parallelism used four independent client processes, each with the existing two BLAS threads. Logical 4,096,000 bootstrap replicas include 3,840,000 new CLS solves and 256,000 immutable prefix solutions. Maximum KKT {phase['max_KKT']:.6g}; maximum simplex-sum error {phase['max_sum_error']:.6g}; maximum active-set updates {phase['max_updates']}. No solver tolerance changed.

Independent local replay passed all 8000 legacy aggregate receipts and separately reconstructed their first128/all256 means from exact binary-rational pi values. All original canonical replica states/masks replayed. Replica256,1024,2048,4095 positions and 4000 CLS solutions replayed across every client/bank/context; maximum pi difference {maxpi:.6g}, objective difference {maxobj:.6g}. Every H1/H2/ALL/final choice and all quarter choices replayed from saved exact means and integer sums ({blockchecks} block checks), including quarter-to-half-to-ALL mean/vote composition. All reported agreement fractions and the five-part convergence gate independently reproduced. All {len(checked)} upstream manifest entries remain unchanged.

The new full streams have per-cell pi/q/position SHA256 receipts. Compact sampled pi/q/position NPZ files are retained locally in `research_log/t020r_receipts/{RUN}/artifacts/t020r_mc_convergence` and in the remote canonical run. Exact block sums, means, regrets, votes and final choices are retained in each `client_NNN_accumulators.json.gz`; no full new 4096-replica float arrays are claimed to be stored. The original full256 arrays are unchanged. `phaseB_choices_freeze.json` binds the new unlabeled outputs. No scientific-table placeholders were created.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f'''# T020R-MC2 — sealed convergence stop

{now}. Lead f0807dd; runtime {summary['runtime']}; run {RUN}. 121 tests PASS; 3,840,000 new CLS plus 256,000 immutable prefix replicas. H1/H2 {100*global_values['H1_H2']:.4f}%, H1/ALL {100*global_values['H1_ALL']:.4f}%, H2/ALL {100*global_values['H2_ALL']:.4f}%. Frozen convergence gate FAIL; {summary['H1_H2_disagreements']}/8000 H1/H2 disagreements. Independent4000 CLS/8000legacy/56000block choices and {len(checked)} upstream manifest entries PASS. Labels/true utilities/query outcomes remain sealed; all scientific gates NOT_EXECUTED; no T020-R/C/F/X diagnosis. Read results/t020r_mc_convergence/RESULTS.md. Await Lead; no further R/seed/policy changes or next stage.
'''
    for name in ('HANDOFF.md','T020R_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table,encoding='utf-8')
    save(ROOT/'research_log/t020r_delivery.json',dict(timestamp=now,lead='f0807dd',runtime=summary['runtime'],run=RUN,status='T020R-MC2',global_agreement=global_values,results=OUT.relative_to(ROOT).as_posix(),next='Await Lead; remain sealed'))
    files=[p for p in (ROOT/'research_log/t020r_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t020r_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


def report_solver_stop(runtime,clients,rows,solved,legacychecks,blockchecks,maxpi,maxobj):
    diagnosis=load(OUT/'solver_cycle_diagnosis.json');assert diagnosis['missing_clients']==[i for i in range(100) if i not in clients]
    checked=[]
    for name in ('t015','t016r','t017r','t018r2','t019','t020'):
        for row in load(ROOT/f'research_log/{name}_artifact_manifest.json'):
            assert sha(ROOT/row['path'])==row['sha256'];checked.append(row)
    cells=[cell for i in clients for cell in gzload(RAW/f'client_{i:03d}_accumulators.json.gz')['cells']]
    verify=dict(status='PARTIAL_RECEIPTS_PASS_COMPLETE_TASK_BLOCKED',completed_clients=clients,missing_clients=diagnosis['missing_clients'],
        independent_position_and_CLS_replays=solved,max_pi_error=maxpi,max_objective_error=maxobj,legacy_all_aggregate_replays=legacychecks,
        independent_legacy_fraction_mean_replays=legacychecks,block_exact_choices=blockchecks,partial_BER4096_choices=len(rows),source_manifest_entries_unchanged=len(checked),source_files=checked,
        labels_parsed=False,query_outcomes_parsed=False,complete_MC_gate='NOT_EXECUTED',cross_platform_entropy_atol=2e-15,entropy_is_diagnostic_only=True)
    save(OUT/'independent_verification.json',verify)
    freeze=dict(status='PARTIAL_ONLY_NOT_PHASE_B_CERTIFICATION',completed_clients=clients,missing_clients=diagnosis['missing_clients'],labels_parsed=False,query_outcomes_parsed=False,
        hashes={p.name:sha(p) for p in RAW.iterdir() if p.is_file()})
    save(OUT/'partial_receipts_hashes.json',freeze)
    summary=dict(status='IMPLEMENTATION_STOP_SOLVER_CYCLE',runtime=runtime['runtime'],run=RUN,complete_clients=len(clients),complete_template_slots=len(rows),
        complete_MC_gate='NOT_EXECUTED',BER_REGRET_A='NOT_EXECUTED',BER_CAP_A='NOT_EXECUTED',BER_SRC_A='NOT_EXECUTED',scientific_diagnosis='NOT_ASSIGNED',
        preserved_new_CLS_solves=sum(c['new_CLS_solves'] for c in cells),preserved_logical_replicas=len(cells)*4096,max_KKT=max(c['max_KKT'] for c in cells),
        max_sum_error=max(c['max_sum_error'] for c in cells),max_updates=max(c['max_updates'] for c in cells),labels_parsed=False,query_outcomes_parsed=False,new_model_forwards=0)
    save(OUT/'summary.json',summary)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    table='| Client | Bank | Context | Replica | Reference KKT |\n|---:|---|---|---:|---:|\n'
    for r in diagnosis['failures']:table+=f"| {r['client']} | {r['bank']} | {r['context']} | {r['replica']} | {r['reference_KKT']:.6g} |\n"
    report=f'''# T020R implementation stop — reproducible CLS working-set cycles

{now}. Lead `f0807dd`; runtime `{runtime['runtime']}`; run `{RUN}`, exit1. **121 tests PASS, but the R4096 extension did not complete.**

The frozen `ActiveSetCLS` raises a working-set-cycle exception on deterministic replicas in clients64 and87. This is an observed solver limitation exposed by the longer stream. It is **not T020R-MC2**, because the complete convergence gate was never computed, and not T020-F or any scientific BER diagnosis. True support composition/utilities/query outcomes remain sealed. BER-REGRET-A, BER-CAP-A and BER-SRC-A remain NOT_EXECUTED.

## Reproduction

{table}

The local diagnostic replays only each missing client's stream until its first failure. Each case also fails with a fresh solver instance, so the failure does not require parallel scheduling or accumulated face-cache history. The recorded coordinate-removal/insertion trace revisits an identical active set; negative face coordinates are on the order of 1e-2, not a 1e-12 rounding violation. The independent exhaustive-face reference finds a feasible KKT-valid optimum for each same C/q. The convex CLS problem remains solvable; the current full-step face-removal/insertion procedure can cycle. Reference solutions are diagnostic only and were never substituted into a BER policy.

Exact C/q/positions/reference vectors and SHA256 are in the two `cycle_client_*.npz` diagnostic artifacts, retained locally/remotely with manifest hashes; human-readable traces and reference residuals are in `solver_cycle_diagnosis.json`. No solver code, tolerance, bootstrap seed, R, or policy was changed.

## Preserved partial computation

{len(clients)}/100 clients have complete per-client files ({len(rows)}/8000 template slots), including {summary['preserved_new_CLS_solves']:,} new CLS solutions plus their immutable first256 prefixes, or {summary['preserved_logical_replicas']:,} logical replicas. These counts exclude partial work in failed clients and the later diagnostic replay. Preserved completed cells have max KKT {summary['max_KKT']:.6g}, max sum error {summary['max_sum_error']:.6g}, maximum {summary['max_updates']} updates. They do not establish that all 4,096,000 target solves passed.

The main process printed62 completed futures before the first exception. Python's executor shutdown waited for already-submitted work, which left98 complete client files on disk. Those files are preserved; no missing client was imputed or omitted to manufacture an agreement score. No complete phaseB choice freeze, full convergence table, or final uncertainty table exists. `partial_receipts_hashes.json` identifies partial evidence only.

Independent replay verified {solved} sampled CLS/position cases spanning all four quarters for the completed clients; max pi error {maxpi:.6g}, objective error {maxobj:.6g}. All {legacychecks} completed-client legacy aggregate rows replay, with separately reconstructed binary-rational first128/all256 mean utilities and original replica canonical states/masks. Exact means, choices and other legacy fields remain exact; local Windows/Linux NumPy log differs by one ULP in diagnostic vote entropy, checked with absolute tolerance2e-15 only for this diagnostic. This changes neither solver nor policy tolerance. All {blockchecks} saved block choices and {len(rows)} partial ALL choices reconstruct from exact means and integer sums; quarter/half/ALL mean and vote composition agrees. All {len(checked)} upstream manifest entries remain unchanged. The complete4000-sample/8000-slot task verification is not claimed.

## Next bounded Lead decision

T020R explicitly freezes the T018R2 direct-RHS solver. Return the reproducible implementation blocker for a solver-specific repair instruction. A targeted next repair would need to remove the demonstrated active-set cycle while preserving the same convex objective/simplex, tolerances, seed stream and immutable first256 outputs, then replay these cases and the old baseline before resuming missing work. This report does not authorize or implement such a repair. No SSL/TTT, new semantic estimator/operator, context detector or FL work was started.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f'''# T020R implementation stop — CLS cycles, still sealed

{now}. Lead f0807dd; runtime {runtime['runtime']}; run {RUN}, exit1. 121testsPASS but only98/100clients complete. Missing64/87 reproduce deterministic ActiveSetCLS cycles; see results/t020r_mc_convergence/solver_cycle_diagnosis.json and RESULTS.md. Complete MC gate NOT_EXECUTED: not MC2, no scientific diagnosis. Labels/true utilities/query remain sealed. Independent3920sampledCLS/7840legacy/54880block checks PASS; {len(checked)}upstream manifests unchanged. Frozen solver unchanged. Await Lead solver-specific repair; do not rerun full task or silently substitute reference solver. Preserve all partial98clients and original T020 prefix.
'''
    for name in ('HANDOFF.md','T020R_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\nExhaustive-face references confirm the same convex problems have feasible optima; they were used only to diagnose the frozen solver, never as policy fallbacks.\n',encoding='utf-8')
    save(ROOT/'research_log/t020r_delivery.json',dict(timestamp=now,lead='f0807dd',runtime=runtime['runtime'],run=RUN,status=summary['status'],results=OUT.relative_to(ROOT).as_posix(),next='Await Lead solver-specific repair; remain sealed'))
    files=[p for p in (ROOT/'research_log/t020r_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t020r_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
