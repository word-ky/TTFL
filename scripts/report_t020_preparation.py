"""Label-free independent T020 bootstrap replay and MC-blocker report."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.constrained_prevalence import ActiveSetCLS,direct_kkt
from src.context.matched_channel import exact_utilities
RUN='20260914-154818-ttfl-t020';RAW=ROOT/f'research_log/t020_receipts/{RUN}/artifacts/t020_bootstrap_expected_regret';OUT=ROOT/'results/t020_bootstrap_expected_regret'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=[f'T013-S{i}' for i in range(4)]
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file() and p.suffix!='.npz':shutil.copyfile(p,OUT/p.name)
    summary=load(RAW/'summary.json');phase=load(RAW/'phaseA_verification.json')
    assert 'Ran 115 tests' in (RAW.parent.parent/'train.log').read_text()
    assert all(sha(RAW/name)==h for name,h in load(RAW/'phaseB_choices_freeze.json')['hashes'].items())
    arrays=dict(np.load(RAW/'bootstrap_arrays.npz'))
    p19=ROOT/'research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity'
    prob=np.load(p19/'real_decomposition.npz')['probabilities'];oldpoint=np.load(p19/'utility_choices.npz')['actual_choices']
    np.testing.assert_array_equal(arrays['point'],oldpoint)
    p16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    cal=np.load(p16/'calibration_counts.npz');mat=cal['soft_numerator']/cal['soft_denominator'][...,None,:]
    p14=ROOT/'results/t014_class_conditional_factorization'
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ctx] for ctx in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    exactrows=gzload(RAW/'utility_means_exact.json.gz');bykey={(r['client'],r['bank'],r['context'],r['salt'],r['train_half']):r for r in exactrows}
    solved=0;max_pi_error=max_objective_error=0.;meanchecks=0;replica_checks=0
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=(i,bi,ti);solver=ActiveSetCLS(mat[loc]);pv=arrays['pi'][loc]
                for k in (0,255):
                    seed=int(hashlib.sha256(f'T020|{i}|{b}|{t}|{k}'.encode()).hexdigest(),16)
                    pos=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=20);np.testing.assert_array_equal(pos,arrays['positions'][loc][k])
                    q=prob[loc][pos].mean(0);np.testing.assert_array_equal(q,arrays['q'][loc][k]);pi,v=solver.solve(q);saved=pv[k]
                    pe=float(np.abs(pi-saved).max());oe=abs(v['objective']-.5*np.sum((mat[loc]@saved-q)**2));assert pe<=1e-7 and oe<=1e-10 and direct_kkt(mat[loc],q,saved)<=1e-10
                    max_pi_error=max(max_pi_error,pe);max_objective_error=max(max_objective_error,float(oe));solved+=1
                m128=[sum((Fraction(float(x)) for x in pv[:128,y]),Fraction())/128 for y in range(10)]
                m256=[sum((Fraction(float(x)) for x in pv[:,y]),Fraction())/256 for y in range(10)]
                for si,s in enumerate(S):
                    for h in (0,1):
                        slot=2*si+h;row=bykey[i,b,t,s,h];table=templates[s,b,h,i][ti]
                        v128,a128=exact_utilities(m128,table);v256,a256=exact_utilities(m256,table)
                        assert v128==[Fraction(x) for x in row['mean_utility128_exact']] and v256==[Fraction(x) for x in row['mean_utility256_exact']]
                        assert a128[0]==arrays['BER128'][loc][slot]==row['BER128'] and a256[0]==arrays['BER256'][loc][slot]==row['BER256'];meanchecks+=1
                        pr=Fraction(row['point_bootstrap_expected_regret_exact']);br=Fraction(row['BER_bootstrap_expected_regret_exact'])
                        assert pr-br==v256[row['BER256']]-v256[row['point_state']] and pr>=br>=0
                        for k in (0,255):
                            vv,aa=exact_utilities(pv[k],table)
                            assert aa[0]==arrays['replica_choices'][loc][k,slot] and sum(1<<x for x in aa)==arrays['replica_argmax_masks'][loc][k,slot];replica_checks+=1
                        choices=arrays['replica_choices'][loc][:,slot];v128freq=np.bincount(choices[:128],minlength=5)/128;v256freq=np.bincount(choices,minlength=5)/256
                        np.testing.assert_array_equal(v128freq,row['vote128']);np.testing.assert_array_equal(v256freq,row['vote256'])
                        assert max(abs(x-y) for x,y in zip(v128freq,v256freq))==row['max_vote_frequency_difference']
    agreement=float(np.mean(arrays['BER128']==arrays['BER256']));assert agreement==summary['global_128_256_agreement']
    checked=[]
    for name in ('t015_artifact_manifest.json','t016r_artifact_manifest.json','t017r_artifact_manifest.json','t018r2_artifact_manifest.json','t019_artifact_manifest.json'):
        for r in load(ROOT/'research_log'/name):assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    verify=dict(status='PASS',independent_position_and_CLS_replays=solved,max_pi_error=max_pi_error,max_objective_error=max_objective_error,
        every_exact_mean_utility_and_BER_choice_replayed=meanchecks,exact_replica_argmax_checks=replica_checks,all_vote_frequencies_replayed=True,
        point_choices_exact=True,global_agreement_replayed=agreement,source_manifest_entries_unchanged=len(checked),source_files=checked,
        target_composition_or_true_utilities_parsed=False,query_counts_or_metrics_parsed=False)
    save(OUT/'independent_verification.json',verify)
    if summary['status']=='PREPARATION_PASS':print('T020_INDEPENDENT_PREPARATION_PASS');return
    assert summary['status']=='MONTE_CARLO_INSTABILITY'
    failures=[r for r in exactrows if not r['agreement']];stable=[r for r in exactrows if r['agreement']]
    failure_rows=[{k:r[k] for k in ('client','bank','context','salt','train_half','BER128','BER256','max_mean_utility_difference','max_vote_frequency_difference','top_mean_utility_margin','p_mode')} for r in failures]
    with (OUT/'MC_stability_failures.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(failure_rows[0]));writer.writeheader();writer.writerows(failure_rows)
    groups=list(csv.DictReader((RAW/'bootstrap_stability.csv').open()))
    table='| Context | Bank A agreement, % | Bank B agreement, % |\n|---|---:|---:|\n'
    for t in C:
        vals=[float(next(r['agreement_fraction'] for r in groups if r['bank']==b and r['context']==t)) for b in B]
        table+=f'| {t} | {vals[0]*100:.3f} | {vals[1]*100:.3f} |\n'
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    report=f'''# T020 STOP — Monte Carlo instability before privileged evaluation

{now}. Lead `5bb4d6e`; runtime `{summary['runtime']}`; run `{RUN}`, exit1 after {summary['seconds']:.2f}s. **115 tests PASS; all 256,000 CLS bootstrap solutions satisfy unchanged invariants.**

## Decision

First128 versus all256 BER state agreement is **{agreement*100:.4f}%**, below the frozen **99%** global requirement: **{len(failures)}/{len(exactrows)}** template-slot decisions differ. This alone triggers the Lead's mandatory stop; the result does not depend on the additional predeclared per-bank/context cluster check. **BER-REGRET-A / BER-CAP-A / BER-SRC-A are NOT EXECUTED. No T020-R/C/F/X scientific diagnosis is assigned.**

{table}

Differences cover {len(set(r['client'] for r in failures))} unique client IDs and {len(set((r['client'],r['bank'],r['context']) for r in failures))} client×bank×context cells. The eight utility-template slots share support draws; they are not independent resampling trials. Exact per-slot mean/vote differences and both decisions are preserved.

Maximum per-state mean-utility difference is {summary['max_mean_utility_difference']:.6g} (fractional utility units); maximum vote-frequency difference is {summary['max_vote_frequency_difference']:.6g}. Median full256 top-mean utility margin is {float(np.median([r['top_mean_utility_margin'] for r in failures])):.6g} among changed slots versus {float(np.median([r['top_mean_utility_margin'] for r in stable])):.6g} among unchanged slots. These are label-free diagnostics only; no claim about whether switching helps true regret or query accuracy is possible from this stopped run.

No R increase, seed retry, class stratification, confidence fallback, tolerance change or alternate state policy was applied. The original 256 draws are retained. Return to Lead for the next bounded decision; do not interpret MC instability as T020-F or permission to move to feature-level work.

## Label-blind implementation

Bootstrap positions 0..19 with replacement using full big-endian SHA256(`T020|client|bank|context|replica`) into PCG64. The preparation script lazily reads only T019 `probabilities` and label-free `actual_choices` array members, frozen soft channels and utility templates. It does not deserialize true support composition, true-template utilities or query counts. Upstream artifact hashing reads bytes solely for provenance. Target labels never enter positions, q, CLS or BER.

Utilities are computed from float64 prevalence as exact binary rationals and the existing rational templates. A shared denominator allows exact integer accumulation for first128/all256 means and expected regrets. Exact ties use the frozen candidate order; no rounding threshold is introduced. All five states' expected regrets share the same average-max term, so maximum mean utility exactly equals minimum expected regret. Both exact and floating mean receipts are saved.

The fixed-context dispatch helper was tested, but the source-context policy was never evaluated. The explicit T020 Section6 precomputed-context dispatch interpretation is recorded in `PROTOCOL_FREEZE.md`; it had no effect on this pre-context MC stop.

## Verification

- T019 posteriors replay byte-identically from frozen T015 logits; all 8,000 point-CLS choices replay the frozen T019/T018R2 baseline.
- 256,000 bootstrap CLS solves: max KKT {phase['max_KKT']:.6g}, max simplex-sum error {phase['max_sum_error']:.6g}, maximum {phase['max_updates']} active-set updates; objective dominance passes at unchanged tolerance.
- Local independent replay checks replica0 and replica255 positions and **2,000 CLS solutions**, max pi difference {max_pi_error:.6g}, objective difference {max_objective_error:.6g}.
- Independently average the exact binary-rational pi values, then evaluate the rational templates: all **8,000** first128/all256 mean-utility decisions agree with saved receipts. Recheck 16,000 replica argmax sets, all vote frequencies, BER/point expected-regret differences, and the global stability fraction.
- All {len(checked)} upstream manifest entries remain unchanged. Zero model/query forwards. No true-regret numerator or final query-count vector was computed because the stability condition failed.

`phaseB_choices_freeze.json` hashes all unlabeled candidate decisions/statistics. `bootstrap_stability.csv`, `bootstrap_uncertainty.csv`, `ber_choices.csv`, exact mean-utility gzip, seed receipts and the independent verification are tracked. Full position/q/pi/invariant/replica-mask arrays remain locally and remotely under `research_log/t020_receipts/{RUN}/artifacts/t020_bootstrap_expected_regret/` (remote canonical run path). No scientific-table placeholders are fabricated.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    save(OUT/'diagnostic_gates.json',dict(MC_STABILITY_PASS=False,global_agreement=agreement,required_agreement=.99,BER_REGRET_A='NOT_EXECUTED',BER_CAP_A='NOT_EXECUTED',BER_SRC_A='NOT_EXECUTED',scientific_diagnosis='NOT_ASSIGNED'))
    hand=f'''# T020 STOP — MC instability

{now}. Lead5bb4d6e/runtime{summary['runtime']}/run{RUN},exit1.115testsPASS;256000CLSsolvesPASS. BER128-vs256 agreement{agreement*100:.4f}%<{99}%;{len(failures)}/8000slots differ. Mandatory MCstop before true labels/utilities/querycounts. All scientific gates NOT_EXECUTED; no T020-R/C/F/X. Independent2000CLS+8000exactmeans+16000replicaargsets replayPASS;{len(checked)}upstream hashes unchanged. Read results/t020_bootstrap_expected_regret/RESULTS.md. No increasedR/newseeds/alternatepolicy;awaitLead.
'''
    for name in ('HANDOFF.md','T020_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\nThis is the declared Monte Carlo stability stop, not evidence about true-regret/capture improvement. No privileged evaluation was opened and no next-stage task started.\n',encoding='utf-8')
    save(ROOT/'research_log/t020_delivery.json',dict(timestamp=now,lead='5bb4d6e',runtime=summary['runtime'],run=RUN,status='MONTE_CARLO_INSTABILITY',global_agreement=agreement,results=OUT.relative_to(ROOT).as_posix(),next='Await Lead bounded instruction; no privileged evaluation'))
    files=[p for p in (ROOT/'research_log/t020_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t020_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
