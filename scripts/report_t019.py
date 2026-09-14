"""Independent T019 receipt replay and conservative mechanism report."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.matched_channel import exact_utilities
from src.context.constrained_prevalence import direct_kkt
RUN='20260914-145145-ttfl-t019';FINISH='20260914-145757-ttfl-t019-finish'
RAW=ROOT/f'research_log/t019_receipts/{RUN}/artifacts/t019_real_channel_heterogeneity';OUT=ROOT/'results/t019_real_channel_heterogeneity'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=[f'T013-S{i}' for i in range(4)]
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def rows(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def macro(c,t):return sum((Fraction(100*int(x),int(y)) for x,y in zip(c,t)),Fraction())/10
def rngstr(values,scale=100):return f'{min(values)*scale:.2f}–{max(values)*scale:.2f}'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file() and p.suffix!='.npz':shutil.copyfile(p,OUT/p.name)
    summary=load(RAW/'summary.json');assert summary['status']=='COMPLETE'
    gate_rows=[dict(scope='global',gate='TASK-MISMATCH-A',passed=summary['TASK_MISMATCH_A']),dict(scope='global',gate='CTX-SPEC-A',passed=summary['CTX_SPEC_A'])]
    gate_rows += [dict(scope=t,gate=gate,passed=summary[key][t]) for gate,key in [('task-context','task_context_pass'),('context-specific','context_specific_pass'),('LOCAL-A','LOCAL_A')] for t in summary[key]]
    with (OUT/'diagnostic_gates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','gate','passed']);writer.writeheader();writer.writerows(gate_rows)
    assert 'Ran 107 tests' in (RAW.parent.parent/'train.log').read_text() and 'exit_code=0' in (RAW.parent.parent/'finish.log').read_text()
    for n,h in load(RAW/'phaseB_choices_freeze.json')['hashes'].items():assert sha(RAW/n)==h
    for n,h in load(RAW/'mechanism_freeze.json')['hashes'].items():assert sha(RAW/n)==h
    real=dict(np.load(RAW/'real_decomposition.npz'));null=dict(np.load(RAW/'null_observations.npz'));u=dict(np.load(RAW/'utility_choices.npz'))
    p16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p14=ROOT/'results/t014_class_conditional_factorization';p13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    truth={r['client']:r for r in load(p16/'support_truth.json')};labels=np.array([truth[i]['labels_in_frozen_support_order'] for i in range(100)]);flatlabels=labels.ravel();counts=real['counts']
    cal=dict(np.load(p16/'calibration_counts.npz'));mat=cal['soft_numerator']/cal['soft_denominator'][...,None,:];probs=real['probabilities'];fp=probs.transpose(1,2,0,3,4).reshape(2,5,2000,10)
    ids=null['source_ids'];assert np.all(ids//20!=np.arange(100)[:,None,None,None,None]);drawlabels=flatlabels[ids]
    for y in range(10):np.testing.assert_array_equal((drawlabels==y).sum(-1),np.broadcast_to(counts[:,y,None,None,None],(100,2,5,128)))
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in ct] for ct in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    seedchecks=0;choicechecks=0;max_q=max_identity=max_kkt=0.
    for i in range(100):
        true=counts[i]/20
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                loc=(i,bi,ti);q=probs[loc].mean(0);max_q=max(max_q,float(np.abs(q-real['q'][loc]).max()))
                w=np.zeros(10)
                for y in np.flatnonzero(counts[i]):w+=true[y]*(probs[loc][labels[i]==y].mean(0)-mat[loc][:,y])
                max_identity=max(max_identity,float(np.abs(q-mat[loc]@true-w).max()))
                for y in np.flatnonzero(counts[i]==0):assert np.isnan(real['mu'][loc][y]).all()
                independent=[]
                for y,n in enumerate(counts[i]):
                    if n:
                        seed=int(hashlib.sha256(f'T019|{i}|{b}|{t}|127|{y}'.encode()).hexdigest(),16)
                        pool=np.array([j*20+k for j in range(100) if j!=i for k in range(20) if labels[j,k]==y])
                        gen=np.random.Generator(np.random.PCG64(seed));independent.extend(pool[gen.integers(0,len(pool),size=int(n))].tolist())
                np.testing.assert_array_equal(ids[loc][127],independent);qnull=fp[bi,ti,independent].mean(0);np.testing.assert_array_equal(qnull,null['q'][loc][127]);seedchecks+=1
                pi=null['pi'][loc][127];max_kkt=max(max_kkt,direct_kkt(mat[loc],qnull,pi));assert abs(pi.sum()-1)<=1e-12
                if ti:
                    qc=fp[bi,0,independent].mean(0);dr=(qnull-mat[loc]@true)-(qc-mat[i,bi,0]@true)
                    np.testing.assert_allclose(null['paired_norm'][loc][127],[np.abs(dr).sum(),np.linalg.norm(dr)],atol=1e-12,rtol=0)
                for si,s in enumerate(S):
                    for h in (0,1):
                        slot=2*si+h;table=templates[s,b,h,i][ti]
                        truevals,truearg=exact_utilities([Fraction(int(n),20) for n in counts[i]],table)
                        for pv,state,sc in [(u['actual_pi'][loc],u['actual_choices'][loc][slot],u['actual_scores'][loc][slot]),(pi,u['null_choices'][loc][127,slot],u['null_scores'][loc][127,slot])]:
                            vv,aa=exact_utilities(pv,table);assert aa[0]==state and sc[0]==float(max(abs(x-y) for x,y in zip(vv,truevals)))
                            assert sc[1]==float(max(truevals)-truevals[int(state)]) and bool(sc[2])==(state in truearg);choicechecks+=1
    assert max_q<=1e-12 and max_identity<=1e-12 and max_kkt<=1e-10
    task=rows(RAW/'task_actual_vs_null_cells.csv');paired=rows(RAW/'paired_context_cells.csv')
    for r in task:
        loc=(int(r['client']),B.index(r['bank']),C.index(r['context']));si=S.index(r['salt']);sl=slice(2*si,2*si+2)
        for j,name in enumerate(('DU','regret')):
            a=u['actual_scores'][loc][sl,j].mean();n=u['null_scores'][loc][:,sl,j].mean(-1)
            assert float(r[name+'_actual'])==a and float(r[name+'_null_p95'])==float(np.quantile(n,.95))
            assert (r[name+'_above_p95']=='True')==bool(a>np.quantile(n,.95)) and float(r[name+'_percentile'])==float(np.mean(n<=a))
    for r in paired:
        i=int(r['client']);bi=B.index(r['bank']);ti=C.index(r['context']);si=S.index(r['salt']);sl=slice(2*si,2*si+2)
        a=(u['actual_scores'][i,bi,ti,sl,0]-u['actual_scores'][i,bi,0,sl,0]).mean();n=null['paired_task_change'][i,bi,ti,:,sl,0].mean(-1)
        assert abs(float(r['delta_DU_actual'])-a)<=1e-15 and float(r['delta_DU_null_p95'])==float(np.quantile(n,.95))
        assert (r['delta_DU_above_p95']=='True')==bool(a>np.quantile(n,.95))
    attribution=rows(RAW/'one_class_repair_attribution.csv');groups={}
    for r in attribution:
        original=Fraction(r['real_regret_exact']);repaired=Fraction(r['repair_regret_exact']);reduction=Fraction(r['regret_reduction_exact']);assert original-repaired==reduction
        if original:assert Fraction(r['recovery_fraction_exact'])==reduction/original
        else:assert r['zero_gap']=='True' and not r['recovery_fraction_exact']
        groups.setdefault((r['client'],r['bank'],r['context'],r['salt']),[]).append(r)
    for g in groups.values():
        best=min(g,key=lambda r:(-Fraction(r['regret_reduction_exact']),int(r['label'])))
        assert sum(r['best_utility_class']=='True' for r in g)==1 and best['best_utility_class']=='True'
    hc=dict(np.load(p13/'half_integer_counts.npz'));cc=hc['correct'];qreceipt=dict(np.load(RAW/'query_count_receipt.npz'));rebuilt=np.zeros_like(qreceipt['null_counts']);actual_counts=np.zeros_like(qreceipt['actual_counts'])
    for bi in range(2):
        for ti in range(5):
            for si in range(4):
                for i in range(100):
                    for h in (0,1):
                        table=cc[si,bi,i,ti,1-h];rebuilt[bi,ti,si]+=table[u['null_choices'][i,bi,ti,:,2*si+h]]
                        actual_counts[bi,ti,si]+=table[u['actual_choices'][i,bi,ti,2*si+h]]
    np.testing.assert_array_equal(rebuilt,qreceipt['null_counts']);np.testing.assert_array_equal(actual_counts,qreceipt['actual_counts'])
    for r in load(RAW/'null_capture_replicas.json'):
        assert macro(rebuilt[B.index(r['bank']),C.index(r['context']),S.index(r['salt']),r['replica']],qreceipt['total'])==Fraction(r['macro_exact'])
    checked=[]
    for name in ('t015_artifact_manifest.json','t016r_artifact_manifest.json','t017r_artifact_manifest.json','t018r2_artifact_manifest.json'):
        for r in load(ROOT/'research_log'/name):assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    verification=dict(status='PASS',independent_last_replica_seed_source_ID_replays=seedchecks,all128000draws_target_excluded_and_exact_counts=True,
        real_q_max_error=max_q,real_residual_identity_max_error=max_identity,independent_null_KKT_max=max_kkt,exact_Fraction_choice_checks=choicechecks,
        task_percentile_cells=len(task),paired_percentile_cells=len(paired),exact_class_repair_attribution_rows=len(attribution),utility_ranked_class_groups=len(groups),
        null_query_count_vectors=5120,actual_query_count_vectors=40,source_manifest_entries_unchanged=len(checked),source_files=checked)
    save(OUT/'independent_verification.json',verification)
    ts=rows(RAW/'task_actual_vs_null_summary.csv');ps=rows(RAW/'paired_context_summary.csv');ls=rows(RAW/'localization_summary.csv');posterior=rows(RAW/'matched_exact_count_null_summary.csv');caps=rows(RAW/'capture_summary.csv');mr=rows(RAW/'utility_margin_quartiles.csv')
    table='| Context | Task tail A, % | Task tail B, % | Paired tail A, % | Paired tail B, % | Single-class half-gap recovery A / B, % |\n|---|---:|---:|---:|---:|---|\n'
    for t in C[1:]:
        get=lambda data,b,key:[float(r[key]) for r in data if r['context']==t and r['bank']==b]
        table+=f"| {t} | {rngstr(get(ts,'A','task_exceedance_fraction'))} | {rngstr(get(ts,'B','task_exceedance_fraction'))} | {rngstr(get(ps,'A','context_exceedance_fraction'))} | {rngstr(get(ps,'B','context_exceedance_fraction'))} | {rngstr(get(ls,'A','fraction_best_class_recovers_half'))} / {rngstr(get(ls,'B','fraction_best_class_recovers_half'))} |\n"
    capturetable='| Context | Actual capture, % | T019 null median capture, % | Actual below null p05 rows / 8 |\n|---|---:|---:|---:|\n'
    for t in C[1:]:
        g=[r for r in caps if r['context']==t];capturetable+=f"| {t} | {rngstr([float(r['actual_capture']) for r in g])} | {rngstr([float(r['null_capture_median']) for r in g])} | {sum(r['actual_below_null_p05']=='True' for r in g)} |\n"
    margintable='| Mean-margin quartile | Actual optimal-set error, % | Mean true regret, pp |\n|---|---:|---:|\n';margin=[]
    for q in range(1,5):
        g=[r for r in mr if int(r['quartile'])==q and r['context'] in C[2:]];error=np.mean([float(r['actual_optimal_error']) for r in g]);regret=np.mean([float(r['actual_regret_mean']) for r in g]);margin.append(regret)
        margintable+=f'| Q{q} | {100*error:.3f} | {100*regret:.3f} |\n'
    maxclass=[]
    for t in C:
        for b in B:
            histogram={y:0 for y in range(10)}
            for r in ls:
                if r['context']==t and r['bank']==b:
                    for y,n in json.loads(r['best_class_histogram']).items():histogram[int(y)]+=n
            total=sum(histogram.values());top=max(histogram,key=lambda y:histogram[y]);maxclass.append(dict(context=t,bank=b,top_class=top,top_share=histogram[top]/total if total else None,counts=histogram))
    save(OUT/'dominant_repair_classes.json',maxclass)
    # Boundary errors are more frequent at small margins, but large margins carry most regret.
    # Keep the mixed branch; neither frequency alone nor privileged repair establishes dominance.
    assert summary['diagnosis']=='T019-X' and not summary['TASK_MISMATCH_A'] and not summary['CTX_SPEC_A']
    now=datetime.now().astimezone().isoformat(timespec='seconds');sv=load(RAW/'solver_verification.json');av=load(RAW/'phaseA_verification.json')
    report=f'''# T019 — T019-X: mixed evidence, not strong channel mismatch

{now}. Lead `d37bbc4`. Preparation `{summary['preparation_runtime']}`, run `{RUN}`; reporting `{summary['runtime']}`, run `{FINISH}`. Both exit0. **107 tests PASS; zero new model/query forwards.**

## Decision

**TASK-MISMATCH-A FAIL (0/3), CTX-SPEC-A FAIL (0/3), LOCAL-A PASS for every context. Final diagnosis: T019-X.** The exact-count matched null does not show the predeclared strong task-relevant or paired context-specific excess. Privileged single-class repairs can recover regret for many clients, but this does not establish a systematic class-specific channel failure. Boundary sensitivity is visible, yet its error frequency alone does not explain the distribution of regret. Do not force C, G or S.

All tail fractions below are per-client strict p95 exceedance, checked separately for each salt after averaging its two template halves. Ranges span four salts; the threshold was 20% in both banks/all salts. Task tail is the union of DU/regret exceedance. Paired tail is the union of shift-minus-clean residual-L1/delta-DU exceedance. LOCAL fractions use only nonzero-regret clients and exact utility-ranked single-class repair, never query-based class selection.

{table}

Clean union excess (posterior-L1 or DU/regret) is 11% / 17% for A/B, below the frozen operational criterion for substantial generic excess. Clean task-only tails are 7% / 10%. Thus G is unsupported. Paired Blur tails reach 17–19%, but remain below the fixed 20% requirement; no threshold was relaxed.

## Capture and the T018 matched-to-real gap

{capturetable}

All actual macro-class values reproduce T018R2 exactly. Client capture deficits are reported only for positive client P00-gain contribution denominators; zero/negative denominators are marked undefined and counted, not silently normalized. Aggregate P00 denominators remain unchanged. The known T018R2 source-context Blur penalty is preserved separately; T019 itself uses oracle context throughout the mechanism audit.

**Protocol clarification:** T017 K20 already used the exact observed class counts. T019 adds per-class decomposition/seed streams, conditional task percentiles, one-class interventions and ID-paired clean/shift contrasts; it does not newly remove composition randomness from the old K20 comparison. Its new null replicas use the predeclared T019 class-level seeds, so small aggregate changes from T018R2 also reflect a new diagnostic Monte Carlo draw, not a new estimator.

For Blur, the below-p05 row count changes from 1/8 in T018R2 to 6/8 in this new 128-replica diagnostic draw, despite identical actual results. This tail-count change is sensitive to the null draw. The eight bank/salt rows share observations and are not eight independent tests. Keep both receipts and the weak client-level task-tail evidence; do not turn the new aggregate row count into a strong mismatch label.

## Why the diagnosis stays mixed

For Contrast/Noise/Blur, small-margin clients have more wrong state choices, but larger-margin errors are less frequent and more costly. The fixed within-bank/context/salt quartiles show:

{margintable}

The upper two margin quartiles contribute {100*sum(margin[2:])/sum(margin):.2f}% of mean true regret in this equal-sized stratification; Q1 contributes {100*margin[0]/sum(margin):.2f}%. Therefore “many errors near boundaries” is descriptive evidence, not sufficient to call the remaining loss boundary-dominated. Conversely, the weak p95 task tails do not justify strong emission mismatch. LOCAL-A identifies label-privileged repairability, not a deployable writer or proof that the same semantic class is responsible everywhere. `dominant_repair_classes.json` gives complete per-context/bank class histograms and top shares rather than asserting a universal bad column.

Recommendation to Lead: a bounded utility-margin/regret-cost and robust neutral-state selection audit could test the observed sensitivity directly; first distinguish frequent small-cost boundary changes from sparse large-cost errors. This is a recommendation only. No feature experiment, new semantic estimator, learned calibration, SSL/TTT, operator expansion or FL has started.

## Implementation and verification

T015 per-example float64 posteriors were reconstructed with the unchanged function and replayed byte-for-byte. Every T016 soft numerator and denominator replays exactly, with its original membership list and target excluded. Real q max error is {av['max_q_reconstruction_error']:.6g}; class-residual identity max error {av['max_residual_identity_error']:.6g}. Absent classes remain undefined (NaN only in masked NPZ slots), not imputed.

For each target/bank/context/replica/class, 128 null replicas draw exactly n_y examples with replacement from the target-excluded pool. The seed is the full big-endian SHA256 integer of `T019|client|bank|context|replica|class`, fed to PCG64. Flat source IDs are stored together with client, support position, original ID and label mapping. Shifted draws fetch clean posteriors for the exact same IDs, with full paired coverage and weights n_y/20.

Solver checks passed for {json.dumps(sv['counts'])}: maximum KKT {sv['max_KKT']:.6g}, sum error {sv['max_sum_error']:.6g}, {sv['max_updates']} maximum updates. All 1,000 all-class-repaired cases recover the noise-free prevalence and exact optimal set with zero true regret. No tolerance, solver, model, state, temperature or K changed.

Preparation independently replays replica0 source IDs in all 1,000 cells; local audit independently reconstructs replica127 source IDs/q in another 1,000 cells and verifies all 128,000 draws for target exclusion/exact class counts. It checks 16,000 exact Fraction state/DU/regret choices, all 4,000 task percentiles, 3,200 paired delta-DU percentile rows, all exact class-repair fractions/rankings, all 5,120 null query-count vectors and 40 actual vectors. All {len(checked)} source-manifest entries from T015/T016/T017/T018R2 remain unchanged.

`phaseB_choices_freeze.json` hashes posterior/null/repair state choices; `mechanism_freeze.json` then hashes null percentiles, paired metrics and class-attribution ranks. Only afterward does `finish_t019.py` open the frozen query-count file. No query outcome selected repairs or formed the null.

## Artifacts

Compact tables and all verification/seed/freeze receipts are in `results/t019_real_channel_heterogeneity/`. Raw posterior/null/choice arrays remain in `research_log/t019_receipts/{RUN}/artifacts/t019_real_channel_heterogeneity/` and the identical remote run path. The finish run writes into that original artifact directory and has a separately retained `finish.log`. Dataset scope remains the inherited PFLlib 100-client baseline on the merged client split, not an official CIFAR-10 test benchmark.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f'''# T019 COMPLETE — T019-X

{now}. Lead d37bbc4. Prep6b5b2c0/run{RUN};finish15ad29f/run{FINISH};both exit0.107testsPASS;zero newforwards. TASK-MISMATCH-A0/3FAIL;CTX-SPEC-A0/3FAIL;LOCAL-AallcontextsPASS. Contrast/Noise/Blur task tails2–11%,pairedtails12–19%;cleanunion11/17%. Small-margin state errors frequent but upperhalf margins carry{100*sum(margin[2:])/sum(margin):.2f}%true regret: mixed, not forced S. T017 K20 already fixed class counts; T019 adds decomposition/paired/repair diagnostics. Read results/t019_real_channel_heterogeneity/RESULTS.md and independent_verification.json. Await Lead; no nextstage.
'''
    for name in ('HANDOFF.md','T019_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\n'+capturetable+'\nRecommendation: bounded margin/regret-cost + robust neutral-state selection audit, without another semantic estimator or learned writer. No next task started.\n',encoding='utf-8')
    save(ROOT/'research_log/t019_delivery.json',dict(timestamp=now,lead='d37bbc4',preparation_run=RUN,finish_run=FINISH,status='COMPLETE',diagnosis='T019-X',results=OUT.relative_to(ROOT).as_posix(),next='Await Lead bounded package'))
    files=[p for p in (ROOT/'research_log/t019_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t019_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
