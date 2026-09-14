"""Independent count replay and scientific report for the frozen T018 completion."""
import csv,gzip,hashlib,json,shutil,sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context.matched_channel import exact_utilities
RUN='20260914-132634-ttfl-t018r2-science';PRERUN='20260914-132551-ttfl-t018r2-preflight'
RAW=ROOT/f'research_log/t018r2_receipts/{RUN}/artifacts/t018r2_constrained_prevalence'
PRE=ROOT/f'research_log/t018r2_receipts/{PRERUN}/artifacts/t018r2_constrained_prevalence'
BASE=ROOT/'results/t018_constrained_prevalence';OUT=BASE/'t018r2'
C=['clean','brightness_dark','contrast_low','gaussian_noise','gaussian_blur'];B=['A','B'];S=[f'T013-S{i}' for i in range(4)]
def load(p):return json.loads(p.read_text())
def gzload(p):return json.loads(gzip.decompress(p.read_bytes()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open()))
def macro(c,t):return sum((Fraction(100*int(x),int(y)) for x,y in zip(c,t)),Fraction())/10
def rr(values,scale=1):return f'{min(values)*scale:.3f}–{max(values)*scale:.3f}'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.is_file() and p.suffix!='.npz':shutil.copyfile(p,OUT/p.name)
    shutil.copytree(PRE,OUT/'preflight',dirs_exist_ok=True)
    summary=load(RAW/'summary.json');assert summary['status']=='COMPLETE'
    assert 'Ran 100 tests' in (PRE.parent.parent/'train.log').read_text() and 'exit_code=0' in (RAW.parent.parent/'train.log').read_text()
    p13=ROOT/'research_log/t013_receipts/20260914-t013-local/artifacts/t013_disjoint_factorization'
    p14=ROOT/'results/t014_class_conditional_factorization'
    p15=ROOT/'research_log/t015_receipts/20260914-043941-ttfl-t015-gpu1/artifacts/t015_unlabeled_semantic_mixture'
    p16=ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics'
    p17=ROOT/'research_log/t017r_receipts/20260914-083609-ttfl-t017r-cached/artifacts/t017_channel_noise_decomposition'
    hc=np.load(p13/'half_integer_counts.npz');cc=hc['correct'];tot=hc['total'];den=tot[0].sum((0,1))
    solved=np.load(RAW/'matched_solver.npz');ar=np.load(RAW/'matched_choices.npz');choices=ar['choices'];stored=ar['aggregate_counts'];pi_array=solved['pi'];du_array=ar['DU']
    assert choices.shape==(100,2,5,128,8)
    np.testing.assert_array_equal(solved['sum_error'],np.abs(solved['pi'].sum(-1)-1.))
    assert solved['sum_error'].max()<=1e-12 and solved['KKT'].max()<=1e-10 and solved['pi'].min()>=-1e-12
    assert solved['objective_improvement'].min()>=-5e-13
    reconstructed=np.zeros_like(stored)
    for si in range(4):
        for bi in range(2):
            for ti in range(5):
                for i in range(100):
                    for h in (0,1):reconstructed[bi,ti,si]+=cc[si,bi,i,ti,1-h][choices[i,bi,ti,:,2*si+h]]
    np.testing.assert_array_equal(reconstructed,stored)
    zero={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p13/'integer_count_receipts.json') if r['policy']=='zero'}
    p00={(r['salt'],r['bank'],r['target']):macro(r['class_correct'],r['class_total']) for r in load(p15/'integer_count_receipts.json') if r['policy']=='P00'}
    original=np.load(p17/'bootstrap_query_counts.npz')['aggregate_counts'][0]
    replicas=gzload(RAW/'matched_replicas.json.gz');assert len(replicas)==5120
    for r in replicas:
        key=(r['salt'],r['bank'],r['context']);bi=B.index(r['bank']);ti=C.index(r['context']);si=S.index(r['salt']);k=r['replica']
        acc=macro(reconstructed[bi,ti,si,k],den);old=macro(original[bi,ti,si,k],den)
        assert acc==Fraction(r['macro_class_exact']) and (acc-zero[key])/(p00[key]-zero[key])==Fraction(r['capture_exact'])
        assert float(old)==r['old_macro_class'] and float((old-zero[key])/(p00[key]-zero[key]))==r['old_capture']
    templates={(r['salt'],r['bank'],r['train_half'],r['target_client']):[[[Fraction(x) for x in row] for row in context] for context in r['utility']] for r in gzload(p14/'class_templates.json.gz')['rows']}
    truth={r['client']:r for r in load(p16/'support_truth.json')};exact_checks=0
    for i in range(100):
        for bi,b in enumerate(B):
            for ti,t in enumerate(C):
                for si,s in enumerate(S):
                    for h in (0,1):
                        table=templates[s,b,h,i][ti];true,_=exact_utilities([Fraction(x,20) for x in truth[i]['counts']],table)
                        for k in (0,127):
                            values,args=exact_utilities(pi_array[i,bi,ti,k],table)
                            assert args[0]==choices[i,bi,ti,k,2*si+h]
                            assert float(max(abs(x-y) for x,y in zip(values,true)))==du_array[i,bi,ti,k,2*si+h];exact_checks+=1
    receipts={(r['mode'],r['salt'],r['bank'],r['context']):r for r in load(RAW/'integer_count_receipts.json')}
    actual={};actual_checks=0
    for mode,name in [('01','actual_oracle_context'),('11','full_source')]:
        accumulated={};episodes=gzload(RAW/(name+'_episodes.json.gz'));actual[mode]=rows(RAW/(name+'.csv'))
        for r in episodes:
            key=(mode,r['salt'],r['bank'],r['context']);si=S.index(r['salt']);bi=B.index(r['bank']);ti=C.index(r['context']);i=r['client'];h=r['train_half']
            counts=accumulated.setdefault(key,np.zeros(10,dtype=np.int64));counts+=cc[si,bi,i,ti,1-h,r['state']]
            table=templates[r['salt'],r['bank'],h,i][C.index(r['estimated_context'])]
            _,args=exact_utilities(r['pi'],table);assert args[0]==r['state'];actual_checks+=1
        for r in actual[mode]:
            key=(mode,r['salt'],r['bank'],r['context']);np.testing.assert_array_equal(accumulated[key],receipts[key]['class_correct'])
            np.testing.assert_array_equal(den,receipts[key]['class_total'])
            acc=macro(accumulated[key],den);assert acc==Fraction(r['macro_class_exact'])
            kk=key[1:];assert (acc-zero[kk])/(p00[kk]-zero[kk])==Fraction(r['capture_exact'])
    matched=rows(RAW/'matched_k20_aggregate.csv');delta=rows(RAW/'matched_k20_delta_vs_t017.csv')
    for r,d in zip(matched,delta):
        group=[x for x in replicas if all(x[k]==r[k] for k in ('bank','context','salt'))]
        for name in ('macro_class','capture','L1','JS','dominant_agreement','DU','true_template_regret','canonical_agreement','optimal_set_agreement'):
            for label,q in zip(('p05','median','p95'),(.05,.5,.95)):
                assert float(r[name+'_'+label])==float(np.quantile([x[name] for x in group],q))
                assert float(d[name+'_delta_'+label])==float(np.quantile([x[name]-x['old_'+name] for x in group],q))
    mp={t:all(float(r['capture_median'])>=.8 for r in matched if r['context']==t) for t in C[1:]}
    rp={t:all(Fraction(r['capture_exact'])>=Fraction(4,5) for r in actual['01'] if r['context']==t) for t in C[1:]}
    sp={t:all(Fraction(r['capture_exact'])>=Fraction(4,5) for r in actual['11'] if r['context']==t) for t in C[1:]}
    assert mp==summary['matched_pass'] and rp==summary['real_pass'] and sp==summary['source_pass']
    checked=[]
    for name in ('t015_artifact_manifest.json','t016r_artifact_manifest.json','t017r_artifact_manifest.json'):
        for r in load(ROOT/'research_log'/name):
            assert sha(ROOT/r['path'])==r['sha256'];checked.append(r)
    verification=dict(status='PASS',matched_count_vectors=5120,matched_count_contributions=1024000,exact_macro_capture_replicas=5120,
        independent_Fraction_matched_choices=exact_checks,independent_Fraction_actual_choices=actual_checks,actual_count_vectors=80,
        all_aggregate_quantiles_and_paired_deltas_replayed=True,source_hash_entries_unchanged=len(checked),source_files=checked,
        max_sum_error=float(solved['sum_error'].max()),max_KKT=float(solved['KKT'].max()),same_q_sha256=sha(p17/'bootstrap_arrays.npz'))
    assert verification['same_q_sha256']==summary['source_q_sha256'];save(OUT/'independent_verification.json',verification)
    now=datetime.now().astimezone().isoformat(timespec='seconds');sv=load(RAW/'matched_solver_verification.json');pv=load(PRE/'solver_verification.json');nf=load(PRE/'noise_free_verification.json')
    table='| Context | Matched median capture, % | Real oracle capture, % | Source capture, % | Matched / real / source |\n|---|---:|---:|---:|---|\n'
    for t in C[1:]:
        table+=f"| {t} | {rr([float(r['capture_median']) for r in matched if r['context']==t],100)} | {rr([float(r['capture']) for r in actual['01'] if r['context']==t],100)} | {rr([float(r['capture']) for r in actual['11'] if r['context']==t],100)} | {mp[t]} / {rp[t]} / {sp[t]} |\n"
    changes='| Context | Matched paired median Δmacro, pp | Real Δmacro vs BBSE, pp | Source Δmacro vs BBSE, pp |\n|---|---:|---:|---:|\n'
    for t in C:
        changes+=f"| {t} | {rr([float(r['macro_class_delta_median']) for r in delta if r['context']==t])} | {rr([float(r['delta_vs_BBSE_pp']) for r in actual['01'] if r['context']==t])} | {rr([float(r['delta_vs_BBSE_pp']) for r in actual['11'] if r['context']==t])} |\n"
    blur=rows(RAW/'blur_decomposition.csv');clean=[r for r in actual['11'] if r['context']=='clean']
    diagnostics='| Context | Matched median L1 | Real L1 | Real dominant agreement, % |\n|---|---:|---:|---:|\n'
    for t in C:
        diagnostics+=f"| {t} | {rr([float(r['L1_median']) for r in matched if r['context']==t])} | {rr([float(r['L1']) for r in actual['01'] if r['context']==t])} | {rr([float(r['dominant_agreement']) for r in actual['01'] if r['context']==t],100)} |\n"
    report=f'''# T018R2: verified exact CLS, Outcome {summary['outcome']}

{now}. Lead `c8aea5b`; frozen science `1744440`. Runtime `{summary['runtime']}`. Preflight `{PRERUN}` and scientific run `{RUN}` both exit0. Science took {summary['seconds']:.2f}s. **100 tests PASS; zero new model forwards; no resampling.**

## Decision

**CLS-MATCH-A PASS (4/4); CLS-REAL-A FAIL (1/4); CLS-SRC-A FAIL (1/4): Outcome B.** Real regression safety and source clean safety both pass. The matched empirical K20 channel becomes task-useful when the simplex constraint is enforced in the measurement fit. This improvement does not transfer sufficiently to real support: output-space CLS alone has not solved real semantic estimation. This supports a remaining real-support/channel heterogeneity hypothesis, not a claim that the affine operator failed. No next phase has started.

Ranges below span both banks and all four salts. Passing a context requires every one of its eight bank×salt rows to reach 80%; no average-row substitution.

{table}

## Paired estimator changes

Same q, checkpoint, states, context decisions, template lookups and P00 denominator; only the optimizer now realizes the unchanged CLS objective. Matched values are p50 of paired per-replica accuracy differences, not differences between independently resampled runs.

{changes}

The real oracle path has no shifted regression beyond 0.5pp. Source clean gain over zero is {rr([float(r['clean_delta_vs_zero']) for r in clean])}pp, passing -0.5pp safety. Each complete bank×context×salt row, its accuracy/capture and semantic diagnostics are in the CSVs; matched tables also include p05/p95 and paired L1, JS, dominant agreement, DU, exact optimal-set/canonical agreement and true-template regret.

{diagnostics}

## Blur decomposition

Matched Blur median capture is {rr([float(r['matched_median']) for r in blur],100)}%; its p05 is {rr([float(r['matched_p05']) for r in blur],100)}%. Real oracle Blur capture is {rr([float(r['actual_oracle_capture']) for r in blur],100)}%, and source Blur is {rr([float(r['source_capture']) for r in blur],100)}%. Real oracle is below the matched p05 in {sum(r['actual_below_matched_p05']=='True' for r in blur)}/8 rows. Source composition adds {rr([float(r['source_minus_oracle_pp']) for r in blur])}pp relative to oracle.

The old 8/8 below-p05 finding narrows to **1/8**, so the **extra-real-Blur residual** is now a row-specific observation, not a uniform anomaly across both banks and every salt. Distinguish this from the additional context-ID composition gap. The old T017 mismatch label was weak and was not redefined here; these task gaps alone do not establish a strong channel-mismatch label. Source context-correct/error subset sample counts and weighted accuracies are saved separately.

## Numerical certification, before metrics

T018R2 changes only the cached face application: cache KKT matrices; use `numpy.linalg.solve(K, actual_rhs)` on each visited face. All working-set logic and tolerances remain unchanged. No unconditional normalization, new dependency or new estimator was added. Old PGD and cached-inverse stops remain preserved in their receipts/history.

- Frozen blocker client35/B/Dark/replica7 passes with zero clipping; the full matched receipt explicitly records its repaired pi, KKT and sum error.
- Noise-free 1,000: max prevalence error {nf['max_pi_error']:.6g}, objective {nf['max_objective']:.6g}; 7,914 singleton and 86 exact-tie checks pass.
- Same 200/reference: max pi error {pv['max_prevalence_error']:.6g}; all 49 historical PGD caps remain repaired.
- All 1,000 real cells pass, max sum error {pv['max_sum_error']:.6g}, max KKT {pv['max_KKT']:.6g}.
- **All 128,000 saved matched q solutions were certified and persisted before query counts opened**: max sum error {sv['max_sum_error']:.6g}, max KKT {sv['max_KKT']:.6g}, max {sv['max_updates']} updates. Arrays save pi, KKT, sum errors, update counts and objective improvements; p95/p99/max summaries are in `matched_solver_verification.json`.

Exact raw CLS prevalence selects the first rational-utility argmax without a true-label tie override. Oracle-context results are written and hashed before full-source composition.

## Independent replay and artifacts

Local independent replay reconstructs 5,120 matched class-count vectors from all 1,024,000 selected half-client contributions, checks all exact macro/capture fractions and paired baselines, all aggregate quantiles/deltas, 16,000 matched boundary-replica Fraction choices and all 16,000 actual Fraction choices plus 80 actual aggregate count vectors. All {len(checked)} upstream T015/T016/T017 manifest entries are byte-identical. The source q file hash is `{summary['source_q_sha256']}`.

The science command's relative output directory initially placed artifacts under release `20260914-132535-ttfl-t018r2-preflight/artifacts/`; after successful completion they were copied unchanged into the canonical `{RUN}/artifacts/` directory. No computation was repeated for this location correction. Raw copies remain remote and under `research_log/t018r2_receipts/`; large NPZ arrays are intentionally not in Git. Compact tables/JSON, hashes and this report are tracked.

Data limitation remains inherited: PFLlib 100-client/10%-participation baseline on the merged CIFAR train/test client split, not the official CIFAR-10 test benchmark. No new FL training occurred.

Return **Outcome B** to Lead. The next question is the matched-to-real observation/channel gap; do not interpret this as permission to start feature semantics, learned calibration, SSL/TTT or federation.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    old=BASE/'t018r/CACHED_INVERSE_STOP_RESULTS.md'
    if not old.exists():shutil.copyfile(BASE/'t018r/RESULTS.md',old)
    (BASE/'t018r/RESULTS.md').write_text(report+'\n\n---\n\n'+old.read_text(encoding='utf-8'),encoding='utf-8')
    (BASE/'RESULTS.md').write_text(report+'\n\nPrevious numerical stops: `PGD_PREFLIGHT_RESULTS.md` and `t018r/CACHED_INVERSE_STOP_RESULTS.md`. Both original raw receipts remain unchanged.\n',encoding='utf-8')
    hand=f'''# T018R2 COMPLETE — Outcome B

{now}. Leadc8aea5b/runtime{summary['runtime']}; preflight{PRERUN}, science{RUN}, bothexit0.100testsPASS; all128000matched numerical solves PASS, maxsum{sv['max_sum_error']:.6g},maxKKT{sv['max_KKT']:.6g}. CLS-MATCH-A4/4PASS; CLS-REAL-A1/4FAIL; CLS-SRC-A1/4FAIL (Darkonly). BothsafetychecksPASS. Zero newforwards/noresampling. Read results/t018_constrained_prevalence/t018r2/RESULTS.md and independent_verification.json. Previousstops preserved. ReturntoLead; no nextstage.
'''
    for name in ('HANDOFF.md','T018R_HANDOFF.md','T018R2_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\n'+changes+f"\nBlur real below matchedp05: {sum(r['actual_below_matched_p05']=='True' for r in blur)}/8. Do not rename old weak mismatch label. Full compact receipts and paired diagnostics tracked; rawNPZ local/remote.\n",encoding='utf-8')
    save(ROOT/'research_log/t018r2_delivery.json',dict(timestamp=now,lead='c8aea5b',runtime=summary['runtime'],preflight_run=PRERUN,science_run=RUN,status='COMPLETE',outcome=summary['outcome'],results=OUT.relative_to(ROOT).as_posix(),next='Await Lead next bounded package'))
    files=[p for p in (ROOT/'research_log/t018r2_receipts').rglob('*') if p.is_file()]+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t018r2_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
