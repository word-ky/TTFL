"""Independent complete R4096 replay and cycle-repair/MC2 delivery."""
import csv,hashlib,json,sys
from datetime import datetime
import numpy as np
import report_t020r as replay
from src.context.constrained_prevalence import ActiveSetCLS,direct_kkt
RUN='20260914-173004-ttfl-t020r2'
ROOT=replay.ROOT;OUT=ROOT/'results/t020r2_mc_convergence';RAW=ROOT/f'research_log/t020r2_receipts/{RUN}/artifacts/t020r2_mc_convergence'
load=replay.load;save=replay.save;sha=replay.sha


def main():
    replay.RUN=RUN;replay.RAW=RAW;replay.OUT=OUT;replay.TEST_COUNT=124;replay.VERIFY_ONLY=True
    replay.main()
    v=load(OUT/'independent_verification.json');summary=load(RAW/'summary.json');resume=load(RAW/'resume_receipt.json')
    assert resume['status']=='COMPLETE_100_CLIENTS' and resume['template_slots']==8000 and resume['cells']==1000
    for name,digest in load(RAW/'preserved_98_verification.json')['files'].items():assert sha(RAW/name)==digest
    for name,digest in resume['hashes'].items():assert sha(RAW/name)==digest
    old={};exec(compile((OUT/'frozen_solver.py').read_text(),'frozen_solver.py','exec'),old)
    prob=np.load(ROOT/'research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity/real_decomposition.npz')['probabilities']
    cal=np.load(ROOT/'research_log/t016r_receipts/20260914-063758-ttfl-t016r-cached/artifacts/t016_confusion_debiased_semantics/calibration_counts.npz')
    mat=cal['soft_numerator']/cal['soft_denominator'][...,None,:];checks=[]
    for r in load(RAW/'solver_fallbacks.json'):
        i=r['client'];bi=replay.B.index(r['bank']);ti=replay.C.index(r['context']);k=r['replica'];C=mat[i,bi,ti]
        seed=int(hashlib.sha256(f"T020|{i}|{r['bank']}|{r['context']}|{k}".encode()).hexdigest(),16)
        positions=np.random.Generator(np.random.PCG64(seed)).integers(0,20,size=20);q=prob[i,bi,ti][positions].mean(0)
        try:old['ActiveSetCLS'](C).solve(q)
        except RuntimeError as exc:assert 'working-set cycle' in str(exc)
        else:raise AssertionError('Reported fallback did not reproduce the frozen cycle')
        ref,obj=old['face_reference'](C,q);pi,rr=ActiveSetCLS(C).solve(q)
        assert rr['solver_path']=='cycle_face_fallback' and rr['selected_face']==r['selected_face'] and rr['cycle_active_set']==r['cycle_active_set']
        assert abs(obj-r['objective'])<=1e-12 and np.max(np.abs(pi-ref))<=1e-9 and direct_kkt(C,q,pi)<=1e-10
        idx=r['selected_face'];m=len(idx);K=np.zeros((m+1,m+1));K[:m,:m]=(C.T@C)[np.ix_(idx,idx)];K[:m,m]=1;K[m,:m]=1
        face=np.zeros(10);face[idx]=np.linalg.solve(K,np.append((C.T@q)[idx],1))[:m]
        assert np.max(np.abs(face-ref))<=1e-9 and direct_kkt(C,q,face)<=1e-10
        checks.append(dict(client=i,bank=r['bank'],context=r['context'],replica=k,selected_face=idx,reference_objective=obj,
            reference_KKT=direct_kkt(C,q,ref),remote_objective_difference=abs(obj-r['objective']),max_pi_difference=float(np.max(np.abs(pi-ref))),frozen_cycle_reproduced=True))
    assert len(checks)==resume['fallback_activations']==2
    extra=[]
    for r in load(ROOT/'research_log/t020r_artifact_manifest.json'):
        assert sha(ROOT/r['path'])==r['sha256'];extra.append(r)
    v.update(fallback_reenumerations=checks,preserved_98_byte_identical=True,full_completeness_verified=True,source_files=v['source_files']+extra,
        source_manifest_entries_unchanged=v['source_manifest_entries_unchanged']+len(extra),cross_platform_entropy_atol=2e-15,scientific_gates='NOT_EXECUTED')
    save(OUT/'independent_verification.json',v)
    if summary['passed']:print('T020R2_INDEPENDENT_PASS_READY_FOR_SCIENCE');return
    assert summary['status']=='T020R-MC2'
    now=datetime.now().astimezone().isoformat(timespec='seconds');g=summary['global_agreement']
    table='| Context | A H1/H2 % | B H1/H2 % | Pooled % |\n|---|---:|---:|---:|\n'
    for t in replay.C:
        a,b=[next(r['agreement'] for r in summary['bank_context'] if r['bank']==bank and r['context']==t) for bank in replay.B]
        pool=next(r['agreement'] for r in summary['pooled_context'] if r['context']==t)
        table+=f'| {t} | {a*100:.3f} | {b*100:.3f} | {pool*100:.3f} |\n'
    cells=load(RAW/'cell_stream_receipts.json');trans=list(csv.DictReader((RAW/'transition_256_to_4096.csv').open()))
    changed=sum(int(r['slots']) for r in trans if r['BER256']!=r['BER4096'])
    report=f'''# T020R2 repair complete; sealed T020R-MC2 stop

{now}. Lead `43d2f85`; runtime `{summary['runtime']}`; run `{RUN}`. **124 remote tests PASS**. Two-client resume and full sealed aggregation completed in {summary['seconds']:.2f}s. Exit1 is the prescribed failed-convergence stop.

## Frozen convergence decision

| Comparison | Agreement | Required | Result |
|---|---:|---:|---|
| Global H1/H2 | {g['H1_H2']*100:.4f}% | 99% | FAIL |
| Global H1/ALL | {g['H1_ALL']*100:.4f}% | 99.5% | FAIL |
| Global H2/ALL | {g['H2_ALL']*100:.4f}% | 99.5% | PASS |

Every bank/context requires98.5%; every pooled context requires99%:

{table}

Three bank/context groups fail (A/Dark, A/Contrast, B/Clean); four pooled contexts fail (all except Noise). H1/H2 differ in **94/8000 slots across38clients/59cells**. Eight template slots share each posterior/bootstrap stream and are correlated. No missing client, imputed decision, rounded-equivalent state or altered threshold enters this result.

**T020R-MC2: persistent MC instability. BER-REGRET-A / BER-CAP-A / BER-SRC-A remain NOT_EXECUTED; no T020-R/C/F/X scientific diagnosis.** True composition, privileged utilities and query outcomes remain sealed. The complete R4096 policy fails the predefined numerical state-identity stability criterion. Its effect on true regret/query accuracy remains unknown; this is not a scientific failure conclusion about the neutral operators. No R increase, seed retry, alternate bootstrap, epsilon tie rule, confidence policy, feature work, SSL/TTT or federation was started.

## Label-free diagnostics

ALL4096 top-two utility-margin p05/median/p95: stable slots {summary['stable_margin_quantiles']}; unstable slots {summary['unstable_margin_quantiles']} (fractional utility units). BER256→BER4096 changes {changed}/8000 slots. Quarter-pair agreements, max/p95 exact-mean utility differences, vote differences, disagreement IDs and the full transition matrix are preserved in the CSV files. These diagnostics do not change the policy.

## Cycle-only repair and reuse

Normal active-set arithmetic remains unchanged. Only a repeated working set invokes the existing deterministic1023-face direct-RHS enumeration, followed by the unchanged finite/simplex/KKT/objective checks. No added clipping, renormalization, ridge, relaxed tolerance or external optimizer. Receipts explicitly distinguish the paths.

Exactly **2 cycle fallbacks** occurred: client64/A/Contrast/replica3071 and client87/A/Contrast/replica2688. No additional cycle appeared. Every fallback trace, selected face, objective and KKT is listed in `solver_fallbacks.csv`/JSON. Independent re-enumeration reproduces both frozen-solver cycles and verifies each returned optimum; no other failure mode is converted to a fallback.

Reused98 client pairs byte-for-byte; computed only64/87. This run added **76,800 CLS solves**, retaining **3,763,200 prior extension solves** and the immutable **256,000 first-prefix replicas**. The assembled result has exactly100clients/1000cells/8000unique slots and4,096,000logical replicas; it does not claim4.096M new solves in this run. Max KKT {max(c['max_KKT'] for c in cells):.6g}, max sum error {max(c['max_sum_error'] for c in cells):.6g}, max updates {max(c['max_updates'] for c in cells)}.

## Verification

Before resume:1000 real observations and4020 normal bootstrap samples yield bitwise-identical pi and unchanged normal receipt fields against the frozen implementation (apart from the explicit path flag), zero fallbacks, and8000 unchanged template decisions/argmax sets. Maximum stored cross-platform pi difference6.71456e-13 is below1e-12. Two observed-cycle fixtures, normal no-enumeration behavior and failed-certification stop behavior pass; full remote suite124testsPASS.

After full assembly: independent local replay checks **4000 CLS/position samples**, **8000 legacy aggregates**, **56000 exact block choices**, every final ALL4096 choice, quarter/half/ALL composition, all agreement rows and the unchanged convergence gate. Both cycle cases are independently re-enumerated and the recorded selected faces reconstructed from C/q. All **{v['source_manifest_entries_unchanged']} upstream manifest entries** remain unchanged. Local entropy alone uses2e-15 absolute tolerance for the observed Windows/Linux NumPy log last-bit difference; utility means, states and ties remain exact. Zero new model forwards and no true-regret/query-count calculation.

`phaseB_choices_freeze.json` binds all complete unlabeled outputs with `convergence_pass=false`, not permission to unseal. `preserved_98_verification.json` binds reused work; `resume_receipt.json` binds the two new clients and completeness. Exact accumulators and sampled NPZ files are retained locally/remotely under the new run. Original T020/T020R reports and raw receipts remain unchanged.

Local D: space exhaustion interrupted the first artifact extraction, not the experiment. The downloaded archive matched remote SHA256 `aae4ebed93f1ed54559ad0ae84a4820806f34aef7369168867228d2decb038db`. The redundant prior delivery tar was removed, the same archive extracted successfully, and its local transport copy removed after successful extraction. All resulting scientific files passed the checks above; no experiment was restarted.

Return this sealed MC2 result to Lead. The authorized solver repair is complete; the frozen scientific evaluation cannot proceed past the convergence stop.
'''
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    hand=f'''# T020R2 repair complete; sealed T020R-MC2 stop

{now}. Lead43d2f85/runtime{summary['runtime']}/run{RUN}.124testsPASS;98clients reused byte-identically;64/87 resumed with exactly2cycle fallbacks;complete100clients/8000slots. H1/H2={g['H1_H2']*100:.4f}%<99%; H1/ALL={g['H1_ALL']*100:.4f}%<99.5%; H2/ALL={g['H2_ALL']*100:.4f}%.94disagreements/38clients/59cells. Independent4000CLS/8000legacy/56000blocks/allfallbacks/{v['source_manifest_entries_unchanged']}upstreamentriesPASS. T020R-MC2; labels/true utilities/query sealed, all scientific gatesNOT_EXECUTED; no T020-R/C/F/X. Read results/t020r2_mc_convergence/RESULTS.md. AwaitLead; no furtherR/seeds/alternatepolicy/nextstage.
'''
    for name in ('HANDOFF.md','T020R2_HANDOFF.md'):(ROOT/'research_log'/name).write_text(hand,encoding='utf-8')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:f.write('\n'+hand)
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(hand+'\n'+table+'\nThe solver repair is complete; the full frozen convergence gate returns MC2. Scientific BER efficacy remains unassessed.\n',encoding='utf-8')
    save(ROOT/'research_log/t020r2_delivery.json',dict(timestamp=now,lead='43d2f85',runtime=summary['runtime'],run=RUN,status='T020R-MC2',fallbacks=2,global_agreement=g,results=OUT.relative_to(ROOT).as_posix(),next='Await Lead; no unsealing'))
    files=[p for p in (ROOT/'research_log/t020r2_receipts').rglob('*') if p.is_file() and p.suffix!='.tar']+[p for p in OUT.rglob('*') if p.is_file()]
    save(ROOT/'research_log/t020r2_artifact_manifest.json',[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print(report)


if __name__=='__main__':main()
