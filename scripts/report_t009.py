"""Independent T009 count/distance checks and failure attribution from saved predictions."""
import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN='20260913-224708-ttfl-t009-gpu1'
FAILED='20260913-224559-ttfl-t009-gpu1'
RECEIPTS=ROOT/'research_log/t009_receipts'
RAW=RECEIPTS/RUN/'artifacts/t009_natural_context'
OLD=ROOT/'research_log/t007r_receipts/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
T008=ROOT/'research_log/t008_receipts/20260913-213117-ttfl-t008-gpu1/artifacts/t008_source_identifiability'
OUT=ROOT/'results/t009_natural_context'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(headers,rows):
    def f(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(f,r))+' |' for r in rows])


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for p in RAW.iterdir():
        if p.suffix in ('.json','.csv') and p.name not in ('source_decisions.json','mixed_query_records.json'):
            shutil.copyfile(p,OUT/p.name)
    load=lambda n:json.loads((RAW/n).read_text())
    s=load('summary.json');m=load('metadata.json');v=load('verification.json');manifest=load('natural_support_manifest.json')
    freeze=load('selection_freeze.json');decisions=load('source_decisions.json');qr=load('mixed_query_records.json')
    skew=load('skew_summary.json');skew_rows=load('skew_labels_receipt.json');ood=load('distance_ood_comparison.json')
    prior=json.loads((RECEIPTS/FAILED/'artifacts/t009_natural_context/natural_support_manifest.json').read_text())
    assert {k:v for k,v in manifest.items() if k!='frozen_at'}=={k:v for k,v in prior.items() if k!='frozen_at'}
    assert sha(RAW/'natural_support_manifest.json')==freeze['manifest_sha256']==m['support_manifest_sha256']
    assert sha(RAW/'source_decisions.json')==freeze['decisions_sha256']==m['decision_sha256']
    prototype=json.loads((T008/'prototypes.json').read_text());contexts=list(prototype['A']['prototypes'])
    distance_error=0.
    for r in decisions:
        d={c:math.fsum((x-y)**2 for x,y in zip(r['signature'],prototype[r['bank']]['prototypes'][c]['vector'])) for c in contexts}
        order=sorted(contexts,key=lambda c:(d[c],contexts.index(c)))
        assert order[:2]==[r['selected'],r['second']]
        signed=min(x for c,x in d.items() if c!=r['true_context'])-d[r['true_context']]
        distance_error=max(distance_error,abs(signed-r['signed_true_margin']),max(abs(d[c]-r['d_'+c]) for c in contexts))
    assert distance_error<1e-8
    pred=np.load(RAW/'mixed_predictions.npz');oldp=np.load(OLD/'predictions.npz')
    oldrows=json.loads((OLD/'raw_records.json').read_text());oi={(r['client'],r['pool'],r['target'],r['context']):r for r in oldrows}
    global_y=np.concatenate([pred[f'c{cid}_labels'] for cid in range(100)]);totals=[int(np.sum(global_y==c)) for c in range(10)]
    def macro(counts):return sum((Fraction(100*n,t) for n,t in zip(counts,totals)),Fraction(0))/10
    old_clean=np.concatenate([oldp[f'c{cid}_clean'] for cid in range(100)])
    clean_counts=[int(np.sum((global_y==c)&(old_clean==global_y))) for c in range(10)];clean=macro(clean_counts)
    audits=[];exact={};harms=[];metric_error=0.
    for r in qr:
        y=pred[f"c{r['client']}_labels"];pr=pred[r['prediction_key']]
        total=np.bincount(y,minlength=10).tolist();counts=np.bincount(y[y==pr],minlength=10).tolist()
        assert total==r['class_total'] and counts==r['class_correct']
        if r['reused_old_key'] is not None:assert np.array_equal(pr,oldp[r['reused_old_key']])
        if r['selected']!=r['target']:
            oracle=oldp[f"c{r['client']}_clean"] if r['target']=='clean' else oldp[oi[r['client'],r['bank'],r['target'],'correct_pair']['prediction_key']]
            oracle_counts=np.bincount(y[y==oracle],minlength=10)
            contribution=macro([a-int(b) for a,b in zip(counts,oracle_counts)])
            harms.append(dict(bank=r['bank'],client=r['client'],target=r['target'],selected=r['selected'],n_query=len(y),
                selected_correct=sum(counts),oracle_correct=int(oracle_counts.sum()),
                aggregate_macroclass_delta_pp=float(contribution),
                entropy=skew_rows[r['client']]['normalized_entropy'],max_class_fraction=skew_rows[r['client']]['max_class_fraction']))
    for r in load('integer_count_receipts.json'):
        ps=np.concatenate([pred[f"{r['bank']}|{r['target']}|c{cid}"] for cid in range(100)])
        counts=[int(np.sum((global_y==c)&(ps==global_y))) for c in range(10)]
        assert counts==r['class_correct'] and totals==r['class_total']
        value=macro(counts);exact[r['bank'],r['target']]=value
        metric_error=max(metric_error,abs(float(value)-r['macro_class']))
        audits.append(dict(bank=r['bank'],target=r['target'],total=totals,correct=counts,macro_class=float(value),sample_weighted=100*sum(counts)/sum(totals)))
    for r in s['retrieval']:
        bank,target=r['bank'],r['target'];oldmetrics={}
        for ctx in ('none','correct_pair','wrong_alt_source'):
            ps=np.concatenate([oldp[oi[cid,bank,target,ctx]['prediction_key']] for cid in range(100)])
            oldmetrics[ctx]=macro([int(np.sum((global_y==c)&(ps==global_y))) for c in range(10)])
        selected=exact[bank,target];gain=oldmetrics['correct_pair']-oldmetrics['none']
        retained=(selected-oldmetrics['none'])/gain;tau=max(Fraction(1,2),(clean-oldmetrics['none'])/4)
        assert r['passed']==(gain>0 and retained>=Fraction(9,10) and selected-oldmetrics['wrong_alt_source']>=tau)
        assert abs(float(retained)-r['retained_recovery'])<1e-10
        assert abs(sum(h['aggregate_macroclass_delta_pp'] for h in harms if (h['bank'],h['target'])==(bank,target))-
                   float(selected-oldmetrics['correct_pair']))<1e-10
    for r in s['clean_safety']:
        assert r['passed']==(exact[r['bank'],'clean']>=clean-Fraction(1,2))
        assert abs(sum(h['aggregate_macroclass_delta_pp'] for h in harms if h['bank']==r['bank'] and h['target']=='clean')-r['delta_pp'])<1e-10
    for r in s['identification']:
        group=[d for d in decisions if d['bank']==r['bank'] and d['K']==r['K']]
        matrix=[[sum(d['true_context']==t and d['selected']==c for d in group) for c in contexts] for t in contexts]
        assert matrix==r['confusion'];correct=[matrix[i][i] for i in range(5)]
        assert r['passed']==(sum(correct)>=450 and min(correct)>=80)
    assert s['NID_A'] and not s['NRET_A']
    # Resolve the launch alias without rewriting the original runtime metadata.
    code=subprocess.check_output(['git','show','fab17bd:scripts/eval_t009_natural_context.py'],cwd=ROOT)
    runtime_bytes=(ROOT/'scripts/eval_t009_natural_context.py').read_bytes()
    code_sha=hashlib.sha256(runtime_bytes).hexdigest()
    assert code_sha=='6684a810489a0ec69dc837627cfa6dc9503073dd13d9e7aec304e65220b43173'
    assert runtime_bytes.replace(b'\r\n',b'\n')==code.replace(b'\r\n',b'\n')
    revision=dict(actual_commit='fab17bd',raw_launch_label=m['code_commit'],deployed_evaluator_sha256=code_sha,
        git_blob_sha256=hashlib.sha256(code).hexdigest(),evidence='Remote sha256sum matches deployed CRLF bytes; LF-normalized source matches git commit. Raw IMPORT_FIX label retained.')
    (ROOT/'research_log/t009_runtime_revision.json').write_text(json.dumps(revision,indent=2),encoding='utf-8')
    m['resolved_runtime_revision']=revision;(OUT/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    v.update(independent_distance_max_error=distance_error,independent_prediction_count_error_pp=metric_error,
        independent_fraction_retrieval_clean_gates_match=True,independent_confusion_gates_match=True,
        retry_support_ids_K_identical=True,freeze_hashes_match=True,mixed_predictions_sha256=sha(RAW/'mixed_predictions.npz'))
    (OUT/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    (OUT/'integer_prediction_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    write_csv(OUT/'harmful_confusions.csv',sorted(harms,key=lambda r:r['aggregate_macroclass_delta_pp']))
    s.update(decision='N-C',decision_rationale='Both NID gates pass; NRET fails clean safety in both banks, and blur retention fails. Localized skew-dependent errors are task-costly.')
    (OUT/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    counts=list(manifest['eligible_counts'].values())
    primary=[r for r in s['identification'] if r['primary']]
    lines=['# CODEX -> CHATGPT','## Timestamp',now,'## Commit / run ID',
        f'Lead 25e6ea7; actual runtime commit fab17bd; run {RUN}; release '
        f'20260913-224704-ttfl-t009-importfix; NVIDIA RTX A6000 GPU1; {m["seconds"]:.2f} seconds. '
        'The launch metadata string IMPORT_FIX was a label, not a commit SHA. '
        't009_runtime_revision.json maps it to fab17bd using the deployed evaluator SHA, '
        'matched against committed source after CRLF/LF normalization. Raw metadata was not silently rewritten. '
        'The enclosing delivery commit contains analysis and report.',
        '## Frozen T008/T007R evidence reused',
        'Checkpoint, 192-scalar operator, neutral writer, source-signature code, T008 prototypes, '
        'T007R pool membership/state tensors, corruptions and PFLlib splits are unchanged. '
        'No prototype rebuild, state fitting, distance change or normalization change. '
        'T007R Noise Pool-B remains a failure under its historical gate; T009 has separate gates.',
        '## Natural-support construction and chosen K',
        f'Eligible counts range {min(counts)}–{max(counts)}. The largest common allowed K is '
        '**20**: at least one client has only 30 eligible images, preventing K=32 or64. '
        'All 100 clients remain included; no per-client size choices or image balancing.',
        'Before loading image pixels or running model features, selection used only train/query '
        'IDs: exclude both calibration pools and all query IDs, shuffle each eligible train-order '
        'list with random.Random seeded by SHA256(7:client:T009-natural-support), then take '
        'the common prefix. natural_support_manifest.json stores counts, K, original IDs and '
        'train indices. The same 20 IDs are used for all five contexts and both banks.',
        f'Manifest frozen at {manifest["frozen_at"]}; all 1,000 decisions frozen at '
        f'{freeze["timestamp"]}, before query metrics or support-label statistics. Support y '
        'was accessed for audit only after query evaluation. The PFLlib container also stores '
        'labels, but the feature phase accesses only its x field; no label value enters selection/signatures.',
        '## Verification',
        '36 existing tests passed before edits. Two label-blind manifest tests passed locally; '
        'the full 40-test suite passed remotely before the successful run. All frozen hashes '
        'match, no support/calibration/query ID overlap, no state fits, model hash unchanged '
        'throughout, all values finite. Decision and manifest hashes are unchanged after analysis.',
        '985 of 1,000 client/target/bank query policies reuse exact old predictions; 15 new '
        'query forwards cover clean clients that selected a corruption state. No secondary-K '
        'retrieval was run. All 24 frozen bank/shift/none-oracle-wrong macro-class metrics '
        'reconstruct exactly from old counts. Every mixed policy is independently recomputed '
        f'from saved predictions; maximum macro-class error {metric_error:.3g} pp. '
        f'Independent signature-distance error {distance_error:.3g}; all confusions and '
        'signed margins match. Fraction arithmetic confirms retrieval and clean-safety gates.',
        '## Bank-A natural identification',
        '488/500 = **97.6%**, NID PASS. Per-context clean/dark/contrast/noise/blur: '
        '92%,100%,100%,100%,96%.',
        '## Bank-B natural identification',
        '490/500 = **98.0%**, NID PASS. Per-context clean/dark/contrast/noise/blur: '
        '93%,100%,100%,100%,97%.',
        '## Per-context confusion / signed margins',
        'Rows true, columns predicted. Signed margin is nearest wrong-context distance '
        'minus true-context distance; negative means misclassification. Full per-decision '
        'distances and margins are in identification.csv.',
        *[f'Bank {r["bank"]}:\n\n'+table(['True / predicted']+contexts,[[c]+r['confusion'][i] for i,c in enumerate(contexts)]) for r in primary],
        table(['Bank','Mean signed margin','Median signed margin','Nearest p50','p90','p95','max'],
              [[r['bank'],r['signed_margin_mean'],r['signed_margin_median']]+[r['nearest_distance_quantiles'][k] for k in ('p50','p90','p95','max')] for r in primary]),
        '## Per-client mixed-state retrieval',
        table(['Bank','Shift','None %','Oracle %','Mixed %','Wrong alt %','tau pp','Retained','Mixed-wrong pp','PASS'],
              [[r[k] for k in ('bank','target','none','oracle','selected_mixed','wrong_alt','tau_pp','retained_recovery','selected_wrong_margin','passed')] for r in s['retrieval']]),
        'Dark, Contrast and Noise retain 100% oracle gain in both banks. Blur retains only '
        '**34.30% /53.89%** because 4/3 clients choose the wrong state. Thus 3/4 shifted '
        'targets pass jointly, but this alone is insufficient: both clean-safety checks fail.',
        table(['Bank','Shift','Mixed weighted %','Mixed macro-client %'],
              [[r['bank'],r['target'],r['sample_weighted'],r['macro_client']] for r in s['retrieval']]),
        '## Clean safety',
        table(['Bank','Zero macro-class %','Mixed macro-class %','Delta pp','Required minimum %','PASS'],
              [[r['bank'],r['none'],r['selected'],r['delta_pp'],r['none']-.5,r['passed']] for r in s['clean_safety']]),
        '**NRET-A fails** because clean macro-class drops 0.610321/0.616185 pp, exceeding '
        'the fixed 0.5 pp allowance. Clean decisions contain 8/7 false adaptations: primarily '
        'clean→blur, plus one clean→noise in each bank. This is not a failure of zero-state '
        'neutrality; it is an incorrect nonzero state choice on clean input.',
        'The following contributions sum exactly to mixed-minus-oracle macro-class accuracy '
        '(for clean, oracle means zero state). They weight each changed class count by the '
        'global class denominator, not by averaging client percentage changes. Client29 alone '
        'selects clean under blur and contributes about -0.77pp in each bank:',
        table(['Bank','Client','True context','Selected','Queries','Mixed correct','Oracle correct','Aggregate delta pp'],
              [[r[k] for k in ('bank','client','target','selected','n_query','selected_correct','oracle_correct','aggregate_macroclass_delta_pp')]
               for r in sorted(harms,key=lambda r:r['aggregate_macroclass_delta_pp'])]),
        '## Label-skew audit',
        'Audit occurs after all decisions/retrieval. Quartiles contain 25 clients sorted by '
        '(normalized entropy, client ID); ties may straddle adjacent quartiles. This rule '
        'was fixed before seeing results. No significance tests or fitted cutoffs.',
        table(['Bank','Context','Entropy quartile','Entropy min','max','ID accuracy %','Median signed margin'],
              [[r[k] for k in ('bank','context','quartile','entropy_min','entropy_max','accuracy','median_signed_margin')] for r in skew['quartiles']]),
        'All Blur identification errors occur in the lowest-entropy quartile (84%/88% '
        'accuracy there vs100% in the other quartiles). Clean identification is 80%/84% '
        'in Q1 and88% in Q3; Q2/Q4 are100%. This supports a localized content-composition '
        'vulnerability, despite passing aggregate NID. It does not justify labeling the '
        'entire natural-support signature unobservable.',
        table(['Bank','Context','Skew feature','Signed-margin Spearman'],
              [[r[k] for k in ('bank','context','feature','spearman')] for r in skew['correlations']]),
        'The per-context correlations are small (absolute rho at most about0.171). '
        'Tail errors and their task cost are more informative here than a strong monotonic '
        'entropy-margin relationship. Associations are descriptive, not causal.',
        'Ten most negative signed-margin decisions:',
        table(['Bank','Client','Context','Selected','Signed margin','Entropy','Max class fraction','Classes'],
              [[r['bank'],r['client'],r['true_context'],r['selected'],r['signed_true_margin'],
                r['skew']['normalized_entropy'],r['skew']['max_class_fraction'],r['skew']['represented_classes']] for r in skew['worst_ten']]),
        '## Secondary K=20 diagnostic (if applicable)',
        'Not applicable: primary K already equals20. No additional smaller-support gate or '
        'secondary query run was created. T008 balanced microbatches also had20 images, '
        'so the comparison is not explained by a different nominal support size alone; '
        'the image/content draws differ.',
        '## OOD-distance comparison',
        table(['Bank','Known p50','p90','p95','max','T008 noise min','median','max'],
              [[b]+[r['known'][k] for k in ('p50','p90','p95','max')]+[r['t008_noise'][k] for k in ('min','median','max')] for b,r in ood.items()]),
        'There is no observed distance overlap in this comparison: natural-known maximum '
        'is48.22, whereas the old balanced noise-image minimum is84.24. This is descriptive '
        'separation for these saved draws, not a validated general rejection threshold. '
        'No threshold was selected; T008\'s high-margin OOD misclassification remains unchanged.',
        '## Failures / uncertainties',
        f'Initial run {FAILED} stopped with a missing read_data import at the first support '
        'pixel load, after manifest persistence and before any decision/query result. The '
        'only repair was importing the existing PFLlib utility. Failed and successful '
        'manifests have exactly the same counts, K, IDs, train indices and order (only '
        'timestamps differ). Both receipts are preserved. No outcome-informed retry occurred.',
        'The retry used launch label IMPORT_FIX; actual deployed commit fab17bd is independently '
        'resolved by source SHA and recorded separately. This provenance defect did not '
        'change computation and was not hidden by rewriting the raw receipt. The known '
        'NVML mismatch warning remains; CUDA execution completed.',
        'Only one label-blind support draw per client and fixed known corruption severities '
        'were tested. Banks remain privileged paired-clean. The checkpoint and PFLlib '
        'merged-original-train/test query split are unchanged, not an official CIFAR-10 '
        'benchmark. High ID accuracy is not a bound on harm from rare, client-dependent '
        'state substitutions. No new state was fit, and no historical result was relabeled.',
        '## NID / NRET decision',
        '**Case N-C: NID-A passes; NRET-A fails.** Natural source context naming largely '
        'survives label skew, but rare wrong state choices are costly: Blur retention fails '
        'and clean safety fails in both banks. Low-entropy concentration is a relevant '
        'localized weakness, not an alternative reason to ignore the passing NID gate.',
        '## Recommended next action',
        'Keep the operator/state bank fixed and return to Research Lead with the harmful '
        'confusion ledger. A next package should address sensitivity of clean↔blur and '
        'blur→contrast substitutions or audit state interpolation/clean preservation. '
        'No such mechanism, threshold, interpolation, SSL, learned writer, federation, '
        'meta-learning or richer operator was implemented. Await the next bounded instruction.']
    report='\n\n'.join(lines)+'\n'
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8');(ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T009 COMPLETE — N-C\n\n'
        'Read results/t009_natural_context/RESULTS.md and t009_delivery.json. K20; NID passes '
        '97.6/98.0%; NRET fails clean safety (-.6103/-.6162pp) and blur retention (.343/.539). '
        'Low-entropy tail errors harmful; do not alter thresholds or rerun. Runtime fab17bd '
        'resolved from IMPORT_FIX alias by SHA.40tests/count/distance audits pass. AwaitLead; '
        'noSSL/interpolation/nextstage.\n',encoding='utf-8')
    delivery=dict(timestamp=now,decision='N-C',lead='25e6ea7',runtime='fab17bd',run=RUN,failed_run=FAILED,
        results='results/t009_natural_context/RESULTS.md',raw=RAW.relative_to(ROOT).as_posix(),
        remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t009_natural_context',next='Await Lead')
    (ROOT/'research_log/t009_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    for path in RECEIPTS.rglob('*.json'):
        if path.name in ('source_decisions.json','mixed_query_records.json'):path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    artifacts=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(RECEIPTS.rglob('*')) if p.is_file()]
    (ROOT/'research_log/t009_artifact_manifest.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision='N-C',distance_error=distance_error,metric_error=metric_error,harmful=harms),indent=2))


if __name__=='__main__':main()
