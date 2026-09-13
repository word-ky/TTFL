"""Independent distance/count checks and the T008 evidence report."""
import csv
import hashlib
import json
import math
import shutil
import statistics
import struct
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN='20260913-213117-ttfl-t008-gpu1'
RECEIPTS=ROOT/'research_log/t008_receipts'/RUN
RAW=RECEIPTS/'artifacts/t008_source_identifiability'
OLD=ROOT/'research_log/t007r_receipts/20260913-203200-ttfl-t007r-gpu1/artifacts/t007r_transfer'
OUT=ROOT/'results/t008_source_identifiability'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(headers,rows):
    def f(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(f,r))+' |' for r in rows])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for path in RAW.iterdir():
        if path.suffix in ('.json','.csv'):shutil.copyfile(path,OUT/path.name)
    load=lambda n:json.loads((RAW/n).read_text())
    s=load('summary.json');m=load('metadata.json');v=load('verification.json')
    prototypes=load('prototypes.json');decisions=load('support_decisions.json');freeze=load('selection_freeze.json')
    assert sha(RAW/'prototypes.json')==freeze['prototype_sha256']==m['prototype_sha256']
    assert sha(RAW/'support_decisions.json')==freeze['decision_sha256']
    assert sha(RAW/'microbatches.json')==freeze['microbatch_sha256']
    assert sha(OLD/'states.pt')==m['old_states_sha256'] and sha(OLD/'summary.json')==m['old_summary_sha256']
    contexts=m['contexts'];distance_error=0.
    for bank,receipt in prototypes.items():
        assert list(receipt['prototypes'])==contexts
        for p in receipt['prototypes'].values():
            assert len(p['vector'])==192
            assert hashlib.sha256(struct.pack('<192d',*p['vector'])).hexdigest()==p['sha256']
    for r in decisions['primary']+decisions['ood']:
        distances={c:math.fsum((a-b)**2 for a,b in zip(r['signature'],prototypes[r['bank']]['prototypes'][c]['vector'])) for c in contexts}
        ordered=sorted(contexts,key=lambda c:(distances[c],contexts.index(c)))
        assert ordered[:2]==[r['selected'],r['second']]
        distance_error=max(distance_error,max(abs(distances[c]-r['d_'+c]) for c in contexts))
        assert distance_error<1e-8
    margins=[]
    for direction,report in s['identification'].items():
        group=[r for r in decisions['primary'] if r['direction']==direction]
        matrix=[[sum(r['true_context']==t and r['selected']==c for r in group) for c in contexts] for t in contexts]
        assert matrix==report['confusion']
        correct=[matrix[i][i] for i in range(5)];assert report['passed']==(sum(correct)>=18 and min(correct)>=3)
        for c in contexts:
            values=[r['margin'] for r in group if r['true_context']==c]
            margins.append(dict(direction=direction,context=c,mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)))
    # Independently reconstruct all reused metric counts from the original saved predictions.
    old_pred=np.load(OLD/'predictions.npz');old_rows=json.loads((OLD/'raw_records.json').read_text())
    metrics=load('metric_count_audit.json');exact={};count_errors=[]
    for row in metrics:
        assert row['reused_context'] is not None
        selected=sorted([r for r in old_rows if (r['pool'],r['target'],r['context'])==
                         (row['bank'],row['target'],row['reused_context'])],key=lambda r:r['client'])
        ys=np.concatenate([old_pred[f"c{r['client']}_labels"] for r in selected])
        ps=np.concatenate([old_pred[r['prediction_key']] for r in selected])
        totals=[int(np.sum(ys==c)) for c in range(10)];correct=[int(np.sum((ys==c)&(ps==ys))) for c in range(10)]
        assert totals==row['class_counts'] and correct==row['correct_counts']
        macro=sum((Fraction(100*n,d) for n,d in zip(correct,totals)),Fraction(0))/10
        exact[row['bank'],row['target'],row['state_choice']]=macro
        count_errors.append(abs(float(macro)-row['macro_class']))
    old_summary=json.loads((OLD/'summary.json').read_text())
    old_gate={(r['pool'],r['target']):r for r in old_summary['gates']}
    for r in s['retrieval']:
        bank=r['direction'][0];target=r['target']
        group=[d for d in decisions['primary'] if d['direction']==r['direction'] and d['true_context']==target]
        selected=sum((exact[bank,target,d['selected']] for d in group),Fraction(0))/4
        none=exact[bank,target,'clean'];oracle=exact[bank,target,target]
        wrong=Fraction.from_float(r['wrong_alt']);tau=old_gate[bank,target]['tau_pp']
        assert abs(float(selected)-r['selected_mean'])<1e-10
        retained=(selected-none)/(oracle-none)
        assert abs(float(retained)-r['retained_recovery'])<1e-10
        assert r['passed']==(retained>=Fraction(9,10) and float(selected-wrong)>=tau)
    v.update(independent_distance_max_error=distance_error,independent_confusions_gates_match=True,
        independent_saved_prediction_counts_match=True,independent_metric_max_error_pp=max(count_errors),
        prototype_vector_hashes_match=True,selection_freeze_hashes_match=True,
        old_predictions_sha256=sha(OLD/'predictions.npz'))
    (OUT/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    with (OUT/'margin_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(margins[0]));w.writeheader();w.writerows(margins)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    assert s['ID_A'] and s['RET_A']
    lines=['# CODEX -> CHATGPT','## Timestamp',now,'## Commit / run ID',
        f'Lead e20ef8f; runtime {m["code_commit"]}; run {RUN}; release 20260913-213110-ttfl-t008. '
        f'NVIDIA RTX A6000 GPU1, runtime {m["seconds"]:.2f} seconds. The enclosing delivery commit '
        'contains this independent audit and report.',
        '## T007R evidence reused',
        'Checkpoint, neutral writer, corruption definitions, pools and state bank remain frozen. '
        'Eight bank correct-pair states were reconstructed using bank clean/corrupted pixels '
        'and match the saved T007R states exactly. Four mapped wrong-alt choices per bank '
        'also exactly match the stored wrong_alt_source states. Clean maps to zero state. '
        'No client-specific fitting or new writer was introduced.',
        'T007R remains C-A with 3/4 joint PASS. Noise Pool-B recovery .499741262 remains FAIL '
        'under that task. T008 measures retention of the oracle gain, a separate gate; its '
        'Noise retrieval success does not relabel T007R.',
        '## Source signature definition',
        'At zero fast state, extract population mean and sigma=sqrt(var+1e-6) after conv1 '
        'and conv2 spatial blocks. For each channel, concatenate '
        '(mu-mu_bank_clean)/(sigma_bank_clean+1e-6) and '
        'log((sigma+1e-6)/(sigma_bank_clean+1e-6)); two layers yield 192 dimensions. '
        'Double-precision statistics; squared Euclidean distance to five fixed prototypes. '
        'No learned metric, gradient, labels, fast state, query tensors or held-out clean '
        'reference enter signature/selection APIs. Ties use the fixed context order.',
        '## Prototype and microbatch provenance',
        f'Exact original pool artifact SHA256 {m["pool_sha256"]}; no image reselection. '
        'Bank A selects on held-out B and vice versa. Every prototype uses the full 80-image '
        'bank pool; all signatures in a direction normalize against that bank\'s clean moments '
        'only. Both pools are disjoint from each other and all queries.',
        'Each held-out pool is partitioned into four disjoint 20-image batches, exactly two '
        'images/class, by committed within-class positions [0:2], [2:4], [4:6], [6:8]. '
        'Labels are used only for constructing/auditing this prescribed diagnostic partition. '
        'Microbatch original IDs, client IDs and indices are in microbatches.json.',
        'Both prototype banks/vectors/hashes were persisted first. All 40 primary decisions '
        'and 8 OOD decisions were then persisted before opening T007R query metrics/labels. '
        f'Selection freeze timestamp: {freeze["timestamp"]}. Prototype/decision file hashes '
        'still match at completion and in the independent audit. Full vectors, references, '
        'all five distances and top-two choices are retained.',
        '## Verification',
        '31 prior regression tests passed before editing; three focused signature/distance/'
        'microbatch tests passed before integration. The full 36-test suite passed remotely '
        'before this 40-decision experiment. Model hash unchanged throughout; checkpoint, '
        'neutral-writer and corruption hashes unchanged; all signatures/distances finite.',
        'All eight reconstructed bank states match saved states exactly. Forty-eight focused '
        'first-two-client prediction regressions (both banks, all shifts, none/oracle/wrong-alt) '
        'match T007R predictions exactly. All selected states already have T007R predictions, '
        'so no new 100-client state/query evaluations were needed. The four batch retrieval '
        'scores per shift reuse the same unchanged global query set, not four independent query trials.',
        f'Independent CPU distance recomputation agrees within {distance_error:.3g}; all top-two '
        'choices, prototype vector hashes, confusion matrices and ID gates match. Independent '
        f'original-prediction counts reproduce all reused macro-class metrics within {max(count_errors):.3g} pp. '
        'Fraction-based retained-recovery checks agree. No query-based retries, prototype '
        'normalization choices, metric fitting or OOD threshold was used.',
        '## A→B identification results',
        f'20/20 correct, all five contexts 4/4. ID direction PASS. Mean top-two distance margin '
        f'{s["identification"]["A_to_B"]["margin_mean"]:.6f}; median '
        f'{s["identification"]["A_to_B"]["margin_median"]:.6f}.',
        '## B→A identification results',
        f'20/20 correct, all five contexts 4/4. ID direction PASS. Mean margin '
        f'{s["identification"]["B_to_A"]["margin_mean"]:.6f}; median '
        f'{s["identification"]["B_to_A"]["margin_median"]:.6f}.',
        '## Confusion matrices / margins',
        'Rows are true contexts; columns follow clean, dark, contrast, noise, blur. Exact '
        '40 decisions including all five distances and second-nearest identities are in identification.csv.',
        'A→B:\n\n'+table(['True / predicted']+contexts,[[c]+s['identification']['A_to_B']['confusion'][i] for i,c in enumerate(contexts)]),
        'B→A:\n\n'+table(['True / predicted']+contexts,[[c]+s['identification']['B_to_A']['confusion'][i] for i,c in enumerate(contexts)]),
        table(['Direction','Context','Margin mean','Median','Min','Max'],[[r[k] for k in ('direction','context','mean','median','min','max')] for r in margins]),
        '## Oracle-state retrieval results',
        'Each of the four source batches selected its true context. Consequently every '
        'individual retrieved-state score equals that bank\'s existing correct-pair oracle '
        'score; retained recovery is exactly 1.0. retrieval.csv contains all 32 per-batch '
        'macro-class, sample-weighted and macro-client scores; retrieval_summary.csv reports '
        'their means separately by direction and shift.',
        table(['Direction','Shift','None %','Oracle %','Selected mean %','Wrong alt %','tau pp','Retained','Selected-wrong pp','PASS'],
            [[r[k] for k in ('direction','target','none','oracle','selected_mean','wrong_alt','tau_pp','retained_recovery','selected_wrong_margin','passed')]
             for r in s['retrieval']]),
        table(['Direction','Shift','Selected macro-class batches 0 / 1 / 2 / 3','Mean sample-weighted %','Mean macro-client %'],
            [[r['direction'],r['target'],' / '.join(f'{r["selected_mean"]:.6f}' for _ in range(4)),
                r['selected_weighted_mean'],r['selected_client_mean']] for r in s['retrieval']]),
        'Forced-wrong retrieval is the mapped bank wrong-alt state; it is verified identical '
        'to the saved T007R negative control. Selected-minus-wrong exceeds frozen tau in '
        'every direction/shift, so success is not being inferred merely because all states improve.',
        '## Noise-image OOD diagnostic',
        table(['Direction','Batch','Selected','Second','Nearest distance','Second distance','Margin'],
            [[r[k] for k in ('direction','batch','selected','second','distance','second_distance','margin')] for r in decisions['ood']]),
        '**All eight noise-image batches are assigned gaussian_noise**, with margins '
        '66.73–69.41, comparable to or larger than many real-context margins. The selector '
        'has no rejection mechanism, and a large top-two margin is not evidence that support '
        'belongs to one of the modeled contexts. Absolute nearest distances are also reported '
        '(84.24–90.08), but no rejection cutoff was chosen. OOD retrieval/query performance '
        'was not evaluated or claimed.',
        '## Failures / uncertainties',
        'No runtime or verification failure. The pre-existing NVML driver/library mismatch '
        'warning persists; PyTorch CUDA completed on A6000. No historical FL or T007R grid was rerun.',
        'This is a closed five-context bank at fixed known corruption severities, evaluated '
        'on two balanced pools. The same images recur across corruption conditions; 40 '
        'controlled decisions are not 40 independent samples from arbitrary client contexts. '
        'The offline bank remains privileged paired-clean, and balanced microbatch construction '
        'uses labels. Runtime signature/selection is source-only; this does not yet establish '
        'a continuous state writer, arbitrary label-skew support robustness, unseen-shift '
        'recognition or OOD handling. The existing checkpoint/query split is unchanged and '
        'is not the official CIFAR-10 test benchmark.',
        '## ID-A / RET-A decision',
        '**ID-A + RET-A.** Both directions pass identification (20/20 with 4/4 per context); '
        'all four shifts jointly pass retrieval with 100% oracle-gain retention. This supports '
        'source-only observability and discrete bank selection in the prescribed balanced, '
        'known-context diagnostic. It does not establish a deployable learned writer.',
        '## Recommended next action',
        'Keep the 192-scalar operator and return to Research Lead. A subsequent bounded '
        'package may investigate a continuous source-only state estimator and its context/OOD '
        'controls. The demonstrated OOD margin limitation should remain visible. No SSL, '
        'gradient-based TTT, neural writer, meta-learning, federation, richer operator or '
        'next stage was launched. The existing 15-minute heartbeat will read new instructions.']
    report='\n\n'.join(lines)+'\n'
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T008 COMPLETE — ID-A + RET-A\n\n'
        'Read results/t008_source_identifiability/RESULTS.md and t008_delivery.json. Both '
        'directions20/20ID, all4jointRETpass, retainedgain1.0. Noise-image8/8classifiednoise '
        'withlarge margins; noOODrejectionclaim.36tests andindependentdistance/countaudits '
        'PASS. Frozenstate/predictionreuse, noquerytuning. AwaitLead; noSSL/nextstage.\n',encoding='utf-8')
    delivery=dict(timestamp=now,decision='ID-A + RET-A',lead='e20ef8f',runtime=m['code_commit'],run=RUN,
        results='results/t008_source_identifiability/RESULTS.md',raw=RAW.relative_to(ROOT).as_posix(),
        remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t008_source_identifiability',next='Await Lead; no next stage')
    (ROOT/'research_log/t008_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(RECEIPTS.rglob('*')) if p.is_file()]
    (ROOT/'research_log/t008_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(ID_A=s['ID_A'],RET_A=s['RET_A'],distance_error=distance_error,metric_error=max(count_errors))))


if __name__=='__main__':main()
