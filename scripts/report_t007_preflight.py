"""Persist the T007 identity-prerequisite diagnosis without interpreting unrun shifts."""
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'research_log/t007_receipts/artifacts/t007_identity_preflight'
OUT=ROOT/'results/t007_balanced_transfer'
RUN='20260913-193525-ttfl-t007-preflight-gpu1'


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def table(headers,rows):
    def fmt(v):
        if isinstance(v,float):return f'{v:.6g}' if 0<abs(v)<1e-4 else f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(fmt,r))+' |' for r in rows])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=json.loads((RAW/'identity_preflight.json').read_text())
    pools=json.loads((RAW/'calibration_pools.json').read_text())
    ds=json.loads((RAW/'restoration_diagnostics.json').read_text())
    pred=np.load(RAW/'identity_predictions.npz')
    # Independent count construction from saved predictions, no model rerun.
    y=np.concatenate([pred[f'c{i}_labels'] for i in range(100)])
    totals=[int(sum(y==c)) for c in range(10)];audits=[]
    for name in ('none','A','B'):
        p=np.concatenate([pred[f'c{i}_{name}'] for i in range(100)])
        counts=[int(sum((y==c)&(p==y))) for c in range(10)]
        macro=sum(100*n/d for n,d in zip(counts,totals))/10
        assert counts==s['metrics'][name]['correct'] and abs(macro-s['metrics'][name]['macro_class'])<1e-10
        audits.append(dict(condition=name,class_counts=totals,correct_counts=counts,macro_class=macro,
            changed_predictions_vs_zero=int(sum(p!=np.concatenate([pred[f'c{i}_none'] for i in range(100)])))))
    vectors={}
    for name in ('A','B'):
        vectors[name]=[v for d in ds if d['pool']==name for vs in ([v-1 for v in d['a']],d['b']) for v in vs]
    va,vb=vectors['A'],vectors['B']
    similarity=dict(condition='clean_identity',cosine=sum(a*b for a,b in zip(va,vb))/
        math.sqrt(sum(a*a for a in va)*sum(b*b for b in vb)),
        l2_distance=math.sqrt(sum((a-b)**2 for a,b in zip(va,vb))),max_abs_difference=max(abs(a-b) for a,b in zip(va,vb)))
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    decision='STOP: clean-identity state prerequisite not satisfied; C-A/C-B/C-C/C-D not evaluated'
    summary=dict(timestamp=now,decision=decision,run=RUN,metadata=s,
        formal_shift_evaluation_run=False,primary_gate_pass_count=None,identity_state_similarity=similarity)
    verification=dict(baseline_tests=24,full_remote_tests=26,balanced_pool_tests_local=2,
        pools_exact_80_and_8_per_class=True,pool_ids_disjoint=True,pool_query_overlap=0,
        all_calibration_ids_from_train=True,same_class_fraction=1.,cross_class_fraction=0.,
        both_derangements_zero_fixed_points=True,exact_target_id_image_feature_multisets=True,
        pool_selection_before_query=True,labels_never_enter_writer=True,
        checkpoint_writer_corruption_hashes_unchanged=True,zero_state_exact=True,model_unchanged=True,
        all_coefficients_finite=True,identity_accuracy_gate_pass=True,
        identity_state_prerequisite_pass=False,independent_macro_class_count_audit_pass=True,
        identity_writer_episodes=2,corruption_writer_episodes=0,no_cap_or_epsilon_change=True)
    for name,value in [('summary.json',summary),('verification.json',verification),('integer_prediction_audit.json',audits)]:
        (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    shutil.copyfile(RAW/'calibration_pools.json',OUT/'calibration_pools.json')
    shutil.copyfile(RAW/'per_client.csv',OUT/'per_client.csv')
    write_csv(OUT/'summary.csv',[dict(condition=n,macro_class=m['macro_class'],sample_weighted=m['sample_weighted'],
        macro_client=m['macro_client']) for n,m in s['metrics'].items()])
    write_csv(OUT/'per_class.csv',[dict(condition=n,**{'class':c},n_query=totals[c],correct=m['correct'][c],accuracy=acc)
        for n,m in s['metrics'].items() for c,acc in enumerate(m['per_class'])])
    write_csv(OUT/'state_similarity.csv',[similarity])
    write_csv(OUT/'restoration_diagnostics.csv',[{k:v for k,v in d.items() if k not in ('a','b')} for d in ds])
    lines=['# CODEX -> CHATGPT','## Timestamp',now,'## Commit / run ID',
        f'Lead 3394d2c; runtime bc07059; run {RUN}; release 20260913-193520-ttfl-t007-preflight. '
        'NVIDIA RTX A6000 GPU1. Remote exit 0 at 2026-09-13 19:35:45 +08. '
        'The enclosing delivery commit contains this diagnosis.',
        '## T007 decision (C-A/C-B/C-C/C-D)',
        '**STOP at the explicitly required clean-identity prerequisite. No C-A/B/C/D assignment.** '
        'Both pools preserve clean macro-class accuracy exactly, but produce materially non-identity '
        'second-layer coefficients. No corrupted-query transfer experiment was run; primary transfer '
        'pass count is N/A, not 0/4. This is a mathematical writer/identity-contract conflict, '
        'not evidence that covariate transfer fails.',
        'The active instruction says: “clean_identity must also change clean macro-class accuracy by '
        '<=0.1 pp and produce no material non-identity state; otherwise stop and diagnose before '
        'interpreting the shift results.” The state half is not satisfied: channel 29 scales are '
        '0.323925 and 0.469380 instead of 1. No new numeric materiality cutoff was invented.',
        '## Why T006 was insufficient',
        'T006 low-semantic pairing still retained 57% same-class pairs. T007 pools and permutations '
        'remove that control limitation exactly. The new transfer question remains unanswered '
        'because its own identity prerequisite is not satisfied.',
        '## Calibration-pool construction and provenance',
        'Enumerated all existing training-partition candidates. Per pool/class, sorted by '
        'SHA256(t007:pool:class:client:original_id), took one per distinct client before any '
        'fallback, excluded previously selected IDs, and selected exactly 8/class. No query '
        'accuracy entered selection. Both pools were persisted before query evaluation. '
        'calibration_pools.json records all original IDs, train indices, client IDs, labels and permutations.',
        table(['Pool','Images','Distinct clients','Distinct clients per class 0..9'],
            [[n,len(p['samples']),p['unique_clients'],','.join(str(p['clients_per_class'][str(c)]) for c in range(10))]
             for n,p in pools.items()]),
        'Pool IDs are disjoint from each other and all 15,039 query IDs. Each same-class permutation '
        'has zero image fixed points and 100% same-class agreement; each cross-class permutation '
        'has zero fixed points and 0% same-class agreement. Exact clean image/feature multisets '
        'were checked at both layers. Labels only construct/audit pools and pairing controls.',
        '## Verification',
        '24 existing tests passed remotely before edits. Two new pool/control tests passed locally; '
        'the full 26-test suite passed remotely before preflight. Two states were fitted, once '
        'per pool, then reused over all 100 clients. Frozen checkpoint, writer and corruption '
        'hashes unchanged; zero-state logits exact; model unchanged; all coefficients finite. '
        'Independent saved-prediction counts reproduce every macro-class score. Both identity '
        'states each change one prediction out of 15,039, without changing class-wise correct counts or accuracy.',
        '## Primary macro-class transfer results',
        'Not run: no corruption states were fitted and no transfer gates are interpreted. '
        'The prerequisite clean-query results are:',
        table(['Condition','Macro-class %','Sample-weighted %','Macro-client %'],
            [[n,m['macro_class'],m['sample_weighted'],m['macro_client']] for n,m in s['metrics'].items()]),
        '## Sample-weighted / macro-client secondary results',
        'Shown above for the clean-identity prerequisite only. The same PFLlib merged '
        'original-train/test client split is retained; this is not the official CIFAR-10 test benchmark.',
        '## Pool-A vs Pool-B reproducibility',
        'Clean accuracy absolute A-B difference is 0 pp for all three metrics. Maximum clean '
        f'query logit differences from zero state are {s["max_query_logit_diff"]["A"]:.9f} '
        f'(A) and {s["max_query_logit_diff"]["B"]:.9f} (B). Prediction equality does not make '
        'the fitted coefficients identity.',
        '## Correct vs semantic/hard controls',
        'Pool permutations verified; their corrupted-source states were not fitted. No comparison is claimed.',
        '## Per-class behavior',
        table(['Class','Query count','Zero-state correct','Accuracy % (identical A/B)'],
            [[c,totals[c],s['metrics']['none']['correct'][c],s['metrics']['none']['per_class'][c]] for c in range(10)]),
        'The pre-existing checkpoint is severely class-uneven (class 9 has zero correct queries, '
        'class 4 has one). This is retained as baseline context; no retraining or label tuning was performed.',
        '## State similarity',table(list(similarity),[list(similarity.values())]),
        'Concatenated state follows the actual model order [gamma1,beta1,gamma2,beta2]. '
        'This describes clean-identity states only, not transferable corruption states.',
        '## Restoration / coefficient diagnostics',
        table(['Pool','Layer','MSE before','MSE after','Ratio','Mean |gamma|','Max |gamma|','Mean |beta|','Max |beta|','Negative fraction','Cap fraction'],
            [[d['pool'],d['layer']]+[d[k] for k in ('mse_before','mse_after','restoration_ratio','gamma_abs_mean',
             'gamma_abs_max','beta_abs_mean','beta_abs_max','negative_scale_fraction','cap_fraction')] for d in ds]),
        'Undefined MSE_after/MSE_before when before=0 is reported as None; it is not a zero restoration error.',
        table(['Pool','Layer','Worst channel (zero-based)','Source variance','EPS','Scale','Max feature difference'],
            [[d['pool'],d['layer'],d['worst_channel'],d['worst_channel_variance'],d['eps'],
              d['worst_channel_scale'],d['max_feature_diff']] for d in s['channel_diagnostics']]),
        'For a clean identity input, the unchanged first-layer fit is '
        '`a = variance / (variance + EPS)` and `b = mean * (1-a)`, not exactly a=1,b=0. '
        'The second layer additionally sees the already-corrected first-layer source. Its channel '
        '29 source variances are 4.78418e-6 (A) and 8.82384e-6 (B), comparable to or smaller '
        'than EPS=1e-5. This explains strong scale shrinkage despite small feature/logit changes '
        'and almost unchanged argmax predictions. No cap is active and no scale is negative in these two identity states. '
        'This repeats the known low-variance mechanism seen in T004 on a new balanced pool; '
        'it is not a newly introduced implementation regression.',
        '## Failures / uncertainties',
        'The process and provenance checks passed. The scientific prerequisite failed at the '
        'state level. The instruction does not quantify “material” state change; a 53–68% '
        'scale contraction is clearly not a small coefficient perturbation, even though all '
        'but one clean prediction agree in each pool. It would be incorrect to silently replace the state requirement '
        'with accuracy equality. Conversely, this observation alone says nothing about corrupted '
        'query transfer. The pre-existing NVML warning remains; PyTorch CUDA completed successfully.',
        '## Recommended next action (recommendation only; do not launch)',
        'Research Lead should reconcile the fixed-writer constraint with the identity-state '
        'prerequisite: explicitly permit the measured low-variance contraction based on functional '
        'evidence, or issue a separate writer revision that is identity-preserving and re-freeze '
        'the comparison. Neither option was selected or implemented here. Preserve these exact '
        'Pool-A/B IDs for continuation. No epsilon/cap edits, pool retries, SSL, FL, richer '
        'operator or T008. The 15-minute heartbeat will check for the next actionable instruction.']
    report='\n\n'.join(lines)+'\n'
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T007 STOPPED — identity prerequisite\n\n'
        'Read results/t007_balanced_transfer/RESULTS.md and T007_HANDOFF.md. Pools fixed and verified; '
        'two clean identity states shrink layer2 channel29 by 53–68% despite identical accuracies (one prediction changes per pool). '
        'No corrupted query transfer run; C-A/B/C/D N/A. Await Lead reconciliation; do not repeat '
        'preflight or change EPS/caps automatically.\n',encoding='utf-8')
    delivery=dict(timestamp=now,status='stopped_at_explicit_identity_prerequisite',lead='3394d2c',runtime='bc07059',
        run=RUN,raw=RAW.relative_to(ROOT).as_posix(),report='results/t007_balanced_transfer/RESULTS.md',
        remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t007_identity_preflight',
        next='Await Lead; preserve pools; no T008')
    (ROOT/'research_log/t007_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted((ROOT/'research_log/t007_receipts').rglob('*')) if p.is_file()]
    (ROOT/'research_log/t007_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=decision,audit=audits,similarity=similarity)))


if __name__=='__main__':main()
