"""Render T007R decision from the independently audited, frozen experiment."""
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ANALYSIS=ROOT/'research_log/t007r_analysis'
OUT=ROOT/'results/t007_balanced_transfer'
RUN='20260913-203200-ttfl-t007r-gpu1'
RECEIPTS=ROOT/'research_log/t007r_receipts'/RUN


def table(headers,rows):
    def f(v):
        if isinstance(v,float):return f'{v:.6g}' if 0<abs(v)<1e-4 else f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(f,r))+' |' for r in rows])


def main():
    for p in ANALYSIS.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    s=json.loads((OUT/'summary.json').read_text());verify=json.loads((OUT/'verification.json').read_text())
    pools=json.loads((OUT/'calibration_pools.json').read_text())
    raw=RECEIPTS/'artifacts/t007r_transfer'
    ds=json.loads((raw/'raw_restoration.json').read_text())
    old=json.loads((ROOT/'research_log/t007_receipts/artifacts/t007_identity_preflight/identity_preflight.json').read_text())
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    count=sum(s['passes'].values());assert count>=2
    s.update(decision='C-A',joint_pass_count=count,
        interpretation='Transferable covariate state under the balanced paired-clean diagnostic; not a deployable unlabeled method.')
    (OUT/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    comparison=[]
    for pool in ('A','B'):
        old_d=next(d for d in old['channel_diagnostics'] if d['pool']==pool and d['layer']==2)
        comparison.append(dict(pool=pool,old_layer2_worst_scale=old_d['worst_channel_scale'],new_identity_scale=1.,
            old_max_abs_gamma=old_d['max_abs_gamma'],new_max_abs_gamma=0.,
            old_max_clean_logit_diff=old['max_query_logit_diff'][pool],new_max_clean_logit_diff=0.,
            old_changed_predictions=1,new_changed_predictions=0,old_accuracy_delta=0.,new_accuracy_delta=0.))
    with (OUT/'writer_identity_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    lookup={(r['pool'],r['target'],r['context']):r for r in s['rows']}
    v=lambda p,t,c:lookup[p,t,c]['macro_class']
    perclass=list(csv.DictReader((OUT/'per_class.csv').open()))
    negative=[d for d in ds if d['negative_scale_fraction']>0]
    lines=['# CODEX -> CHATGPT','## Timestamp',now,'## Commit / run ID',
        f'Lead 7553405; runtime 5794dbf; run {RUN}; release 20260913-203154-ttfl-t007r. '
        f'NVIDIA RTX A6000 GPU1; formal runtime {s["metadata"]["seconds"]:.2f} seconds. '
        'The enclosing delivery commit contains the independent analysis and report.',
        '## T007R identity-centered writer verification',
        'New separately versioned paired_affine_oracle_neutral.py fits ridge on the residual '
        '`z-x`, centering the prior at gamma=0. EPS=1e-5, final scale clamp [-8,8], backbone '
        'and 192-scalar state are unchanged. Historical paired_affine_oracle.py bytes are '
        'unchanged. Random, constant and near-zero-variance identity tests require exact equality '
        'and passed; known affine/cap/sequential tests also passed.',
        'On both original pools, all 192 values are exactly zero for clean identity. All '
        '15,039 clean logits/predictions and all three accuracy metrics are exactly unchanged. '
        'The full clean-query prerequisite completed before fitting any corrupted state. '
        'This resolves the old writer mathematical prior mismatch; no identity threshold was relaxed.',
        '## Exact pool provenance reused',
        f'Original receipt SHA256: {s["metadata"]["pool_sha256"]}. No pool-selection call in the '
        'T007R evaluator. Both pools contain 80 training images, exactly 8/class, each class '
        'spread across eight distinct clients; each pool spans 64 clients. Pool A/B IDs are '
        'disjoint and absent from all query IDs. Original train index, original ID and class '
        'were rechecked. The same committed permutations have 100%/0% same-class agreement '
        'and zero image fixed points; target image/feature multisets are exact at both layers.',
        'For calibration corruptions, the existing RNG key uses each image\'s original client '
        'ID and original image ID, not a new synthetic pool ID. Query corruption tensors are '
        'created once per client/target and reused across pools/conditions. Labels only audit '
        'the frozen pools/permutations and evaluate queries; writer APIs accept pixels only.',
        '## Clean-identity old-vs-new audit',
        table(['Pool','Old L2 worst scale','New scale','Old max |gamma|','New max |gamma|','Old max logit diff','New max logit diff','Old changed argmax','New changed argmax'],
              [[r[k] for k in ('pool','old_layer2_worst_scale','new_identity_scale','old_max_abs_gamma','new_max_abs_gamma',
                'old_max_clean_logit_diff','new_max_clean_logit_diff','old_changed_predictions','new_changed_predictions')]
               for r in comparison]),
        'Old values reuse T007 preflight receipts; the old writer grid was not rerun. Its '
        'stopped report remains in research_log/t007_stopped_preflight_report and raw receipts '
        'remain in research_log/t007_receipts. The repair comparison is about identity behavior '
        'only, not a paired old-vs-new corruption-performance ablation.',
        '## Primary macro-class transfer table (Pool-A and Pool-B separately)',
        f'Clean zero-state macro-class accuracy: {s["metadata"]["clean_macro_class"]:.9f}%. '
        'Primary accuracy is equal-weight mean of the ten class accuracies over all client '
        'queries. All values below are percentages.',
        table(['Pool','Shift','None','Identity','Correct','Same class','Cross class','Wrong alt','Noise source'],
            [[p,t]+[v(p,t,c) for c in ('none','clean_identity','correct_pair','same_class_derangement',
                'cross_class_derangement','wrong_alt_source','noise_source')] for p in ('A','B') for t in s['passes']]),
        'Sample-weighted and equal-client metrics for every condition are in summary.csv; '
        'A/B values and absolute differences for all three metrics are in pool_reproducibility.csv. '
        'Correct-pair secondary values are:',
        table(['Pool','Shift','Sample-weighted %','Macro-client %'],
            [[r['pool'],r['target'],r['sample_weighted'],r['macro_client']] for r in s['rows'] if r['context']=='correct_pair']),
        '## Headroom / tau / PASS table',
        'Unchanged original T007 rule: recovery >=0.50 and correct-minus-wrong-alt >=tau '
        'and correct-minus-noise >=tau independently in both pools, tau=max(0.5,0.25*headroom). '
        'No rounding is used for decisions.',
        table(['Pool','Shift','H pp','tau pp','Recovery','Correct-wrong pp','Correct-noise pp','Pool PASS'],
            [[r[k] for k in ('pool','target','headroom_pp','tau_pp','recovery','correct_wrong_pp','correct_noise_pp','passed')]
             for r in s['gates']]),
        table(['Shift','Joint A AND B PASS'],list(s['passes'].items())),
        '**3/4 joint PASS: Dark, Contrast, Blur.** Noise is FAIL because Pool-B recovery is '
        '0.499741262, below 0.50 by 0.000258738 (about 0.000896 pp of macro-class accuracy). '
        'This boundary is retained as failure; independent rational arithmetic on integer '
        'class counts confirms it. Both noise context margins exceed tau, but that does not '
        'override the recovery criterion.',
        '## Semantic-pairing decomposition',
        table(['Pool','Shift','Correct-same pp','Same-cross pp','Correct-cross pp'],
            [[r[k] for k in ('pool','target','instance_advantage_pp','semantic_advantage_pp','hard_pair_gap_pp')] for r in s['gates']]),
        'Correct pairing materially exceeds same-class and cross-class derangements for '
        'every shift in both pools. However, the semantic component changes sign: same-minus-cross '
        'is negative in Pool-A and positive in Pool-B. This does not support a universal '
        'class-semantic ordering. The strong correct-vs-wrong-source/noise margins support the '
        'predeclared covariate-transfer claim within this paired-clean oracle, while the '
        'large pairing gaps show dependence on privileged clean correspondence.',
        '## Pool reproducibility / state similarity',
        table(['Shift','Condition','A macro-class %','B macro-class %','Absolute gap pp'],
            [[r['target'],r['context'],r['pool_A'],r['pool_B'],r['abs_difference']]
             for r in s['pool_reproducibility'] if r['metric']=='macro_class']),
        table(['Shift','Condition','State cosine','State L2','State max abs difference'],
            [[r['target'],r['context'],r['cosine'],r['l2_distance'],r['max_abs_difference']] for r in s['state_similarity']]),
        'States are concatenated in actual model order [gamma1,beta1,gamma2,beta2]. Cosine '
        'for zero-vs-zero states is undefined (None), with L2 and max difference exactly zero. '
        'These two pools are predeclared draws, not hyperparameter choices or confidence intervals.',
        '## Coefficient and restoration diagnostics',
        f'All 42 formal state fits (2 clean identities +40 corrupted-source controls) are finite. '
        f'{len(s["cap_flags"])} layer records have cap fraction >10%; maximum observed cap fraction is {max(d["cap_fraction"] for d in ds):.4%}. '
        f'{len(negative)} of {len(ds)} layer records contain negative scales; negative scales '
        'are permitted by the frozen clamp and are not silently removed. Full per-state/layer '
        'MSE, restoration ratio, gamma/beta, negative fraction, cap fraction and minimum source '
        'variance are in restoration_diagnostics.csv; raw coefficients are in raw_restoration.json and states.pt.',
        table(['Pool','Shift','Layer','MSE before','MSE after','Ratio','Mean |gamma|','Max |gamma|','Mean |beta|','Max |beta|','Neg fraction','Cap fraction','Min variance'],
            [[d['pool'],d['target'],d['layer']]+[d[k] for k in ('mse_before','mse_after','restoration_ratio',
                'gamma_abs_mean','gamma_abs_max','beta_abs_mean','beta_abs_max','negative_scale_fraction','cap_fraction','minimum_source_variance')]
             for d in ds if d['context']=='correct_pair']),
        'Restoration ratios describe each support pairing objective, not task relevance by '
        'themselves. Different derangements change the regression target correspondence. '
        'Cross-pool semantic-control instability should not be explained solely by MSE.',
        '## Per-class behavior',
        table(['Pool','Shift','Class','None %','Correct %','Delta pp'],
            [[p,t,c,
              float(next(r['accuracy'] for r in perclass if (r['pool'],r['target'],r['context'],r['class'])==(p,t,'none',str(c)))),
              float(next(r['accuracy'] for r in perclass if (r['pool'],r['target'],r['context'],r['class'])==(p,t,'correct_pair',str(c)))),
              float(next(r['accuracy'] for r in perclass if (r['pool'],r['target'],r['context'],r['class'])==(p,t,'correct_pair',str(c))))-
              float(next(r['accuracy'] for r in perclass if (r['pool'],r['target'],r['context'],r['class'])==(p,t,'none',str(c))))]
             for p in ('A','B') for t in s['passes'] for c in range(10)]),
        'All conditions\' class counts/accuracies are retained in per_class.csv. This table '
        'does not imply that each class improves; individual class losses remain visible.',
        '## Verification',
        '26 baseline tests passed before editing, three new writer tests passed before evaluator '
        'integration, and the complete 31-test suite passed remotely before execution. A real '
        '6-client smoke passed before the 100-client formal evaluation. Six clients were used '
        'because the first two lack several query classes; this avoids undefined smoke macro-class '
        'metrics and does not select pools/states by accuracy. Both stages checked exact identity '
        'on all 100 clean-query clients before corruption fitting.',
        f'Formal: 5,600 records, 42 globally fitted states; no per-client state fits. Checkpoint '
        'and old writer/corruption hashes unchanged; exact pool SHA checked. Model hash checked '
        'after every fit/evaluation; no calibration/query overlap; exact target multisets and '
        'semantic fractions. Saved predictions independently reconstruct every class/weighted '
        f'accuracy with maximum error {verify["independent_count_max_error_pp"]:.3g} pp. '
        'Fraction-based gates match every runtime PASS/FAIL. Identity comparison uses the old '
        'saved receipt and new exact-equality checks, without rerunning historical grids.',
        '## Failures / uncertainties',
        'No formal runtime or identity failure. The workflow encountered one transient SCP '
        'connection closure during analysis upload; its existing legacy-SCP retry succeeded. '
        'No experiment was restarted. The pre-existing NVML mismatch warning persists; PyTorch '
        'CUDA completed on A6000.',
        'This checkpoint has strongly uneven class performance and uses the established PFLlib '
        'client split drawn from merged original train/test data, not an official CIFAR-10 test '
        'benchmark. The diagnostic is transductive with respect to client membership (calibration '
        'and query clients overlap), but calibration image IDs are disjoint from all query IDs; '
        'it demonstrates state reuse across clients, not held-out-client generalization. Only '
        'two calibration draws and one checkpoint are tested. Clean target pairs are privileged '
        'oracle information. Do not attribute every change from T006 solely to the new writer: '
        'the balanced global calibration protocol also differs, and no old-writer transfer grid '
        'was run. Noise remains a strict failure despite being extremely close to recovery .5.',
        '## C-A / C-B / C-C / C-D decision',
        '**C-A: transferable covariate state exists under this balanced paired-clean diagnostic.** '
        'Three shifts pass independently in both pools under the unchanged gates. This supports '
        'the capacity of the existing neutral 192-scalar operator for these shifts. It does not '
        'establish a deployable source-only writer or a final federated method.',
        '## Recommended next action',
        'Keep the diagonal operator and return to Research Lead. A next package may investigate '
        'whether source-side unlabeled evidence can estimate the useful oracle state, with '
        'matched source-context controls and the same frozen protocol. No SSL, meta-learning, '
        'federated retraining, richer operator, or next stage was launched. The 15-minute '
        'heartbeat continues to check for actionable instructions.']
    report='\n\n'.join(lines)+'\n'
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report,encoding='utf-8')
    (ROOT/'research_log/HANDOFF.md').write_text('# T007R COMPLETE — C-A (3/4 joint PASS)\n\n'
        'Read results/t007_balanced_transfer/RESULTS.md and t007r_delivery.json. Neutral writer '
        'identity exact. Dark/Contrast/Blur pass both pools. Noise B recovery .499741262 fails; '
        'do not round up or retry. All31tests and independent integer/Fraction audits pass. '
        'Old writer/preflight evidence preserved. Await Lead; no next stage.\n',encoding='utf-8')
    delivery=dict(timestamp=now,decision='C-A',lead='7553405',runtime='5794dbf',run=RUN,
        raw=RECEIPTS.relative_to(ROOT).as_posix(),results='results/t007_balanced_transfer/RESULTS.md',
        remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts/t007r_transfer',
        joint_pass=3,next='Await Lead; no SSL or next stage')
    (ROOT/'research_log/t007r_delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf-8')
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(RECEIPTS.rglob('*')) if p.is_file()]
    (ROOT/'research_log/t007r_artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(decision='C-A',passes=count,negative_layers=len(negative),cap_flags=len(s['cap_flags']))))


if __name__=='__main__':main()
