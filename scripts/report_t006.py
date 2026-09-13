"""Render the completed T006 audit from measured summary artifacts (stdlib only)."""
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/t006_semantic_pairing'
RUN = '20260913-183307-ttfl-t006-gpu1'
RECEIPTS = ROOT / 'research_log/t006_receipts' / RUN


def table(headers, rows):
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, float) else str(x)
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(fmt, row)) + ' |' for row in rows])


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    s.pop('numerical_case', None)
    s['numeric_candidates'] = ['S-B', 'S-C']
    s.pop('interpretation_pending', None)
    s['decision'] = 'S-B'
    s['decision_rationale'] = ('Noise and Blur satisfy the predeclared S-B inequalities. '
        'Dark and Contrast satisfy only the numerical S-C inequalities; random aggregate '
        'same-label fractions span 62.90% to 64.12%, which does not establish the broad '
        'semantic variation required by the full S-C interpretation. S-A is 0/4. '
        'Effects are shift-specific and do not establish universal semantic dominance.')
    (OUT / 'summary.json').write_text(json.dumps(s, indent=2))
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    acc = {(r['target'], r['context']): r['accuracy'] for r in s['rows']}
    e = s['effects']
    lines = ['# CODEX -> CHATGPT', '## Timestamp', now,
        '## Commit / run ID',
        f'Lead instruction: f93c0ff. Runtime code: {s["metadata"]["code_commit"]}. Run: {RUN}. '
        'The enclosing delivery commit contains the analysis and report. NVIDIA RTX A6000 GPU1; '
        'formal evaluation 39.98 seconds, 0.0674 GiB peak PyTorch allocated memory (not whole-device memory).',
        '## T006 decision (S-A/S-B/S-C/S-D)',
        '**S-B, with shift-specific limitations.** ' + s['decision_rationale'],
        'The numerical counts are S-A 0/4, S-B 2/4, S-C 2/4. These are overlapping evidence '
        'checks, not mutually exclusive labels. No new threshold or tie-break rule is introduced. '
        'Noise semantic advantage exceeds tau by only 0.0366 pp; Blur instance advantage '
        'misses tau by 0.0216 pp. Integer prediction counts confirm these boundary decisions.',
        '## Why T005 P-B needed a semantic-pairing audit',
        'T005 removed image fixed points but retained 63.1735% query-weighted same-class pairing '
        '(59.6789% equal-client). Thus a label-free pairing algorithm did not remove class semantics '
        'under the existing alpha=0.1 partition. This audit manipulates pairing only, preserving '
        'the target multiset. It does not evaluate a deployable label-free writer.',
        '## Verification',
        '22 runtime tests and the real 2-client smoke passed before the 100-client run; two '
        'additional statistical-helper tests passed remotely during analysis. 5,600 records, '
        '4,000 new writer episodes. All 24 focused old-writer regressions reproduce predictions '
        'exactly; reused accuracy error is 0 pp. The first two clients cover all four old '
        'conditions and corruptions; remaining old rows reuse T005 receipts.',
        'All new permutations have zero fixed points and identical ID/image/feature multisets. '
        'Low/high offsets independently recomputed from support labels; random offsets distinct '
        'and deterministic. Support/query overlap 0. Frozen model hash unchanged in every '
        'episode, zero-state logits exact, no nonfinite coefficients. Labels only construct/audit '
        'controls, never enter writer/state fitting. Independent prediction accuracy error '
        '<6.48e-6 pp. Clean reference is 5,146/15,039 = 34.217700645%; integer counts reproduce '
        'all decision inequalities. Checkpoint, writer, corruption hashes and eps/caps are in metadata.',
        'The same PFLlib client split is used throughout. It was made from merged original '
        'train/test data with disjoint per-client support/query partitions; these numbers are '
        'not an official CIFAR-10 test benchmark.',
        '## Same-label fraction audit',
        'All fractions below are percentages. Query weighting matches the main accuracy metric; '
        'equal-client statistics are also reported. Low/high select extrema over nonzero cyclic '
        'offsets, not over every possible permutation.',
        table(['Pairing', 'Query-weighted %', 'Client mean %', 'Client SD %', 'Client min %', 'Client max %'],
              [[r['context'], 100*r['query_weighted_mean']] +
               [100*r['client_distribution'][k] for k in ('mean','sd','min','max')]
               for r in s['semantic_summary']]),
        f'Within-client high-minus-low same-class fraction: mean {100*s["semantic_spread"]["mean"]:.4f} pp, '
        f'SD {100*s["semantic_spread"]["sd"]:.4f}, range 0 to {100*s["semantic_spread"]["max"]:.4f}. '
        'Low still averages 57.1933% query-weighted same-class agreement, so semantic leakage is reduced, not removed.',
        '## Main headroom-normalized results',
        table(['Shift','No adapt %','Correct %','H pp','Recovery','tau pp','S-A','S-B','S-C numeric'],
              [[r['target'],acc[r['target'],'none'],acc[r['target'],'correct_pair'],r['headroom_pp'],
                r['recovery_correct'],r['tau_pp'],r['S_A'],r['S_B'],r['S_C']] for r in e]),
        '## Correct vs low/high semantic derangements',
        table(['Shift','Low %','High %','Correct-high pp','High-low pp','Correct-low pp','Wrong alt %','Noise support %'],
              [[r['target'],acc[r['target'],'low_semantic_derangement'],acc[r['target'],'high_semantic_derangement'],
                r['instance_advantage_pp'],r['semantic_advantage_pp'],r['marginal_gap_pp'],
                acc[r['target'],'wrong_alt'],acc[r['target'],'noise_source']] for r in e]),
        '## Random-derangement distribution and centered correlations',
        'Eight distinct offsets per client. SD is descriptive population SD over eight aggregate '
        'replicates, not a confidence interval. Pearson/Spearman use 800 observations per shift '
        'after centering accuracy and same-class fraction within each client across random offsets. '
        'Spearman ranks the centered values with average ties. No p-values or independent-trial claims.',
        table(['Shift','Random mean %','SD','Min','Max','Correct-mean pp','Fraction below correct','Centered r','Centered rho'],
              [[r['target']] + [r['accuracy'][k] for k in ('mean','sd','min','max')] +
               [r['correct_minus_random_mean'],r['fraction_random_below_correct'],r['centered_pearson'],r['centered_spearman']]
               for r in s['random']]),
        'Random aggregate same-class fraction (identical across corruptions): query-weighted '
        'mean 63.3297%, SD 0.3582 pp, min 62.8980%, max 64.1232%; equal-client mean 60.1501%, '
        'SD 0.3566 pp, min 59.7472%, max 61.0122%. Positive centered correlations are modest '
        '(r 0.105–0.163; rho 0.051–0.145). S-B is assigned from its preset inequalities, '
        'not by inventing a material-correlation threshold.',
        '## Per-client paired summaries',
        'Equal-client deltas in pp; fraction positive excludes ties. Aggregate query-weighted '
        'deltas above differ because client query counts differ.',
        table(['Shift','Delta','Mean','Median','Fraction >0','p10','p25','p75','p90'],
              [[r['target'],name]+[d[k] for k in ('mean','median','fraction_positive','p10','p25','p75','p90')]
               for r in e for name,d in r['paired'].items()]),
        '## Restoration / coefficient diagnostics',
        'Layer summaries below are equal-client means except explicitly marked maxima. '
        'Restoration ratio is mean per-client MSE_after/MSE_before; undefined zero-denominator '
        'ratios are excluded. Caps are unchanged at scale [-8,8].',
        table(['Shift','Pairing','Layer','MSE before','MSE after','Ratio','Mean |gamma|','Max |gamma|','Mean |beta|','Max |beta|','Negative fraction','Cap mean','Cap max'],
              [[r['target'],r['context'],r['layer']]+[r[k] for k in
                ('mse_before','mse_after','restoration_ratio','gamma_abs_mean','gamma_abs_max_max',
                 'beta_abs_mean','beta_abs_max_max','negative_scale_fraction','cap_fraction','cap_fraction_max')]
               for r in s['restoration'] if r['context'] in
               ('correct_pair','low_semantic_derangement','high_semantic_derangement')]),
        'Lower restoration MSE alone is not task evidence: in Low contrast, correct pairing '
        'has worse query accuracy than both semantic derangements. Different target pairings '
        'also define different reconstruction targets. The present controls do not isolate '
        'a causal relation between MSE and query accuracy.',
        '## Failures / uncertainties',
        'No formal runtime failure. Local analysis initially failed due to duplicate OpenMP '
        'libraries; the unchanged analysis calculations and two helper tests ran successfully '
        'in the established remote environment, without the unsafe duplicate-library override. '
        'The pre-existing NVML warning persists; PyTorch CUDA on A6000 completed the run.',
        'There are 9 formal layer episodes with cap fraction >10%; all are retained and listed '
        'in cap_flags.json. This can affect attribution. Small headroom in Noise/Blur makes '
        'threshold decisions close. Client heterogeneity is large, with negative lower-tail '
        'deltas. Same-class agreement in low controls remains high, and the cyclic-offset '
        'family explores limited pairings. Results support the requested S-B diagnostic, '
        'not a universal decomposition of all oracle utility into semantics.',
        '## Recommended next action (recommendation only; do not launch)',
        'Return to Research Lead for a class-balanced, cross-client corruption-transfer '
        'diagnostic that separates client semantic marginals from covariate context. '
        'Do not use T004/T005 oracle gains alone as proof of context reading. No T007, SSL, '
        'operator expansion, or FL retraining was launched. The existing 15-minute heartbeat '
        'should read new instructions and remain quiet when no actionable change exists.']
    report = '\n\n'.join(lines) + '\n'
    (OUT/'RESULTS.md').write_text(report, encoding='utf-8')
    (ROOT/'coordination/CODEX_TO_CHATGPT.md').write_text(report, encoding='utf-8')
    diag = json.loads((RECEIPTS/'artifacts/t006_semantic_pairing/raw_restoration.json').read_text())
    flags = [{k:v for k,v in d.items() if k not in ('a','b')} for d in diag if d['cap_fraction']>.1]
    (OUT/'cap_flags.json').write_text(json.dumps(flags,indent=2))
    manifest = []
    for path in sorted(RECEIPTS.rglob('*')):
        if path.is_file() and path.suffix != '.gz':
            data = path.read_bytes()
            manifest.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest()))
            if path.name == 'raw_restoration.json':
                path.with_suffix('.json.gz').write_bytes(gzip.compress(data,mtime=0))
    (ROOT/'research_log/t006_artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    delivery = dict(timestamp=now,decision='S-B',lead='f93c0ff',runtime=s['metadata']['code_commit'],
                    run=RUN,results='results/t006_semantic_pairing/RESULTS.md',
                    remote=f'/home/wenchang/asdasdsad/wjq/TTFL/runs/{RUN}/artifacts',
                    heartbeat='ttfl-chatgpt: existing ACTIVE 15-minute heartbeat',next_task='Await Lead; no T007 launched')
    (ROOT/'research_log/t006_delivery.json').write_text(json.dumps(delivery,indent=2))
    (ROOT/'research_log/HANDOFF.md').write_text('# T006 COMPLETE — S-B\n\n'
        'Read results/t006_semantic_pairing/RESULTS.md and t006_delivery.json. '
        'S-A 0/4, S-B 2/4; S-C numeric 2/4 lacks the broad random semantic variation qualifier. '
        'All evaluation and integer checks passed. Await Research Lead; do not launch T007.\n', encoding='utf-8')
    with (ROOT/'research_log/T006_HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {now} — COMPLETE\n\nS-B; see results/t006_semantic_pairing/RESULTS.md. '
                '5600 records/4000 new episodes; 22 runtime +2 analysis tests passed remotely. '
                'Old rows exact; integer audit passed. Raw receipts preserved locally and remotely. No T007.\n')
    with (ROOT/'research_log/progress.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {now} — T006 complete\n\nRun {RUN}; S-B on Noise/Blur. '
                'S-A0/4; Dark/Contrast numerical S-C only, random semantic variation narrow. '
                'Local OpenMP analysis conflict resolved by remote analysis (no override). '
                'Report, raw receipts, integer audit and manifest persisted. Await Lead, no T007.\n')
    print(json.dumps(dict(decision='S-B',cap_flags=len(flags),files=len(manifest))))


if __name__ == '__main__':
    main()
