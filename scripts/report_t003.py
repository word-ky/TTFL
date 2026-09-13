"""Render T003 evidence with explicit post-run M-B interpretation; no new inference."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();raw,out=Path(a.raw),Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    s=json.loads((raw/'summary.json').read_text())
    assert s['passed_corruptions']==1 and s['metadata']['clients']==100
    s['case']='M-B'
    s['case_interpretation']='Post-run qualitative adjudication: partial specificity, strongest on dark and contrast, but only1/4 frozen gates pass. Clean-reference calibration harms clean clients; this does not isolate affine capacity from reference mismatch or moment insufficiency.'
    verify=json.loads((raw/'verification.json').read_text())
    sanity=list(csv.DictReader((raw/'clean_sanity.csv').open()))
    predictions=np.load(raw/'predictions.npz')
    errors=[]
    for r in sanity:
        cid=int(r['client']);labels=predictions[f'c{cid}_labels']
        for key,field in [('none','before'),('moment','after')]:
            accuracy=100*np.mean(predictions[f'c{cid}_clean_{key}']==labels)
            errors.append(abs(accuracy-float(r[field])))
    assert max(errors)<1e-4
    verify['clean_sanity_prediction_max_error_pp']=max(errors)
    s['clean_sanity'].update(clients_worse=sum(float(r['delta'])<0 for r in sanity),
        median_delta=float(np.median([float(r['delta']) for r in sanity])),
        max_logit_abs_difference=max(float(r['logit_abs_difference_max']) for r in sanity))
    for name in ('summary.csv','per_client.csv','clean_sanity.csv','moment_diagnostics.csv',
                 'reference_moments.pt','reference_moments.json'):
        shutil.copyfile(raw/name,out/name)
    (out/'summary.json').write_text(json.dumps(s,indent=2))
    (out/'verification.json').write_text(json.dumps(verify,indent=2))
    provenance=json.loads((raw/'reference_moments.json').read_text())
    lines=['# T003 — Moment-Written Fast Context Operator','',
        '**Case M-B; 1/4 PASS, below the required2/4.** Dark passes all four frozen2pp tests. Contrast beats both wrong contexts and noise but fails to improve no-adaptation accuracy. Noise/blur lose accuracy; this is partial moment-context signal, not an overall success.',
        '', '**Clean sanity warning:** clean-support calibration changes clean-query accuracy '
        f"{s['clean_sanity']['before']:.3f}% -> {s['clean_sanity']['after']:.3f}% ({s['clean_sanity']['delta']:.3f}pp), "
        f"with {s['clean_sanity']['clients_worse']}/100 clients worse. Both clean layers have zero scale clamping. This reference/writer can harm clean client predictions even when support moments are matched closely. Affine capacity alone is not identified as the cause.",
        '', '## Main results', '',
        '| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    targets=[r['target'] for r in s['specificity']]
    for target in targets:
        lines.append('| '+target+' | '+' | '.join(f"{r['accuracy']:.3f}" for r in s['rows'] if r['target']==target)+' |')
    lines+=['','Accuracies are sample-weighted percentages; macro-client accuracy is in summary.csv. CE rows are copied from T002, not recomputed.', '',
        '| Target | Gain pp | Correct-clean pp | Correct-alt pp | Correct-noise pp | PASS |',
        '|---|---:|---:|---:|---:|:---:|']
    for r in s['specificity']:
        lines.append('| '+r['target']+' | '+' | '.join(f"{r[k]:.3f}" for k in
            ('correct_gain_pp','correct_minus_wrong_clean_pp','correct_minus_wrong_alt_pp','correct_minus_noise_pp'))+f" | {r['passed']} |")
    lines+=['','Frozen gate: every gain/gap above>=2pp; at least2/4targets required. No query tuning or new operator. M-A is arithmetic; the M-B interpretation is post-run descriptive adjudication, recorded separately from immutable raw outputs.',
        '', '## Per-client specificity', '',
        '| Target / wrong | Mean | Median | Fraction>0 | P10 | P25 | P75 | P90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['specificity']:
        for wrong in ('clean','alt'):
            d=r[f'wrong_{wrong}_paired']
            lines.append(f"| {r['target']} / {wrong} | "+' | '.join(f"{d[k]:.3f}" for k in
                ('mean','median','fraction_positive','p10','p25','p75','p90'))+' |')
    lines+=['', '## Moment and clamp diagnostics', '',
        '**Clamp warning:** Dark support layer1 hits scale=4 in5/32channels (15.625%) for all100clients, appearing in Dark/correct and Noise/wrong-alt. Contrast support layer1 exceeds10%clamping in2clients, appearing in Contrast/correct and Blur/wrong-alt. Total204/3400layer episodes exceed10%. Bounds remain fixed. These are repeated conditions, not204distinct clients.',
        '', 'Before/after mismatch is mean_channel(|mu-mu_global|+|sigma-sigma_global|), measured on actual sequential calibrated support activations. All3400layer episodes decrease; max change is negative. Matching low-order moments is not sufficient to preserve task accuracy.', '',
        '| Correct support / layer | Mean abs gamma | Max abs gamma | Mean abs beta | Max abs beta | Clamp fraction | Mismatch before | Mismatch after |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in s['diagnostics']:
        if d['context']=='moment_correct' or d['target']=='clean_sanity':
            lines.append(f"| {d['target']} / {d['layer']} | "+' | '.join(f"{d[k]:.6f}" for k in
                ('gamma_abs_mean','gamma_abs_max_max','beta_abs_mean','beta_abs_max_max','clamp_fraction','moment_mismatch_before','moment_mismatch_after'))+' |')
    lines+=['','All layer/context diagnostic summaries are in moment_diagnostics.csv; raw per-channel moments are preserved in the run receipts.',
        '', '## Reference and verification', '',
        f"Reference uses all {provenance['samples']} training images across100clients, no cap, no query IDs and no labels forwarded. Collection order is client0..99/stored sample order; all sample/spatial activations weighted equally. Double-precision first/second sums, variance clamp>=0, sigma=sqrt(var+1e-5). Sequential writer uses sigma_g/(sigma_s+1e-5), scale clamped[.25,4], and calibrated layer1 before collecting layer2.",
        '', f"Checkpoint SHA256: `{s['metadata']['checkpoint_sha256']}`.",
        '', f"Reference SHA256: `{s['metadata']['reference_sha256']}`. reference_moments.json contains source IDs, activation counts and numeric reference vectors.",
        '', '15tests passed locally and onA6000; real2-client smoke preceded100-client formal evaluation. 2000query records,1700writer episodes including100clean-sanity episodes. Model hashes unchanged, zero-state logit difference0, T002 no-adapt reproduction error0pp, no support/query or reference/query overlap. Writer accepts no labels and uses no optimizer/gradients. Corruption source SHA identical across deployedT002/T003.',
        '', '```json',json.dumps(verify,indent=2),'```',
        '', '## Interpretation and next action', '',
        'T003 removes label inputs from the writer but image moments can still reflect client class composition. The clean sanity degradation makes global-reference mismatch under label skew a plausible competing explanation (inference, not isolated causal evidence). Dark demonstrates useful correction with the same192scalars; broad sufficiency of those scalars or of moments is unproved. Do not infer that a larger operator alone will fix this.',
        '', 'Stop here for Research Lead review. Recommend explicitly accounting for the clean-reference harm when designing the next richer-operator/oracle diagnostic; no next-stage implementation was launched. No SSL, meta-learning, retraining, severity or clamp sweep.',
        '', f"Runtime code `{s['metadata']['code_commit']}`; run `20260913-153752-ttfl-t003-gpu0`; reference collection {provenance['seconds']:.2f}s; formal evaluation {s['metadata']['seconds']:.2f}s on RTX A6000. Two SSH connection timeouts during monitoring recovered; same job finished normally at2026-09-13T15:39:10+08, exit0. No rerun. PFLlib merged-split/client75:25 diagnostic, not an official CIFAR10 test benchmark."]
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(case=s['case'],clean_sanity=s['clean_sanity'],verification=verify),indent=2))


if __name__=='__main__':
    main()
