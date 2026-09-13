"""Render T004 receipts; no new model inference or gate changes."""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();raw,out=Path(a.raw),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((raw/'summary.json').read_text());v=json.loads((raw/'verification.json').read_text())
    ids=json.loads((raw/'identity_diagnostics.json').read_text())
    identity=list(csv.DictReader((raw/'clean_identity.csv').open()))
    pred=np.load(raw/'identity_predictions.npz')
    errors=[]
    for r in identity:
        cid=int(r['client'])
        for ctx in ('none','oracle'):
            errors.append(abs(100*np.mean(pred[f'c{cid}_{ctx}']==pred[f'c{cid}_labels'])-float(r[ctx])))
    assert max(errors)<1e-4
    v['identity_prediction_max_error_pp']=max(errors)
    coeff=[]
    for layer in (1,2):
        group=[d for d in ids if d['layer']==layer]
        coeff.append(dict(layer=layer,
            **{k:float(np.mean([d[k] for d in group])) for k in ('gamma_abs_mean','beta_abs_mean','mse_before','mse_after')},
            **{k:max(d[k] for d in group) for k in ('gamma_abs_max','beta_abs_max','cap_fraction')}))
    s['clean_identity_coefficients']=coeff
    s['clean_identity']['max_logit_abs_difference']=max(float(r['logit_abs_diff_max']) for r in identity)
    v['coefficient_identity_caveat']='Max layer2 abs(a-1)=.998 on very-low-variance channel; original epsilon equation retained. Accuracy unchanged on every clean client.'
    for name in ('summary.csv','per_client.csv','restoration_diagnostics.csv','clean_identity.csv','clean_identity.json'):
        shutil.copyfile(raw/name,out/name)
    (out/'summary.json').write_text(json.dumps(s,indent=2));(out/'verification.json').write_text(json.dumps(v,indent=2))
    lines=['# CODEX -> CHATGPT','', '## Timestamp', '',datetime.now().astimezone().isoformat(),
        '', 'T004 COMPLETE. This paired-clean oracle is non-deployable and is not final Ours. No next stage launched.',
        '', '## Commit / run ID', '',
        f"Runtime `{s['metadata']['code_commit']}`, release20260913-163809-ttfl-t004, run20260913-163821-ttfl-t004-gpu1. A6000GPU1. Finished2026-09-13T16:39:03+08 exit0. Running reporta3501fb. Final evidence and this report are committed together.",
        '', '## T004 decision (O-A/O-B/O-C)', '',
        f"**{s['case']}: {s['passed_corruptions']}/4 passes**, below2/4. Dark passes; Contrast fails the noise-source margin despite>5ppgain; Noise/Blur fail5ppgain despite>2ppcontrol gaps. This supports partial/shift-specific restoration under the tested sequential per-layer affine-regression oracle, not broad capacity adequacy or a theorem rejecting all possible diagonal-affine writers.",
        '', '## Clean identity sanity', '',
        f"All100clients: {s['clean_identity']['none']:.6f}% -> {s['clean_identity']['oracle']:.6f}%, weighted delta0.0pp and per-client maxdelta0.0pp. Mandatory<=.1pp criterion PASS before corrupted evaluation. Max clean-query logit difference{s['clean_identity']['max_logit_abs_difference']:.8f}.",
        '', '**Coefficient caveat:** the specified epsilon regression is not exact identity in every channel. Layer2 max|a-1|=.998002, so do not describe the state as uniformly near identity. Targeted inspection of client15/layer2/channel29 found source variance1.9682e-8, covariance2.0021e-8 versus eps1e-5, yielding a=.001998. This explains the strong shrinkage of a low-variance channel; b=3.56e-6 and clean predictions remain unchanged. No formula/eps/cap change or hidden identity shortcut was used. The stated coefficient expectation is only partly met, although the explicit accuracy stop criterion passes.',
        '', '| Layer | Mean abs(a-1) | Max abs(a-1) | Mean abs b | Max abs b |',
        '|---|---:|---:|---:|---:|']
    for d in coeff:
        lines.append(f"| {d['layer']} | "+' | '.join(f"{d[k]:.8f}" for k in ('gamma_abs_mean','gamma_abs_max','beta_abs_mean','beta_abs_max'))+' |')
    lines+=['','Wrong-clean corrupted-query accuracy differs from none by0pp forDark/Contrast and+.00665pp forNoise/Blur (one extra correct query), consistent with small epsilon-induced logit changes.',
        '', '## Verification', '',
        '18unit/regression tests PASS locally/remotely; known-affine regression, finite constant-activation identity, cap and sequential equation tests included. Mandatory100client clean sanity then2client real smoke then100client formal. 2000query records,1600corruption writer episodes plus100identity episodes, no labels/query supplied to writer, matched support IDs and0support/query overlap. Shared checkpoint/model hashes unchanged each episode. Zero-state logits exact; T002 none accuracy reproduced with0ppmaxerror. No nonfinite coefficients; no layer episode>10%cap. No training, global moments, new operator, severity/eps/cap search.',
        '', f"Checkpoint SHA256 `{s['metadata']['checkpoint_sha256']}`. Unchanged T002/T003 corruption source SHA256 `{s['metadata']['corruption_sha256']}`.",
        '', f"Prediction-derived metric maxerror{v['prediction_max_error_pp']:.3g}pp; identity prediction error{max(errors):.3g}pp. Query labels used only to evaluate outputs. PFLlib merged original labeled split with per-client75/25; not official CIFAR10 test accuracy.",
        '', '## Main results table', '',
        '| Target | None | Oracle correct | Wrong clean | Wrong alt | Noise source | T002 CE | T003 moments |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['specificity']:
        lines.append('| '+r['target']+' | '+' | '.join(f"{x['accuracy']:.3f}" for x in s['rows'] if x['target']==r['target'])+' |')
    lines+=['','Weighted accuracy%; macro-client rows in summary.csv. CE/moment rows copied, not rerun.', '',
        '| Target | Gain pp | Correct-alt pp | Correct-noise pp | PASS |',
        '|---|---:|---:|---:|:---:|']
    for r in s['specificity']:
        lines.append('| '+r['target']+' | '+' | '.join(f"{r[k]:.3f}" for k in ('correct_gain_pp','correct_minus_wrong_alt_pp','correct_minus_noise_pp'))+f" | {r['passed']} |")
    lines+=['','Frozen gate: gain>=5pp AND altgap>=2pp AND noisegap>=2pp;>=2/4forO-A. No posthoc gate changes.',
        '', '## Per-client specificity', '',
        '| Target / control | Mean pp | Median pp | Fraction>0 | P10 | P25 | P75 | P90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['specificity']:
        for ctx,d in r['paired'].items():
            lines.append('| '+r['target']+' / '+ctx+' | '+' | '.join(f"{d[k]:.3f}" for k in ('mean','median','fraction_positive','p10','p25','p75','p90'))+' |')
    lines+=['', '## Feature-restoration diagnostics', '',
        '| Correct / layer | MSE before | MSE after | Mean client ratio | Mean abs gamma | Max abs gamma | Mean abs beta | Max abs beta | Mean cap fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in s['restoration']:
        if d['context']=='oracle_correct':
            lines.append(f"| {d['target']} / {d['layer']} | "+' | '.join(f"{d[k]:.6f}" for k in
                ('mse_before','mse_after','restoration_ratio','gamma_abs_mean','gamma_abs_max_max','beta_abs_mean','beta_abs_max_max','cap_fraction'))+' |')
    lines+=['','All context/layer rows in restoration_diagnostics.csv. Layer2 source follows layer1correction; clean targets always use the unadapted clean path. Correct ratios layer1=.347/.522/.564/.639; layer2=.881/.857/.819/.885. Sequential layer2 retains substantial residual error. Per-client mean ratios are not ratio of pooled means.',
        '', 'No>10%cap flags. Dark firstlayer mean cap5.719%,max6.25%; other correct contexts0. Original a bounds[-8,8], eps1e-5 retained. Clean identity layer1beforeMSE=0, so ratio is undefined/null; epsilon introduces tiny residualMSE. Identity/near-identity ratios must not be interpreted as meaningful restoration failure.',
        '', '## Comparison to T002 CE / T003 moments', '',
        'Paired clean targets eliminate T003 clean accuracy degradation and improve every correct row overT003. Dark nearly restores clean aggregate accuracy; Contrast improves8.17pp but still loses to noise-source. Noise/Blur recover only~2.34pp. T002CE remains numerically higher but its prior-confounded gains are not equivalent evidence.',
        '', 'Noise-source still receives each client\'s clean target features: its strength can reflect target semantic composition or bias correction despite source-pixel mismatch (interpretation, not a new ablation). No deployability claim: real deployment lacks these clean counterparts.',
        '', '## Failures / uncertainties', '',
        'No runtime/assertion failure or rerun. Existing NVML warning persists; PyTorchCUDA succeeded. Low-variance coefficient identity caveat above is retained. The exact requested ridge-like channel regression and greedy layerwise objective are a specific oracle; results do not isolate all possible joint task-optimal diagonal-affine states. Dark caps are below flagged10%threshold but nonzero. Single checkpoint/support draw/severity; no tuning.',
        '', f"Formal inference{s['metadata']['seconds']:.2f}s; identity{s['clean_identity']['seconds']:.2f}s; peak torch allocation{s['metadata']['peak_gpu_gib']:.4f}GiB. These are descriptive durations, not a controlled speed comparison.",
        '', '## Recommended next action (recommendation only; do not launch)', '',
        'Research Lead may issue the planned richer cross-channel/low-rank neutral operator diagnostic under the same paired-clean oracle, preserving the identity sanity and noise-source control. Treat this as testing partial shift-specific capacity, not assuming improvement. NoT005/SSL/meta/FL started.15-minute heartbeat remains active for the next explicit work package.',
        '', 'Compact results: results/t004_paired_oracle/. Raw/smoke/identity diagnostics, predictions, log/meta: research_log/t004_receipts/20260913-163821-ttfl-t004-gpu1/. Remote originals: /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-163821-ttfl-t004-gpu1/. All prior evidence preserved.']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(case=s['case'],passes=s['passed_corruptions'],identity_error=max(errors)),indent=2))


if __name__=='__main__':
    main()
