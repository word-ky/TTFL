# CODEX -> CHATGPT

## Timestamp
2026-09-13 15:26 +08. T002 COMPLETE, overall FAIL / Case B. Formal job finished15:21:13+08 exit0. Requested15-minute execution heartbeat `ttfl-chatgpt` is ACTIVE; it reads new instructions and resumes unfinished work. T002 is now finished: do not rerun it merely because the incoming file still calls it active.

## Commit
Runtime `e09a4a8`; release `20260913-151830-ttfl-t002`; run `20260913-151900-ttfl-t002-gpu0`. Lead4455f46 RESEARCH_PROTOCOL.md read/merged aseda9c6e with no experiment change. Running receipt pushed in9f6d278. This final report and evidence are committed together; enclosing Git commit identifies delivery.

## What changed
Evaluation-only scripts/eval_pfllib_covariate_context.py, deterministic src/data/covariate.py, three focused tests, CUDA launch script. Reused upstream PFLlib loader, ContextFedAvgCNN, existing affine.adapt and evaluate/state_hash. No training/model code changes, FedAvg retraining, severity search, SSL, meta-learning or new FL.

## Checkpoint / data reused
Existing formal CIFAR10 run: /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10.

- Checkpoint SHA256: 260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb.
- Existing support_indices.json SHA256: 855f12223d1164d14c3dd13835ac8a9a73522834c6c49b4228015a3afba2ae85.
- Existing split_manifest.json SHA256: 79b8b6d245ad6785208588c2687b4016be1bd4d3f9da475fca61610c3bc64296.
- All100 clients; exact saved support indices up to64/client, exact query splits. Full-batch SGD lr0.1/10steps; zero gamma/beta each episode; identical settings across conditions.
- Existing PFLlib merged original labeled CIFAR10 splits then client75/25 split; not official CIFAR10 test accuracy.

## Corruption definitions and tensor range
Before implementation saved client0 range observed[-1,1]; evaluated supports/queries across100 also[-1,1]. Convert z=(x+1)/2, transform in[0,1], return2*z-1:

- dark: .45*z;
- contrast: .35*(z-mean)+mean, per-image/channel spatial mean;
- noise: clamp(z+.15*N(0,1),0,1);
- blur: torchvision Gaussian5x5/sigma1.5, reflect padding;
- noise-image: independent uniform[0,1] pixels, preserving exact label vector.

CPU torch.Generator per-image seed: first8 little-endian bytes of SHA256(7:client:original_sample_id:transform) modulo(2^63-1). Noise is deterministic by original ID and independent of ordering. Fixed label permutation per client. Definitions frozen before smoke/formal metrics. Wrong-alt mapping exactly follows Lead: noise/blur/dark/contrast respectively; wrong-clean uses same IDs/labels.

## Verification / matched-label controls
Baseline9 tests passed remotely before edits. Final12 tests passed remotely;2-client real smoke passed before100-client formal run. Smoke did not select configuration.

Formal:2400records,2000adaptation episodes, support/query ID overlap0, identical support IDs/labels across matched contexts, all2000shared model hashes unchanged, checkpoint SHA unchanged. Zero-state max logit difference0.0 across all corrupted query batches. Clean baseline reproduces all old per-client accuracies, max error0.0pp. Prediction-derived accuracy max discrepancy6.99e-6pp. Losses/norms finite; transform repeat/order tests pass. A6000 evaluation115.95s after setup, peak torch allocation0.0811GiB (descriptive, not an efficiency benchmark).

## Results table
Weighted accuracy%; macro-client accuracy, support CE before/after and gamma/beta norm means for every row are in summary.csv.

| Target | None | Correct | Wrong clean | Wrong alt | Shuffled | Noise image | PASS |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Dark |17.641|52.051|44.325|44.431|52.796|50.409|Yes|
| Low contrast |20.088|55.290|52.856|54.066|56.061|57.125|No|
| Gaussian noise |30.747|55.629|52.324|46.991|56.134|56.606|No|
| Gaussian blur |30.993|57.038|58.155|54.299|57.570|60.297|No|

| Target | Correct gain pp | Correct-clean pp | Correct-alt pp | Correct-noise pp | Failure |
|---|---:|---:|---:|---:|---|
| Dark |34.411|7.727|7.620|1.642|Passes frozen rule|
| Contrast |35.202|2.434|1.223|-1.835|Alt gap<2; noise stronger|
| Noise |24.882|3.305|8.638|-0.977|Noise-image stronger|
| Blur |26.046|-1.117|2.740|-3.258|Clean gap<2; noise stronger|

Only1/4 passes; required>=2/4 not met. Even dark has shuffled>correct. It supports limited covariate sensitivity under the stipulated rule, not proof that correct image-label correspondence drives the gain.

## Per-client paired specificity summary
Delta=correct-wrong accuracy pp, equal client weight. Full400target/client rows in per_client.csv.

| Target / wrong | Mean | Median | Fraction>0 | P10 | P25 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dark / clean |7.491|3.871|.68|-1.172|0|11.111|23.676|
| Dark / noise |7.696|3.265|.65|-2.601|0|13.775|25.182|
| Contrast / clean |2.767|1.937|.59|-11.491|-2.156|9.949|17.651|
| Contrast / blur |1.219|0|.48|-7.454|-2.992|5.063|11.241|
| Noise / clean |3.252|2.548|.78|0|.605|4.711|6.974|
| Noise / dark |8.449|7.824|.81|-.243|2.727|12.992|17.875|
| Blur / clean |-1.276|0|.38|-8.079|-2.949|1.260|3.203|
| Blur / contrast |2.345|1.231|.60|-2.329|-.110|5.912|7.883|

## Existing-receipt prior-correlation analysis (completed)
Old static-client clean-query rows joined with exact support labels. Entropy/max-class fraction/four gains in prior_per_client.csv. Descriptive Pearson/Spearman, mean ranks for ties, no significance tests:

| Affine gain | vs max-class fraction r/rho | vs prior gain r/rho |
|---|---:|---:|
| Correct |.509/.488|.457/.397|
| Shuffled |.496/.473|.492/.433|
| Noise |.547/.531|.574/.507|

Moderately positive, consistent with a strong prior contribution, not causal proof. Completed within minutes with no model evaluation.

## Failures / uncertainties
No T002 evaluation/runtime failure. Existing NVML mismatch persists but actual CUDA works. Concurrent Lead protocol commit caused a push rejection, resolved by normal merge. Local log-writing interpolation error and one report patch format error occurred before writes and were corrected; neither affected execution.

One checkpoint, one fixed severity, one support draw/client, static label-skew. Matched supports eliminate between-condition histogram/content differences but do not eliminate general prior gains. Three noise-image controls beat correct; all4shuffled rows beat correct. Brightness has partial signal; multi-corruption gate fails. This is method/diagnostic evidence, not an observed implementation bug.

## Case A / B / C decision
**Case B, with one partial brightness PASS.** Gains on every corrupted query, but only1/4passes all controls. Robust current-covariate context reading independent of prior-driven adaptation is not established. Do not equate this with universal incapacity or overall contextual success.

## Recommended next action
Research Lead reviews and selects the next bounded operator/sufficient-statistic diagnostic. No next stage launched. The15-minute heartbeat remains active for the explicit next work package and must not repeat completed T002.

Compact outputs: results/t002_covariate/{RESULTS.md,summary.json,summary.csv,verification.json,per_client.csv,prior_correlations.json,prior_per_client.csv}.
Raw records/identities/predictions/smoke/logs/meta: research_log/t002_receipts/20260913-151900-ttfl-t002-gpu0/. NPZ predictions retained locally/remotely, Git-ignored. Remote originals: /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-151900-ttfl-t002-gpu0/.
