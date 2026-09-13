# CODEX -> CHATGPT

## Timestamp
2026-09-13 11:44 +08 — T001 COMPLETE / ACCEPTANCE FAIL / WAITING FOR RESEARCH LEAD

## Commit
Tested experimental implementation: `c0f6717de9722a7d9a983bb5769ab1788234c765`.
Interim coordination commit: `cf3e87c`. Final evidence commit is the commit containing this report; delivery SHA recorded in `research_log/delivery.json` after push.

## What changed
Implemented a small3-block Conv-BN-ReLU-Pool CNN and448-scalar channel affine operator after spatial blocks. Fast gamma/beta start exactly at zero, are reset for every context/configuration, and alone receive SGD updates; backbone/classifier/BN buffers stay frozen in eval mode. Built official torchvision Digits loaders plus documented MNIST-M image mirror, deterministic splits, shared-checkpoint comparisons, raw prediction/JSON/CSV receipts and aggregate table.

No V1 memory implementation, SSL, meta-learning or federation. No post-result parameter sweep or next task executed.

## Experiments run
- Remote new project: `/home/wenchang/asdasdsad/wjq/TTFL`, host `wenchang-PR4904W1`, physical NVIDIA RTX A6000 GPU0. Python3.12.12, torch2.4.0+cu121, torchvision0.19.0.
- Release: `20260913-113613-ttfl-t001`.
- Smoke: `20260913-113707-ttfl-t001-smoke`, exit0 at11:37:33+08,7 tests PASS0.973s,21 miniature real-data records. Smoke scores are not scientific evidence.
- Full: `20260913-113757-ttfl-t001-full`, wrapper11:38:04–11:39:01+08, exit0.7 tests PASS0.956s,20 source epochs,446 evaluation records.
- Source: MNIST/USPS/MNIST-M,6000 images each; seed7, Adam.001,batch128,20 epochs, final checkpoint selected without target evaluation. Final online training accuracy99.978%, CE.001296; no source validation/early stopping.
- Target: SVHN,3000 fixed official test queries(seed101). Five balanced200-image training-split support draws(seeds11,22,33,44,55;20/class). Wrong-domain supports from each of3 sources have identical label counts. Shuffled labels preserve correct images; noise uses uniform[-1,1]. Wrong supports can overlap source training, disclosed in frozen plan.
- All12 affine grid cells: steps1/3/5/10 x LR.001/.01/.1. Full-model SGD comparison: LR.001,steps1/3/5/10,BN eval. Random no-write affine state std.01. BN replaces running moments from one full support batch(momentum1). Prior uses Laplace1 support/source histograms.

## Results table
Accuracy mean±SD over five support draws. One backbone checkpoint. Affine rows below all use the same best correct-support grid cell: LR.1,10steps.

| Method / support | Query accuracy (%) | Gain (pp) |
|---|---:|---:|
| No adaptation |41.633|0.000|
| BN / correct |45.200±0.373|+3.567|
| BN / wrong MNIST |36.573±0.153|-5.060|
| BN / wrong USPS |38.667±0.193|-2.967|
| BN / wrong MNIST-M |43.840±0.130|+2.207|
| Prior / correct |41.233±0.000|-0.400|
| Random state / no write |41.580±0.126|-0.053|
| Affine / correct |42.933±0.379|+1.300|
| Affine / wrong MNIST |42.147±0.267|+0.513|
| Affine / wrong USPS |41.693±0.049|+0.060|
| Affine / wrong MNIST-M |41.173±1.220|-0.460|
| Affine / shuffled labels |34.520±2.220|-7.113|
| Affine / noise |40.833±0.520|-0.800|
| Full-model / correct / LR.001,10steps |42.913±0.673|+1.280|

Exact machine-readable values and every method/context/grid cell: `results/t001/summary.json`, `summary.csv`, `RESULTS.md`. Raw446 rows, split IDs, training/config/environment/checkpoint hashes: `results/t001/raw/`. Full prediction archive and checkpoint retained remotely and in local ignored `research_log/remote_runs/20260913-113757-ttfl-t001-full/artifacts/t001`; hashes/bytes in `results/t001/artifact_manifest.json`.

### Acceptance
**FAIL, all12 cells.** Best correct affine gain+1.300pp < required+5pp. At that same cell correct-minus-wrong is+0.787(MNIST),+1.240(USPS),+1.760(MNIST-M)pp, all below2pp. Correct-minus-shuffled+8.413pp does not rescue wrong-context failure. Even the largest individual correct-support gain across all cells/draws is only+1.90pp(seed11,LR.1,10steps). No SSL/meta/FL/T002 may follow this result.

## Diagnostics
- Neutral operator max absolute logit difference: **0.0**. No baseline entry penalty.
- Shared checkpoint SHA256: `ffcf2dfa205c72094a54f592b1fef8b44dc830a3242acc6d917909b85c1df997`; downloaded file hash verified. Recorded starting tensor SHA identical for every row. Runtime asserts verify affine backbone+BN buffers remain unchanged and shared model is unchanged after every method.
- Receipt check PASS:446 unique configurations/draws,3000 query labels, support/query and source/query ID intersections0. Independent prediction-derived accuracy max difference1.8915e-6pp; per-class max4.3437e-6pp (float32 rounding). `results/t001/verification.json`.
- Correct affine at LR.1/10steps: support CE2.467726→2.041584; query CE2.044170→1.832588. Mean||gamma||=.415126,||beta||=.473865. Worst-class accuracy18.651→21.667%. Operator updates and reduces loss, but accuracy/context specificity remain inadequate.
- BN improves accuracy more(+3.567pp) but worsens query CE2.044170→2.711888 and worst-class accuracy18.651→17.222%. BN correct and shuffled are identical because BN ignores labels. Correct BN exceeds wrongMNIST-M only1.36pp; BN is useful here but not a clean stronger-than-all-controls result.
- Balanced support makes prior correction identical across domain/shuffle/noise contexts; it underperforms by.4pp. This deliberately removes support class-count differences as explanation of affine specificity.

## Failures / uncertainties
Engineering pipeline passed; the tested V2-A read-operator/configuration failed the scientific gate. Do not equate this with proof that every affine operator is impossible: only one source checkpoint, one held-out domain, fixed balanced200-shot support, one exploratory query grid. Five support seeds are not five independent backbone seeds. The modest full-model gain also means this limited10-step/.001 comparison is not a converged supervised ceiling and cannot isolate operator expressiveness from optimization budget/checkpoint behavior.

Acquisition failures retained in research_log: obsolete MNIST URL404, conda CA trust issue(repaired by OS CA bundle), Stanford HTTPS hostname mismatch(use original published HTTP+matchingMD5), MNIST-M Git LFS pointer resolved to actual archive with matchingSHA256. Remote torchvision/Pillow imports initially missing; installed only in project venv. Existing NVML driver/library mismatch does not prevent tested CUDA; no global repair. No existing data/checkpoint was present in this new repo, so this is not a numerical reproduction of Claude's previous Digits run.

## Recommended next action
Return to Research Lead for a new read-operator/diagnostic package. Keep neutral initialization and same-checkpoint controls. Possible question for the Lead: does an adequately optimized supervised full-model comparison separate write-budget limits from affine read capacity? This is a recommendation only; no extra run or new grid was launched. T001 result stays FAIL; SSL/meta/FL remain stopped.
