# T023 — T023-F / CTX-

2026-09-14T21:41:32+08:00. Scientific Lead89466b3; completion Lead38b01e8. All four canonical stages exit0; final full suite138tests PASS; independent verification PASS. This is a verified mechanism outcome, not an implementation blocker.

## Decision

| Gate | Shift contexts passing | Overall |
|---|---:|---|
| ROT-TASK-A | 0/4 | False |
| ROT-REGRET-A | 0/4 | False |
| ROT-CTX-A | 0/4 | False |


Shifted regression safety: False. The fixed rotation auxiliary objective does not meet the predeclared task-proximity criteria. No alternate objective, normalization, regularization, angle, fold or state subset was tried. No actual writer/model update or federation was run. T023's context flag concerns this rotation selector only; it does not invalidate the earlier T021/T022 CTX+ evidence.

## Capture and regret

All ranges span both banks/all4salts. Capture uses the identical P00 denominator and opposite-half integer counts. It is a percentage of available oracle gain, not accuracy percentage. TASK requires>=80% in every bank/salt for>=3/4shifts and all shifted rows>=BBSE-S-01−0.5pp.

| Shift/policy | Capture %, min to max | Median % | Delta vs BBSE-S-01 pp |
|---|---:|---:|---:|
| brightness_dark/ROT-CURRENT | 36.960 to 40.117 | 38.614 | -9.090 to -8.546 |
| brightness_dark/ROT-CLEAN-SURROGATE | 24.723 to 28.364 | 26.593 | -11.268 to -10.653 |
| brightness_dark/P | 89.138 to 91.783 | 90.851 | 0.180 to 0.673 |
| contrast_low/ROT-CURRENT | 34.353 to 45.620 | 40.180 | -5.099 to -3.577 |
| contrast_low/ROT-CLEAN-SURROGATE | 34.289 to 44.241 | 38.951 | -5.143 to -3.700 |
| contrast_low/P | 75.729 to 83.422 | 81.244 | 0.368 to 1.547 |
| gaussian_noise/ROT-CURRENT | -140.992 to -126.430 | -133.946 | -14.505 to -13.246 |
| gaussian_noise/ROT-CLEAN-SURROGATE | -143.537 to -86.551 | -115.157 | -14.681 to -10.504 |
| gaussian_noise/P | 78.334 to 81.889 | 79.891 | 0.772 to 0.892 |
| gaussian_blur/ROT-CURRENT | -11.343 to -1.632 | -6.457 | -7.142 to -6.150 |
| gaussian_blur/ROT-CLEAN-SURROGATE | -12.722 to -0.478 | -6.560 | -7.266 to -6.044 |
| gaussian_blur/P | 74.911 to 76.690 | 75.694 | 0.609 to 0.970 |


Negative capture for Noise and Blur means the selected states perform worse than the frozen zero-state baseline; it does not mean negative accuracy. Noise CURRENT capture is about−141% to−126%, making this a large harmful selection effect rather than a near miss of the80% threshold.

| Shift/comparison (CURRENT) | Mean regret reduction % | New p90 | Baseline p90 | Clients better /100 |
|---|---:|---:|---:|---:|
| brightness_dark/vs_P | -508.467 to -377.919 | 0.332 to 0.345 | 0.048 to 0.081 | 15.000 to 17.000 |
| brightness_dark/current_vs_clean | 2.808 to 4.241 | 0.332 to 0.345 | 0.317 to 0.360 | 30.000 to 32.000 |
| contrast_low/vs_P | -316.060 to -233.736 | 0.294 to 0.326 | 0.082 to 0.100 | 11.000 to 17.000 |
| contrast_low/current_vs_clean | -14.602 to -4.873 | 0.294 to 0.326 | 0.281 to 0.314 | 16.000 to 20.000 |
| gaussian_noise/vs_P | -1251.823 to -1049.138 | 0.384 to 0.412 | 0.038 to 0.057 | 7.000 to 9.000 |
| gaussian_noise/current_vs_clean | -11.499 to 1.452 | 0.384 to 0.412 | 0.383 to 0.458 | 16.000 to 17.000 |
| gaussian_blur/vs_P | -556.406 to -465.630 | 0.298 to 0.315 | 0.058 to 0.066 | 17.000 to 19.000 |
| gaussian_blur/current_vs_clean | -6.026 to -5.409 | 0.298 to 0.315 | 0.292 to 0.302 | 23.000 to 24.000 |


Positive reduction means less regret. REGRET vsP and CTX vsCLEAN each additionally require p90<=1.05baseline and>=15% mean reduction in every bank/salt for>=3/4shifts. Exact-zero baseline means remain noneligible for relative gain. The two halves are paired within each client before better/equal/worse counting. Historical P integer numerators reproduce T018 exactly. Both CURRENT and CLEAN have complete comparisons against P in paired_regret_context.csv.

## Rotation sanity and state ranking

| Bank/context | True MSE mean | Scramble mean MSE p05/median/p95 | Clients true < own scramble median % |
|---|---:|---:|---:|
| A/clean | 0.34370 | 0.41944/0.42808/0.43650 | 81.0 |
| A/brightness_dark | 0.34352 | 0.41893/0.42775/0.43395 | 82.0 |
| A/contrast_low | 0.45429 | 0.50283/0.51674/0.52436 | 75.0 |
| A/gaussian_noise | 0.31705 | 0.38484/0.38969/0.39791 | 79.0 |
| A/gaussian_blur | 0.38739 | 0.46472/0.47464/0.48445 | 82.0 |
| B/clean | 0.34370 | 0.41944/0.42808/0.43650 | 81.0 |
| B/brightness_dark | 0.34352 | 0.41893/0.42775/0.43395 | 82.0 |
| B/contrast_low | 0.45429 | 0.50283/0.51674/0.52436 | 75.0 |
| B/gaussian_noise | 0.31705 | 0.38484/0.38969/0.39791 | 79.0 |
| B/gaussian_blur | 0.38739 | 0.46472/0.47464/0.48445 | 82.0 |


True rotation scores beat each client's scramble median in75–82% of cells, so the observed failure cannot be summarized as “true labels are indistinguishable from scrambling.” This generated-label structure does not translate into useful state ranking. For Noise, CURRENT median Spearman is−0.4/−0.5 across the two banks, with only about19–21% positive correlations. The auxiliary-task difference alone does not qualify the objective for a writer.

Each client has16 independent-per-image label permutations with balanced4-way counts. Table scramble quantiles are across16 replica means over100clients; the final column compares each client to its own16-replica median. Per-client p05/median/p95 are in phase_a/scramble_cells.csv. This diagnostic was frozen before task scoring and was not a task-selection gate.

| Policy/bank/context | Median Spearman | Q25/Q75 | Positive % | Undefined /800 |
|---|---:|---:|---:|---:|
| ROT-CURRENT/A/clean | -0.3 | -0.7/0.35909242322980395 | 36.4 | 3 |
| ROT-CURRENT/A/brightness_dark | 0.09999999999999999 | -0.3/0.6 | 58.3 | 15 |
| ROT-CURRENT/A/contrast_low | 0.09999999999999999 | -0.3/0.6 | 55.1 | 8 |
| ROT-CURRENT/A/gaussian_noise | -0.39999999999999997 | -0.7/0.0 | 21.1 | 8 |
| ROT-CURRENT/A/gaussian_blur | 0.0 | -0.39999999999999997/0.49999999999999994 | 47.1 | 0 |
| ROT-CURRENT/B/clean | -0.3 | -0.7/0.3 | 32.9 | 3 |
| ROT-CURRENT/B/brightness_dark | 0.09999999999999999 | -0.3/0.6 | 57.7 | 15 |
| ROT-CURRENT/B/contrast_low | 0.19999999999999998 | -0.3/0.6 | 56.7 | 8 |
| ROT-CURRENT/B/gaussian_noise | -0.49999999999999994 | -0.7/0.0 | 18.8 | 11 |
| ROT-CURRENT/B/gaussian_blur | 0.0 | -0.49999999999999994/0.39999999999999997 | 45.0 | 0 |
| ROT-CLEAN-SURROGATE/A/clean | -0.3 | -0.7/0.35909242322980395 | 36.4 | 3 |
| ROT-CLEAN-SURROGATE/A/brightness_dark | 0.19999999999999998 | -0.49999999999999994/0.7 | 57.6 | 15 |
| ROT-CLEAN-SURROGATE/A/contrast_low | 0.19999999999999998 | -0.3/0.7 | 59.7 | 8 |
| ROT-CLEAN-SURROGATE/A/gaussian_noise | -0.3 | -0.7/0.19999999999999998 | 30.7 | 8 |
| ROT-CLEAN-SURROGATE/A/gaussian_blur | 0.0 | -0.49999999999999994/0.6 | 45.6 | 0 |
| ROT-CLEAN-SURROGATE/B/clean | -0.3 | -0.7/0.3 | 32.9 | 3 |
| ROT-CLEAN-SURROGATE/B/brightness_dark | 0.09999999999999999 | -0.49999999999999994/0.6 | 53.5 | 15 |
| ROT-CLEAN-SURROGATE/B/contrast_low | 0.3 | -0.19999999999999998/0.7 | 63.6 | 8 |
| ROT-CLEAN-SURROGATE/B/gaussian_noise | -0.3 | -0.7/0.09999999999999999 | 28.1 | 11 |
| ROT-CLEAN-SURROGATE/B/gaussian_blur | -0.09999999999999999 | -0.49999999999999994/0.49999999999999994 | 42.8 | 0 |


Spearman uses average ranks, with the exact rational utility order and negative rotation MSE. Each bank/context summary contains100clients×4salts×2halves; these800utility comparisons reuse the same support and are not independent trials. Constant vectors are undefined and excluded from the defined-correlation denominator with explicit counts.

| Policy/bank/context | clean/Dark/Contrast/Noise/Blur counts | Exact tie cells /100 |
|---|---|---:|
| ROT-CURRENT/A/clean | [3, 37, 43, 12, 5] | 0 |
| ROT-CURRENT/A/brightness_dark | [4, 47, 27, 20, 2] | 0 |
| ROT-CURRENT/A/contrast_low | [7, 40, 33, 13, 7] | 0 |
| ROT-CURRENT/A/gaussian_noise | [4, 31, 54, 9, 2] | 0 |
| ROT-CURRENT/A/gaussian_blur | [3, 44, 26, 27, 0] | 0 |
| ROT-CURRENT/B/clean | [2, 39, 50, 6, 3] | 0 |
| ROT-CURRENT/B/brightness_dark | [4, 47, 30, 17, 2] | 0 |
| ROT-CURRENT/B/contrast_low | [9, 38, 40, 8, 5] | 0 |
| ROT-CURRENT/B/gaussian_noise | [6, 29, 54, 8, 3] | 0 |
| ROT-CURRENT/B/gaussian_blur | [3, 45, 26, 25, 1] | 0 |
| ROT-CLEAN-SURROGATE/A/clean | [3, 37, 43, 12, 5] | 0 |
| ROT-CLEAN-SURROGATE/A/brightness_dark | [3, 37, 43, 12, 5] | 0 |
| ROT-CLEAN-SURROGATE/A/contrast_low | [3, 37, 43, 12, 5] | 0 |
| ROT-CLEAN-SURROGATE/A/gaussian_noise | [3, 37, 43, 12, 5] | 0 |
| ROT-CLEAN-SURROGATE/A/gaussian_blur | [3, 37, 43, 12, 5] | 0 |
| ROT-CLEAN-SURROGATE/B/clean | [2, 39, 50, 6, 3] | 0 |
| ROT-CLEAN-SURROGATE/B/brightness_dark | [2, 39, 50, 6, 3] | 0 |
| ROT-CLEAN-SURROGATE/B/contrast_low | [2, 39, 50, 6, 3] | 0 |
| ROT-CLEAN-SURROGATE/B/gaussian_noise | [2, 39, 50, 6, 3] | 0 |
| ROT-CLEAN-SURROGATE/B/gaussian_blur | [2, 39, 50, 6, 3] | 0 |


## Fixed implementation and independent audit

H is the unchanged512-D fc1/ReLU representation. Rotations are exact quarter turns after deterministic corruption. The same SHA256-sorted10/10support split is reused across states, banks, contexts and rotations. Raw float64 H plus bias is fitted to generated4-way rotation one-hot labels by minimum-norm least squares with fixed rcond1e-12. Score is the mean over40held-out views×4output coordinates, averaged over both directions. Disposable coefficients never touch query data and are not persisted or backpropagated. Only x is accessed from the PFLlib support storage; class-label fields are not accessed during extraction/probe fit/choice.

Extraction reused5000unrotated paths, ran15000rotated batches and250unrotated replay batches on A6000. Probe stage froze5000state scores,1000CURRENT choices and1000CLEAN-SURROGATE choices, all tie masks/ranks and16scrambles. Before privileged scoring, 2000 cross-fitted scores (4000 individual fits) were independently reproduced by numpy.linalg.lstsq with max error1.4210854715202004e-14; observed rank range40–40.

The separate verifier uses original model forward with a final-classifier input hook, not the main extractor; SciPy LAPACK least-squares, not the main SVD helper; independent fold/scramble SHA code; and historical integer/Fraction arithmetic, not the main scorer. It verifies5000 unrotated and3000 rotated original-model H samples, 100000 reused cache sample vectors, 40000 quarter-turn sample tensors; maximum H difference0.0. It reconstructs2000 cross-fitted probe scores (4000 fits), max score error1.2878587085651816e-14; all16replicas in200 cells (3200 cell-replicas), max error3.9968028886505635e-15; all1000choice/tie cells and all scramble summaries.

Explicit cardinalities precede gate reconstruction:120 aggregate rows,24000regret episodes with exactly200unique client-halves per policy/bank/context/salt,120paired rows,16000rank correlations,20rank summaries and20histogram cells. All integer numerators/P00 capture, exact mean/p90 regret, paired counts, histograms, ranks and three gates are independently reconstructed. Model/state hashes and all Phase-A/privileged input hashes remain unchanged. Scoring performs zero new model/query forwards. 1026 local upstream manifest entries pass.

## Runtime and recovery

- Extraction `c73385b58e0879e64570e0935de5ddd9bf443989` / `20260914-212025-ttfl-t023-extract-gpu1`.
- Phase-A `01038b4d7baadd7f330287b25bde2d2a17f25e3a` / `20260914-212245-ttfl-t023-phase-a`.
- Scoring `6d995574f5f452b18b02edc250af14bfd96c73a6` / `20260914-213328-ttfl-t023-score-final`.
- Verification `dbf90212afb60d86bf459398f116227872f37339` / `20260914-213803-ttfl-t023-verify-gpu1`.

Initial scorer20260914-212711-ttfl-t023-score completed atd9490a3. Lead38b01e8 then required explicit sealed-input assertions and Phase-A runtime metadata; the scoring-only repeat is byte-identical on all scientific outputs (completion_scoring_replay.json). Frozen extraction/probe choices were not regenerated. One verifier deployment upload lost SSH connectivity before any verifier job started; the same scientific inputs were retained on recovery.

Compact receipts and logs are under research_log/t023_receipts; extraction/ and phase_a/ retain stage receipts. Large H/pixel/score NPZ arrays remain in canonical remote project run directories with exact byte sizes, paths and hashes in remote_artifact_manifest.json. The existing PFLlib CIFAR10 split is44961train/15039query after merged-source client partitioning,100clients and the historical10% participation checkpoint; this is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining occurred.

Return T023-F/CTX- for Lead review. Request the next bounded research decision; do not implement a writer, alternate SSL objective or T024 autonomously.
