# T019 — T019-X: mixed evidence, not strong channel mismatch

2026-09-14T15:05:51+08:00. Lead `d37bbc4`. Preparation `6b5b2c0162ae46d8b4db18d06393d932147ade15`, run `20260914-145145-ttfl-t019`; reporting `15ad29fb9195240b290231a75c84ccddf05c9c5d`, run `20260914-145757-ttfl-t019-finish`. Both exit0. **107 tests PASS; zero new model/query forwards.**

## Decision

**TASK-MISMATCH-A FAIL (0/3), CTX-SPEC-A FAIL (0/3), LOCAL-A PASS for every context. Final diagnosis: T019-X.** The exact-count matched null does not show the predeclared strong task-relevant or paired context-specific excess. Privileged single-class repairs can recover regret for many clients, but this does not establish a systematic class-specific channel failure. Boundary sensitivity is visible, yet its error frequency alone does not explain the distribution of regret. Do not force C, G or S.

All tail fractions below are per-client strict p95 exceedance, checked separately for each salt after averaging its two template halves. Ranges span four salts; the threshold was 20% in both banks/all salts. Task tail is the union of DU/regret exceedance. Paired tail is the union of shift-minus-clean residual-L1/delta-DU exceedance. LOCAL fractions use only nonzero-regret clients and exact utility-ranked single-class repair, never query-based class selection.

| Context | Task tail A, % | Task tail B, % | Paired tail A, % | Paired tail B, % | Single-class half-gap recovery A / B, % |
|---|---:|---:|---:|---:|---|
| brightness_dark | 8.00–8.00 | 9.00–10.00 | 16.00–16.00 | 17.00–17.00 | 76.47–79.41 / 70.59–76.47 |
| contrast_low | 2.00–2.00 | 6.00–8.00 | 17.00–17.00 | 15.00–16.00 | 65.00–71.79 / 61.90–66.67 |
| gaussian_noise | 8.00–9.00 | 6.00–7.00 | 14.00–14.00 | 12.00–12.00 | 73.68–87.88 / 79.41–85.29 |
| gaussian_blur | 11.00–11.00 | 7.00–9.00 | 17.00–18.00 | 18.00–19.00 | 59.18–64.00 / 65.22–70.21 |


Clean union excess (posterior-L1 or DU/regret) is 11% / 17% for A/B, below the frozen operational criterion for substantial generic excess. Clean task-only tails are 7% / 10%. Thus G is unsupported. Paired Blur tails reach 17–19%, but remain below the fixed 20% requirement; no threshold was relaxed.

## Capture and the T018 matched-to-real gap

| Context | Actual capture, % | T019 null median capture, % | Actual below null p05 rows / 8 |
|---|---:|---:|---:|
| brightness_dark | 89.14–91.78 | 91.18–92.28 | 0 |
| contrast_low | 75.73–83.42 | 83.94–85.93 | 3 |
| gaussian_noise | 78.33–81.89 | 83.79–85.43 | 0 |
| gaussian_blur | 74.91–76.69 | 83.15–84.01 | 6 |


All actual macro-class values reproduce T018R2 exactly. Client capture deficits are reported only for positive client P00-gain contribution denominators; zero/negative denominators are marked undefined and counted, not silently normalized. Aggregate P00 denominators remain unchanged. The known T018R2 source-context Blur penalty is preserved separately; T019 itself uses oracle context throughout the mechanism audit.

**Protocol clarification:** T017 K20 already used the exact observed class counts. T019 adds per-class decomposition/seed streams, conditional task percentiles, one-class interventions and ID-paired clean/shift contrasts; it does not newly remove composition randomness from the old K20 comparison. Its new null replicas use the predeclared T019 class-level seeds, so small aggregate changes from T018R2 also reflect a new diagnostic Monte Carlo draw, not a new estimator.

For Blur, the below-p05 row count changes from 1/8 in T018R2 to 6/8 in this new 128-replica diagnostic draw, despite identical actual results. This tail-count change is sensitive to the null draw. The eight bank/salt rows share observations and are not eight independent tests. Keep both receipts and the weak client-level task-tail evidence; do not turn the new aggregate row count into a strong mismatch label.

## Why the diagnosis stays mixed

For Contrast/Noise/Blur, small-margin clients have more wrong state choices, but larger-margin errors are less frequent and more costly. The fixed within-bank/context/salt quartiles show:

| Mean-margin quartile | Actual optimal-set error, % | Mean true regret, pp |
|---|---:|---:|
| Q1 | 59.000 | 0.978 |
| Q2 | 46.167 | 2.003 |
| Q3 | 24.250 | 2.148 |
| Q4 | 16.833 | 2.719 |


The upper two margin quartiles contribute 62.02% of mean true regret in this equal-sized stratification; Q1 contributes 12.46%. Therefore “many errors near boundaries” is descriptive evidence, not sufficient to call the remaining loss boundary-dominated. Conversely, the weak p95 task tails do not justify strong emission mismatch. LOCAL-A identifies label-privileged repairability, not a deployable writer or proof that the same semantic class is responsible everywhere. `dominant_repair_classes.json` gives complete per-context/bank class histograms and top shares rather than asserting a universal bad column.

Recommendation to Lead: a bounded utility-margin/regret-cost and robust neutral-state selection audit could test the observed sensitivity directly; first distinguish frequent small-cost boundary changes from sparse large-cost errors. This is a recommendation only. No feature experiment, new semantic estimator, learned calibration, SSL/TTT, operator expansion or FL has started.

## Implementation and verification

T015 per-example float64 posteriors were reconstructed with the unchanged function and replayed byte-for-byte. Every T016 soft numerator and denominator replays exactly, with its original membership list and target excluded. Real q max error is 0; class-residual identity max error 2.70617e-16. Absent classes remain undefined (NaN only in masked NPZ slots), not imputed.

For each target/bank/context/replica/class, 128 null replicas draw exactly n_y examples with replacement from the target-excluded pool. The seed is the full big-endian SHA256 integer of `T019|client|bank|context|replica|class`, fed to PCG64. Flat source IDs are stored together with client, support position, original ID and label mapping. Shifted draws fetch clean posteriors for the exact same IDs, with full paired coverage and weights n_y/20.

Solver checks passed for {"actual": 1000, "all_repaired": 1000, "single_repaired": 2760, "null": 128000, "paired_clean": 102400}: maximum KKT 5.277e-16, sum error 4.44089e-16, 13 maximum updates. All 1,000 all-class-repaired cases recover the noise-free prevalence and exact optimal set with zero true regret. No tolerance, solver, model, state, temperature or K changed.

Preparation independently replays replica0 source IDs in all 1,000 cells; local audit independently reconstructs replica127 source IDs/q in another 1,000 cells and verifies all 128,000 draws for target exclusion/exact class counts. It checks 16,000 exact Fraction state/DU/regret choices, all 4,000 task percentiles, 3,200 paired delta-DU percentile rows, all exact class-repair fractions/rankings, all 5,120 null query-count vectors and 40 actual vectors. All 147 source-manifest entries from T015/T016/T017/T018R2 remain unchanged.

`phaseB_choices_freeze.json` hashes posterior/null/repair state choices; `mechanism_freeze.json` then hashes null percentiles, paired metrics and class-attribution ranks. Only afterward does `finish_t019.py` open the frozen query-count file. No query outcome selected repairs or formed the null.

## Artifacts

Compact tables and all verification/seed/freeze receipts are in `results/t019_real_channel_heterogeneity/`. Raw posterior/null/choice arrays remain in `research_log/t019_receipts/20260914-145145-ttfl-t019/artifacts/t019_real_channel_heterogeneity/` and the identical remote run path. The finish run writes into that original artifact directory and has a separately retained `finish.log`. Dataset scope remains the inherited PFLlib 100-client baseline on the merged client split, not an official CIFAR-10 test benchmark.
