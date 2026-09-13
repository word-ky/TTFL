# CODEX -> CHATGPT

## Timestamp

2026-09-13T18:43:52+08:00

## Commit / run ID

Lead instruction: f93c0ff. Runtime code: c53c9be6d901e54430136fa721437b9480827c01. Run: 20260913-183307-ttfl-t006-gpu1. The enclosing delivery commit contains the analysis and report. NVIDIA RTX A6000 GPU1; formal evaluation 39.98 seconds, 0.0674 GiB peak PyTorch allocated memory (not whole-device memory).

## T006 decision (S-A/S-B/S-C/S-D)

**S-B, with shift-specific limitations.** Noise and Blur satisfy the predeclared S-B inequalities. Dark and Contrast satisfy only the numerical S-C inequalities; random aggregate same-label fractions span 62.90% to 64.12%, which does not establish the broad semantic variation required by the full S-C interpretation. S-A is 0/4. Effects are shift-specific and do not establish universal semantic dominance.

The numerical counts are S-A 0/4, S-B 2/4, S-C 2/4. These are overlapping evidence checks, not mutually exclusive labels. No new threshold or tie-break rule is introduced. Noise semantic advantage exceeds tau by only 0.0366 pp; Blur instance advantage misses tau by 0.0216 pp. Integer prediction counts confirm these boundary decisions.

## Why T005 P-B needed a semantic-pairing audit

T005 removed image fixed points but retained 63.1735% query-weighted same-class pairing (59.6789% equal-client). Thus a label-free pairing algorithm did not remove class semantics under the existing alpha=0.1 partition. This audit manipulates pairing only, preserving the target multiset. It does not evaluate a deployable label-free writer.

## Verification

22 runtime tests and the real 2-client smoke passed before the 100-client run; two additional statistical-helper tests passed remotely during analysis. 5,600 records, 4,000 new writer episodes. All 24 focused old-writer regressions reproduce predictions exactly; reused accuracy error is 0 pp. The first two clients cover all four old conditions and corruptions; remaining old rows reuse T005 receipts.

All new permutations have zero fixed points and identical ID/image/feature multisets. Low/high offsets independently recomputed from support labels; random offsets distinct and deterministic. Support/query overlap 0. Frozen model hash unchanged in every episode, zero-state logits exact, no nonfinite coefficients. Labels only construct/audit controls, never enter writer/state fitting. Independent prediction accuracy error <6.48e-6 pp. Clean reference is 5,146/15,039 = 34.217700645%; integer counts reproduce all decision inequalities. Checkpoint, writer, corruption hashes and eps/caps are in metadata.

The same PFLlib client split is used throughout. It was made from merged original train/test data with disjoint per-client support/query partitions; these numbers are not an official CIFAR-10 test benchmark.

## Same-label fraction audit

All fractions below are percentages. Query weighting matches the main accuracy metric; equal-client statistics are also reported. Low/high select extrema over nonzero cyclic offsets, not over every possible permutation.

| Pairing | Query-weighted % | Client mean % | Client SD % | Client min % | Client max % |
| --- | --- | --- | --- | --- | --- |
| low_semantic_derangement | 57.1933 | 53.4694 | 25.2704 | 10.9375 | 100.0000 |
| high_semantic_derangement | 71.1450 | 68.3391 | 18.1365 | 35.9375 | 100.0000 |
| random_derangement_0 | 62.8980 | 59.7472 | 22.1101 | 18.7500 | 100.0000 |
| random_derangement_1 | 64.1232 | 61.0122 | 21.3489 | 25.0000 | 100.0000 |
| random_derangement_2 | 63.0887 | 59.8994 | 21.4952 | 23.4375 | 100.0000 |
| random_derangement_3 | 63.4349 | 60.1236 | 21.6462 | 14.0625 | 100.0000 |
| random_derangement_4 | 63.4874 | 60.1989 | 21.1902 | 21.8750 | 100.0000 |
| random_derangement_5 | 63.1689 | 59.9934 | 22.2046 | 17.1875 | 100.0000 |
| random_derangement_6 | 63.3992 | 60.0167 | 21.8280 | 15.6250 | 100.0000 |
| random_derangement_7 | 63.0372 | 60.2092 | 21.8549 | 18.7500 | 100.0000 |
| correct_pair | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |
| t005_original | 63.1735 | 59.6789 | 21.6230 | 10.9375 | 100.0000 |

Within-client high-minus-low same-class fraction: mean 14.8696 pp, SD 8.2242, range 0 to 31.2500. Low still averages 57.1933% query-weighted same-class agreement, so semantic leakage is reduced, not removed.

## Main headroom-normalized results

| Shift | No adapt % | Correct % | H pp | Recovery | tau pp | S-A | S-B | S-C numeric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_dark | 17.6408 | 33.5395 | 16.5769 | 0.9591 | 4.1442 | False | False | True |
| contrast_low | 20.0878 | 28.2599 | 14.1299 | 0.5784 | 3.5325 | False | False | True |
| gaussian_noise | 30.7467 | 33.0807 | 3.4710 | 0.6724 | 0.8677 | False | True | False |
| gaussian_blur | 30.9928 | 33.3400 | 3.2249 | 0.7278 | 0.8062 | False | True | False |

## Correct vs low/high semantic derangements

| Shift | Low % | High % | Correct-high pp | High-low pp | Correct-low pp | Wrong alt % | Noise support % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_dark | 31.0393 | 32.7083 | 0.8312 | 1.6690 | 2.5002 | 15.7125 | 31.2188 |
| contrast_low | 30.4409 | 31.5845 | -3.3247 | 1.1437 | -2.1810 | 23.4125 | 31.4582 |
| gaussian_noise | 31.4582 | 32.3625 | 0.7181 | 0.9043 | 1.6224 | 28.3397 | 30.5406 |
| gaussian_blur | 31.4117 | 32.5554 | 0.7846 | 1.1437 | 1.9283 | 27.2625 | 30.9994 |

## Random-derangement distribution and centered correlations

Eight distinct offsets per client. SD is descriptive population SD over eight aggregate replicates, not a confidence interval. Pearson/Spearman use 800 observations per shift after centering accuracy and same-class fraction within each client across random offsets. Spearman ranks the centered values with average ties. No p-values or independent-trial claims.

| Shift | Random mean % | SD | Min | Max | Correct-mean pp | Fraction below correct | Centered r | Centered rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_dark | 31.8123 | 0.4713 | 31.1457 | 32.6750 | 1.7272 | 1.0000 | 0.1634 | 0.1055 |
| contrast_low | 31.0875 | 0.5974 | 30.1416 | 32.2561 | -2.8276 | 0.0000 | 0.1051 | 0.0505 |
| gaussian_noise | 31.6045 | 0.5563 | 30.4542 | 32.2295 | 1.4762 | 1.0000 | 0.1368 | 0.1454 |
| gaussian_blur | 31.6369 | 0.4629 | 30.6869 | 32.2960 | 1.7031 | 1.0000 | 0.1459 | 0.0974 |

Random aggregate same-class fraction (identical across corruptions): query-weighted mean 63.3297%, SD 0.3582 pp, min 62.8980%, max 64.1232%; equal-client mean 60.1501%, SD 0.3566 pp, min 59.7472%, max 61.0122%. Positive centered correlations are modest (r 0.105–0.163; rho 0.051–0.145). S-B is assigned from its preset inequalities, not by inventing a material-correlation threshold.

## Per-client paired summaries

Equal-client deltas in pp; fraction positive excludes ties. Aggregate query-weighted deltas above differ because client query counts differ.

| Shift | Delta | Mean | Median | Fraction >0 | p10 | p25 | p75 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_dark | correct_low | 3.7049 | 5.0454 | 0.6600 | -21.0536 | -5.7357 | 14.4861 | 23.5114 |
| brightness_dark | correct_high | 2.4176 | 4.0327 | 0.6500 | -20.1695 | -6.2261 | 12.3850 | 19.3026 |
| brightness_dark | high_low | 1.2873 | 0.0000 | 0.2200 | -0.7381 | 0.0000 | 0.0000 | 6.8104 |
| brightness_dark | correct_random_mean | 2.9458 | 5.0454 | 0.6600 | -20.5841 | -7.0881 | 13.7678 | 20.2300 |
| contrast_low | correct_low | -0.7452 | 2.5173 | 0.5600 | -27.9081 | -8.3157 | 10.6437 | 17.1414 |
| contrast_low | correct_high | -1.9541 | 0.9783 | 0.5100 | -22.3824 | -8.4530 | 7.6090 | 14.6095 |
| contrast_low | high_low | 1.2089 | 0.0000 | 0.3000 | -1.6121 | 0.0000 | 0.8853 | 8.4239 |
| contrast_low | correct_random_mean | -1.5880 | 1.6088 | 0.5700 | -25.0809 | -8.7824 | 8.4564 | 15.0013 |
| gaussian_noise | correct_low | 2.8336 | 4.3499 | 0.6100 | -21.3476 | -5.6053 | 14.2414 | 21.6028 |
| gaussian_noise | correct_high | 2.1090 | 3.9392 | 0.6100 | -21.1987 | -5.6053 | 11.1395 | 18.9588 |
| gaussian_noise | high_low | 0.7246 | 0.0000 | 0.2500 | -0.5950 | 0.0000 | 0.0947 | 4.0724 |
| gaussian_noise | correct_random_mean | 2.5630 | 5.0054 | 0.6100 | -21.1557 | -5.3427 | 12.9691 | 19.3954 |
| gaussian_blur | correct_low | 3.2189 | 3.5849 | 0.6100 | -19.6496 | -5.2061 | 15.1899 | 23.0769 |
| gaussian_blur | correct_high | 2.4273 | 3.5849 | 0.5900 | -19.4296 | -5.2815 | 13.9651 | 19.7318 |
| gaussian_blur | high_low | 0.7916 | 0.0000 | 0.2900 | -0.6905 | 0.0000 | 0.8133 | 3.8073 |
| gaussian_blur | correct_random_mean | 2.9963 | 4.7456 | 0.6300 | -19.5784 | -3.7406 | 14.9138 | 20.4065 |

## Restoration / coefficient diagnostics

Layer summaries below are equal-client means except explicitly marked maxima. Restoration ratio is mean per-client MSE_after/MSE_before; undefined zero-denominator ratios are excluded. Caps are unchanged at scale [-8,8].

| Shift | Pairing | Layer | MSE before | MSE after | Ratio | Mean |gamma| | Max |gamma| | Mean |beta| | Max |beta| | Negative fraction | Cap mean | Cap max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brightness_dark | correct_pair | 1 | 0.1233 | 0.0432 | 0.3471 | 1.4184 | 7.0000 | 0.2410 | 1.7579 | 0.0000 | 0.0572 | 0.0625 |
| brightness_dark | correct_pair | 2 | 0.0593 | 0.0522 | 0.8812 | 0.2534 | 1.2982 | 0.1263 | 1.7317 | 0.0116 | 0.0000 | 0.0000 |
| brightness_dark | low_semantic_derangement | 1 | 0.1838 | 0.1006 | 0.5467 | 1.0004 | 3.4671 | 0.2516 | 1.0653 | 0.3522 | 0.0000 | 0.0000 |
| brightness_dark | low_semantic_derangement | 2 | 0.1972 | 0.1681 | 0.8529 | 1.6250 | 9.0000 | 0.9813 | 15.4509 | 0.2172 | 0.0138 | 0.1562 |
| brightness_dark | high_semantic_derangement | 1 | 0.1830 | 0.1006 | 0.5490 | 0.9834 | 5.7723 | 0.2466 | 0.9397 | 0.2584 | 0.0000 | 0.0000 |
| brightness_dark | high_semantic_derangement | 2 | 0.1969 | 0.1679 | 0.8533 | 1.5188 | 9.0000 | 0.9118 | 16.1535 | 0.1872 | 0.0103 | 0.0625 |
| contrast_low | correct_pair | 1 | 0.0494 | 0.0254 | 0.5221 | 0.8891 | 3.4869 | 0.0474 | 0.2338 | 0.0000 | 0.0000 | 0.0000 |
| contrast_low | correct_pair | 2 | 0.0518 | 0.0443 | 0.8573 | 0.1692 | 1.1929 | 0.0924 | 0.6349 | 0.0034 | 0.0000 | 0.0000 |
| contrast_low | low_semantic_derangement | 1 | 0.1463 | 0.1003 | 0.6859 | 0.9870 | 1.4466 | 0.2558 | 0.9030 | 0.4512 | 0.0000 | 0.0000 |
| contrast_low | low_semantic_derangement | 2 | 0.1970 | 0.1677 | 0.8520 | 1.1443 | 9.0000 | 0.6355 | 11.9104 | 0.1675 | 0.0022 | 0.0469 |
| contrast_low | high_semantic_derangement | 1 | 0.1442 | 0.1002 | 0.6952 | 0.9619 | 1.3340 | 0.2523 | 0.8576 | 0.3631 | 0.0000 | 0.0000 |
| contrast_low | high_semantic_derangement | 2 | 0.1965 | 0.1673 | 0.8523 | 1.0606 | 8.2187 | 0.5991 | 10.3373 | 0.1431 | 0.0017 | 0.0312 |
| gaussian_noise | correct_pair | 1 | 0.0236 | 0.0132 | 0.5639 | 0.2358 | 0.6890 | 0.0182 | 0.0710 | 0.0000 | 0.0000 | 0.0000 |
| gaussian_noise | correct_pair | 2 | 0.0120 | 0.0098 | 0.8193 | 0.1215 | 1.0007 | 0.0683 | 0.4643 | 0.0034 | 0.0000 | 0.0000 |
| gaussian_noise | low_semantic_derangement | 1 | 0.2157 | 0.1003 | 0.4638 | 0.9888 | 1.2005 | 0.2528 | 0.8600 | 0.3928 | 0.0000 | 0.0000 |
| gaussian_noise | low_semantic_derangement | 2 | 0.1964 | 0.1680 | 0.8560 | 1.2425 | 8.9786 | 0.7206 | 11.7189 | 0.1742 | 0.0037 | 0.0781 |
| gaussian_noise | high_semantic_derangement | 1 | 0.2124 | 0.1003 | 0.4705 | 0.9786 | 1.1202 | 0.2491 | 0.7984 | 0.2731 | 0.0000 | 0.0000 |
| gaussian_noise | high_semantic_derangement | 2 | 0.1959 | 0.1678 | 0.8569 | 1.1011 | 8.2053 | 0.6370 | 10.9187 | 0.1547 | 0.0014 | 0.0312 |
| gaussian_blur | correct_pair | 1 | 0.0128 | 0.0082 | 0.6393 | 0.2269 | 1.2409 | 0.0228 | 0.0615 | 0.0000 | 0.0000 | 0.0000 |
| gaussian_blur | correct_pair | 2 | 0.0097 | 0.0085 | 0.8851 | 0.0578 | 1.0023 | 0.0361 | 0.1927 | 0.0014 | 0.0000 | 0.0000 |
| gaussian_blur | low_semantic_derangement | 1 | 0.1744 | 0.1003 | 0.5747 | 0.9817 | 1.2182 | 0.2531 | 0.8559 | 0.3887 | 0.0000 | 0.0000 |
| gaussian_blur | low_semantic_derangement | 2 | 0.1964 | 0.1679 | 0.8550 | 1.1243 | 9.0000 | 0.6438 | 12.6731 | 0.1691 | 0.0023 | 0.0312 |
| gaussian_blur | high_semantic_derangement | 1 | 0.1713 | 0.1002 | 0.5845 | 0.9644 | 1.1946 | 0.2496 | 0.7967 | 0.2822 | 0.0000 | 0.0000 |
| gaussian_blur | high_semantic_derangement | 2 | 0.1959 | 0.1676 | 0.8560 | 0.9813 | 7.0000 | 0.5583 | 9.7265 | 0.1447 | 0.0013 | 0.0469 |

Lower restoration MSE alone is not task evidence: in Low contrast, correct pairing has worse query accuracy than both semantic derangements. Different target pairings also define different reconstruction targets. The present controls do not isolate a causal relation between MSE and query accuracy.

## Failures / uncertainties

No formal runtime failure. Local analysis initially failed due to duplicate OpenMP libraries; the unchanged analysis calculations and two helper tests ran successfully in the established remote environment, without the unsafe duplicate-library override. The pre-existing NVML warning persists; PyTorch CUDA on A6000 completed the run.

There are 9 formal layer episodes with cap fraction >10%; all are retained and listed in cap_flags.json. This can affect attribution. Small headroom in Noise/Blur makes threshold decisions close. Client heterogeneity is large, with negative lower-tail deltas. Same-class agreement in low controls remains high, and the cyclic-offset family explores limited pairings. Results support the requested S-B diagnostic, not a universal decomposition of all oracle utility into semantics.

## Recommended next action (recommendation only; do not launch)

Return to Research Lead for a class-balanced, cross-client corruption-transfer diagnostic that separates client semantic marginals from covariate context. Do not use T004/T005 oracle gains alone as proof of context reading. No T007, SSL, operator expansion, or FL retraining was launched. The existing 15-minute heartbeat should read new instructions and remain quiet when no actionable change exists.
