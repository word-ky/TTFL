# CODEX -> CHATGPT

## Timestamp

2026-09-13T20:38:02+08:00

## Commit / run ID

Lead 7553405; runtime 5794dbf; run 20260913-203200-ttfl-t007r-gpu1; release 20260913-203154-ttfl-t007r. NVIDIA RTX A6000 GPU1; formal runtime 37.17 seconds. The enclosing delivery commit contains the independent analysis and report.

## T007R identity-centered writer verification

New separately versioned paired_affine_oracle_neutral.py fits ridge on the residual `z-x`, centering the prior at gamma=0. EPS=1e-5, final scale clamp [-8,8], backbone and 192-scalar state are unchanged. Historical paired_affine_oracle.py bytes are unchanged. Random, constant and near-zero-variance identity tests require exact equality and passed; known affine/cap/sequential tests also passed.

On both original pools, all 192 values are exactly zero for clean identity. All 15,039 clean logits/predictions and all three accuracy metrics are exactly unchanged. The full clean-query prerequisite completed before fitting any corrupted state. This resolves the old writer mathematical prior mismatch; no identity threshold was relaxed.

## Exact pool provenance reused

Original receipt SHA256: c6b1692e6fae8db52dec5dfe036af03bdc696de359a7b35f5bf64fe1152e49d0. No pool-selection call in the T007R evaluator. Both pools contain 80 training images, exactly 8/class, each class spread across eight distinct clients; each pool spans 64 clients. Pool A/B IDs are disjoint and absent from all query IDs. Original train index, original ID and class were rechecked. The same committed permutations have 100%/0% same-class agreement and zero image fixed points; target image/feature multisets are exact at both layers.

For calibration corruptions, the existing RNG key uses each image's original client ID and original image ID, not a new synthetic pool ID. Query corruption tensors are created once per client/target and reused across pools/conditions. Labels only audit the frozen pools/permutations and evaluate queries; writer APIs accept pixels only.

## Clean-identity old-vs-new audit

| Pool | Old L2 worst scale | New scale | Old max |gamma| | New max |gamma| | Old max logit diff | New max logit diff | Old changed argmax | New changed argmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.323925 | 1.000000 | 0.676075 | 0.000000 | 0.006222 | 0.000000 | 1 | 0 |
| B | 0.469380 | 1.000000 | 0.530620 | 0.000000 | 0.004875 | 0.000000 | 1 | 0 |

Old values reuse T007 preflight receipts; the old writer grid was not rerun. Its stopped report remains in research_log/t007_stopped_preflight_report and raw receipts remain in research_log/t007_receipts. The repair comparison is about identity behavior only, not a paired old-vs-new corruption-performance ablation.

## Primary macro-class transfer table (Pool-A and Pool-B separately)

Clean zero-state macro-class accuracy: 34.138689920%. Primary accuracy is equal-weight mean of the ten class accuracies over all client queries. All values below are percentages.

| Pool | Shift | None | Identity | Correct | Same class | Cross class | Wrong alt | Noise source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 17.546026 | 32.296536 | 14.528763 | 18.710409 | 14.903865 | 12.884809 |
| A | contrast_low | 19.955264 | 19.955264 | 28.561473 | 14.460475 | 18.925132 | 22.733171 | 10.145171 |
| A | gaussian_noise | 30.676644 | 30.676644 | 32.499558 | 12.953750 | 16.952166 | 30.382434 | 10.336917 |
| A | gaussian_blur | 30.901579 | 30.901579 | 33.048500 | 12.977576 | 17.996638 | 29.798137 | 10.198125 |
| B | brightness_dark | 17.546026 | 17.546026 | 32.424808 | 22.126559 | 11.733552 | 15.489587 | 10.000000 |
| B | contrast_low | 19.955264 | 19.955264 | 28.111626 | 20.008427 | 12.215961 | 22.693556 | 8.717600 |
| B | gaussian_noise | 30.676644 | 30.676644 | 32.406771 | 21.963064 | 11.914205 | 30.537680 | 8.497081 |
| B | gaussian_blur | 30.901579 | 30.901579 | 33.009919 | 22.234302 | 12.329852 | 28.354235 | 8.572245 |

Sample-weighted and equal-client metrics for every condition are in summary.csv; A/B values and absolute differences for all three metrics are in pool_reproducibility.csv. Correct-pair secondary values are:

| Pool | Shift | Sample-weighted % | Macro-client % |
| --- | --- | --- | --- |
| A | brightness_dark | 32.362524 | 31.268559 |
| A | contrast_low | 28.638872 | 27.361031 |
| A | gaussian_noise | 32.575304 | 31.387282 |
| A | gaussian_blur | 33.120553 | 31.950703 |
| B | brightness_dark | 32.488862 | 31.338011 |
| B | contrast_low | 28.186715 | 26.830968 |
| B | gaussian_noise | 32.482213 | 31.267717 |
| B | gaussian_blur | 33.080657 | 31.880620 |

## Headroom / tau / PASS table

Unchanged original T007 rule: recovery >=0.50 and correct-minus-wrong-alt >=tau and correct-minus-noise >=tau independently in both pools, tau=max(0.5,0.25*headroom). No rounding is used for decisions.

| Pool | Shift | H pp | tau pp | Recovery | Correct-wrong pp | Correct-noise pp | Pool PASS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 16.592663 | 4.148166 | 0.888978 | 17.392670 | 19.411726 | True |
| A | contrast_low | 14.183426 | 3.545856 | 0.606779 | 5.828302 | 18.416302 | True |
| A | gaussian_noise | 3.462046 | 0.865512 | 0.526542 | 2.117123 | 22.162640 | True |
| A | gaussian_blur | 3.237111 | 0.809278 | 0.663221 | 3.250362 | 22.850375 | True |
| B | brightness_dark | 16.592663 | 4.148166 | 0.896708 | 16.935221 | 22.424808 | True |
| B | contrast_low | 14.183426 | 3.545856 | 0.575063 | 5.418069 | 19.394026 | True |
| B | gaussian_noise | 3.462046 | 0.865512 | 0.499741 | 1.869091 | 23.909690 | False |
| B | gaussian_blur | 3.237111 | 0.809278 | 0.651303 | 4.655684 | 24.437674 | True |

| Shift | Joint A AND B PASS |
| --- | --- |
| brightness_dark | True |
| contrast_low | True |
| gaussian_noise | False |
| gaussian_blur | True |

**3/4 joint PASS: Dark, Contrast, Blur.** Noise is FAIL because Pool-B recovery is 0.499741262, below 0.50 by 0.000258738 (about 0.000896 pp of macro-class accuracy). This boundary is retained as failure; independent rational arithmetic on integer class counts confirms it. Both noise context margins exceed tau, but that does not override the recovery criterion.

## Semantic-pairing decomposition

| Pool | Shift | Correct-same pp | Same-cross pp | Correct-cross pp |
| --- | --- | --- | --- | --- |
| A | brightness_dark | 17.767773 | -4.181647 | 13.586126 |
| A | contrast_low | 14.100999 | -4.464658 | 9.636341 |
| A | gaussian_noise | 19.545807 | -3.998416 | 15.547391 |
| A | gaussian_blur | 20.070923 | -5.019062 | 15.051861 |
| B | brightness_dark | 10.298249 | 10.393007 | 20.691256 |
| B | contrast_low | 8.103198 | 7.792466 | 15.895664 |
| B | gaussian_noise | 10.443707 | 10.048859 | 20.492567 |
| B | gaussian_blur | 10.775617 | 9.904450 | 20.680067 |

Correct pairing materially exceeds same-class and cross-class derangements for every shift in both pools. However, the semantic component changes sign: same-minus-cross is negative in Pool-A and positive in Pool-B. This does not support a universal class-semantic ordering. The strong correct-vs-wrong-source/noise margins support the predeclared covariate-transfer claim within this paired-clean oracle, while the large pairing gaps show dependence on privileged clean correspondence.

## Pool reproducibility / state similarity

| Shift | Condition | A macro-class % | B macro-class % | Absolute gap pp |
| --- | --- | --- | --- | --- |
| brightness_dark | none | 17.546026 | 17.546026 | 0.000000 |
| brightness_dark | clean_identity | 17.546026 | 17.546026 | 0.000000 |
| brightness_dark | correct_pair | 32.296536 | 32.424808 | 0.128272 |
| brightness_dark | same_class_derangement | 14.528763 | 22.126559 | 7.597796 |
| brightness_dark | cross_class_derangement | 18.710409 | 11.733552 | 6.976857 |
| brightness_dark | wrong_alt_source | 14.903865 | 15.489587 | 0.585721 |
| brightness_dark | noise_source | 12.884809 | 10.000000 | 2.884809 |
| contrast_low | none | 19.955264 | 19.955264 | 0.000000 |
| contrast_low | clean_identity | 19.955264 | 19.955264 | 0.000000 |
| contrast_low | correct_pair | 28.561473 | 28.111626 | 0.449848 |
| contrast_low | same_class_derangement | 14.460475 | 20.008427 | 5.547953 |
| contrast_low | cross_class_derangement | 18.925132 | 12.215961 | 6.709171 |
| contrast_low | wrong_alt_source | 22.733171 | 22.693556 | 0.039615 |
| contrast_low | noise_source | 10.145171 | 8.717600 | 1.427572 |
| gaussian_noise | none | 30.676644 | 30.676644 | 0.000000 |
| gaussian_noise | clean_identity | 30.676644 | 30.676644 | 0.000000 |
| gaussian_noise | correct_pair | 32.499558 | 32.406771 | 0.092786 |
| gaussian_noise | same_class_derangement | 12.953750 | 21.963064 | 9.009314 |
| gaussian_noise | cross_class_derangement | 16.952166 | 11.914205 | 5.037962 |
| gaussian_noise | wrong_alt_source | 30.382434 | 30.537680 | 0.155245 |
| gaussian_noise | noise_source | 10.336917 | 8.497081 | 1.839836 |
| gaussian_blur | none | 30.901579 | 30.901579 | 0.000000 |
| gaussian_blur | clean_identity | 30.901579 | 30.901579 | 0.000000 |
| gaussian_blur | correct_pair | 33.048500 | 33.009919 | 0.038581 |
| gaussian_blur | same_class_derangement | 12.977576 | 22.234302 | 9.256726 |
| gaussian_blur | cross_class_derangement | 17.996638 | 12.329852 | 5.666787 |
| gaussian_blur | wrong_alt_source | 29.798137 | 28.354235 | 1.443902 |
| gaussian_blur | noise_source | 10.198125 | 8.572245 | 1.625880 |

| Shift | Condition | State cosine | State L2 | State max abs difference |
| --- | --- | --- | --- | --- |
| brightness_dark | none | None | 0.000000 | 0.000000 |
| brightness_dark | clean_identity | None | 0.000000 | 0.000000 |
| brightness_dark | correct_pair | 0.997125 | 0.920631 | 0.634401 |
| brightness_dark | same_class_derangement | 0.358100 | 28.661219 | 12.663203 |
| brightness_dark | cross_class_derangement | -0.002083 | 42.051315 | 11.286189 |
| brightness_dark | wrong_alt_source | 0.989348 | 0.308539 | 0.099241 |
| brightness_dark | noise_source | 0.409785 | 26.427911 | 9.644997 |
| contrast_low | none | None | 0.000000 | 0.000000 |
| contrast_low | clean_identity | None | 0.000000 | 0.000000 |
| contrast_low | correct_pair | 0.978617 | 1.285448 | 0.588573 |
| contrast_low | same_class_derangement | 0.533265 | 14.324533 | 4.606547 |
| contrast_low | cross_class_derangement | 0.087233 | 23.444934 | 8.290230 |
| contrast_low | wrong_alt_source | 0.988889 | 0.294799 | 0.146079 |
| contrast_low | noise_source | 0.409785 | 26.427911 | 9.644997 |
| gaussian_noise | none | None | 0.000000 | 0.000000 |
| gaussian_noise | clean_identity | None | 0.000000 | 0.000000 |
| gaussian_noise | correct_pair | 0.989348 | 0.308539 | 0.099241 |
| gaussian_noise | same_class_derangement | 0.512935 | 13.542943 | 4.722561 |
| gaussian_noise | cross_class_derangement | -0.077732 | 22.115994 | 9.053641 |
| gaussian_noise | wrong_alt_source | 0.997125 | 0.920631 | 0.634401 |
| gaussian_noise | noise_source | 0.409785 | 26.427911 | 9.644997 |
| gaussian_blur | none | None | 0.000000 | 0.000000 |
| gaussian_blur | clean_identity | None | 0.000000 | 0.000000 |
| gaussian_blur | correct_pair | 0.988889 | 0.294799 | 0.146079 |
| gaussian_blur | same_class_derangement | 0.484636 | 11.098656 | 2.479361 |
| gaussian_blur | cross_class_derangement | 0.112544 | 19.348906 | 7.208911 |
| gaussian_blur | wrong_alt_source | 0.978617 | 1.285448 | 0.588573 |
| gaussian_blur | noise_source | 0.409785 | 26.427911 | 9.644997 |

States are concatenated in actual model order [gamma1,beta1,gamma2,beta2]. Cosine for zero-vs-zero states is undefined (None), with L2 and max difference exactly zero. These two pools are predeclared draws, not hyperparameter choices or confidence intervals.

## Coefficient and restoration diagnostics

All 42 formal state fits (2 clean identities +40 corrupted-source controls) are finite. 0 layer records have cap fraction >10%; maximum observed cap fraction is 6.2500%. 48 of 84 layer records contain negative scales; negative scales are permitted by the frozen clamp and are not silently removed. Full per-state/layer MSE, restoration ratio, gamma/beta, negative fraction, cap fraction and minimum source variance are in restoration_diagnostics.csv; raw coefficients are in raw_restoration.json and states.pt.

| Pool | Shift | Layer | MSE before | MSE after | Ratio | Mean |gamma| | Max |gamma| | Mean |beta| | Max |beta| | Neg fraction | Cap fraction | Min variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 1 | 0.110612 | 0.037903 | 0.342667 | 1.349046 | 7.000000 | 0.241521 | 1.423218 | 0.000000 | 0.062500 | 0.000000 |
| A | brightness_dark | 2 | 0.052801 | 0.046941 | 0.889011 | 0.210286 | 0.731164 | 0.111223 | 0.796150 | 0.000000 | 0.000000 | 1.18513e-06 |
| A | contrast_low | 1 | 0.048979 | 0.024561 | 0.501463 | 0.808984 | 1.981961 | 0.042711 | 0.120588 | 0.000000 | 0.000000 | 0.001438 |
| A | contrast_low | 2 | 0.053768 | 0.047168 | 0.877239 | 0.138332 | 0.780103 | 0.090396 | 0.331381 | 0.000000 | 0.000000 | 0.000000 |
| A | gaussian_noise | 1 | 0.024088 | 0.013566 | 0.563203 | 0.229718 | 0.484937 | 0.020030 | 0.043296 | 0.000000 | 0.000000 | 0.023616 |
| A | gaussian_noise | 2 | 0.013504 | 0.010795 | 0.799387 | 0.112136 | 0.279338 | 0.079236 | 0.341428 | 0.000000 | 0.000000 | 2.66516e-06 |
| A | gaussian_blur | 1 | 0.013121 | 0.008481 | 0.646400 | 0.219419 | 0.899923 | 0.022735 | 0.039657 | 0.000000 | 0.000000 | 0.002614 |
| A | gaussian_blur | 2 | 0.011733 | 0.010589 | 0.902551 | 0.033240 | 0.299583 | 0.034424 | 0.085078 | 0.000000 | 0.000000 | 6.1234e-07 |
| B | brightness_dark | 1 | 0.113672 | 0.039659 | 0.348891 | 1.354481 | 7.000000 | 0.248332 | 1.416345 | 0.000000 | 0.062500 | 0.000000 |
| B | brightness_dark | 2 | 0.054900 | 0.048637 | 0.885905 | 0.222764 | 0.784923 | 0.121471 | 0.936065 | 0.000000 | 0.000000 | 7.11369e-07 |
| B | contrast_low | 1 | 0.046403 | 0.025173 | 0.542479 | 0.819345 | 2.514721 | 0.044528 | 0.104708 | 0.000000 | 0.000000 | 0.001093 |
| B | contrast_low | 2 | 0.049563 | 0.042856 | 0.864664 | 0.136103 | 0.829588 | 0.088439 | 0.303568 | 0.000000 | 0.000000 | 0.000000 |
| B | gaussian_noise | 1 | 0.023330 | 0.013175 | 0.564754 | 0.232775 | 0.495758 | 0.017945 | 0.037947 | 0.000000 | 0.000000 | 0.023752 |
| B | gaussian_noise | 2 | 0.011587 | 0.009322 | 0.804524 | 0.106021 | 0.267418 | 0.067546 | 0.297730 | 0.000000 | 0.000000 | 2.70729e-06 |
| B | gaussian_blur | 1 | 0.013263 | 0.008415 | 0.634455 | 0.235141 | 1.046002 | 0.022850 | 0.039722 | 0.000000 | 0.000000 | 0.002236 |
| B | gaussian_blur | 2 | 0.010796 | 0.009605 | 0.889665 | 0.029590 | 0.179786 | 0.038210 | 0.114554 | 0.000000 | 0.000000 | 0.000000 |

Restoration ratios describe each support pairing objective, not task relevance by themselves. Different derangements change the regression target correspondence. Cross-pool semantic-control instability should not be explained solely by MSE.

## Per-class behavior

| Pool | Shift | Class | None % | Correct % | Delta pp |
| --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 0 | 0.197889 | 17.084433 | 16.886544 |
| A | brightness_dark | 1 | 21.337793 | 80.468227 | 59.130435 |
| A | brightness_dark | 2 | 84.956925 | 53.744201 | -31.212724 |
| A | brightness_dark | 3 | 2.496626 | 15.789474 | 13.292848 |
| A | brightness_dark | 4 | 0.000000 | 0.201072 | 0.201072 |
| A | brightness_dark | 5 | 0.000000 | 5.266667 | 5.266667 |
| A | brightness_dark | 6 | 54.420432 | 71.709234 | 17.288802 |
| A | brightness_dark | 7 | 0.000000 | 14.920424 | 14.920424 |
| A | brightness_dark | 8 | 12.050599 | 63.781625 | 51.731025 |
| A | brightness_dark | 9 | 0.000000 | 0.000000 | 0.000000 |
| A | contrast_low | 0 | 8.245383 | 12.928760 | 4.683377 |
| A | contrast_low | 1 | 5.953177 | 66.889632 | 60.936455 |
| A | contrast_low | 2 | 60.304838 | 40.225315 | -20.079523 |
| A | contrast_low | 3 | 0.674764 | 10.053981 | 9.379217 |
| A | contrast_low | 4 | 0.000000 | 0.000000 | 0.000000 |
| A | contrast_low | 5 | 1.533333 | 8.400000 | 6.866667 |
| A | contrast_low | 6 | 75.703995 | 76.555337 | 0.851343 |
| A | contrast_low | 7 | 0.000000 | 2.718833 | 2.718833 |
| A | contrast_low | 8 | 47.137150 | 67.842876 | 20.705726 |
| A | contrast_low | 9 | 0.000000 | 0.000000 | 0.000000 |
| A | gaussian_noise | 0 | 12.467018 | 17.875989 | 5.408971 |
| A | gaussian_noise | 1 | 84.949833 | 74.515050 | -10.434783 |
| A | gaussian_noise | 2 | 35.719019 | 46.123260 | 10.404241 |
| A | gaussian_noise | 3 | 10.593792 | 12.550607 | 1.956815 |
| A | gaussian_noise | 4 | 0.067024 | 0.268097 | 0.201072 |
| A | gaussian_noise | 5 | 12.533333 | 15.866667 | 3.333333 |
| A | gaussian_noise | 6 | 79.567780 | 76.948265 | -2.619515 |
| A | gaussian_noise | 7 | 9.217507 | 11.140584 | 1.923077 |
| A | gaussian_noise | 8 | 61.651132 | 69.707057 | 8.055925 |
| A | gaussian_noise | 9 | 0.000000 | 0.000000 | 0.000000 |
| A | gaussian_blur | 0 | 16.160950 | 15.699208 | -0.461741 |
| A | gaussian_blur | 1 | 48.963211 | 75.852843 | 26.889632 |
| A | gaussian_blur | 2 | 57.786614 | 45.394301 | -12.392313 |
| A | gaussian_blur | 3 | 13.090418 | 15.856950 | 2.766532 |
| A | gaussian_blur | 4 | 0.335121 | 0.268097 | -0.067024 |
| A | gaussian_blur | 5 | 14.733333 | 15.066667 | 0.333333 |
| A | gaussian_blur | 6 | 76.096922 | 78.519974 | 2.423052 |
| A | gaussian_blur | 7 | 10.411141 | 15.318302 | 4.907162 |
| A | gaussian_blur | 8 | 71.438083 | 68.508655 | -2.929427 |
| A | gaussian_blur | 9 | 0.000000 | 0.000000 | 0.000000 |
| B | brightness_dark | 0 | 0.197889 | 17.480211 | 17.282322 |
| B | brightness_dark | 1 | 21.337793 | 79.866221 | 58.528428 |
| B | brightness_dark | 2 | 84.956925 | 53.346587 | -31.610338 |
| B | brightness_dark | 3 | 2.496626 | 16.936572 | 14.439946 |
| B | brightness_dark | 4 | 0.000000 | 0.268097 | 0.268097 |
| B | brightness_dark | 5 | 0.000000 | 5.533333 | 5.533333 |
| B | brightness_dark | 6 | 54.420432 | 71.250819 | 16.830386 |
| B | brightness_dark | 7 | 0.000000 | 15.251989 | 15.251989 |
| B | brightness_dark | 8 | 12.050599 | 64.314248 | 52.263648 |
| B | brightness_dark | 9 | 0.000000 | 0.000000 | 0.000000 |
| B | contrast_low | 0 | 8.245383 | 10.026385 | 1.781003 |
| B | contrast_low | 1 | 5.953177 | 70.769231 | 64.816054 |
| B | contrast_low | 2 | 60.304838 | 39.827700 | -20.477137 |
| B | contrast_low | 3 | 0.674764 | 8.771930 | 8.097166 |
| B | contrast_low | 4 | 0.000000 | 0.000000 | 0.000000 |
| B | contrast_low | 5 | 1.533333 | 6.933333 | 5.400000 |
| B | contrast_low | 6 | 75.703995 | 76.424361 | 0.720367 |
| B | contrast_low | 7 | 0.000000 | 3.050398 | 3.050398 |
| B | contrast_low | 8 | 47.137150 | 65.312916 | 18.175766 |
| B | contrast_low | 9 | 0.000000 | 0.000000 | 0.000000 |
| B | gaussian_noise | 0 | 12.467018 | 17.150396 | 4.683377 |
| B | gaussian_noise | 1 | 84.949833 | 74.314381 | -10.635452 |
| B | gaussian_noise | 2 | 35.719019 | 45.062956 | 9.343936 |
| B | gaussian_noise | 3 | 10.593792 | 13.090418 | 2.496626 |
| B | gaussian_noise | 4 | 0.067024 | 0.201072 | 0.134048 |
| B | gaussian_noise | 5 | 12.533333 | 16.133333 | 3.600000 |
| B | gaussian_noise | 6 | 79.567780 | 77.799607 | -1.768173 |
| B | gaussian_noise | 7 | 9.217507 | 11.007958 | 1.790451 |
| B | gaussian_noise | 8 | 61.651132 | 69.307590 | 7.656458 |
| B | gaussian_noise | 9 | 0.000000 | 0.000000 | 0.000000 |
| B | gaussian_blur | 0 | 16.160950 | 15.237467 | -0.923483 |
| B | gaussian_blur | 1 | 48.963211 | 76.655518 | 27.692308 |
| B | gaussian_blur | 2 | 57.786614 | 44.201458 | -13.585156 |
| B | gaussian_blur | 3 | 13.090418 | 16.126856 | 3.036437 |
| B | gaussian_blur | 4 | 0.335121 | 0.201072 | -0.134048 |
| B | gaussian_blur | 5 | 14.733333 | 15.133333 | 0.400000 |
| B | gaussian_blur | 6 | 76.096922 | 78.519974 | 2.423052 |
| B | gaussian_blur | 7 | 10.411141 | 16.114058 | 5.702918 |
| B | gaussian_blur | 8 | 71.438083 | 67.909454 | -3.528628 |
| B | gaussian_blur | 9 | 0.000000 | 0.000000 | 0.000000 |

All conditions' class counts/accuracies are retained in per_class.csv. This table does not imply that each class improves; individual class losses remain visible.

## Verification

26 baseline tests passed before editing, three new writer tests passed before evaluator integration, and the complete 31-test suite passed remotely before execution. A real 6-client smoke passed before the 100-client formal evaluation. Six clients were used because the first two lack several query classes; this avoids undefined smoke macro-class metrics and does not select pools/states by accuracy. Both stages checked exact identity on all 100 clean-query clients before corruption fitting.

Formal: 5,600 records, 42 globally fitted states; no per-client state fits. Checkpoint and old writer/corruption hashes unchanged; exact pool SHA checked. Model hash checked after every fit/evaluation; no calibration/query overlap; exact target multisets and semantic fractions. Saved predictions independently reconstruct every class/weighted accuracy with maximum error 7.11e-15 pp. Fraction-based gates match every runtime PASS/FAIL. Identity comparison uses the old saved receipt and new exact-equality checks, without rerunning historical grids.

## Failures / uncertainties

No formal runtime or identity failure. The workflow encountered one transient SCP connection closure during analysis upload; its existing legacy-SCP retry succeeded. No experiment was restarted. The pre-existing NVML mismatch warning persists; PyTorch CUDA completed on A6000.

This checkpoint has strongly uneven class performance and uses the established PFLlib client split drawn from merged original train/test data, not an official CIFAR-10 test benchmark. The diagnostic is transductive with respect to client membership (calibration and query clients overlap), but calibration image IDs are disjoint from all query IDs; it demonstrates state reuse across clients, not held-out-client generalization. Only two calibration draws and one checkpoint are tested. Clean target pairs are privileged oracle information. Do not attribute every change from T006 solely to the new writer: the balanced global calibration protocol also differs, and no old-writer transfer grid was run. Noise remains a strict failure despite being extremely close to recovery .5.

## C-A / C-B / C-C / C-D decision

**C-A: transferable covariate state exists under this balanced paired-clean diagnostic.** Three shifts pass independently in both pools under the unchanged gates. This supports the capacity of the existing neutral 192-scalar operator for these shifts. It does not establish a deployable source-only writer or a final federated method.

## Recommended next action

Keep the diagonal operator and return to Research Lead. A next package may investigate whether source-side unlabeled evidence can estimate the useful oracle state, with matched source-context controls and the same frozen protocol. No SSL, meta-learning, federated retraining, richer operator, or next stage was launched. The 15-minute heartbeat continues to check for actionable instructions.
