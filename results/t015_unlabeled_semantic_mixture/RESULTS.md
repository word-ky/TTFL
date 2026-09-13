# CODEX -> CHATGPT

## Timestamp / commit / run

2026-09-14T04:46:07+08:00. Leadffb85ee; runtime 099c5f1b5fa07bf930d94aac46a5a4a8957d0786; run 20260914-043941-ttfl-t015-gpu1; release20260914-043936-ttfl-t015; NVIDIA RTX A6000 GPU1; 31.13s. One formal run, exit0. Enclosing delivery commit contains this independent report.

## Preflight / hash / model verification

57 tests PASS including four new focused tests: mixture normalization/branch isolation, clean-zero regression, exact utility ties with label-free API, and mandatory Phase-A freeze before privileged label access. Checkpoint, states, support manifest, exact T009 decisions, T011 query artifact and T014 template hashes match. All support IDs are disjoint from calibration pools and every query ID. 1,820 support-only forwards; model/candidate-state hashes unchanged after each forward;0 query forwards.

Preflight reconstructs120 historical aggregate offline rows: T014 class_context_only and T013 zero/half-oracle. These are existing supervised aggregate references required by the task; individual target query arrays, candidate outcomes and labels remain unopened until after Phase B. Every1,600 target-excluded T014 template is reconstructed from half counts after the freezes and matches exactly. No new template, classifier calibration, state fit or temperature tuning.

## Exact T009 context regression

Saved T009 context-name decisions are reused, never replaced by T010/T011 selection. Both confusion matrices reconstruct exactly: A488/500=97.6%, B490/500=98.0%; per-context clean/dark/contrast/noise/blur are92/100/100/100/96% and93/100/100/100/97%. context_decisions.csv preserves all1,000 decisions.

## Source and privileged freeze order

Phase A frozen at 2026-09-13T20:40:09.724655+00:00: four mixtures from T=1 support posteriors;40,000 P01/P11/zero-soft/zero-hard/uniform choices, full utilities and argmax sets; source labels and target query outcomes unopened. Mixture SHA f49ea4416779eb86ec6186f405f6b67fbf314913c6fde1a6eeb67b3a70f2d018; choices SHA e1662d9110b5b55f5d9538d2d0a57fb2d73806640f2732d6bf51d4e366f12896. Raw support logits are preserved in the formal NPZ.

Only after require_phaseA_freeze verifies the persisted receipt/hash are support y values accessed. Phase B freezes16,000 P00/P10 choices at 2026-09-13T20:40:13.224834+00:00, before opening T011 target query predictions/labels. No final-state re-estimation of mixture or iteration occurs.

P00 uses support labels and oracle context; P01 uses posterior mixture under the true frozen context state with oracle context; P10 uses support labels and T009 source context; P11 uses posterior mixture under T009’s provisional state and source context. Controls use zero-soft, zero-hard and uniform mixture with the same source context. P01 is context-privileged even though support-label-free; P00/P10 are label-privileged.

Softmax and probability aggregation are float64 after unchanged-precision logits. Utility evaluation preserves exact T014 template fractions, converts each binary float64 mixture mass exactly to Fraction and sums without rounding. True support histograms and uniform pi are exact rationals. Independent NumPy reconstruction from logits has max mixture error2.22e-16. All56,000 utility vectors/argmax choices reproduce exactly; all280 reported metrics reconstruct from saved per-example predictions/counts; gates use exact fractions.

## Semantic-mixture quality

| bank | target | estimator | n | L1_mean | L1_median | JS_mean | JS_median | top_class_agreement | spearman_mean | spearman_median | spearman_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | pi_zero_soft | 100 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 0.318658 | 100 |
| A | clean | pi_zero_hard | 100 | 1.264000 | 1.300000 | 0.360963 | 0.343529 | 0.290000 | 0.206758 | 0.207182 | 100 |
| A | clean | pi_oracle_state_soft | 100 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 0.318658 | 100 |
| A | clean | pi_source_state_soft | 100 | 1.295678 | 1.331933 | 0.335741 | 0.332573 | 0.270000 | 0.249138 | 0.296440 | 100 |
| A | brightness_dark | pi_zero_soft | 100 | 1.475594 | 1.509093 | 0.399958 | 0.404229 | 0.150000 | 0.103716 | 0.172791 | 100 |
| A | brightness_dark | pi_zero_hard | 100 | 1.574000 | 1.700000 | 0.492859 | 0.538267 | 0.140000 | 0.074194 | 0.073403 | 100 |
| A | brightness_dark | pi_oracle_state_soft | 100 | 1.310229 | 1.303101 | 0.341273 | 0.327241 | 0.290000 | 0.237988 | 0.307075 | 100 |
| A | brightness_dark | pi_source_state_soft | 100 | 1.310229 | 1.303101 | 0.341273 | 0.327241 | 0.290000 | 0.237988 | 0.307075 | 100 |
| A | contrast_low | pi_zero_soft | 100 | 1.451654 | 1.470378 | 0.388083 | 0.380322 | 0.240000 | 0.128128 | 0.169214 | 100 |
| A | contrast_low | pi_zero_hard | 100 | 1.531000 | 1.700000 | 0.479520 | 0.518892 | 0.200000 | 0.076256 | 0.095019 | 100 |
| A | contrast_low | pi_oracle_state_soft | 100 | 1.345680 | 1.349090 | 0.356656 | 0.349470 | 0.270000 | 0.186676 | 0.223449 | 100 |
| A | contrast_low | pi_source_state_soft | 100 | 1.345680 | 1.349090 | 0.356656 | 0.349470 | 0.270000 | 0.186676 | 0.223449 | 100 |
| A | gaussian_noise | pi_zero_soft | 100 | 1.325835 | 1.403100 | 0.351475 | 0.351416 | 0.240000 | 0.200670 | 0.257296 | 100 |
| A | gaussian_noise | pi_zero_hard | 100 | 1.319000 | 1.400000 | 0.384594 | 0.380952 | 0.260000 | 0.174382 | 0.197595 | 100 |
| A | gaussian_noise | pi_oracle_state_soft | 100 | 1.302194 | 1.331811 | 0.337775 | 0.331711 | 0.270000 | 0.243293 | 0.302750 | 100 |
| A | gaussian_noise | pi_source_state_soft | 100 | 1.302194 | 1.331811 | 0.337775 | 0.331711 | 0.270000 | 0.243293 | 0.302750 | 100 |
| A | gaussian_blur | pi_zero_soft | 100 | 1.329352 | 1.310639 | 0.340808 | 0.324553 | 0.300000 | 0.241997 | 0.290129 | 100 |
| A | gaussian_blur | pi_zero_hard | 100 | 1.323000 | 1.400000 | 0.384538 | 0.378375 | 0.300000 | 0.172711 | 0.195272 | 100 |
| A | gaussian_blur | pi_oracle_state_soft | 100 | 1.302992 | 1.328046 | 0.338005 | 0.337836 | 0.270000 | 0.245529 | 0.296440 | 100 |
| A | gaussian_blur | pi_source_state_soft | 100 | 1.311582 | 1.328046 | 0.340862 | 0.337836 | 0.270000 | 0.242048 | 0.296440 | 100 |
| B | clean | pi_zero_soft | 100 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 0.318658 | 100 |
| B | clean | pi_zero_hard | 100 | 1.264000 | 1.300000 | 0.360963 | 0.343529 | 0.290000 | 0.206758 | 0.207182 | 100 |
| B | clean | pi_oracle_state_soft | 100 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 0.318658 | 100 |
| B | clean | pi_source_state_soft | 100 | 1.295334 | 1.331933 | 0.335466 | 0.334570 | 0.270000 | 0.252023 | 0.296440 | 100 |
| B | brightness_dark | pi_zero_soft | 100 | 1.475594 | 1.509093 | 0.399958 | 0.404229 | 0.150000 | 0.103716 | 0.172791 | 100 |
| B | brightness_dark | pi_zero_hard | 100 | 1.574000 | 1.700000 | 0.492859 | 0.538267 | 0.140000 | 0.074194 | 0.073403 | 100 |
| B | brightness_dark | pi_oracle_state_soft | 100 | 1.308314 | 1.298764 | 0.340602 | 0.326238 | 0.290000 | 0.238952 | 0.307075 | 100 |
| B | brightness_dark | pi_source_state_soft | 100 | 1.308314 | 1.298764 | 0.340602 | 0.326238 | 0.290000 | 0.238952 | 0.307075 | 100 |
| B | contrast_low | pi_zero_soft | 100 | 1.451654 | 1.470378 | 0.388083 | 0.380322 | 0.240000 | 0.128128 | 0.169214 | 100 |
| B | contrast_low | pi_zero_hard | 100 | 1.531000 | 1.700000 | 0.479520 | 0.518892 | 0.200000 | 0.076256 | 0.095019 | 100 |
| B | contrast_low | pi_oracle_state_soft | 100 | 1.347697 | 1.361463 | 0.358087 | 0.352300 | 0.260000 | 0.183474 | 0.214087 | 100 |
| B | contrast_low | pi_source_state_soft | 100 | 1.347697 | 1.361463 | 0.358087 | 0.352300 | 0.260000 | 0.183474 | 0.214087 | 100 |
| B | gaussian_noise | pi_zero_soft | 100 | 1.325835 | 1.403100 | 0.351475 | 0.351416 | 0.240000 | 0.200670 | 0.257296 | 100 |
| B | gaussian_noise | pi_zero_hard | 100 | 1.319000 | 1.400000 | 0.384594 | 0.380952 | 0.260000 | 0.174382 | 0.197595 | 100 |
| B | gaussian_noise | pi_oracle_state_soft | 100 | 1.304167 | 1.335631 | 0.338605 | 0.333953 | 0.270000 | 0.240809 | 0.292115 | 100 |
| B | gaussian_noise | pi_source_state_soft | 100 | 1.304167 | 1.335631 | 0.338605 | 0.333953 | 0.270000 | 0.240809 | 0.292115 | 100 |
| B | gaussian_blur | pi_zero_soft | 100 | 1.329352 | 1.310639 | 0.340808 | 0.324553 | 0.300000 | 0.241997 | 0.290129 | 100 |
| B | gaussian_blur | pi_zero_hard | 100 | 1.323000 | 1.400000 | 0.384538 | 0.378375 | 0.300000 | 0.172711 | 0.195272 | 100 |
| B | gaussian_blur | pi_oracle_state_soft | 100 | 1.303075 | 1.328956 | 0.338155 | 0.339476 | 0.280000 | 0.249099 | 0.311400 | 100 |
| B | gaussian_blur | pi_source_state_soft | 100 | 1.310458 | 1.328956 | 0.340605 | 0.339476 | 0.280000 | 0.245617 | 0.311400 | 100 |

Mixture quality is measured against the true K20 support histogram, not query labels. Top-class agreement uses deterministic first-class argmax; the additional top_class_in_true_argmax field preserves true-histogram ties. All quartile strata appear in mixture_quality.csv. Constant vectors produce NA Spearman with valid counts retained. These distances are descriptive and are not classifier-calibration guarantees.

Primary source-state mixture mean L1 error is about1.30–1.35, and top-class agreement only26–29%. Thus the frozen model does not reliably expose the dominant semantic class even after provisional context correction.

| bank | target | stratum | n | L1_mean | JS_mean | top_class_agreement | spearman_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | context_True | 92 | 1.282549 | 0.331292 | 0.260870 | 0.257961 |
| A | clean | context_False | 8 | 1.446667 | 0.386912 | 0.375000 | 0.147673 |
| A | brightness_dark | context_True | 100 | 1.310229 | 0.341273 | 0.290000 | 0.237988 |
| A | brightness_dark | context_False | 0 | None | None | None | None |
| A | contrast_low | context_True | 100 | 1.345680 | 0.356656 | 0.270000 | 0.186676 |
| A | contrast_low | context_False | 0 | None | None | None | None |
| A | gaussian_noise | context_True | 100 | 1.302194 | 0.337775 | 0.270000 | 0.243293 |
| A | gaussian_noise | context_False | 0 | None | None | None | None |
| A | gaussian_blur | context_True | 96 | 1.309989 | 0.340534 | 0.250000 | 0.237803 |
| A | gaussian_blur | context_False | 4 | 1.349793 | 0.348739 | 0.750000 | 0.343922 |
| B | clean | context_True | 93 | 1.287771 | 0.333182 | 0.258065 | 0.258307 |
| B | clean | context_False | 7 | 1.395808 | 0.365811 | 0.428571 | 0.168536 |
| B | brightness_dark | context_True | 100 | 1.308314 | 0.340602 | 0.290000 | 0.238952 |
| B | brightness_dark | context_False | 0 | None | None | None | None |
| B | contrast_low | context_True | 100 | 1.347697 | 0.358087 | 0.260000 | 0.183474 |
| B | contrast_low | context_False | 0 | None | None | None | None |
| B | gaussian_noise | context_True | 100 | 1.304167 | 0.338605 | 0.270000 | 0.240809 |
| B | gaussian_noise | context_False | 0 | None | None | None | None |
| B | gaussian_blur | context_True | 97 | 1.307604 | 0.339607 | 0.268041 | 0.243044 |
| B | gaussian_blur | context_False | 3 | 1.402745 | 0.372859 | 0.666667 | 0.328813 |

## Policy accuracy / regret

Macro-class accuracy averaged across four salts for readability (all per-salt/bank/context/policy rows, weighted and macro-client metrics, choice frequencies, half-oracle hit rates and regret statistics remain in policy_metrics.csv):

| bank | target | zero | T014_class_context | P00 | P01 | P10 | P11 | zero_soft | zero_hard | uniform |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | 34.138690 | 40.411844 | 40.311665 | 36.245939 | 40.176567 | 35.403716 | 35.577668 | 35.802859 | 34.320986 |
| A | brightness_dark | 17.546026 | 35.466289 | 35.477773 | 32.296536 | 35.477773 | 32.296536 | 30.415726 | 22.223445 | 32.296536 |
| A | contrast_low | 19.955264 | 33.739205 | 33.519650 | 28.585856 | 33.519650 | 28.585856 | 28.118911 | 23.155531 | 28.561473 |
| A | gaussian_noise | 30.676644 | 37.615401 | 37.528393 | 33.032858 | 37.528393 | 33.032858 | 32.898874 | 32.793003 | 32.499558 |
| A | gaussian_blur | 30.901579 | 39.995770 | 40.019018 | 34.079180 | 40.019018 | 32.220486 | 33.296082 | 33.854137 | 31.263385 |
| B | clean | 34.138690 | 40.275622 | 40.268839 | 36.015444 | 40.089245 | 35.125628 | 35.361697 | 35.589144 | 34.202018 |
| B | brightness_dark | 17.546026 | 35.443927 | 35.292164 | 32.424808 | 35.292164 | 32.424808 | 30.540209 | 22.152906 | 32.424808 |
| B | contrast_low | 19.955264 | 33.604517 | 33.311370 | 28.209162 | 33.311370 | 28.209162 | 26.458303 | 22.943772 | 28.111626 |
| B | gaussian_noise | 30.676644 | 37.658753 | 37.580895 | 32.922221 | 37.580895 | 32.922221 | 32.730445 | 32.589541 | 32.406771 |
| B | gaussian_blur | 30.901579 | 39.981846 | 39.879851 | 34.103375 | 39.879851 | 32.649585 | 33.158257 | 33.873781 | 31.716664 |

The source support remains disjoint from query halves and is identical across salts; only the frozen offline template training orientation changes. Both halves are concatenated for full-query metrics. No query forward pass or query-based choice is added. The half-oracle is a historical optimistic comparison, selected on its own evaluated outcomes.

## Predeclared gates, all salts and banks

| gate | salt | bank | target | capture | passed |
| --- | --- | --- | --- | --- | --- |
| SUPPORT_COMP | T013-S0 | A | brightness_dark | 0.998327 | True |
| SEM_EST | T013-S0 | A | brightness_dark | 0.822583 | True |
| SEM_SRC | T013-S0 | A | brightness_dark | 0.822583 | True |
| SUPPORT_COMP | T013-S0 | A | contrast_low | 0.986908 | True |
| SEM_EST | T013-S0 | A | contrast_low | 0.639771 | False |
| SEM_SRC | T013-S0 | A | contrast_low | 0.639771 | False |
| SUPPORT_COMP | T013-S0 | A | gaussian_noise | 0.993184 | True |
| SEM_EST | T013-S0 | A | gaussian_noise | 0.345595 | False |
| SEM_SRC | T013-S0 | A | gaussian_noise | 0.345595 | False |
| SUPPORT_COMP | T013-S0 | A | gaussian_blur | 1.001485 | True |
| SEM_EST | T013-S0 | A | gaussian_blur | 0.339737 | False |
| SEM_SRC | T013-S0 | A | gaussian_blur | 0.136582 | False |
| SUPPORT_COMP | T013-S0 | B | brightness_dark | 0.986806 | True |
| SEM_EST | T013-S0 | B | brightness_dark | 0.840166 | True |
| SEM_SRC | T013-S0 | B | brightness_dark | 0.840166 | True |
| SUPPORT_COMP | T013-S0 | B | contrast_low | 0.981641 | True |
| SEM_EST | T013-S0 | B | contrast_low | 0.616021 | False |
| SEM_SRC | T013-S0 | B | contrast_low | 0.616021 | False |
| SUPPORT_COMP | T013-S0 | B | gaussian_noise | 0.994311 | True |
| SEM_EST | T013-S0 | B | gaussian_noise | 0.329452 | False |
| SEM_SRC | T013-S0 | B | gaussian_noise | 0.329452 | False |
| SUPPORT_COMP | T013-S0 | B | gaussian_blur | 0.991892 | True |
| SEM_EST | T013-S0 | B | gaussian_blur | 0.350111 | False |
| SEM_SRC | T013-S0 | B | gaussian_blur | 0.187822 | False |
| SUPPORT_COMP | T013-S1 | A | brightness_dark | 1.000898 | True |
| SEM_EST | T013-S1 | A | brightness_dark | 0.822913 | True |
| SEM_SRC | T013-S1 | A | brightness_dark | 0.822913 | True |
| SUPPORT_COMP | T013-S1 | A | contrast_low | 0.978663 | True |
| SEM_EST | T013-S1 | A | contrast_low | 0.636357 | False |
| SEM_SRC | T013-S1 | A | contrast_low | 0.636357 | False |
| SUPPORT_COMP | T013-S1 | A | gaussian_noise | 0.980917 | True |
| SEM_EST | T013-S1 | A | gaussian_noise | 0.341052 | False |
| SEM_SRC | T013-S1 | A | gaussian_noise | 0.341052 | False |
| SUPPORT_COMP | T013-S1 | A | gaussian_blur | 1.020124 | True |
| SEM_EST | T013-S1 | A | gaussian_blur | 0.353964 | False |
| SEM_SRC | T013-S1 | A | gaussian_blur | 0.150960 | False |
| SUPPORT_COMP | T013-S1 | B | brightness_dark | 0.990117 | True |
| SEM_EST | T013-S1 | B | brightness_dark | 0.838264 | True |
| SEM_SRC | T013-S1 | B | brightness_dark | 0.838264 | True |
| SUPPORT_COMP | T013-S1 | B | contrast_low | 0.974498 | True |
| SEM_EST | T013-S1 | B | contrast_low | 0.623059 | False |
| SEM_SRC | T013-S1 | B | contrast_low | 0.623059 | False |
| SUPPORT_COMP | T013-S1 | B | gaussian_noise | 0.983873 | True |
| SEM_EST | T013-S1 | B | gaussian_noise | 0.323619 | False |
| SEM_SRC | T013-S1 | B | gaussian_noise | 0.323619 | False |
| SUPPORT_COMP | T013-S1 | B | gaussian_blur | 0.995697 | True |
| SEM_EST | T013-S1 | B | gaussian_blur | 0.359085 | False |
| SEM_SRC | T013-S1 | B | gaussian_blur | 0.196677 | False |
| SUPPORT_COMP | T013-S2 | A | brightness_dark | 1.000183 | True |
| SEM_EST | T013-S2 | A | brightness_dark | 0.822589 | True |
| SEM_SRC | T013-S2 | A | brightness_dark | 0.822589 | True |
| SUPPORT_COMP | T013-S2 | A | contrast_low | 0.984693 | True |
| SEM_EST | T013-S2 | A | contrast_low | 0.632545 | False |
| SEM_SRC | T013-S2 | A | contrast_low | 0.632545 | False |
| SUPPORT_COMP | T013-S2 | A | gaussian_noise | 0.989269 | True |
| SEM_EST | T013-S2 | A | gaussian_noise | 0.343991 | False |
| SEM_SRC | T013-S2 | A | gaussian_noise | 0.343991 | False |
| SUPPORT_COMP | T013-S2 | A | gaussian_blur | 0.993368 | True |
| SEM_EST | T013-S2 | A | gaussian_blur | 0.345352 | False |
| SEM_SRC | T013-S2 | A | gaussian_blur | 0.140552 | False |
| SUPPORT_COMP | T013-S2 | B | brightness_dark | 0.994977 | True |
| SEM_EST | T013-S2 | B | brightness_dark | 0.836060 | True |
| SEM_SRC | T013-S2 | B | brightness_dark | 0.836060 | True |
| SUPPORT_COMP | T013-S2 | B | contrast_low | 0.973886 | True |
| SEM_EST | T013-S2 | B | contrast_low | 0.619704 | False |
| SEM_SRC | T013-S2 | B | contrast_low | 0.619704 | False |
| SUPPORT_COMP | T013-S2 | B | gaussian_noise | 0.994379 | True |
| SEM_EST | T013-S2 | B | gaussian_noise | 0.325921 | False |
| SEM_SRC | T013-S2 | B | gaussian_noise | 0.325921 | False |
| SUPPORT_COMP | T013-S2 | B | gaussian_blur | 0.982045 | True |
| SEM_EST | T013-S2 | B | gaussian_blur | 0.360550 | False |
| SEM_SRC | T013-S2 | B | gaussian_blur | 0.198869 | False |
| SUPPORT_COMP | T013-S3 | A | brightness_dark | 1.003166 | True |
| SEM_EST | T013-S3 | A | brightness_dark | 0.822282 | True |
| SEM_SRC | T013-S3 | A | brightness_dark | 0.822282 | True |
| SUPPORT_COMP | T013-S3 | A | contrast_low | 0.986048 | True |
| SEM_EST | T013-S3 | A | contrast_low | 0.636418 | False |
| SEM_SRC | T013-S3 | A | contrast_low | 0.636418 | False |
| SUPPORT_COMP | T013-S3 | A | gaussian_noise | 0.986588 | True |
| SEM_EST | T013-S3 | A | gaussian_noise | 0.344914 | False |
| SEM_SRC | T013-S3 | A | gaussian_noise | 0.344914 | False |
| SUPPORT_COMP | T013-S3 | A | gaussian_blur | 0.995552 | True |
| SEM_EST | T013-S3 | A | gaussian_blur | 0.355036 | False |
| SEM_SRC | T013-S3 | A | gaussian_blur | 0.150537 | False |
| SUPPORT_COMP | T013-S3 | B | brightness_dark | 0.994209 | True |
| SEM_EST | T013-S3 | B | brightness_dark | 0.839215 | True |
| SEM_SRC | T013-S3 | B | brightness_dark | 0.839215 | True |
| SUPPORT_COMP | T013-S3 | B | contrast_low | 0.984036 | True |
| SEM_EST | T013-S3 | B | contrast_low | 0.613268 | False |
| SEM_SRC | T013-S3 | B | contrast_low | 0.613268 | False |
| SUPPORT_COMP | T013-S3 | B | gaussian_noise | 0.982956 | True |
| SEM_EST | T013-S3 | B | gaussian_noise | 0.321996 | False |
| SEM_SRC | T013-S3 | B | gaussian_noise | 0.321996 | False |
| SUPPORT_COMP | T013-S3 | B | gaussian_blur | 0.985600 | True |
| SEM_EST | T013-S3 | B | gaussian_blur | 0.356705 | False |
| SEM_SRC | T013-S3 | B | gaussian_blur | 0.195385 | False |

**SUPPORT-COMP-A PASS:**4/4 shifts satisfy P00/T014 class_context gain capture in every salt/bank, and clean safety passes. The true K20 support sample itself is sufficiently informative for this frozen mechanism under the task’s gate; support sampling is not the first bottleneck here.

**SEM-EST-A FAIL:** only Dark jointly retains80% of P00 gain under oracle context. Contrast retains roughly61–64%, Noise32–35%, Blur34–36%. Even with context identity supplied, frozen posterior mixtures cannot replace support labels.

**SEM-SRC-A FAIL:** again only Dark passes gain capture. Source P11 Blur capture drops further to about14–20%. Clean safety passes, and P11 beats uniform by>=.5pp on Noise/Blur in both banks, but these do not compensate for the failed gain-capture condition.

| bank | target | mean_P11_minus_uniform | passed |
| --- | --- | --- | --- |
| A | brightness_dark | 0.000000 | False |
| B | brightness_dark | 0.000000 | False |
| A | contrast_low | 0.024383 | False |
| B | contrast_low | 0.097536 | False |
| A | gaussian_noise | 0.533300 | True |
| B | gaussian_noise | 0.515450 | True |
| A | gaussian_blur | 0.957101 | True |
| B | gaussian_blur | 0.932921 | True |

All capture denominators and raw gains are retained in capture_ratios.csv. Ratios are omitted if the denominator is nonpositive; no ratio is clipped or averaged before deciding the gate.

## Gap attribution: sampling / semantic estimation / context / interaction

| salt | bank | target | semantic_only_gap | context_only_gap | full_source_gap |
| --- | --- | --- | --- | --- | --- |
| T013-S0 | A | brightness_dark | 3.181422 | 0.000000 | 3.181422 |
| T013-S0 | A | contrast_low | 4.878837 | 0.000000 | 4.878837 |
| T013-S0 | A | gaussian_noise | 4.477637 | 0.000000 | 4.477637 |
| T013-S0 | A | gaussian_blur | 6.040848 | 0.000000 | 7.899542 |
| T013-S0 | B | brightness_dark | 2.830550 | 0.000000 | 2.830550 |
| T013-S0 | B | contrast_low | 5.148913 | 0.000000 | 5.148913 |
| T013-S0 | B | gaussian_noise | 4.625108 | 0.000000 | 4.625108 |
| T013-S0 | B | gaussian_blur | 5.821729 | 0.000000 | 7.275519 |
| T013-S1 | A | brightness_dark | 3.174233 | 0.000000 | 3.174233 |
| T013-S1 | A | contrast_low | 4.917975 | 0.000000 | 4.917975 |
| T013-S1 | A | gaussian_noise | 4.530232 | 0.000000 | 4.530232 |
| T013-S1 | A | gaussian_blur | 5.915068 | 0.000000 | 7.773762 |
| T013-S1 | B | brightness_dark | 2.870742 | 0.000000 | 2.870742 |
| T013-S1 | B | contrast_low | 4.981596 | 0.000000 | 4.981596 |
| T013-S1 | B | gaussian_noise | 4.665423 | 0.000000 | 4.665423 |
| T013-S1 | B | gaussian_blur | 5.737144 | 0.000000 | 7.190934 |
| T013-S2 | A | brightness_dark | 3.181309 | 0.000000 | 3.181309 |
| T013-S2 | A | contrast_low | 4.999483 | 0.000000 | 4.999483 |
| T013-S2 | A | gaussian_noise | 4.483347 | 0.000000 | 4.483347 |
| T013-S2 | A | gaussian_blur | 5.941341 | 0.000000 | 7.800035 |
| T013-S2 | B | brightness_dark | 2.917517 | 0.000000 | 2.917517 |
| T013-S2 | B | contrast_low | 5.069216 | 0.000000 | 5.069216 |
| T013-S2 | B | gaussian_noise | 4.658196 | 0.000000 | 4.658196 |
| T013-S2 | B | gaussian_blur | 5.749747 | 0.000000 | 7.203537 |
| T013-S3 | A | brightness_dark | 3.187988 | 0.000000 | 3.187988 |
| T013-S3 | A | contrast_low | 4.938879 | 0.000000 | 4.938879 |
| T013-S3 | A | gaussian_noise | 4.490922 | 0.000000 | 4.490922 |
| T013-S3 | A | gaussian_blur | 5.862094 | 0.000000 | 7.720788 |
| T013-S3 | B | brightness_dark | 2.850616 | 0.000000 | 2.850616 |
| T013-S3 | B | contrast_low | 5.209109 | 0.000000 | 5.209109 |
| T013-S3 | B | gaussian_noise | 4.685971 | 0.000000 | 4.685971 |
| T013-S3 | B | gaussian_blur | 5.797284 | 0.000000 | 7.251073 |

P10 equals P00 on all four shifted contexts in both banks/every salt: correct support composition makes the existing context-ID errors harmless to these choices. Thus context identification alone is not the dominant aggregate bottleneck. The large P00−P01 gap establishes semantic-mixture estimation failure independently of context errors.

On Blur, P11 is another~1.86/1.45pp below P01 on average, despite P10 preserving P00. This is consistent with an additional coupled provisional-state/posterior/template-context effect, rather than a simple context-only loss. P01 already fails, so this is a secondary interaction, not the taxonomy case where both single-error branches were individually adequate. Gaps are not forced to add.

## Clean safety and worst episodes

| salt | bank | policy | delta_pp | passed |
| --- | --- | --- | --- | --- |
| T013-S0 | A | P00 | 6.126316 | True |
| T013-S0 | A | P01 | 2.049333 | True |
| T013-S0 | A | P10 | 5.989519 | True |
| T013-S0 | A | P11 | 1.218697 | True |
| T013-S0 | A | zero_soft | 1.396940 | True |
| T013-S0 | A | zero_hard | 1.567397 | True |
| T013-S0 | A | uniform | 0.182296 | True |
| T013-S0 | B | P00 | 6.168043 | True |
| T013-S0 | B | P01 | 1.863350 | True |
| T013-S0 | B | P10 | 5.985098 | True |
| T013-S0 | B | P11 | 0.960201 | True |
| T013-S0 | B | zero_soft | 1.144800 | True |
| T013-S0 | B | zero_hard | 1.335409 | True |
| T013-S0 | B | uniform | 0.063328 | True |
| T013-S1 | A | P00 | 6.158686 | True |
| T013-S1 | A | P01 | 2.181847 | True |
| T013-S1 | A | P10 | 6.028637 | True |
| T013-S1 | A | P11 | 1.324703 | True |
| T013-S1 | A | zero_soft | 1.497225 | True |
| T013-S1 | A | zero_hard | 1.627779 | True |
| T013-S1 | A | uniform | 0.182296 | True |
| T013-S1 | B | P00 | 6.054856 | True |
| T013-S1 | B | P01 | 1.910055 | True |
| T013-S1 | B | P10 | 5.878613 | True |
| T013-S1 | B | P11 | 1.026906 | True |
| T013-S1 | B | zero_soft | 1.297661 | True |
| T013-S1 | B | zero_hard | 1.375531 | True |
| T013-S1 | B | uniform | 0.063328 | True |
| T013-S2 | A | P00 | 6.159544 | True |
| T013-S2 | A | P01 | 2.062530 | True |
| T013-S2 | A | P10 | 6.022747 | True |
| T013-S2 | A | P11 | 1.218680 | True |
| T013-S2 | A | zero_soft | 1.391202 | True |
| T013-S2 | A | zero_hard | 1.587583 | True |
| T013-S2 | A | uniform | 0.182296 | True |
| T013-S2 | B | P00 | 6.102567 | True |
| T013-S2 | B | P01 | 1.863626 | True |
| T013-S2 | B | P10 | 5.926323 | True |
| T013-S2 | B | P11 | 0.947223 | True |
| T013-S2 | B | zero_soft | 1.126295 | True |
| T013-S2 | B | zero_hard | 1.388931 | True |
| T013-S2 | B | uniform | 0.063328 | True |
| T013-S3 | A | P00 | 6.247356 | True |
| T013-S3 | A | P01 | 2.135287 | True |
| T013-S3 | A | P10 | 6.110605 | True |
| T013-S3 | A | P11 | 1.298024 | True |
| T013-S3 | A | zero_soft | 1.470546 | True |
| T013-S3 | A | zero_hard | 1.873918 | True |
| T013-S3 | A | uniform | 0.182296 | True |
| T013-S3 | B | P00 | 6.195131 | True |
| T013-S3 | B | P01 | 1.869987 | True |
| T013-S3 | B | P10 | 6.012186 | True |
| T013-S3 | B | P11 | 1.013425 | True |
| T013-S3 | B | zero_soft | 1.323271 | True |
| T013-S3 | B | zero_hard | 1.701944 | True |
| T013-S3 | B | uniform | 0.063328 | True |

P11 clean accuracy averages35.404/35.126% versus zero34.139%, so safety passes while much of P00’s supervised clean gain remains unrecovered. Zero-hard can outperform source-soft on Clean/Blur, but the predeclared primary policy is unchanged; no estimator is selected post hoc.

| salt | bank | client | target | source_context | entropy | max_class_fraction | macroclass_contribution_P11_minus_P00 | P11_states | P00_states |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T013-S0 | B | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.257525 | clean|clean | contrast_low|contrast_low |
| T013-S1 | B | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.257525 | clean|clean | contrast_low|contrast_low |
| T013-S2 | B | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.257525 | clean|clean | contrast_low|contrast_low |
| T013-S3 | B | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.257525 | clean|clean | contrast_low|contrast_low |
| T013-S0 | A | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.217391 | clean|clean | contrast_low|contrast_low |
| T013-S1 | A | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.217391 | clean|clean | contrast_low|contrast_low |
| T013-S2 | A | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.217391 | clean|clean | contrast_low|contrast_low |
| T013-S3 | A | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | -1.217391 | clean|clean | contrast_low|contrast_low |
| T013-S1 | B | 94 | clean | clean | 0.086214 | 0.950000 | -0.639822 | clean|clean | brightness_dark|brightness_dark |
| T013-S2 | B | 94 | clean | clean | 0.086214 | 0.950000 | -0.639822 | clean|clean | brightness_dark|brightness_dark |
| T013-S0 | B | 94 | gaussian_blur | gaussian_blur | 0.086214 | 0.950000 | -0.626248 | gaussian_blur|gaussian_blur | brightness_dark|brightness_dark |
| T013-S1 | B | 94 | gaussian_blur | gaussian_blur | 0.086214 | 0.950000 | -0.626248 | gaussian_blur|gaussian_blur | brightness_dark|brightness_dark |
| T013-S2 | B | 94 | gaussian_blur | gaussian_blur | 0.086214 | 0.950000 | -0.626248 | gaussian_blur|gaussian_blur | brightness_dark|brightness_dark |
| T013-S3 | B | 94 | gaussian_blur | gaussian_blur | 0.086214 | 0.950000 | -0.626248 | gaussian_blur|gaussian_blur | brightness_dark|brightness_dark |
| T013-S0 | B | 94 | clean | clean | 0.086214 | 0.950000 | -0.619911 | clean|gaussian_noise | brightness_dark|brightness_dark |
| T013-S0 | A | 94 | clean | clean | 0.086214 | 0.950000 | -0.619822 | clean|clean | brightness_dark|brightness_dark |

episode_loss_ledger.csv preserves every client/context/salt/bank loss contribution, including historical low-entropy Clean/Blur clients. These are post-freeze audits only. A contribution is weighted by global class denominators and is not the client’s own percentage-point accuracy delta.

## Mechanism vs implementation conclusion

All model-path tests, hashes, boundaries, freeze receipts, posterior/utility replays and exact count gates pass. The failure is mechanistic: target-side class composition is present in K20 support when labels are supplied, but this frozen classifier’s posteriors do not recover it well enough. Existing T009 context identification is not the dominant bottleneck with true support composition; it can still compound pseudo-semantic error on Blur.

These conclusions concern the existing low-accuracy checkpoint and synthetic PFLlib CIFAR-10 split/state bank. They do not establish that semantic mixture is fundamentally unobservable from images, and do not validate a learned writer. T014 templates remain offline supervised/context-defined objects; even P11 is a target-side observability diagnostic, not a final deployable method.

## Recommended next action

Return SUPPORT-COMP-A PASS / SEM-EST-A FAIL / SEM-SRC-A FAIL to Research Lead. The next bounded question should address semantic-mixture observability under the frozen model or a separately authorized estimator diagnostic, with correct-context and support-label controls retained. No SSL, TTT update, writer, calibration sweep, new federation, operator expansion, threshold change or T016 was started.
