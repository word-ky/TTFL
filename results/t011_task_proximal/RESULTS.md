# CODEX -> CHATGPT

## Timestamp / commit / run

2026-09-14T00:41:15+08:00. Lead 941731c; runtime f32367413cae9f6629b5648d2dd665a73c8c078c; successful run 20260914-003521-ttfl-t011-gpu1-lf; release 20260914-003432-ttfl-t011; NVIDIA RTX A6000 GPU1; 59.12s. The enclosing delivery commit contains this report.

The first launch 20260914-003444-ttfl-t011-gpu1 exited before tests or scoring because the new shell file had CRLF line endings. Converting that file to LF repaired the launch; its normalized Git content and runtime revision are unchanged. One formal evaluation completed; no sweep or alternate policy.

## Frozen objects and verification

46 tests PASS. Fixed checkpoint/operator/neutral writer/corruptions/source signature/T007R states/T008 reference hashes and exact T009 K20 manifest verified. 5240 source candidate calls and 1048 zero-logit regressions; exact max difference 0. Model and candidate states unchanged after every source/query episode. No labels or query objects enter the scoring API. All scores finite; no gradient, parameter fitting, augmentation/temperature search or new FL.

This uses the existing 100-client CIFAR-10 PFLlib merged-data per-client query split, not the official CIFAR-10 test benchmark. Earlier training used 10% participation; T011 performs only frozen-model evaluation.

## Prediction-consistency definition

C is mean Jensen-Shannon divergence (natural logarithm) between original and horizontally flipped T=1 softmax predictions. Model logits retain their precision; probabilities/reductions use double. PC-safe admits a nonzero state only when C<C_zero and flip top-1 agreement >= zero agreement, then chooses minimum C with fixed clean/dark/contrast/noise/blur tie order. Otherwise zero. Entropies, probability margins and JS-to-zero are descriptive only.

## Source-score freeze

5,000 candidate rows and 1,000 choices frozen at 2026-09-13T16:36:05.316704+00:00; scores SHA cddb195b9bd65ad93300257ddd4dfcb98301ae0bda12bb614d74e12af391e19a; decisions SHA 3b5259e5036601c89147f20c106199e5fe5a974e0d7d57a21d006d6db012799c. Historical query predictions, T009/T010 choices and T010 moment scores were opened only afterward. Query matrix frozen at 2026-09-13T16:36:30.169390+00:00. Raw source JSON stays immutable; source_candidate_scores.csv in the report adds the historical moment comparison after freezing.

## PC-safe selection specificity

Natural context-name agreement is only 54/500 (10.8%) and 48/500 (9.6%). This is descriptive, not an extra gate: task utility need not require choosing the named state. Balanced sanity is also weak, 3/20 and 2/20. Neither bank ever selects Dark, even on Dark targets.

Natural Bank A:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 30 | 0 | 14 | 51 | 5 |
| brightness_dark | 28 | 0 | 3 | 69 | 0 |
| contrast_low | 30 | 0 | 0 | 70 | 0 |
| gaussian_noise | 24 | 0 | 46 | 23 | 7 |
| gaussian_blur | 43 | 0 | 3 | 53 | 1 |

Natural Bank B:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 30 | 0 | 17 | 48 | 5 |
| brightness_dark | 26 | 0 | 3 | 71 | 0 |
| contrast_low | 35 | 0 | 0 | 65 | 0 |
| gaussian_noise | 23 | 0 | 57 | 17 | 3 |
| gaussian_blur | 41 | 0 | 3 | 55 | 1 |

Balanced Bank A:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 3 | 0 | 0 | 1 | 0 |
| brightness_dark | 1 | 0 | 0 | 3 | 0 |
| contrast_low | 2 | 0 | 0 | 2 | 0 |
| gaussian_noise | 2 | 0 | 2 | 0 | 0 |
| gaussian_blur | 2 | 0 | 0 | 2 | 0 |

Balanced Bank B:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 2 | 0 | 0 | 1 | 1 |
| brightness_dark | 2 | 0 | 0 | 2 | 0 |
| contrast_low | 1 | 0 | 0 | 3 | 0 |
| gaussian_noise | 0 | 0 | 4 | 0 | 0 |
| gaussian_blur | 1 | 0 | 0 | 3 | 0 |

## Full candidate utility matrix verification

All 5,000 client/context/bank/candidate policies evaluated: 2616 exact historical predictions reused and 2384 missing combinations computed. Each saved prediction vector independently reproduces integer per-class correct/total counts; all reused arrays match the historical source. T007R and T009/T010 aggregate policies reconstruct with max error 7.11e-15 pp. Fraction arithmetic verifies retrieval and clean gates. Matrix never changes the selector.

## Retrieval / clean safety

| bank | target | none | oracle | selected_mixed | wrong_alt | tau_pp | retained_recovery | selected_wrong_margin | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 32.296536 | 18.851161 | 14.903865 | 4.148166 | 0.088481 | 3.947295 | False |
| A | contrast_low | 19.955264 | 28.561473 | 19.389820 | 22.733171 | 3.545856 | -0.065702 | -3.343351 | False |
| A | gaussian_noise | 30.676644 | 32.499558 | 25.256486 | 30.382434 | 0.865512 | -2.973349 | -5.125949 | False |
| A | gaussian_blur | 30.901579 | 33.048500 | 32.176630 | 29.798137 | 0.809278 | 0.593898 | 2.378493 | False |
| B | brightness_dark | 17.546026 | 32.424808 | 18.766266 | 15.489587 | 4.148166 | 0.082012 | 3.276679 | False |
| B | contrast_low | 19.955264 | 28.111626 | 19.622645 | 22.693556 | 3.545856 | -0.040780 | -3.070911 | False |
| B | gaussian_noise | 30.676644 | 32.406771 | 22.532556 | 30.537680 | 0.865512 | -4.707219 | -8.005124 | False |
| B | gaussian_blur | 30.901579 | 33.009919 | 32.170821 | 28.354235 | 0.809278 | 0.602010 | 3.816586 | False |

| bank | none | selected | delta_pp | passed |
| --- | --- | --- | --- | --- |
| A | 34.138690 | 35.303839 | 1.165149 | True |
| B | 34.138690 | 34.408068 | 0.269378 | True |

**PC-RET-A FAIL: 0/4 joint shifted targets pass.** Clean improves +1.165149 / +0.269378 pp, but Dark retains only 8.85% /8.20%, Contrast has negative gain, Noise loses 5.420158 /8.144088 pp against zero, and Blur retains only 59.39% /60.20%. Better clean and slightly better Blur than T010 come with severe losses on the other shifts.

## Within-context task-alignment comparison: DeltaC vs DeltaJ

Both metrics below use the identical full five-candidate matrix, including zero: 500 rows per bank/context, 2,500 pooled. This differs from T010 selected-state-only correlations and must not be presented as a contradiction of that older analysis. Positive fractions and conditional harm exclude zero improvements by their strict definition. Correlations are descriptive; repeated candidates within clients are not independent samples, and no significance claim is made.

| bank | target | metric | pearson | spearman | positive_fraction | harmful_fraction | median_delta_acc_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | DeltaC | -0.014649 | 0.257117 | 0.212000 | 0.301887 | 0.686090 |
| A | clean | DeltaJ | -0.147076 | 0.087892 | 0.010000 | 0.600000 | -6.545455 |
| A | brightness_dark | DeltaC | -0.168867 | -0.374560 | 0.208000 | 0.509615 | -0.389394 |
| A | brightness_dark | DeltaJ | 0.291000 | 0.360973 | 0.590000 | 0.233898 | 3.571429 |
| A | contrast_low | DeltaC | -0.072166 | -0.273299 | 0.200000 | 0.420000 | 0.000000 |
| A | contrast_low | DeltaJ | 0.007717 | 0.136186 | 0.400000 | 0.220000 | 2.650053 |
| A | gaussian_noise | DeltaC | -0.132090 | 0.121046 | 0.300000 | 0.413333 | 0.000000 |
| A | gaussian_noise | DeltaJ | -0.174774 | 0.263060 | 0.200000 | 0.200000 | 0.934906 |
| A | gaussian_blur | DeltaC | 0.057126 | 0.074418 | 0.180000 | 0.366667 | 0.638211 |
| A | gaussian_blur | DeltaJ | -0.043085 | -0.042611 | 0.242000 | 0.404959 | 0.000000 |
| A | all | DeltaC | 0.016118 | 0.015681 | 0.220000 | 0.403636 | 0.000000 |
| A | all | DeltaJ | -0.020160 | 0.257563 | 0.288400 | 0.256588 | 1.768173 |
| B | clean | DeltaC | -0.031655 | 0.263202 | 0.238000 | 0.302521 | 0.613497 |
| B | clean | DeltaJ | -0.156473 | 0.123267 | 0.010000 | 0.600000 | -6.909091 |
| B | brightness_dark | DeltaC | -0.147494 | -0.331350 | 0.218000 | 0.449541 | 0.000000 |
| B | brightness_dark | DeltaJ | 0.262610 | 0.333406 | 0.578000 | 0.242215 | 2.525253 |
| B | contrast_low | DeltaC | -0.063616 | -0.256918 | 0.200000 | 0.420000 | 0.000000 |
| B | contrast_low | DeltaJ | 0.011656 | 0.123021 | 0.400000 | 0.235000 | 2.226620 |
| B | gaussian_noise | DeltaC | -0.159859 | 0.030420 | 0.334000 | 0.449102 | 0.000000 |
| B | gaussian_noise | DeltaJ | -0.153744 | 0.254861 | 0.200000 | 0.200000 | 0.934906 |
| B | gaussian_blur | DeltaC | 0.055246 | 0.102242 | 0.192000 | 0.322917 | 0.726819 |
| B | gaussian_blur | DeltaJ | -0.062379 | -0.053903 | 0.242000 | 0.429752 | 0.000000 |
| B | all | DeltaC | 0.001115 | 0.012151 | 0.236400 | 0.394247 | 0.000000 |
| B | all | DeltaJ | -0.016111 | 0.251966 | 0.286000 | 0.268531 | 1.428571 |

| bank | nonnegative_spearman_contexts | lower_harm_contexts | passed |
| --- | --- | --- | --- |
| A | 2 | 1 | False |
| B | 2 | 1 | False |

**PC-ALIGN FAIL in both banks:** only 2/4 shifts have nonnegative consistency Spearman; only 1/4 has lower harmful fraction than matched moment restoration. Dark and Contrast consistency correlations are negative. Pooled Spearman is barely positive (0.0157 /0.0122) and cannot establish useful within-context alignment. The full candidate universe changes moment statistics too; no pooled-only rescue is justified.

## True-context-state regret / query-best-state analysis

Argmax sets use tied integer correct counts, not floating tolerances. Regret thresholds >2 and >5 pp use exact fractions. All tied best names are preserved; their histogram can count more than one name per client. Query-best is a post-hoc diagnostic on these same finite query sets, not a deployable or held-out selector.

| bank | target | quartile | true_in_argmax_fraction | regret_positive_fraction | regret_gt2_fraction | regret_gt5_fraction | median | p75 | p90 | median_best_gain_vs_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | all | 0.200000 | 0.800000 | 0.610000 | 0.460000 | 4.177741 | 8.277628 | 18.826220 | 4.177741 |
| A | clean | 1 | 0.280000 | 0.720000 | 0.480000 | 0.440000 | 1.694915 | 8.796296 | 19.613639 | 1.694915 |
| A | clean | 2 | 0.120000 | 0.880000 | 0.680000 | 0.440000 | 4.545455 | 7.438017 | 17.777778 | 4.545455 |
| A | clean | 3 | 0.120000 | 0.880000 | 0.680000 | 0.560000 | 5.494505 | 10.563380 | 17.726190 | 5.494505 |
| A | clean | 4 | 0.280000 | 0.720000 | 0.600000 | 0.400000 | 3.105590 | 5.747126 | 16.056261 | 3.105590 |
| A | brightness_dark | all | 0.710000 | 0.290000 | 0.240000 | 0.210000 | 0.000000 | 1.180199 | 15.555245 | 11.437908 |
| A | contrast_low | all | 0.470000 | 0.530000 | 0.460000 | 0.340000 | 0.968013 | 8.433180 | 21.144902 | 10.702529 |
| A | gaussian_noise | all | 0.510000 | 0.490000 | 0.390000 | 0.290000 | 0.000000 | 6.177755 | 19.028143 | 4.520945 |
| A | gaussian_blur | all | 0.200000 | 0.800000 | 0.690000 | 0.470000 | 4.881086 | 11.816487 | 18.831348 | 5.555556 |
| A | gaussian_blur | 1 | 0.280000 | 0.720000 | 0.640000 | 0.400000 | 3.389831 | 13.333333 | 22.461268 | 4.166667 |
| A | gaussian_blur | 2 | 0.080000 | 0.920000 | 0.680000 | 0.480000 | 4.545455 | 10.743802 | 21.955923 | 6.145251 |
| A | gaussian_blur | 3 | 0.160000 | 0.840000 | 0.800000 | 0.640000 | 10.000000 | 13.333333 | 19.436364 | 5.555556 |
| A | gaussian_blur | 4 | 0.280000 | 0.720000 | 0.640000 | 0.360000 | 3.500000 | 7.874016 | 14.050064 | 6.832298 |
| B | clean | all | 0.220000 | 0.780000 | 0.610000 | 0.430000 | 3.529900 | 8.468710 | 18.314856 | 3.529900 |
| B | clean | 1 | 0.320000 | 0.680000 | 0.480000 | 0.440000 | 1.271186 | 7.425743 | 20.132904 | 1.271186 |
| B | clean | 2 | 0.160000 | 0.840000 | 0.640000 | 0.400000 | 4.469274 | 8.522727 | 17.777778 | 4.469274 |
| B | clean | 3 | 0.120000 | 0.880000 | 0.680000 | 0.480000 | 3.750000 | 11.971831 | 17.909091 | 3.750000 |
| B | clean | 4 | 0.280000 | 0.720000 | 0.640000 | 0.400000 | 3.105590 | 6.211180 | 15.805808 | 3.105590 |
| B | brightness_dark | all | 0.740000 | 0.260000 | 0.240000 | 0.190000 | 0.000000 | 1.180199 | 15.482517 | 11.879869 |
| B | contrast_low | all | 0.430000 | 0.570000 | 0.500000 | 0.360000 | 1.857559 | 8.561776 | 21.462185 | 10.000000 |
| B | gaussian_noise | all | 0.500000 | 0.500000 | 0.380000 | 0.310000 | 0.310559 | 6.331342 | 19.560976 | 3.955307 |
| B | gaussian_blur | all | 0.190000 | 0.810000 | 0.690000 | 0.470000 | 4.700597 | 12.000851 | 20.327273 | 5.262250 |
| B | gaussian_blur | 1 | 0.240000 | 0.760000 | 0.600000 | 0.400000 | 3.389831 | 12.500000 | 22.750535 | 3.669725 |
| B | gaussian_blur | 2 | 0.080000 | 0.920000 | 0.680000 | 0.480000 | 4.545455 | 11.046512 | 21.625344 | 6.703911 |
| B | gaussian_blur | 3 | 0.160000 | 0.840000 | 0.800000 | 0.640000 | 10.000000 | 13.636364 | 21.963636 | 5.555556 |
| B | gaussian_blur | 4 | 0.280000 | 0.720000 | 0.680000 | 0.360000 | 4.000000 | 7.142857 | 13.562259 | 6.211180 |

| bank | target | query_best_policy_macro_class | zero_macro_class | best_gain_pp |
| --- | --- | --- | --- | --- |
| A | clean | 40.723160 | 34.138690 | 6.584470 |
| A | brightness_dark | 35.712121 | 17.546026 | 18.166095 |
| A | contrast_low | 33.993166 | 19.955264 | 14.037902 |
| A | gaussian_noise | 37.825842 | 30.676644 | 7.149199 |
| A | gaussian_blur | 40.343614 | 30.901579 | 9.442035 |
| B | clean | 40.572038 | 34.138690 | 6.433348 |
| B | brightness_dark | 35.686070 | 17.546026 | 18.140043 |
| B | contrast_low | 33.862340 | 19.955264 | 13.907076 |
| B | gaussian_noise | 37.894827 | 30.676644 | 7.218183 |
| B | gaussian_blur | 40.374218 | 30.901579 | 9.472638 |

True Clean is query-optimal in only 20% /22% of clients, and true Blur in 20% /19%. In the lowest entropy quartile, 40% of Blur clients have true-state regret >5 pp in both banks; median regret is 3.39 pp. Alternative fixed states offer headroom: the descriptive best-of-five policy gains 9.44 /9.47 pp on Blur and 6.58 /6.43 pp on Clean versus zero. For the aggregate query-best table only, ties use fixed candidate order; per-client optimality/regret preserves every tie. This supports state/client mismatch, but does not prove interpolation will solve it. Query-best selection is optimistically biased by maximizing on the same finite query outcomes.

## Clean/Blur low-entropy ledger

All 338 Clean/Blur choices differing from the named true state are preserved in clean_blur_error_ledger.csv, with prior choices, source JS/agreement, true/selected query accuracy and all query-best states. Historically difficult clients:

| client | bank | target | entropy | max_class_fraction | T009 | T010 | T011 | true_query_accuracy | selected_query_accuracy | query_best_states | query_best_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | B | clean | -0.000000 | 1.000000 | gaussian_blur | gaussian_blur | gaussian_noise | 0.000000 | 0.847458 | brightness_dark | 1.271186 |
| 15 | A | gaussian_blur | -0.000000 | 1.000000 | contrast_low | contrast_low | clean | 0.000000 | 0.000000 | brightness_dark | 3.389831 |
| 15 | B | gaussian_blur | -0.000000 | 1.000000 | contrast_low | contrast_low | clean | 0.000000 | 0.000000 | brightness_dark | 3.389831 |
| 29 | A | clean | -0.000000 | 1.000000 | gaussian_noise | gaussian_noise | contrast_low | 75.916230 | 98.691099 | contrast_low | 98.691099 |
| 29 | B | clean | -0.000000 | 1.000000 | gaussian_noise | gaussian_noise | contrast_low | 75.916230 | 99.214660 | contrast_low | 99.214660 |
| 29 | A | gaussian_blur | -0.000000 | 1.000000 | clean | clean | contrast_low | 75.916230 | 93.455497 | contrast_low | 93.455497 |
| 29 | B | gaussian_blur | -0.000000 | 1.000000 | clean | clean | contrast_low | 76.178010 | 95.026178 | contrast_low | 95.026178 |

Context misidentification and task harm are distinct here: some non-true states improve clean accuracy, whereas source-consistency ranking badly sacrifices Dark/Noise. The failure is not confined to clients 15/29 or to low-entropy supports.

## OOD descriptive result

[
  {
    "bank": "A",
    "selected_histogram": {
      "clean": 0,
      "brightness_dark": 0,
      "contrast_low": 4,
      "gaussian_noise": 0,
      "gaussian_blur": 0
    },
    "fraction_zero": 0.0
  },
  {
    "bank": "B",
    "selected_histogram": {
      "clean": 0,
      "brightness_dark": 0,
      "contrast_low": 4,
      "gaussian_noise": 0,
      "gaussian_blur": 0
    },
    "fraction_zero": 0.0
  }
]

| bank | batch | selected | C_clean | C_contrast_low | agreement_zero | agreement_selected | entropy_zero | entropy_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | contrast_low | 0.003677 | 0.000049 | 0.900000 | 1.000000 | 1.798570 | 0.025892 |
| A | 1 | contrast_low | 0.003626 | 0.000051 | 0.950000 | 1.000000 | 1.731886 | 0.022732 |
| A | 2 | contrast_low | 0.002016 | 0.000030 | 1.000000 | 1.000000 | 1.732963 | 0.023536 |
| A | 3 | contrast_low | 0.003161 | 0.000039 | 1.000000 | 1.000000 | 1.791828 | 0.026578 |
| B | 0 | contrast_low | 0.004372 | 0.000026 | 1.000000 | 1.000000 | 1.677822 | 0.011669 |
| B | 1 | contrast_low | 0.002811 | 0.000013 | 1.000000 | 1.000000 | 1.671253 | 0.010329 |
| B | 2 | contrast_low | 0.003061 | 0.000021 | 0.950000 | 1.000000 | 1.769335 | 0.011597 |
| B | 3 | contrast_low | 0.004399 | 0.000028 | 0.950000 | 1.000000 | 1.710036 | 0.012406 |

All 8/8 random-noise supports choose Contrast, with selected flip agreement 1.0 and very low mean entropy (about 0.010–0.027 nats), versus zero entropy about 1.67–1.80. The score accepts highly confident and flip-stable predictions on pure noise. Prediction consistency is not an OOD detector. No threshold or OOD success claim.

## Mechanism vs implementation conclusion

Regression, immutability, freezes and complete prediction/count reconstruction pass. The fixed horizontal-flip consistency criterion fails as a per-client utility selector; stability can reward predictions unrelated to task correctness. At the same time, the named true-context state often is not the client-optimal fixed state. These are two separate mechanism findings. The utility matrix shows substantial headroom, so this run does not establish insufficient affine capacity.

## Case decision

**Primary P-C, with failed PC-safe ranking as a secondary finding. PC-ALIGN FAIL and PC-RET-A FAIL.** P-A/P-B conditions do not hold. P-D is not supported because best-of-five headroom remains substantial. P-C is a descriptive diagnosis from the regret matrix, not a claim that a new method has succeeded.

## Recommended next action

Return the full utility matrix to Research Lead. Preserve the operator and state bank. A subsequent bounded package may test client-conditioned state amplitude/interpolation or another task-proximal criterion, but this flip-consistency score should not be promoted into a TTT loss. No next-stage work, learned writer, SSL optimization, federation or richer operator was started.
