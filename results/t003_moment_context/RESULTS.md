# T003 ¡ª Moment-Written Fast Context Operator

**Case M-B; 1/4 PASS, below the required2/4.** Dark passes all four frozen2pp tests. Contrast beats both wrong contexts and noise but fails to improve no-adaptation accuracy. Noise/blur lose accuracy; this is partial moment-context signal, not an overall success.

**Clean sanity warning:** clean-support calibration changes clean-query accuracy 34.218% -> 24.131% (-10.087pp), with 77/100 clients worse. Both clean layers have zero scale clamping. This reference/writer can harm clean client predictions even when support moments are matched closely. Affine capacity alone is not identified as the cause.

## Main results

| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark | 17.641 | 23.791 | 11.749 | 11.104 | 13.152 | 52.051 | 50.409 |
| contrast_low | 20.088 | 19.376 | 13.598 | 15.200 | 15.533 | 55.290 | 57.125 |
| gaussian_noise | 30.747 | 22.934 | 19.543 | 23.918 | 36.804 | 55.629 | 56.606 |
| gaussian_blur | 30.993 | 23.100 | 21.338 | 21.065 | 31.751 | 57.038 | 60.297 |

Accuracies are sample-weighted percentages; macro-client accuracy is in summary.csv. CE rows are copied from T002, not recomputed.

| Target | Gain pp | Correct-clean pp | Correct-alt pp | Correct-noise pp | PASS |
|---|---:|---:|---:|---:|:---:|
| brightness_dark | 6.151 | 12.042 | 12.687 | 10.639 | True |
| contrast_low | -0.711 | 5.778 | 4.176 | 3.843 | False |
| gaussian_noise | -7.813 | 3.391 | -0.984 | -13.871 | False |
| gaussian_blur | -7.893 | 1.762 | 2.035 | -8.651 | False |

Frozen gate: every gain/gap above>=2pp; at least2/4targets required. No query tuning or new operator. M-A is arithmetic; the M-B interpretation is post-run descriptive adjudication, recorded separately from immutable raw outputs.

## Per-client specificity

| Target / wrong | Mean | Median | Fraction>0 | P10 | P25 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark / clean | 10.501 | 9.789 | 0.850 | -9.548 | 3.061 | 22.601 | 33.145 |
| brightness_dark / alt | 10.829 | 11.593 | 0.840 | -10.544 | 2.778 | 26.241 | 36.151 |
| contrast_low / clean | 5.358 | 3.828 | 0.680 | -9.895 | 0.000 | 12.442 | 21.765 |
| contrast_low / alt | 3.960 | 3.180 | 0.650 | -9.428 | 0.000 | 8.533 | 18.332 |
| gaussian_noise / clean | 3.380 | 2.691 | 0.700 | -0.856 | 0.000 | 5.471 | 9.836 |
| gaussian_noise / alt | -1.255 | -3.339 | 0.370 | -17.335 | -9.845 | 5.367 | 19.500 |
| gaussian_blur / clean | 1.235 | 0.534 | 0.510 | -6.292 | -1.111 | 3.237 | 8.813 |
| gaussian_blur / alt | 2.319 | 1.246 | 0.530 | -10.686 | -1.997 | 9.027 | 18.206 |

## Moment and clamp diagnostics

**Clamp warning:** Dark support layer1 hits scale=4 in5/32channels (15.625%) for all100clients, appearing in Dark/correct and Noise/wrong-alt. Contrast support layer1 exceeds10%clamping in2clients, appearing in Contrast/correct and Blur/wrong-alt. Total204/3400layer episodes exceed10%. Bounds remain fixed. These are repeated conditions, not204distinct clients.

Before/after mismatch is mean_channel(|mu-mu_global|+|sigma-sigma_global|), measured on actual sequential calibrated support activations. All3400layer episodes decrease; max change is negative. Matching low-order moments is not sufficient to preserve task accuracy.

| Correct support / layer | Mean abs gamma | Max abs gamma | Mean abs beta | Max abs beta | Clamp fraction | Mismatch before | Mismatch after |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_sanity / 1 | 0.084384 | 0.779420 | 0.024189 | 0.337998 | 0.000000 | 0.056734 | 0.000012 |
| clean_sanity / 2 | 0.066799 | 0.690344 | 0.039924 | 0.486526 | 0.000000 | 0.043824 | 0.000026 |
| brightness_dark / 1 | 1.453333 | 3.000000 | 0.264907 | 2.201978 | 0.156250 | 0.377267 | 0.073256 |
| brightness_dark / 2 | 0.463945 | 3.000000 | 0.264210 | 4.566708 | 0.000156 | 0.136769 | 0.000077 |
| contrast_low / 1 | 1.183175 | 3.000000 | 0.061121 | 0.497909 | 0.039688 | 0.237483 | 0.000847 |
| contrast_low / 2 | 0.148788 | 1.394431 | 0.069503 | 0.702889 | 0.000000 | 0.085681 | 0.000046 |
| gaussian_noise / 1 | 0.119156 | 0.570132 | 0.056865 | 0.393036 | 0.000000 | 0.118382 | 0.000007 |
| gaussian_noise / 2 | 0.109787 | 0.574501 | 0.083793 | 0.900591 | 0.000000 | 0.058141 | 0.000026 |
| gaussian_blur / 1 | 0.400665 | 2.892506 | 0.028315 | 0.326472 | 0.000000 | 0.120109 | 0.000045 |
| gaussian_blur / 2 | 0.078754 | 0.745820 | 0.043846 | 0.398624 | 0.000000 | 0.052146 | 0.000051 |

All layer/context diagnostic summaries are in moment_diagnostics.csv; raw per-channel moments are preserved in the run receipts.

## Reference and verification

Reference uses all 44961 training images across100clients, no cap, no query IDs and no labels forwarded. Collection order is client0..99/stored sample order; all sample/spatial activations weighted equally. Double-precision first/second sums, variance clamp>=0, sigma=sqrt(var+1e-5). Sequential writer uses sigma_g/(sigma_s+1e-5), scale clamped[.25,4], and calibrated layer1 before collecting layer2.

Checkpoint SHA256: `260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb`.

Reference SHA256: `bc46da4b4e958880df3ee440e466d27b5ee35ce0523ad8577f94c0aed7788acb`. reference_moments.json contains source IDs, activation counts and numeric reference vectors.

15tests passed locally and onA6000; real2-client smoke preceded100-client formal evaluation. 2000query records,1700writer episodes including100clean-sanity episodes. Model hashes unchanged, zero-state logit difference0, T002 no-adapt reproduction error0pp, no support/query or reference/query overlap. Writer accepts no labels and uses no optimizer/gradients. Corruption source SHA identical across deployedT002/T003.

```json
{
  "records": 2000,
  "writer_episodes": 1700,
  "reference_training_only": true,
  "reference_query_overlap": 0,
  "support_query_overlap": 0,
  "matched_support_ids": true,
  "label_free_writer_api": true,
  "shared_model_unchanged_each_episode": true,
  "neutral_max_abs_diff": 0.0,
  "t002_baseline_max_error_pp": 0.0,
  "prediction_max_error_pp": 5.796414995984378e-06,
  "moment_mismatch_increases_gt1e_minus5": 0,
  "moment_mismatch_max_increase": -0.014726875834655383,
  "layer_episodes_clamp_gt10pct": 204,
  "clean_sanity_prediction_max_error_pp": 3.8359424863188e-06
}
```

## Interpretation and next action

T003 removes label inputs from the writer but image moments can still reflect client class composition. The clean sanity degradation makes global-reference mismatch under label skew a plausible competing explanation (inference, not isolated causal evidence). Dark demonstrates useful correction with the same192scalars; broad sufficiency of those scalars or of moments is unproved. Do not infer that a larger operator alone will fix this.

Stop here for Research Lead review. Recommend explicitly accounting for the clean-reference harm when designing the next richer-operator/oracle diagnostic; no next-stage implementation was launched. No SSL, meta-learning, retraining, severity or clamp sweep.

Runtime code `7bc31ded5e2a0dc6ad7e20b7688240c34c24c366`; run `20260913-153752-ttfl-t003-gpu0`; reference collection 3.33s; formal evaluation 54.64s on RTX A6000. Two SSH connection timeouts during monitoring recovered; same job finished normally at2026-09-13T15:39:10+08, exit0. No rerun. PFLlib merged-split/client75:25 diagnostic, not an official CIFAR10 test benchmark.