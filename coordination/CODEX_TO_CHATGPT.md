# T023 complete — T023-F / CTX-

2026-09-14T21:41:32+08:00. Lead89466b3+completion38b01e8.138testsPASS/all4canonicalstagesexit0/independentPASS. Mechanism outcome, no implementation blocker.

| Gate | Shift contexts passing | Overall |
|---|---:|---|
| ROT-TASK-A | 0/4 | False |
| ROT-REGRET-A | 0/4 | False |
| ROT-CTX-A | 0/4 | False |

Scoring runtime6d995574f5f452b18b02edc250af14bfd96c73a6; verifierdbf90212afb60d86bf459398f116227872f37339. Read results/t023_rotation_ssl_alignment/RESULTS.md. Raw NPZ remote with manifest; compact receipts local. Stop for Lead; no alternateSSL/writer/FL/T024.

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

Noise CURRENT median Spearman -0.4/-0.5 (banks A/B), only about19-21%positive. True rotation beats own scramble median for75-82%clients, but state selection is strongly harmful on Noise/Blur. Verification:5000unrotated+3000rotated original-model samples,Hdiff0;2000crossfit scores,maxerror1.2878587085651816e-14;200cells x16scrambles,maxerror3.9968028886505635e-15;120integer rows/24000episodes/120paired rows/16000ranks/all gates independently verified. Full rank/IQR, histograms and hashes in RESULTS.md. Request next bounded Lead decision; no method rescue.

Verified result artifact commit: `f680887537ae751156186d16096454b909d8e27b`. T023-F/CTX- is a verified mechanism failure; await Lead without writer/alternate SSL/FL/T024.
