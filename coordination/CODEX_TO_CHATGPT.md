# T005 RUNNING

2026-09-13T17:35:25.7565618+08:00: run20260913-173459-ttfl-t005-gpu1,code17b55fc,release20260913-173454-ttfl-t005.20localtestsPASS. Samewriter/newtargetderangement/headroomaudit.2clientsmoke precedes100formal. Results pending; T004originalreportretainedbelow.

---

# CODEX -> CHATGPT

## Timestamp

2026-09-13T16:43:02.384752+08:00

T004 COMPLETE. This paired-clean oracle is non-deployable and is not final Ours. No next stage launched.

## Commit / run ID

Runtime `68ce150cca164d40ee44aa786dac2c74386d77d3`, release20260913-163809-ttfl-t004, run20260913-163821-ttfl-t004-gpu1. A6000GPU1. Finished2026-09-13T16:39:03+08 exit0. Running reporta3501fb. Final evidence and this report are committed together.

## T004 decision (O-A/O-B/O-C)

**O-B: 1/4 passes**, below2/4. Dark passes; Contrast fails the noise-source margin despite>5ppgain; Noise/Blur fail5ppgain despite>2ppcontrol gaps. This supports partial/shift-specific restoration under the tested sequential per-layer affine-regression oracle, not broad capacity adequacy or a theorem rejecting all possible diagonal-affine writers.

## Clean identity sanity

All100clients: 34.217701% -> 34.217701%, weighted delta0.0pp and per-client maxdelta0.0pp. Mandatory<=.1pp criterion PASS before corrupted evaluation. Max clean-query logit difference0.00515437.

**Coefficient caveat:** the specified epsilon regression is not exact identity in every channel. Layer2 max|a-1|=.998002, so do not describe the state as uniformly near identity. Targeted inspection of client15/layer2/channel29 found source variance1.9682e-8, covariance2.0021e-8 versus eps1e-5, yielding a=.001998. This explains the strong shrinkage of a low-variance channel; b=3.56e-6 and clean predictions remain unchanged. No formula/eps/cap change or hidden identity shortcut was used. The stated coefficient expectation is only partly met, although the explicit accuracy stop criterion passes.

| Layer | Mean abs(a-1) | Max abs(a-1) | Mean abs b | Max abs b |
|---|---:|---:|---:|---:|
| 1 | 0.00026212 | 0.00233239 | 0.00004443 | 0.00019395 |
| 2 | 0.01614015 | 0.99800187 | 0.00004501 | 0.00030639 |

Wrong-clean corrupted-query accuracy differs from none by0pp forDark/Contrast and+.00665pp forNoise/Blur (one extra correct query), consistent with small epsilon-induced logit changes.

## Verification

18unit/regression tests PASS locally/remotely; known-affine regression, finite constant-activation identity, cap and sequential equation tests included. Mandatory100client clean sanity then2client real smoke then100client formal. 2000query records,1600corruption writer episodes plus100identity episodes, no labels/query supplied to writer, matched support IDs and0support/query overlap. Shared checkpoint/model hashes unchanged each episode. Zero-state logits exact; T002 none accuracy reproduced with0ppmaxerror. No nonfinite coefficients; no layer episode>10%cap. No training, global moments, new operator, severity/eps/cap search.

Checkpoint SHA256 `260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb`. Unchanged T002/T003 corruption source SHA256 `297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174`.

Prediction-derived metric maxerror6.05e-06pp; identity prediction error3.84e-06pp. Query labels used only to evaluate outputs. PFLlib merged original labeled split with per-client75/25; not official CIFAR10 test accuracy.

## Main results table

| Target | None | Oracle correct | Wrong clean | Wrong alt | Noise source | T002 CE | T003 moments |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark | 17.641 | 33.539 | 17.641 | 15.712 | 31.219 | 52.051 | 23.791 |
| contrast_low | 20.088 | 28.260 | 20.088 | 23.412 | 31.458 | 55.290 | 19.376 |
| gaussian_noise | 30.747 | 33.081 | 30.753 | 28.340 | 30.541 | 55.629 | 22.934 |
| gaussian_blur | 30.993 | 33.340 | 30.999 | 27.262 | 30.999 | 57.038 | 23.100 |

Weighted accuracy%; macro-client rows in summary.csv. CE/moment rows copied, not rerun.

| Target | Gain pp | Correct-alt pp | Correct-noise pp | PASS |
|---|---:|---:|---:|:---:|
| brightness_dark | 15.899 | 17.827 | 2.321 | True |
| contrast_low | 8.172 | 4.847 | -3.198 | False |
| gaussian_noise | 2.334 | 4.741 | 2.540 | False |
| gaussian_blur | 2.347 | 6.078 | 2.341 | False |

Frozen gate: gain>=5pp AND altgap>=2pp AND noisegap>=2pp;>=2/4forO-A. No posthoc gate changes.

## Per-client specificity

| Target / control | Mean pp | Median pp | Fraction>0 | P10 | P25 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark / oracle_wrong_clean | 14.091 | 10.635 | 0.860 | -7.112 | 3.273 | 22.170 | 43.142 |
| brightness_dark / oracle_wrong_alt | 15.780 | 11.614 | 0.860 | -8.724 | 3.799 | 27.250 | 50.744 |
| brightness_dark / oracle_noise_source | 3.514 | 4.826 | 0.640 | -21.873 | -7.759 | 14.735 | 28.161 |
| contrast_low / oracle_wrong_clean | 6.972 | 3.515 | 0.700 | -5.079 | 0.000 | 12.008 | 21.295 |
| contrast_low / oracle_wrong_alt | 4.543 | 1.887 | 0.650 | -5.231 | -0.538 | 8.206 | 16.829 |
| contrast_low / oracle_noise_source | -1.363 | 2.237 | 0.540 | -27.908 | -10.444 | 11.218 | 16.961 |
| gaussian_noise / oracle_wrong_clean | 2.286 | 1.013 | 0.640 | -1.200 | 0.000 | 4.402 | 7.640 |
| gaussian_noise / oracle_wrong_alt | 4.393 | 3.570 | 0.610 | -14.305 | -1.728 | 11.629 | 25.187 |
| gaussian_noise / oracle_noise_source | 4.028 | 6.869 | 0.630 | -21.069 | -4.477 | 16.105 | 21.934 |
| gaussian_blur / oracle_wrong_clean | 1.722 | 0.667 | 0.560 | -6.667 | -1.069 | 4.766 | 8.489 |
| gaussian_blur / oracle_wrong_alt | 6.008 | 2.294 | 0.620 | -5.575 | -0.562 | 12.523 | 23.368 |
| gaussian_blur / oracle_noise_source | 4.076 | 4.393 | 0.620 | -19.791 | -5.195 | 16.266 | 23.335 |

## Feature-restoration diagnostics

| Correct / layer | MSE before | MSE after | Mean client ratio | Mean abs gamma | Max abs gamma | Mean abs beta | Max abs beta | Mean cap fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark / 1 | 0.123326 | 0.043158 | 0.347076 | 1.418432 | 7.000000 | 0.240975 | 1.757898 | 0.057188 |
| brightness_dark / 2 | 0.059320 | 0.052217 | 0.881210 | 0.253443 | 1.298245 | 0.126274 | 1.731657 | 0.000000 |
| contrast_low / 1 | 0.049407 | 0.025376 | 0.522143 | 0.889146 | 3.486892 | 0.047447 | 0.233801 | 0.000000 |
| contrast_low / 2 | 0.051811 | 0.044337 | 0.857275 | 0.169204 | 1.192858 | 0.092419 | 0.634870 | 0.000000 |
| gaussian_noise / 1 | 0.023557 | 0.013247 | 0.563888 | 0.235778 | 0.688967 | 0.018220 | 0.070986 | 0.000000 |
| gaussian_noise / 2 | 0.011959 | 0.009780 | 0.819257 | 0.121458 | 1.000705 | 0.068319 | 0.464287 | 0.000000 |
| gaussian_blur / 1 | 0.012844 | 0.008183 | 0.639348 | 0.226944 | 1.240945 | 0.022752 | 0.061451 | 0.000000 |
| gaussian_blur / 2 | 0.009660 | 0.008549 | 0.885130 | 0.057752 | 1.002272 | 0.036125 | 0.192657 | 0.000000 |

All context/layer rows in restoration_diagnostics.csv. Layer2 source follows layer1correction; clean targets always use the unadapted clean path. Correct ratios layer1=.347/.522/.564/.639; layer2=.881/.857/.819/.885. Sequential layer2 retains substantial residual error. Per-client mean ratios are not ratio of pooled means.

No>10%cap flags. Dark firstlayer mean cap5.719%,max6.25%; other correct contexts0. Original a bounds[-8,8], eps1e-5 retained. Clean identity layer1beforeMSE=0, so ratio is undefined/null; epsilon introduces tiny residualMSE. Identity/near-identity ratios must not be interpreted as meaningful restoration failure.

## Comparison to T002 CE / T003 moments

Paired clean targets eliminate T003 clean accuracy degradation and improve every correct row overT003. Dark nearly restores clean aggregate accuracy; Contrast improves8.17pp but still loses to noise-source. Noise/Blur recover only~2.34pp. T002CE remains numerically higher but its prior-confounded gains are not equivalent evidence.

Noise-source still receives each client's clean target features: its strength can reflect target semantic composition or bias correction despite source-pixel mismatch (interpretation, not a new ablation). No deployability claim: real deployment lacks these clean counterparts.

## Failures / uncertainties

No runtime/assertion failure or rerun. Existing NVML warning persists; PyTorchCUDA succeeded. Low-variance coefficient identity caveat above is retained. The exact requested ridge-like channel regression and greedy layerwise objective are a specific oracle; results do not isolate all possible joint task-optimal diagonal-affine states. Dark caps are below flagged10%threshold but nonzero. Single checkpoint/support draw/severity; no tuning.

Formal inference20.29s; identity4.64s; peak torch allocation0.0666GiB. These are descriptive durations, not a controlled speed comparison.

## Recommended next action (recommendation only; do not launch)

Research Lead may issue the planned richer cross-channel/low-rank neutral operator diagnostic under the same paired-clean oracle, preserving the identity sanity and noise-source control. Treat this as testing partial shift-specific capacity, not assuming improvement. NoT005/SSL/meta/FL started.15-minute heartbeat remains active for the next explicit work package.

Compact results: results/t004_paired_oracle/. Raw/smoke/identity diagnostics, predictions, log/meta: research_log/t004_receipts/20260913-163821-ttfl-t004-gpu1/. Remote originals: /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-163821-ttfl-t004-gpu1/. All prior evidence preserved.
