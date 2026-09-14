# T022 — T022-N

2026-09-14T19:53:07+08:00. Lead `06647a0`. All five stages exit0; final full suite **133 tests PASS**. Independent verification **PASS**.

## Frozen decision

| Response | MATCH contexts | REAL contexts | Regression safety | REGRET vs P | CTX | SRC |
|---|---:|---:|---|---:|---:|---|
| Phi_L | 1/4 | 1/4 | False | 0/4 | 4/4 | NOT_EXECUTED |
| Phi_H | 1/4 | 1/4 | False | 0/4 | 4/4 | NOT_EXECUTED |


Flags per response: {"L": {"CTX": "CTX+", "SRC": "NOT_EXECUTED"}, "H": {"CTX": "CTX+", "SRC": "NOT_EXECUTED"}}. The diagnosis uses the prespecified preference for response logits before response H. For M/N there is no arbitrary winning representation: per-representation CTX flags remain explicit. Source choices were frozen for both representations, but source scoring is authorized only when RESP-REAL-A passes.

The frozen first-moment observer branch stops here under the predeclared T022-M/N rule. No further observer trick, rescue or next task was started.

Beyond Dark, response matched capture falls to roughly67–76%, while real capture is about50–67%. Both absolute T021 observers had passed the matched gate, so response failure is already visible under its own matched class-conditional model. Removing the state-invariant component did not preserve enough useful signal for this specific raw first-moment CLS procedure. This is not proof that every possible response observer lacks semantic information. CTX+ establishes an advantage over clean prototypes; it does not offset failed matched/real criteria.

## Observation and controls

Phi_L is40-D and Phi_H is2048-D. Each concatenates raw float32 differences in fixed order Dark−clean, Contrast−clean, Noise−clean, Blur−clean for the same sample and bank. There is no normalization, scaling, projection, whitening, ridge, learned metric, gradient update, new state/operator or federation. The target's exact same signature mean is used for oracle, source and clean-prototype policies; only the prototype context and predeclared utility context differ. Source uses the frozen T009 decision. Clean-prototype control uses clean other-client signatures and target-context utility. Offline other-client labels calibrate M; target-i contributes zero samples or labels to its own M/estimator.

The checkpoint, banks, K20 support IDs, deterministic corruptions, target-excluded T014 utility templates, solver tolerances and canonical exact tie order are unchanged. Phase-A froze6000 estimates and48000 choices before composition/query scoring. Matched draws preserve exact target class counts, use only the other99clients, and use full big-endian SHA256(`T022|matched|L-or-H|client|bank|context|replica`) into PCG64; actual seed tokens are `L` and `H`. R=128 is fixed, with256000 solutions and2048000 matched choices.

## Capture against frozen baselines

Ranges span both banks/all four salts. Matched values are medians over128 exact-count replicas. P and T021 absolute L/H are historical baselines, not retrained or refitted. Their real integer numerators are reproduced exactly before comparison.

| Context/observer | Matched capture %, min to max | Real capture %, min to max |
|---|---:|---:|
| brightness_dark/P | 90.783 to 92.351 | 89.138 to 91.783 |
| brightness_dark/T021_L | 90.541 to 91.936 | 89.289 to 90.281 |
| brightness_dark/T021_H | 91.895 to 93.506 | 87.110 to 89.612 |
| brightness_dark/L | 90.309 to 92.036 | 90.314 to 92.408 |
| brightness_dark/H | 92.283 to 93.657 | 88.274 to 90.617 |
| contrast_low/P | 83.432 to 85.872 | 75.729 to 83.422 |
| contrast_low/T021_L | 82.346 to 83.775 | 72.333 to 76.391 |
| contrast_low/T021_H | 83.060 to 84.545 | 68.940 to 79.791 |
| contrast_low/L | 71.090 to 73.995 | 54.232 to 59.649 |
| contrast_low/H | 73.658 to 76.435 | 57.323 to 66.641 |
| gaussian_noise/P | 84.478 to 85.748 | 78.334 to 81.889 |
| gaussian_noise/T021_L | 80.402 to 82.363 | 71.239 to 76.240 |
| gaussian_noise/T021_H | 82.256 to 83.527 | 70.227 to 72.586 |
| gaussian_noise/L | 67.495 to 68.825 | 52.805 to 58.516 |
| gaussian_noise/H | 69.167 to 72.284 | 49.517 to 53.265 |
| gaussian_blur/P | 82.684 to 84.908 | 74.911 to 76.690 |
| gaussian_blur/T021_L | 79.010 to 81.961 | 68.244 to 74.730 |
| gaussian_blur/T021_H | 80.809 to 82.533 | 72.997 to 75.658 |
| gaussian_blur/L | 69.528 to 69.861 | 60.387 to 61.767 |
| gaussian_blur/H | 70.991 to 73.079 | 51.651 to 54.278 |


RESP-MATCH-A requires matched median capture>=80% in every bank/salt for at least3/4shifts. RESP-REAL-A requires real capture>=80% under the same aggregation and every shifted row>=BBSE-S-01−0.5pp. The source clean safety rule remains>=zero−0.5pp. These are the original T018/T021 expressions. Opposite-half integer query counts and P00 denominators are unchanged. No query forward is added.

## Paired regret and context specificity

| Response/context/comparison | Mean regret reduction % | New p90 regret | Baseline p90 regret | Clients better /100 |
|---|---:|---:|---:|---:|
| L/brightness_dark/vs_P | 9.069 to 19.347 | 0.049 to 0.077 | 0.048 to 0.081 | 15.000 to 17.000 |
| L/brightness_dark/vs_T021_absolute | 3.780 to 17.166 | 0.049 to 0.077 | 0.053 to 0.077 | 2.000 to 5.000 |
| L/brightness_dark/context_vs_clean_prototype | 75.504 to 76.866 | 0.049 to 0.077 | 0.240 to 0.275 | 24.000 to 28.000 |
| L/contrast_low/vs_P | -135.754 to -88.004 | 0.233 to 0.252 | 0.082 to 0.100 | 11.000 to 17.000 |
| L/contrast_low/vs_T021_absolute | -111.672 to -70.174 | 0.233 to 0.252 | 0.092 to 0.121 | 11.000 to 17.000 |
| L/contrast_low/context_vs_clean_prototype | 33.919 to 42.967 | 0.233 to 0.252 | 0.264 to 0.306 | 45.000 to 50.000 |
| L/gaussian_noise/vs_P | -145.244 to -98.655 | 0.087 to 0.097 | 0.038 to 0.057 | 11.000 to 17.000 |
| L/gaussian_noise/vs_T021_absolute | -70.834 to -43.629 | 0.087 to 0.097 | 0.069 to 0.077 | 13.000 to 15.000 |
| L/gaussian_noise/context_vs_clean_prototype | 24.920 to 37.784 | 0.087 to 0.097 | 0.151 to 0.179 | 27.000 to 30.000 |
| L/gaussian_blur/vs_P | -123.735 to -109.926 | 0.112 to 0.127 | 0.058 to 0.066 | 19.000 to 22.000 |
| L/gaussian_blur/vs_T021_absolute | -61.055 to -41.878 | 0.112 to 0.127 | 0.086 to 0.094 | 13.000 to 16.000 |
| L/gaussian_blur/context_vs_clean_prototype | 28.852 to 34.851 | 0.112 to 0.127 | 0.164 to 0.179 | 36.000 to 42.000 |
| H/brightness_dark/vs_P | -9.610 to -0.153 | 0.058 to 0.081 | 0.048 to 0.081 | 12.000 to 13.000 |
| H/brightness_dark/vs_T021_absolute | -0.069 to 4.402 | 0.058 to 0.081 | 0.053 to 0.080 | 3.000 to 6.000 |
| H/brightness_dark/context_vs_clean_prototype | 44.416 to 51.970 | 0.058 to 0.081 | 0.147 to 0.154 | 13.000 to 14.000 |
| H/contrast_low/vs_P | -110.730 to -79.569 | 0.234 to 0.257 | 0.082 to 0.100 | 14.000 to 18.000 |
| H/contrast_low/vs_T021_absolute | -52.335 to -16.196 | 0.234 to 0.257 | 0.133 to 0.181 | 14.000 to 19.000 |
| H/contrast_low/context_vs_clean_prototype | 48.949 to 58.464 | 0.234 to 0.257 | 0.350 to 0.387 | 51.000 to 57.000 |
| H/gaussian_noise/vs_P | -142.603 to -100.980 | 0.088 to 0.095 | 0.038 to 0.057 | 15.000 to 17.000 |
| H/gaussian_noise/vs_T021_absolute | -68.393 to -49.601 | 0.088 to 0.095 | 0.074 to 0.083 | 7.000 to 9.000 |
| H/gaussian_noise/context_vs_clean_prototype | 28.037 to 41.925 | 0.088 to 0.095 | 0.158 to 0.192 | 24.000 to 27.000 |
| H/gaussian_blur/vs_P | -140.059 to -110.485 | 0.145 to 0.156 | 0.058 to 0.066 | 18.000 to 21.000 |
| H/gaussian_blur/vs_T021_absolute | -101.220 to -65.654 | 0.145 to 0.156 | 0.072 to 0.113 | 9.000 to 13.000 |
| H/gaussian_blur/context_vs_clean_prototype | 37.772 to 41.935 | 0.145 to 0.156 | 0.189 to 0.206 | 39.000 to 45.000 |


Positive reduction means less true-template regret. RESP-REGRET-A vs P and RESP-CTX-A vs clean-prototype require>=15% mean reduction with p90 no more than5% worse in both banks/all salts for>=3/4shifts. A zero baseline mean is noneligible for relative improvement. The T021 absolute comparison is diagnostic only, never an additional gate. Client better/equal/worse counts aggregate the two held-out halves per client. Complete mean/p90, exact regret, L1, optimal/canonical agreement and paired counts are retained in CSV/gzip receipts. Salts and halves reuse support and are not independent sample replications.

## Numerical and independent verification

Extraction reused1807 historical paths and computed3193 missing A6000 paths; all5000 logical state paths/100000 sample outputs are available through old+new artifacts. Model/state hashes are unchanged and support/calibration/query IDs are disjoint. Independent reconstruction checks40000 complete response vectors directly from original cached and newly frozen state outputs, with exact arithmetic equality; checks72280 cached sample arrays by their frozen hashes. This is a cache/state-output audit, not a new model-forward replay. T021's independent original-model replay remains unchanged upstream evidence.

Independent checks: 280 prototype cells, 6000 actual CLS replays, 6000 target means, 48000 actual choice/mask checks, 4000 matched CLS replays, 32000 sampled matched choice checks, all256000 draw count/exclusion checks, 3 exhaustive fallback re-enumerations, 56000 exact regrets, 280 actual integer rows and 10240 matched integer rows. Every gate and paired count is reconstructed. Maximum replay pi difference=0.0. All input hashes and953 local upstream manifest entries pass.

Prototype numerical diagnostics (rank/condition are descriptive only): `{"L": {"rank_range": [10, 10], "condition_range": [190.43873830140333, 775.8188019538923], "actual_max_KKT": 1.1191048088221578e-13, "matched_max_KKT": 1.5010215292932116e-13}, "H": {"rank_range": [10, 10], "condition_range": [119.42170824202283, 224.5490164074652], "actual_max_KKT": 5.4001247917767614e-12, "matched_max_KKT": 8.15347789284715e-12}}`. The solver forms only a10×10 Gram matrix. No2048×2048 covariance is constructed. Existing cycle-only exhaustive-face handling is unchanged; no new tolerance or fallback rule was introduced.

## Runtime and artifacts

The first scoring launch (`20260914-194816-ttfl-t022-score`) carried an accidental placeholder runtime string. It was retained and superseded by a scoring-only rerun with the actual deployed commit. All compact scientific outputs are byte-identical; `scoring_receipt_correction.json` records the comparison hashes. Metadata/runtime duration differs. No extraction or matched draws were rerun.

- Extraction `5abcf355f38022fe57e09f058c8b133e3f79859b` / `20260914-193443-ttfl-t022-extract-gpu1`.
- Phase-A `4cc03ffe71b1e70f8fb2cb94328407f6ff9f3796` / `20260914-194148-ttfl-t022-phase-a`.
- Matched `d114147734cfbdc354c112c1403d3218af825561` / `20260914-194335-ttfl-t022-matched`.
- Scoring `c5ba764761561e953680519dbdc27a62c5b915c1` / `20260914-194845-ttfl-t022-score-receipt`.
- Verification `9caac731bd42a9e8e6968f7a5204337f0997756d` / `20260914-194946-ttfl-t022-verify`.

Compact stage receipts are under extraction/, phase_a/, matched/ and the result root; train logs and raw compact receipts are in research_log/t022_receipts. Large NPZ arrays remain in canonical remote project runs with full paths, byte sizes and SHA256 in remote_artifact_manifest.json. This report does not claim local copies of those arrays. The inherited PFLlib CIFAR10 split has44961train/15039query samples after merged-source client partitioning,100clients and the historical10% participation checkpoint. It is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining occurred.
