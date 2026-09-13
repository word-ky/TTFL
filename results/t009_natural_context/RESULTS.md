# CODEX -> CHATGPT

## Timestamp

2026-09-13T22:54:20+08:00

## Commit / run ID

Lead 25e6ea7; actual runtime commit fab17bd; run 20260913-224708-ttfl-t009-gpu1; release 20260913-224704-ttfl-t009-importfix; NVIDIA RTX A6000 GPU1; 11.41 seconds. The launch metadata string IMPORT_FIX was a label, not a commit SHA. t009_runtime_revision.json maps it to fab17bd using the deployed evaluator SHA, matched against committed source after CRLF/LF normalization. Raw metadata was not silently rewritten. The enclosing delivery commit contains analysis and report.

## Frozen T008/T007R evidence reused

Checkpoint, 192-scalar operator, neutral writer, source-signature code, T008 prototypes, T007R pool membership/state tensors, corruptions and PFLlib splits are unchanged. No prototype rebuild, state fitting, distance change or normalization change. T007R Noise Pool-B remains a failure under its historical gate; T009 has separate gates.

## Natural-support construction and chosen K

Eligible counts range 30–1521. The largest common allowed K is **20**: at least one client has only 30 eligible images, preventing K=32 or64. All 100 clients remain included; no per-client size choices or image balancing.

Before loading image pixels or running model features, selection used only train/query IDs: exclude both calibration pools and all query IDs, shuffle each eligible train-order list with random.Random seeded by SHA256(7:client:T009-natural-support), then take the common prefix. natural_support_manifest.json stores counts, K, original IDs and train indices. The same 20 IDs are used for all five contexts and both banks.

Manifest frozen at 2026-09-13T14:47:18.440609+00:00; all 1,000 decisions frozen at 2026-09-13T14:47:26.696829+00:00, before query metrics or support-label statistics. Support y was accessed for audit only after query evaluation. The PFLlib container also stores labels, but the feature phase accesses only its x field; no label value enters selection/signatures.

## Verification

36 existing tests passed before edits. Two label-blind manifest tests passed locally; the full 40-test suite passed remotely before the successful run. All frozen hashes match, no support/calibration/query ID overlap, no state fits, model hash unchanged throughout, all values finite. Decision and manifest hashes are unchanged after analysis.

985 of 1,000 client/target/bank query policies reuse exact old predictions; 15 new query forwards cover clean clients that selected a corruption state. No secondary-K retrieval was run. All 24 frozen bank/shift/none-oracle-wrong macro-class metrics reconstruct exactly from old counts. Every mixed policy is independently recomputed from saved predictions; maximum macro-class error 7.11e-15 pp. Independent signature-distance error 1.14e-13; all confusions and signed margins match. Fraction arithmetic confirms retrieval and clean-safety gates.

## Bank-A natural identification

488/500 = **97.6%**, NID PASS. Per-context clean/dark/contrast/noise/blur: 92%,100%,100%,100%,96%.

## Bank-B natural identification

490/500 = **98.0%**, NID PASS. Per-context clean/dark/contrast/noise/blur: 93%,100%,100%,100%,97%.

## Per-context confusion / signed margins

Rows true, columns predicted. Signed margin is nearest wrong-context distance minus true-context distance; negative means misclassification. Full per-decision distances and margins are in identification.csv.

Bank A:

| True / predicted | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 92 | 0 | 0 | 1 | 7 |
| brightness_dark | 0 | 100 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 100 | 0 | 0 |
| gaussian_noise | 0 | 0 | 0 | 100 | 0 |
| gaussian_blur | 1 | 0 | 3 | 0 | 96 |

Bank B:

| True / predicted | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 93 | 0 | 0 | 1 | 6 |
| brightness_dark | 0 | 100 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 100 | 0 | 0 |
| gaussian_noise | 0 | 0 | 0 | 100 | 0 |
| gaussian_blur | 1 | 0 | 2 | 0 | 97 |

| Bank | Mean signed margin | Median signed margin | Nearest p50 | p90 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- |
| A | 53.804278 | 21.283314 | 6.212052 | 16.670310 | 23.395590 | 42.404145 |
| B | 56.445602 | 22.127916 | 5.942520 | 16.382872 | 25.292304 | 48.220022 |

## Per-client mixed-state retrieval

| Bank | Shift | None % | Oracle % | Mixed % | Wrong alt % | tau pp | Retained | Mixed-wrong pp | PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 32.296536 | 32.296536 | 14.903865 | 4.148166 | 1.000000 | 17.392670 | True |
| A | contrast_low | 19.955264 | 28.561473 | 28.561473 | 22.733171 | 3.545856 | 1.000000 | 5.828302 | True |
| A | gaussian_noise | 30.676644 | 32.499558 | 32.499558 | 30.382434 | 0.865512 | 1.000000 | 2.117123 | True |
| A | gaussian_blur | 30.901579 | 33.048500 | 31.637967 | 29.798137 | 0.809278 | 0.342997 | 1.839830 | False |
| B | brightness_dark | 17.546026 | 32.424808 | 32.424808 | 15.489587 | 4.148166 | 1.000000 | 16.935221 | True |
| B | contrast_low | 19.955264 | 28.111626 | 28.111626 | 22.693556 | 3.545856 | 1.000000 | 5.418069 | True |
| B | gaussian_noise | 30.676644 | 32.406771 | 32.406771 | 30.537680 | 0.865512 | 1.000000 | 1.869091 | True |
| B | gaussian_blur | 30.901579 | 33.009919 | 32.037735 | 28.354235 | 0.809278 | 0.538886 | 3.683500 | False |

Dark, Contrast and Noise retain 100% oracle gain in both banks. Blur retains only **34.30% /53.89%** because 4/3 clients choose the wrong state. Thus 3/4 shifted targets pass jointly, but this alone is insufficient: both clean-safety checks fail.

| Bank | Shift | Mixed weighted % | Mixed macro-client % |
| --- | --- | --- | --- |
| A | brightness_dark | 32.362524 | 31.268559 |
| A | contrast_low | 28.638872 | 27.361031 |
| A | gaussian_noise | 32.575304 | 31.387282 |
| A | gaussian_blur | 31.704236 | 31.229869 |
| B | brightness_dark | 32.488862 | 31.338011 |
| B | contrast_low | 28.186715 | 26.830967 |
| B | gaussian_noise | 32.482213 | 31.267717 |
| B | gaussian_blur | 32.109848 | 31.394028 |

## Clean safety

| Bank | Zero macro-class % | Mixed macro-class % | Delta pp | Required minimum % | PASS |
| --- | --- | --- | --- | --- | --- |
| A | 34.138690 | 33.528369 | -0.610321 | 33.638690 | False |
| B | 34.138690 | 33.522505 | -0.616185 | 33.638690 | False |

**NRET-A fails** because clean macro-class drops 0.610321/0.616185 pp, exceeding the fixed 0.5 pp allowance. Clean decisions contain 8/7 false adaptations: primarily clean→blur, plus one clean→noise in each bank. This is not a failure of zero-state neutrality; it is an incorrect nonzero state choice on clean input.

The following contributions sum exactly to mixed-minus-oracle macro-class accuracy (for clean, oracle means zero state). They weight each changed class count by the global class denominator, not by averaging client percentage changes. Client29 alone selects clean under blur and contributes about -0.77pp in each bank:

| Bank | Client | True context | Selected | Queries | Mixed correct | Oracle correct | Aggregate delta pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 29 | gaussian_blur | clean | 382 | 175 | 291 | -0.775920 |
| A | 29 | gaussian_blur | clean | 382 | 175 | 290 | -0.769231 |
| A | 26 | gaussian_blur | contrast_low | 284 | 147 | 216 | -0.451586 |
| A | 29 | clean | gaussian_noise | 382 | 245 | 290 | -0.301003 |
| B | 29 | clean | gaussian_noise | 382 | 251 | 290 | -0.260870 |
| B | 74 | gaussian_blur | contrast_low | 164 | 88 | 118 | -0.196265 |
| A | 74 | gaussian_blur | contrast_low | 164 | 87 | 116 | -0.189716 |
| B | 26 | clean | gaussian_blur | 284 | 187 | 215 | -0.182946 |
| A | 26 | clean | gaussian_blur | 284 | 192 | 215 | -0.150202 |
| B | 30 | clean | gaussian_blur | 275 | 62 | 81 | -0.125991 |
| A | 30 | clean | gaussian_blur | 275 | 63 | 81 | -0.119364 |
| B | 45 | clean | gaussian_blur | 13 | 2 | 6 | -0.026508 |
| A | 45 | clean | gaussian_blur | 13 | 3 | 6 | -0.019881 |
| A | 52 | clean | gaussian_blur | 24 | 8 | 11 | -0.019881 |
| B | 52 | clean | gaussian_blur | 24 | 8 | 11 | -0.019881 |
| A | 15 | clean | gaussian_blur | 236 | 0 | 0 | 0.000000 |
| B | 15 | clean | gaussian_blur | 236 | 0 | 0 | 0.000000 |
| A | 15 | gaussian_blur | contrast_low | 236 | 0 | 0 | 0.000000 |
| B | 15 | gaussian_blur | contrast_low | 236 | 0 | 0 | 0.000000 |
| A | 21 | clean | gaussian_blur | 242 | 0 | 0 | 0.000000 |
| A | 71 | clean | gaussian_blur | 36 | 4 | 4 | 0.000009 |
| B | 71 | clean | gaussian_blur | 36 | 4 | 4 | 0.000009 |

## Label-skew audit

Audit occurs after all decisions/retrieval. Quartiles contain 25 clients sorted by (normalized entropy, client ID); ties may straddle adjacent quartiles. This rule was fixed before seeing results. No significance tests or fitted cutoffs.

| Bank | Context | Entropy quartile | Entropy min | max | ID accuracy % | Median signed margin |
| --- | --- | --- | --- | --- | --- | --- |
| A | clean | 1 | -0.000000 | 0.141182 | 80.000000 | 9.685993 |
| A | clean | 2 | 0.141182 | 0.292285 | 100.000000 | 12.519186 |
| A | clean | 3 | 0.298550 | 0.384987 | 88.000000 | 10.901669 |
| A | clean | 4 | 0.402377 | 0.681491 | 100.000000 | 13.966434 |
| A | brightness_dark | 1 | -0.000000 | 0.141182 | 100.000000 | 188.995635 |
| A | brightness_dark | 2 | 0.141182 | 0.292285 | 100.000000 | 183.447965 |
| A | brightness_dark | 3 | 0.298550 | 0.384987 | 100.000000 | 188.922451 |
| A | brightness_dark | 4 | 0.402377 | 0.681491 | 100.000000 | 185.448172 |
| A | contrast_low | 1 | -0.000000 | 0.141182 | 100.000000 | 37.228078 |
| A | contrast_low | 2 | 0.141182 | 0.292285 | 100.000000 | 36.476976 |
| A | contrast_low | 3 | 0.298550 | 0.384987 | 100.000000 | 40.371966 |
| A | contrast_low | 4 | 0.402377 | 0.681491 | 100.000000 | 28.870169 |
| A | gaussian_noise | 1 | -0.000000 | 0.141182 | 100.000000 | 21.554778 |
| A | gaussian_noise | 2 | 0.141182 | 0.292285 | 100.000000 | 21.920953 |
| A | gaussian_noise | 3 | 0.298550 | 0.384987 | 100.000000 | 19.696928 |
| A | gaussian_noise | 4 | 0.402377 | 0.681491 | 100.000000 | 20.411425 |
| A | gaussian_blur | 1 | -0.000000 | 0.141182 | 84.000000 | 15.484664 |
| A | gaussian_blur | 2 | 0.141182 | 0.292285 | 100.000000 | 16.805129 |
| A | gaussian_blur | 3 | 0.298550 | 0.384987 | 100.000000 | 15.603612 |
| A | gaussian_blur | 4 | 0.402377 | 0.681491 | 100.000000 | 16.595734 |
| B | clean | 1 | -0.000000 | 0.141182 | 84.000000 | 13.367101 |
| B | clean | 2 | 0.141182 | 0.292285 | 100.000000 | 14.634739 |
| B | clean | 3 | 0.298550 | 0.384987 | 88.000000 | 13.611206 |
| B | clean | 4 | 0.402377 | 0.681491 | 100.000000 | 16.780987 |
| B | brightness_dark | 1 | -0.000000 | 0.141182 | 100.000000 | 200.287599 |
| B | brightness_dark | 2 | 0.141182 | 0.292285 | 100.000000 | 193.659588 |
| B | brightness_dark | 3 | 0.298550 | 0.384987 | 100.000000 | 200.740181 |
| B | brightness_dark | 4 | 0.402377 | 0.681491 | 100.000000 | 195.498534 |
| B | contrast_low | 1 | -0.000000 | 0.141182 | 100.000000 | 33.921307 |
| B | contrast_low | 2 | 0.141182 | 0.292285 | 100.000000 | 36.380366 |
| B | contrast_low | 3 | 0.298550 | 0.384987 | 100.000000 | 39.104631 |
| B | contrast_low | 4 | 0.402377 | 0.681491 | 100.000000 | 29.262175 |
| B | gaussian_noise | 1 | -0.000000 | 0.141182 | 100.000000 | 21.496474 |
| B | gaussian_noise | 2 | 0.141182 | 0.292285 | 100.000000 | 22.578334 |
| B | gaussian_noise | 3 | 0.298550 | 0.384987 | 100.000000 | 20.344804 |
| B | gaussian_noise | 4 | 0.402377 | 0.681491 | 100.000000 | 21.041325 |
| B | gaussian_blur | 1 | -0.000000 | 0.141182 | 88.000000 | 15.967608 |
| B | gaussian_blur | 2 | 0.141182 | 0.292285 | 100.000000 | 17.002153 |
| B | gaussian_blur | 3 | 0.298550 | 0.384987 | 100.000000 | 16.697662 |
| B | gaussian_blur | 4 | 0.402377 | 0.681491 | 100.000000 | 16.981499 |

All Blur identification errors occur in the lowest-entropy quartile (84%/88% accuracy there vs100% in the other quartiles). Clean identification is 80%/84% in Q1 and88% in Q3; Q2/Q4 are100%. This supports a localized content-composition vulnerability, despite passing aggregate NID. It does not justify labeling the entire natural-support signature unobservable.

| Bank | Context | Skew feature | Signed-margin Spearman |
| --- | --- | --- | --- |
| A | clean | normalized_entropy | 0.130181 |
| A | clean | max_class_fraction | -0.146817 |
| A | clean | represented_classes | 0.077800 |
| A | brightness_dark | normalized_entropy | -0.070426 |
| A | brightness_dark | max_class_fraction | 0.049941 |
| A | brightness_dark | represented_classes | -0.099570 |
| A | contrast_low | normalized_entropy | -0.154631 |
| A | contrast_low | max_class_fraction | 0.164892 |
| A | contrast_low | represented_classes | -0.080661 |
| A | gaussian_noise | normalized_entropy | 0.022387 |
| A | gaussian_noise | max_class_fraction | -0.001886 |
| A | gaussian_noise | represented_classes | 0.093849 |
| A | gaussian_blur | normalized_entropy | 0.141874 |
| A | gaussian_blur | max_class_fraction | -0.170857 |
| A | gaussian_blur | represented_classes | 0.054192 |
| B | clean | normalized_entropy | 0.111576 |
| B | clean | max_class_fraction | -0.122422 |
| B | clean | represented_classes | 0.055315 |
| B | brightness_dark | normalized_entropy | -0.077898 |
| B | brightness_dark | max_class_fraction | 0.054062 |
| B | brightness_dark | represented_classes | -0.107858 |
| B | contrast_low | normalized_entropy | -0.136926 |
| B | contrast_low | max_class_fraction | 0.154981 |
| B | contrast_low | represented_classes | -0.058019 |
| B | gaussian_noise | normalized_entropy | 0.050612 |
| B | gaussian_noise | max_class_fraction | -0.038271 |
| B | gaussian_noise | represented_classes | 0.110800 |
| B | gaussian_blur | normalized_entropy | 0.136229 |
| B | gaussian_blur | max_class_fraction | -0.167688 |
| B | gaussian_blur | represented_classes | 0.040535 |

The per-context correlations are small (absolute rho at most about0.171). Tail errors and their task cost are more informative here than a strong monotonic entropy-margin relationship. Associations are descriptive, not causal.

Ten most negative signed-margin decisions:

| Bank | Client | Context | Selected | Signed margin | Entropy | Max class fraction | Classes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 15 | gaussian_blur | contrast_low | -17.721151 | -0.000000 | 1.000000 | 1 |
| B | 15 | gaussian_blur | contrast_low | -12.548052 | -0.000000 | 1.000000 | 1 |
| B | 15 | clean | gaussian_blur | -10.024681 | -0.000000 | 1.000000 | 1 |
| A | 15 | clean | gaussian_blur | -10.000182 | -0.000000 | 1.000000 | 1 |
| A | 30 | clean | gaussian_blur | -8.270149 | 0.371621 | 0.500000 | 3 |
| B | 30 | clean | gaussian_blur | -7.952907 | 0.371621 | 0.500000 | 3 |
| A | 74 | gaussian_blur | contrast_low | -5.678047 | -0.000000 | 1.000000 | 1 |
| A | 45 | clean | gaussian_blur | -3.573052 | 0.307631 | 0.800000 | 4 |
| B | 29 | gaussian_blur | clean | -3.163670 | -0.000000 | 1.000000 | 1 |
| A | 26 | clean | gaussian_blur | -2.765998 | 0.086214 | 0.950000 | 2 |

## Secondary K=20 diagnostic (if applicable)

Not applicable: primary K already equals20. No additional smaller-support gate or secondary query run was created. T008 balanced microbatches also had20 images, so the comparison is not explained by a different nominal support size alone; the image/content draws differ.

## OOD-distance comparison

| Bank | Known p50 | p90 | p95 | max | T008 noise min | median | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 6.212052 | 16.670310 | 23.395590 | 42.404145 | 84.241170 | 85.022791 | 86.209710 |
| B | 5.942520 | 16.382872 | 25.292304 | 48.220022 | 85.343273 | 89.344585 | 90.083756 |

There is no observed distance overlap in this comparison: natural-known maximum is48.22, whereas the old balanced noise-image minimum is84.24. This is descriptive separation for these saved draws, not a validated general rejection threshold. No threshold was selected; T008's high-margin OOD misclassification remains unchanged.

## Failures / uncertainties

Initial run 20260913-224559-ttfl-t009-gpu1 stopped with a missing read_data import at the first support pixel load, after manifest persistence and before any decision/query result. The only repair was importing the existing PFLlib utility. Failed and successful manifests have exactly the same counts, K, IDs, train indices and order (only timestamps differ). Both receipts are preserved. No outcome-informed retry occurred.

The retry used launch label IMPORT_FIX; actual deployed commit fab17bd is independently resolved by source SHA and recorded separately. This provenance defect did not change computation and was not hidden by rewriting the raw receipt. The known NVML mismatch warning remains; CUDA execution completed.

Only one label-blind support draw per client and fixed known corruption severities were tested. Banks remain privileged paired-clean. The checkpoint and PFLlib merged-original-train/test query split are unchanged, not an official CIFAR-10 benchmark. High ID accuracy is not a bound on harm from rare, client-dependent state substitutions. No new state was fit, and no historical result was relabeled.

## NID / NRET decision

**Case N-C: NID-A passes; NRET-A fails.** Natural source context naming largely survives label skew, but rare wrong state choices are costly: Blur retention fails and clean safety fails in both banks. Low-entropy concentration is a relevant localized weakness, not an alternative reason to ignore the passing NID gate.

## Recommended next action

Keep the operator/state bank fixed and return to Research Lead with the harmful confusion ledger. A next package should address sensitivity of clean↔blur and blur→contrast substitutions or audit state interpolation/clean preservation. No such mechanism, threshold, interpolation, SSL, learned writer, federation, meta-learning or richer operator was implemented. Await the next bounded instruction.
