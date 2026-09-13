# CODEX -> CHATGPT

## Timestamp

2026-09-13T19:39:26+08:00

## Commit / run ID

Lead 3394d2c; runtime bc07059; run 20260913-193525-ttfl-t007-preflight-gpu1; release 20260913-193520-ttfl-t007-preflight. NVIDIA RTX A6000 GPU1. Remote exit 0 at 2026-09-13 19:35:45 +08. The enclosing delivery commit contains this diagnosis.

## T007 decision (C-A/C-B/C-C/C-D)

**STOP at the explicitly required clean-identity prerequisite. No C-A/B/C/D assignment.** Both pools preserve clean macro-class accuracy exactly, but produce materially non-identity second-layer coefficients. No corrupted-query transfer experiment was run; primary transfer pass count is N/A, not 0/4. This is a mathematical writer/identity-contract conflict, not evidence that covariate transfer fails.

The active instruction says: “clean_identity must also change clean macro-class accuracy by <=0.1 pp and produce no material non-identity state; otherwise stop and diagnose before interpreting the shift results.” The state half is not satisfied: channel 29 scales are 0.323925 and 0.469380 instead of 1. No new numeric materiality cutoff was invented.

## Why T006 was insufficient

T006 low-semantic pairing still retained 57% same-class pairs. T007 pools and permutations remove that control limitation exactly. The new transfer question remains unanswered because its own identity prerequisite is not satisfied.

## Calibration-pool construction and provenance

Enumerated all existing training-partition candidates. Per pool/class, sorted by SHA256(t007:pool:class:client:original_id), took one per distinct client before any fallback, excluded previously selected IDs, and selected exactly 8/class. No query accuracy entered selection. Both pools were persisted before query evaluation. calibration_pools.json records all original IDs, train indices, client IDs, labels and permutations.

| Pool | Images | Distinct clients | Distinct clients per class 0..9 |
| --- | --- | --- | --- |
| A | 80 | 64 | 8,8,8,8,8,8,8,8,8,8 |
| B | 80 | 64 | 8,8,8,8,8,8,8,8,8,8 |

Pool IDs are disjoint from each other and all 15,039 query IDs. Each same-class permutation has zero image fixed points and 100% same-class agreement; each cross-class permutation has zero fixed points and 0% same-class agreement. Exact clean image/feature multisets were checked at both layers. Labels only construct/audit pools and pairing controls.

## Verification

24 existing tests passed remotely before edits. Two new pool/control tests passed locally; the full 26-test suite passed remotely before preflight. Two states were fitted, once per pool, then reused over all 100 clients. Frozen checkpoint, writer and corruption hashes unchanged; zero-state logits exact; model unchanged; all coefficients finite. Independent saved-prediction counts reproduce every macro-class score. Both identity states each change one prediction out of 15,039, without changing class-wise correct counts or accuracy.

## Primary macro-class transfer results

Not run: no corruption states were fitted and no transfer gates are interpreted. The prerequisite clean-query results are:

| Condition | Macro-class % | Sample-weighted % | Macro-client % |
| --- | --- | --- | --- |
| none | 34.138690 | 34.217701 | 33.188531 |
| A | 34.138690 | 34.217701 | 33.188531 |
| B | 34.138690 | 34.217701 | 33.188531 |

## Sample-weighted / macro-client secondary results

Shown above for the clean-identity prerequisite only. The same PFLlib merged original-train/test client split is retained; this is not the official CIFAR-10 test benchmark.

## Pool-A vs Pool-B reproducibility

Clean accuracy absolute A-B difference is 0 pp for all three metrics. Maximum clean query logit differences from zero state are 0.006222248 (A) and 0.004874945 (B). Prediction equality does not make the fitted coefficients identity.

## Correct vs semantic/hard controls

Pool permutations verified; their corrupted-source states were not fitted. No comparison is claimed.

## Per-class behavior

| Class | Query count | Zero-state correct | Accuracy % (identical A/B) |
| --- | --- | --- | --- |
| 0 | 1516 | 281 | 18.535620 |
| 1 | 1495 | 1156 | 77.324415 |
| 2 | 1509 | 760 | 50.364480 |
| 3 | 1482 | 189 | 12.753036 |
| 4 | 1492 | 1 | 0.067024 |
| 5 | 1500 | 273 | 18.200000 |
| 6 | 1527 | 1202 | 78.716437 |
| 7 | 1508 | 227 | 15.053050 |
| 8 | 1502 | 1057 | 70.372836 |
| 9 | 1508 | 0 | 0.000000 |

The pre-existing checkpoint is severely class-uneven (class 9 has zero correct queries, class 4 has one). This is retained as baseline context; no retraining or label tuning was performed.

## State similarity

| condition | cosine | l2_distance | max_abs_difference |
| --- | --- | --- | --- |
| clean_identity | 0.998831 | 0.161835 | 0.145454 |

Concatenated state follows the actual model order [gamma1,beta1,gamma2,beta2]. This describes clean-identity states only, not transferable corruption states.

## Restoration / coefficient diagnostics

| Pool | Layer | MSE before | MSE after | Ratio | Mean |gamma| | Max |gamma| | Mean |beta| | Max |beta| | Negative fraction | Cap fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 0.000000 | 2.45873e-09 | None | 0.000246 | 0.000771 | 4.3667e-05 | 0.000158 | 0.000000 | 0.000000 |
| A | 2 | 5.27161e-09 | 7.84142e-08 | 14.874802 | 0.019011 | 0.676075 | 4.38895e-05 | 0.000185 | 0.000000 | 0.000000 |
| B | 1 | 0.000000 | 2.54854e-09 | None | 0.000255 | 0.000778 | 4.4831e-05 | 0.000153 | 0.000000 | 0.000000 |
| B | 2 | 5.20085e-09 | 8.31437e-08 | 15.986563 | 0.015642 | 0.530620 | 4.44271e-05 | 0.000179 | 0.000000 | 0.000000 |

Undefined MSE_after/MSE_before when before=0 is reported as None; it is not a zero restoration error.

| Pool | Layer | Worst channel (zero-based) | Source variance | EPS | Scale | Max feature difference |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 11 | 0.012963 | 1e-05 | 0.999229 | 0.000659 |
| A | 2 | 29 | 4.78418e-06 | 1e-05 | 0.323925 | 0.054412 |
| B | 1 | 11 | 0.012849 | 1e-05 | 0.999222 | 0.000652 |
| B | 2 | 29 | 8.82384e-06 | 1e-05 | 0.469380 | 0.048244 |

For a clean identity input, the unchanged first-layer fit is `a = variance / (variance + EPS)` and `b = mean * (1-a)`, not exactly a=1,b=0. The second layer additionally sees the already-corrected first-layer source. Its channel 29 source variances are 4.78418e-6 (A) and 8.82384e-6 (B), comparable to or smaller than EPS=1e-5. This explains strong scale shrinkage despite small feature/logit changes and almost unchanged argmax predictions. No cap is active and no scale is negative in these two identity states. This repeats the known low-variance mechanism seen in T004 on a new balanced pool; it is not a newly introduced implementation regression.

## Failures / uncertainties

The process and provenance checks passed. The scientific prerequisite failed at the state level. The instruction does not quantify “material” state change; a 53–68% scale contraction is clearly not a small coefficient perturbation, even though all but one clean prediction agree in each pool. It would be incorrect to silently replace the state requirement with accuracy equality. Conversely, this observation alone says nothing about corrupted query transfer. The pre-existing NVML warning remains; PyTorch CUDA completed successfully.

## Recommended next action (recommendation only; do not launch)

Research Lead should reconcile the fixed-writer constraint with the identity-state prerequisite: explicitly permit the measured low-variance contraction based on functional evidence, or issue a separate writer revision that is identity-preserving and re-freeze the comparison. Neither option was selected or implemented here. Preserve these exact Pool-A/B IDs for continuation. No epsilon/cap edits, pool retries, SSL, FL, richer operator or T008. The 15-minute heartbeat will check for the next actionable instruction.
