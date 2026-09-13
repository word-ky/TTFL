# CODEX -> CHATGPT

## Timestamp

2026-09-13T21:35:45+08:00

## Commit / run ID

Lead e20ef8f; runtime 4a71d7d; run 20260913-213117-ttfl-t008-gpu1; release 20260913-213110-ttfl-t008. NVIDIA RTX A6000 GPU1, runtime 6.70 seconds. The enclosing delivery commit contains this independent audit and report.

## T007R evidence reused

Checkpoint, neutral writer, corruption definitions, pools and state bank remain frozen. Eight bank correct-pair states were reconstructed using bank clean/corrupted pixels and match the saved T007R states exactly. Four mapped wrong-alt choices per bank also exactly match the stored wrong_alt_source states. Clean maps to zero state. No client-specific fitting or new writer was introduced.

T007R remains C-A with 3/4 joint PASS. Noise Pool-B recovery .499741262 remains FAIL under that task. T008 measures retention of the oracle gain, a separate gate; its Noise retrieval success does not relabel T007R.

## Source signature definition

At zero fast state, extract population mean and sigma=sqrt(var+1e-6) after conv1 and conv2 spatial blocks. For each channel, concatenate (mu-mu_bank_clean)/(sigma_bank_clean+1e-6) and log((sigma+1e-6)/(sigma_bank_clean+1e-6)); two layers yield 192 dimensions. Double-precision statistics; squared Euclidean distance to five fixed prototypes. No learned metric, gradient, labels, fast state, query tensors or held-out clean reference enter signature/selection APIs. Ties use the fixed context order.

## Prototype and microbatch provenance

Exact original pool artifact SHA256 c6b1692e6fae8db52dec5dfe036af03bdc696de359a7b35f5bf64fe1152e49d0; no image reselection. Bank A selects on held-out B and vice versa. Every prototype uses the full 80-image bank pool; all signatures in a direction normalize against that bank's clean moments only. Both pools are disjoint from each other and all queries.

Each held-out pool is partitioned into four disjoint 20-image batches, exactly two images/class, by committed within-class positions [0:2], [2:4], [4:6], [6:8]. Labels are used only for constructing/auditing this prescribed diagnostic partition. Microbatch original IDs, client IDs and indices are in microbatches.json.

Both prototype banks/vectors/hashes were persisted first. All 40 primary decisions and 8 OOD decisions were then persisted before opening T007R query metrics/labels. Selection freeze timestamp: 2026-09-13T13:31:36.275761+00:00. Prototype/decision file hashes still match at completion and in the independent audit. Full vectors, references, all five distances and top-two choices are retained.

## Verification

31 prior regression tests passed before editing; three focused signature/distance/microbatch tests passed before integration. The full 36-test suite passed remotely before this 40-decision experiment. Model hash unchanged throughout; checkpoint, neutral-writer and corruption hashes unchanged; all signatures/distances finite.

All eight reconstructed bank states match saved states exactly. Forty-eight focused first-two-client prediction regressions (both banks, all shifts, none/oracle/wrong-alt) match T007R predictions exactly. All selected states already have T007R predictions, so no new 100-client state/query evaluations were needed. The four batch retrieval scores per shift reuse the same unchanged global query set, not four independent query trials.

Independent CPU distance recomputation agrees within 5.68e-14; all top-two choices, prototype vector hashes, confusion matrices and ID gates match. Independent original-prediction counts reproduce all reused macro-class metrics within 7.11e-15 pp. Fraction-based retained-recovery checks agree. No query-based retries, prototype normalization choices, metric fitting or OOD threshold was used.

## A→B identification results

20/20 correct, all five contexts 4/4. ID direction PASS. Mean top-two distance margin 56.133132; median 21.548573.

## B→A identification results

20/20 correct, all five contexts 4/4. ID direction PASS. Mean margin 58.977433; median 23.923831.

## Confusion matrices / margins

Rows are true contexts; columns follow clean, dark, contrast, noise, blur. Exact 40 decisions including all five distances and second-nearest identities are in identification.csv.

A→B:

| True / predicted | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 4 | 0 | 0 | 0 | 0 |
| brightness_dark | 0 | 4 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 4 | 0 | 0 |
| gaussian_noise | 0 | 0 | 0 | 4 | 0 |
| gaussian_blur | 0 | 0 | 0 | 0 | 4 |

B→A:

| True / predicted | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 4 | 0 | 0 | 0 | 0 |
| brightness_dark | 0 | 4 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 4 | 0 | 0 |
| gaussian_noise | 0 | 0 | 0 | 4 | 0 |
| gaussian_blur | 0 | 0 | 0 | 0 | 4 |

| Direction | Context | Margin mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| A_to_B | clean | 15.566026 | 17.084210 | 9.094343 | 19.001344 |
| A_to_B | brightness_dark | 189.618626 | 188.404940 | 184.923403 | 196.741221 |
| A_to_B | contrast_low | 34.749612 | 33.397163 | 25.469343 | 46.734779 |
| A_to_B | gaussian_noise | 20.468543 | 19.646192 | 18.492760 | 24.089028 |
| A_to_B | gaussian_blur | 20.262851 | 20.084762 | 18.166718 | 22.715162 |
| B_to_A | clean | 14.925931 | 14.394482 | 12.175854 | 18.738907 |
| B_to_A | brightness_dark | 203.012904 | 203.593463 | 194.116589 | 210.748103 |
| B_to_A | contrast_low | 35.944474 | 34.415760 | 25.369005 | 49.577373 |
| B_to_A | gaussian_noise | 22.002249 | 22.257522 | 16.211367 | 27.282584 |
| B_to_A | gaussian_blur | 19.001606 | 22.064831 | 7.197762 | 24.679000 |

## Oracle-state retrieval results

Each of the four source batches selected its true context. Consequently every individual retrieved-state score equals that bank's existing correct-pair oracle score; retained recovery is exactly 1.0. retrieval.csv contains all 32 per-batch macro-class, sample-weighted and macro-client scores; retrieval_summary.csv reports their means separately by direction and shift.

| Direction | Shift | None % | Oracle % | Selected mean % | Wrong alt % | tau pp | Retained | Selected-wrong pp | PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_to_B | brightness_dark | 17.546026 | 32.296536 | 32.296536 | 14.903865 | 4.148166 | 1.000000 | 17.392670 | True |
| A_to_B | contrast_low | 19.955264 | 28.561473 | 28.561473 | 22.733171 | 3.545856 | 1.000000 | 5.828302 | True |
| A_to_B | gaussian_noise | 30.676644 | 32.499558 | 32.499558 | 30.382434 | 0.865512 | 1.000000 | 2.117123 | True |
| A_to_B | gaussian_blur | 30.901579 | 33.048500 | 33.048500 | 29.798137 | 0.809278 | 1.000000 | 3.250362 | True |
| B_to_A | brightness_dark | 17.546026 | 32.424808 | 32.424808 | 15.489587 | 4.148166 | 1.000000 | 16.935221 | True |
| B_to_A | contrast_low | 19.955264 | 28.111626 | 28.111626 | 22.693556 | 3.545856 | 1.000000 | 5.418069 | True |
| B_to_A | gaussian_noise | 30.676644 | 32.406771 | 32.406771 | 30.537680 | 0.865512 | 1.000000 | 1.869091 | True |
| B_to_A | gaussian_blur | 30.901579 | 33.009919 | 33.009919 | 28.354235 | 0.809278 | 1.000000 | 4.655684 | True |

| Direction | Shift | Selected macro-class batches 0 / 1 / 2 / 3 | Mean sample-weighted % | Mean macro-client % |
| --- | --- | --- | --- | --- |
| A_to_B | brightness_dark | 32.296536 / 32.296536 / 32.296536 / 32.296536 | 32.362524 | 31.268559 |
| A_to_B | contrast_low | 28.561473 / 28.561473 / 28.561473 / 28.561473 | 28.638872 | 27.361031 |
| A_to_B | gaussian_noise | 32.499558 / 32.499558 / 32.499558 / 32.499558 | 32.575304 | 31.387282 |
| A_to_B | gaussian_blur | 33.048500 / 33.048500 / 33.048500 / 33.048500 | 33.120553 | 31.950703 |
| B_to_A | brightness_dark | 32.424808 / 32.424808 / 32.424808 / 32.424808 | 32.488862 | 31.338011 |
| B_to_A | contrast_low | 28.111626 / 28.111626 / 28.111626 / 28.111626 | 28.186715 | 26.830967 |
| B_to_A | gaussian_noise | 32.406771 / 32.406771 / 32.406771 / 32.406771 | 32.482213 | 31.267717 |
| B_to_A | gaussian_blur | 33.009919 / 33.009919 / 33.009919 / 33.009919 | 33.080657 | 31.880620 |

Forced-wrong retrieval is the mapped bank wrong-alt state; it is verified identical to the saved T007R negative control. Selected-minus-wrong exceeds frozen tau in every direction/shift, so success is not being inferred merely because all states improve.

## Noise-image OOD diagnostic

| Direction | Batch | Selected | Second | Nearest distance | Second distance | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| A_to_B | 0 | gaussian_noise | clean | 86.209710 | 154.500035 | 68.290325 |
| A_to_B | 1 | gaussian_noise | clean | 84.611926 | 152.328754 | 67.716828 |
| A_to_B | 2 | gaussian_noise | clean | 84.241170 | 150.971262 | 66.730091 |
| A_to_B | 3 | gaussian_noise | clean | 85.433655 | 153.217036 | 67.783381 |
| B_to_A | 0 | gaussian_noise | clean | 90.083756 | 158.559449 | 68.475693 |
| B_to_A | 1 | gaussian_noise | clean | 89.858387 | 159.268501 | 69.410114 |
| B_to_A | 2 | gaussian_noise | clean | 88.830784 | 158.075346 | 69.244562 |
| B_to_A | 3 | gaussian_noise | clean | 85.343273 | 154.451523 | 69.108250 |

**All eight noise-image batches are assigned gaussian_noise**, with margins 66.73–69.41, comparable to or larger than many real-context margins. The selector has no rejection mechanism, and a large top-two margin is not evidence that support belongs to one of the modeled contexts. Absolute nearest distances are also reported (84.24–90.08), but no rejection cutoff was chosen. OOD retrieval/query performance was not evaluated or claimed.

## Failures / uncertainties

No runtime or verification failure. The pre-existing NVML driver/library mismatch warning persists; PyTorch CUDA completed on A6000. No historical FL or T007R grid was rerun.

This is a closed five-context bank at fixed known corruption severities, evaluated on two balanced pools. The same images recur across corruption conditions; 40 controlled decisions are not 40 independent samples from arbitrary client contexts. The offline bank remains privileged paired-clean, and balanced microbatch construction uses labels. Runtime signature/selection is source-only; this does not yet establish a continuous state writer, arbitrary label-skew support robustness, unseen-shift recognition or OOD handling. The existing checkpoint/query split is unchanged and is not the official CIFAR-10 test benchmark.

## ID-A / RET-A decision

**ID-A + RET-A.** Both directions pass identification (20/20 with 4/4 per context); all four shifts jointly pass retrieval with 100% oracle-gain retention. This supports source-only observability and discrete bank selection in the prescribed balanced, known-context diagnostic. It does not establish a deployable learned writer.

## Recommended next action

Keep the 192-scalar operator and return to Research Lead. A subsequent bounded package may investigate a continuous source-only state estimator and its context/OOD controls. The demonstrated OOD margin limitation should remain visible. No SSL, gradient-based TTT, neural writer, meta-learning, federation, richer operator or next stage was launched. The existing 15-minute heartbeat will read new instructions.
