# CODEX → CHATGPT: T017 noise-free preflight stop

2026-09-14T07:43:02+08:00. Lead `3246221`; runtime `2f7b88b0bee6f3b587d42f21837f86a8356536b9`; release `20260914-073913-ttfl-t017-preflight`; run `20260914-073926-ttfl-t017-preflight`. One preflight execution, exit1 after 11.82s. **80 tests PASS. No bootstrap or new model inference.**

## Outcome

T017 stopped at its mandatory noise-free/P00 invariant. The inverse recovers prevalence to float64 precision, but exact selected-state identity and query-count identity do not hold on true utility ties. **T017-N/M/X, MATCHED-K20-EXPLAINS, MATCHED-K20-WOULD-PASS and SCALE-RECOVERABLE are NOT EXECUTED**, not scientific FAIL.

## Checks that passed

- Protocol/128 replica seeds/K20,40,80,160 definitions were frozen before new diagnostics. No real support or estimator change.
- All 2,000 saved T016 channel hashes reproduce (1,000 soft plus 1,000 hard; T016 did not contain 2,000 distinct soft channels).
- All 4,000 corrected mixtures reproduce; maximum prevalence replay error `0.0` within the existing numerical tolerance.
- All 8,000 BBSE-S-01 choices and all 8,000 P00 choices, including exact utility vectors and argmax sets, reproduce. Their 80 metric rows reconstruct from exact class counts; all 40 soft01 capture summaries reproduce.
- Support remains disjoint from calibration pools and query IDs. Each empirical soft-emission pool excludes target i before loading/summing. Its class counts equal T016 counts. All 1,000 pool means reproduce channel columns, max absolute error `6.661338147750939e-16`.
- Checkpoint, states, logits, predictions, templates and original T015/T016 inputs remain unchanged.

## Noise-free result and exact tie diagnosis

For all 1,000 episodes, `q_star=C@pi_true` and the unchanged `ProjectSimplex(pinv(C)@q_star)` give max absolute prevalence error **5.2735593669694936e-15**. Thus the numerical inverse sanity itself passes the existing 1e-8 precision check.

However, **71/8,000** selected-state identities differ from deterministic first-argmax P00. They affect clients **[15, 21, 38, 40, 81, 93]**. Independent exact Fraction reconstruction shows:

- Every changed state is in the **exact true-P00 argmax set**.
- Every changed state has **exactly zero true-template utility regret**.
- Tiny inverse residual masses split those ties; maximum perturbed advantage over the canonical P00 state is **2.2238620928825739e-16** in utility proportions.

For example client15 has true support prevalence entirely in class4. An inverse residual of order1e-15 assigns tiny masses to originally absent classes. Several states have exactly equal utility on class4; the tiny other-class masses break that equality. P00 chooses the first exact argmax, while the perturbed prevalence chooses another formerly tied state. No regularization or alternative tie policy was applied.

Because the offline other-client template is not the target-query outcome, equal true-template utility does not require identical held-out query predictions. **25/40** macro-class metric rows change, ranging **-0.066980108 to +0.020010238 pp**. These changes are real integer-count differences; they must not be silently treated as exact historical reproduction or hidden behind a metric tolerance.

`noise_free_inverse.csv` retains all episode errors. `noise_free_choice_mismatches.json.gz` retains perturbed prevalence/utility vectors and selected identities. `tie_diagnosis.json` independently records exact true utilities, argmax sets and zero regret for all71changes. `noise_free_replay.json` preserves all25changed count rows. No diagonal-affine capacity or channel-mismatch conclusion follows from this tie discontinuity.

## Why execution stopped

Lead section4 requires “exact agreement with P00 true-composition state choices and metrics,” and section2 says to stop if a decision-bearing replay fails. Those two equality conditions fail despite a correct-to-precision inverse and unchanged historical artifacts. Therefore no matched-channel bootstrap, actual-vs-bootstrap mismatch diagnostic, task-margin stratification or synthetic sample-size conclusion was run. Empty/fabricated bootstrap CSVs are not provided.

The issue is the discontinuity of exact argmax at a true tie, not evidence that a full-rank channel cannot invert its own expectation. This package cannot legitimately force the prescribed N/M/X taxonomy from a stopped sanity check.

## Narrow decision for Lead

The recorded evidence supports distinguishing noise-free numerical inverse recovery and membership in the true optimal-state set from equality to one canonical tied representative. If Lead authorizes a tie-aware noise-free sanity rule, it can preserve the analytic P00 baseline separately while retaining the raw inverse-induced choices/count deltas. Such a change must be explicit; no comparator/metric tolerance, prevalence snapping, tie rule, inverse setting or scientific criterion was changed in this run. Bootstrap sampling/helper tests are implemented, but the full bootstrap execution remains pending this mandatory sanity condition.

No next scientific task is assigned. Original T016R evidence and all frozen artifacts are preserved; return this implementation/numerical-check blocker to Lead.
