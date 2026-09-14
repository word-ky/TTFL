# T018R: real-cell preflight passes; matched solver stops on sum feasibility

2026-09-14T12:50:42+08:00. Lead `8d12997` / unchanged scientific protocol `1744440`.

**Status: numerical blocker. CLS-MATCH-A / CLS-REAL-A / CLS-SRC-A remain NOT EXECUTED; no Outcome A/B/C.** The historical T018 PGD stop is preserved in its original raw receipts and the report below. This is a distinct T018R result.

## What changed and passed

Implemented the prescribed deterministic active-set simplex QP, retaining old PGD unchanged. The objective, warm start, observations, templates, context decisions, K20, tolerances and gates did not change. Face systems are lazily cached as `solve(K, identity)` mappings, as authorized; no added dependency, regularization or statistical tuning. Most-negative removal and greatest-dual-violation insertion use smallest-index tie breaking. Cycles, >100 updates, nonfinite values and failed feasibility stop execution.

Runtime `5f5e1a5a845f743646aea7f67c1d47463f1b9832`; run `20260914-124138-ttfl-t018r-preflight`, exit0. **98 tests pass.** Added tests cover condition numbers 3/50/100/200, 12 fixed synthetic face-reference comparisons, boundary zeros, insertion/removal, replay and input immutability.

- Noise-free 1,000: max prevalence error 4.57634e-13, max objective 3.9975e-27; all 7,914 singleton and 86 tied exact-optimum checks pass.
- Same fixed 200 actual cases: **all 49 old PGD caps repaired**, all 151 old converged cases reference-equivalent; max prevalence error 5.78725e-13, max objective error 5.01498e-16.
- All 1,000 real cells: KKT / simplex feasibility / BBSE objective dominance pass; max KKT 4.83169e-13, maximum 11 working-set updates. Full traces and pi vectors are saved.

## New stopped case

Runtime `315a984ee4ba5c26495b48f3a73c081c5f0340bc`; run `20260914-124709-ttfl-t018r-science`, exit1. Before opening query counts or accuracy, the program attempted numerical certification of the saved 128,000 matched-K20 observations. It accepted **45,575** cases then stopped at **client35 / bankB / Dark / replica7**. The remaining cases were not attempted; no partial accuracy or gate was computed. The accepted prefix was in memory and is not a complete persisted result array; its length follows the deterministic loop index and preserved log.

Failing face: classes `[0,1,2,3,6,8,9]`. All coordinates are nonnegative, but cached inverse-map application returns a sum of **1.000000000001048**, absolute error **1.048050535246148e-12**, exceeding the fixed `1e-12` sum tolerance by about 4.8%. `direct_kkt` is 1.04805e-12, which passes its separate `1e-10` bound. Thus this is specifically a **simplex sum feasibility rejection**, not a KKT threshold failure, cycle or update cap.

Same-runtime read-only replay reproduces the exception. The independently solved face and exhaustive reference give sum error 0.0, KKT 2.63834e-17, and prevalence difference from the cached result 4.77188e-13. Objective difference is 3.17781e-17. This evidence localizes the issue to roundoff in the cached inverse-map application; it is very different from the old PGD error of 0.04343. These direct solutions were **diagnostics only**, never substituted into the policy evaluation.

No normalization, tolerance relaxation, alternative solver or second scientific run was applied. Under the specified roundoff rule, clipping/renormalization occurs only when the face solve has a tiny negative coordinate; this failing face had none. The implementation therefore rejected the sum error as required.

## Scope and next decision

The exact original matched q was replayed from saved T017R arrays; no resampling. Zero new model forwards; no query metrics opened. All 85 T015/T016/T017 manifest entries remain byte-identical. No estimator or scientific conclusion follows from this stop.

Return to Lead for a narrow engineering decision on the face-system numerical application, e.g. direct RHS solve while preserving the same active-set trajectory/objective and all current tolerances. This is a suggestion only; no repair or rerun is authorized by this report itself. Do not move to feature semantics, SSL/TTT or FL.

Receipts: `research_log/t018r_receipts/20260914-124138-ttfl-t018r-preflight/` and `research_log/t018r_receipts/20260914-124709-ttfl-t018r-science/`; compact copies under `results/t018_constrained_prevalence/t018r/`. `blocker_diagnosis.json` contains the exact q-derived case, pi vectors and independent numerical comparisons. Scientific CSVs were not fabricated for unexecuted stages.
