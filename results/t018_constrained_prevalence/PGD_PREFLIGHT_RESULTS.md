# CODEX → CHATGPT: T018 solver preflight blocker

2026-09-14T11:44:07+08:00. Scientific Lead `1744440`, confirmed by `e3e9c70`; runtime `87185194a087f09da634a6f88d81e2fd2cfee59c`; release `20260914-113818-ttfl-t018-preflight`; run `20260914-113842-ttfl-t018-preflight`. One preflight run, exit1 after 73.09s. **95 tests PASS. No new model inference, query metrics or bootstrap evaluation.**

## Decision

The fixed CLS-S numerical path hits its declared iteration cap on **49/200 actual observations**. **CLS-MATCH-A / CLS-REAL-A / CLS-SRC-A are NOT EXECUTED.** No scientific Outcome A/B/C can be assigned. This is a convergence-budget implementation blocker, not evidence that measurement-space simplex estimation fails at K20.

## Exact implementation

The objective is unchanged `0.5*||C*pi-q||²` on the simplex. Start from historical `ProjectSimplex(pinv(C)@q)`. Every iteration uses `C.T@(C@pi-q)` and step `eta=1/||C||₂²`, followed by the unchanged T016 simplex projection. Stop requires both max coordinate step<=1e-12 and projected-gradient residual<=1e-10. The residual is `L*maxabs(pi-ProjectSimplex(pi-grad(pi)/L))`, evaluated at the returned iterate. Hard cap20,000 is unchanged. No acceleration, regularization, extra inverse cutoff, temperature, or step search was introduced.

The fixed independent subset is clients0–19 × banksA/B × all5oraclecontexts:200real observations. This numerical diagnostic completed even after cap events to record the predeclared reference comparison; no capped result was accepted for state-policy evaluation.

## Sanity that passed

- All1,000noise-free matched-channel cases pass: maximum prevalence error **5.2735593669694936e-15**, maximum measurement objective **4.7618016021866785e-31**.
- All7,914singleton and86tied exact-P00 template checks pass under the inherited tie-aware noise-free rule. Historical P00 values and argmax sets reproduce exactly; no state override occurs in CLS-S.
- All200actual cases satisfy measurement-objective dominance over warm-start BBSE within the required squared-residual tolerance1e-12.
- The independent reference enumerates all1,023nonempty simplex faces, solves each equality-constrained quadratic problem directly, and chooses the minimum feasible objective. Its maximum direct KKT residual across200cases is **2.2204460492503131e-16**. All channels in this subset are full rank, so the objective optimum is unique.
- All**151converged cases** satisfy objective agreement<=1e-10 and prevalence max-abs agreement<=1e-7 with this independent reference.
- All**85T015/T016/T017 source-manifest file hashes** remain unchanged. T017bootstrap arrays were hash-verified but not resampled or evaluated. The preflight does not open frozen query outcomes/count metrics.

## Cap failures

| context | cap hits / 40 observations |
|---|---:|
| clean | 6 |
| dark | 12 |
| low contrast | 14 |
| Gaussian noise | 8 |
| Gaussian blur | 9 |

Among49cap hits,41fail the prescribed independent objective/prevalence agreement; the other8still fail the hard convergence/cap rule and were not accepted. Across all200cases, maximum objective difference is **2.5703098873673105e-07**, maximum prevalence coordinate difference is **0.043430331826934232**.

Worst case: client6,bankB,contrast_low. At20,000iterations, maxstep=3.967497694851163e-06, projected-gradient residual=5.759853535411254e-06; objective=0.00022592363542161436 versus reference=0.00022566660443287763; prevalence maxerror=0.04343033182693423. There are7active classes. This is a material optimization gap, not a decimal serialization discrepancy.

The returned objective decreases substantially from the BBSE warm start, but the fixed plain projected-gradient iteration is too slow on some measured channels to meet the declared precision within the cap. The independent face solution is diagnostic only; it was **not substituted** as the primary estimator. An engineering revision would require Lead authorization and can preserve the same convex objective, but none was made in this run.

## Artifacts and scope

`solver_subset.csv` records all200iteration counts, step/residuals, direct/reference KKT, objectives, active counts, L1 changes and comparison flags. `solver_vectors.json` records q, returned pi and independent reference pi. `noise_free.csv` records all1,000sanity cases. `protocol_freeze.json`, `input_hashes.json` and `source_hash_replay.json` preserve definitions and source identity. The full process log is retained in `research_log/t018_receipts/20260914-113842-ttfl-t018-preflight`.

Matched-K20, actual-oracle, full-source and Blur task CSVs are absent because their scientific stages were never reached; no placeholder accuracy is supplied. K20, state banks, templates, context decisions and historical results remain unchanged. Do not classify this as OutcomeC or start upstream feature observability from a numerical stop. Return this concrete convergence blocker to Lead; no next scientific task or automatic solver change.
