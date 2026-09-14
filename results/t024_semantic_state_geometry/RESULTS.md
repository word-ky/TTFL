# T024 — T024-S: semantic-state oracle geometry

2026-09-14T23:32:46+08:00. Lead d55a0f9 (unchanged heartbeat96682c0). **Clean visual context + semantic label-composition shift**, with anonymous fixed neutral operators `s0..s4`; `s0` is the strict no-op. Target composition is privileged. This is an oracle-template diagnostic, not a deployable adaptation method or a new sampled benchmark.

## Verified decision

| Gate | Bank A | Bank B | Overall |
|---|---|---|---|
| SEM-HEADROOM-A | True | True | True |
| SEM-SPEC-A | True | True | True |
| SEM-TREND-A | True | True | True |


All three frozen gates pass in both banks: **T024-S**. Severe semantic change makes the historical oracle action materially stale, and a mismatched current composition also selects materially worse actions under the same target template. This supports the specified semantic-state abstraction within the existing fixed bank and the predeclared cyclic-shift construction. It is a mechanism/geometry result, not an implementation blocker.

| Bank | Stale mean / median / p90 pp | Stale >=1pp % | Switch % | Mismatch mean pp | Mismatch >=1pp % | Current−zero / historical−zero pp |
|---|---:|---:|---:|---:|---:|---:|
| A | 9.644 / 5.883 / 24.942 | 78.250 | 86.625 | 7.562 | 65.375 | 5.835 / -3.809 |
| B | 9.536 / 5.503 / 24.571 | 76.500 | 86.625 | 7.478 | 64.250 | 5.671 / -3.865 |


Both banks switch canonical state in693/800source cells (86.625%). Stale regret exceeds1pp in78.25%/76.5% of cells; mismatched-current regret exceeds1pp in65.375%/64.25%. Under severe shift the current oracle has mean positive gain versus no-op of5.835/5.671pp, while the historical choice has mean gain−3.809/−3.865pp. Thus headroom is not just an oracle beating no-op: retaining an otherwise historically optimal action is itself harmful on average after the semantic change.

## Frozen construction and units

The800historical compositions are100clients×4salts×2train-half orientations. Each normalizes the exact T014 saved class counts, verified against T013 counts. For every source composition, the nine nonzero right-cyclic class permutations are compared by exact TV; maximum TV wins and smallest offset breaks ties. No utility is used to construct the counter-state. There are zero uniform degenerate compositions. All four fixed lambdas, resulting3200current compositions and800SHA256 next-in-sorted-order mismatch pairings were saved before class-template utility access. Pairing is bijective/self-excluding within salt/half and shared across banks.

The mismatch has the same lambda as the target, but its realized TV magnitude can differ because its source composition differs. Both current and mismatched compositions are evaluated using the *target's same* leave-one-client-out class template. All scoring uses clean visual context; these operator columns are anonymous actions, not visual client-state labels.

T014 class templates store candidate class accuracy **minus no-op accuracy**. Consequently stored U values are expected template gains relative to no-op, not absolute accuracies. This common baseline cancels exactly in choices and stale/mismatch regret. Reported pp are expected accuracy differences under the frozen other99-client template, not newly measured current-client test accuracy. All vectors, choices, full argmax sets, regrets and gate comparisons use Fraction arithmetic. Top-two margins preserve tied-zero margins. Floats are used only for descriptive quantiles, JS and correlations.

Semantic freeze SHA256: `c7d8ed7ff15893ce0efb97e8e0ed44daba9b86191f2d3f98ddde46827f932a71`. JS denotes square root of base2 Jensen-Shannon divergence. The protocol and generator were unchanged after scoring.

## Shift-strength trend and zero-state safety

| Bank / lambda | Mean stale pp | Median / p75 / p90 pp | Switch % | Mismatch mean pp | Oracle−zero / historical−zero pp |
|---|---:|---:|---:|---:|---:|
| A / 1/4 | 0.548 | 0.000 / 0.000 / 1.820 | 24.125 | 5.608 | 4.810 / 4.262 |
| A / 1/2 | 2.189 | 0.000 / 2.978 / 8.205 | 42.250 | 4.888 | 3.761 / 1.572 |
| A / 3/4 | 5.385 | 2.494 / 7.591 / 16.210 | 68.375 | 5.587 | 4.266 / -1.119 |
| A / 1 | 9.644 | 5.883 / 14.353 / 24.942 | 86.625 | 7.562 | 5.835 / -3.809 |
| B / 1/4 | 0.576 | 0.000 / 0.000 / 1.969 | 24.750 | 5.495 | 4.728 / 4.153 |
| B / 1/2 | 2.159 | 0.000 / 2.779 / 8.174 | 43.250 | 4.601 | 3.639 / 1.480 |
| B / 3/4 | 5.303 | 2.340 / 7.405 / 16.099 | 69.000 | 5.573 | 4.111 / -1.192 |
| B / 1 | 9.536 | 5.503 / 14.458 / 24.571 | 86.625 | 7.478 | 5.671 / -3.865 |


Mean stale regret increases from0.548→9.644pp in bankA and0.576→9.536pp in bankB. Severe-minus-mild gaps are9.096/8.960pp, exceeding the0.5pp gate. Nondecreasing stale regret is partly an algebraic consequence of this fixed-bank linear mixture with a historically optimal action; the substantive empirical requirements are the magnitude, affected-cell fraction and mismatch specificity. Positive oracle−zero is also structurally nonnegative because no-op belongs to the bank; the measured magnitude and historical−zero remain informative.

| Bank | Distance | Spearman vs stale regret | Cell×lambda rows |
|---|---|---:|---:|
| A | TV | 0.514303 | 3200 |
| A | JS_distance | 0.514093 | 3200 |
| B | TV | 0.509597 | 3200 |
| B | JS_distance | 0.509451 | 3200 |


The3200rows per bank reuse800compositions across4lambdas. Likewise800source cells are repeated views of100clients, not800independent clients. These correlations are descriptive; they never replace a gate.

## Current-state specificity and transitions

| Bank / salt | Severe mean mismatch pp | >=0.5pp |
|---|---:|---|
| A / T013-S0 | 8.638 | True |
| A / T013-S1 | 7.152 | True |
| A / T013-S2 | 6.662 | True |
| A / T013-S3 | 7.795 | True |
| B / T013-S0 | 8.621 | True |
| B / T013-S1 | 7.272 | True |
| B / T013-S2 | 6.352 | True |
| B / T013-S3 | 7.667 | True |


Both banks pass all4salts (required>=3) and their pooled >=1pp affected-cell fractions exceed25%. Complete per-lambda/salt controls are in mismatch_by_salt.csv. Full8transition matrices and all cell-level current/historical/mismatch argmax sets are retained.


Bank A, lambda1: rows historical, columns current. Total800 source cells.

| Historical / current | s0 | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|---:|
| s0 | 4 | 36 | 7 | 48 | 5 |
| s1 | 53 | 49 | 15 | 178 | 7 |
| s2 | 13 | 15 | 0 | 30 | 0 |
| s3 | 28 | 114 | 60 | 54 | 36 |
| s4 | 0 | 14 | 0 | 34 | 0 |

Bank B, lambda1: rows historical, columns current. Total800 source cells.

| Historical / current | s0 | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|---:|
| s0 | 4 | 43 | 6 | 47 | 6 |
| s1 | 56 | 55 | 13 | 177 | 8 |
| s2 | 12 | 9 | 0 | 26 | 0 |
| s3 | 27 | 109 | 53 | 48 | 38 |
| s4 | 1 | 22 | 0 | 40 | 0 |


## Independent verification and execution

**142 tests PASS**, including focused cyclic target, exact TV scaling, pairing, tie, oracle and gate-boundary tests. The separate verifier does not import the main geometry/gate functions. It reconstructs all800pi0/q vectors,3200pi_lambda vectors,800pairings,1600clean target-excluded templates directly from the other99clients' integer counts,19200utility vectors,6400choice/regret rows,8lambda aggregates,32mismatch aggregates and8transition matrices. Every choice/full argmax set, stale/mismatch/zero regret, switch/overlap bit and all three gates agree exactly. Maximum exact discrepancy0; maximum JS difference3.3306690738754696e-16; maximum descriptive Spearman difference1.1102230246251565e-16.

All 112 source manifest entries and all analysis input hashes remain unchanged. Expected source/template/row cardinalities, state order and no-op index pass. Analysis and verification are pure local CPU saved-evidence calculations: no self-supervised loss, writer, gradient update, scientific model forward, new training or federation was executed. The remote job ran only the regression suite.

- Analysis runtime `740dda4135c0e985a9eedeab47990baebb03adee`, local receipt `research_log/t024_receipts/20260914-t024-local`, seconds9.637.
- Independent verifier runtime `28e9590af58ac619efa3ed9c59987b990c13878b`, local receipt `research_log/t024_receipts/20260914-t024-verify`, seconds31.286.
- Full suite remote run `20260914-232500-ttfl-t024-tests`, source release `20260914-232433-ttfl-t024-tests` from analysis runtime; exit0,142tests.

## Scope of the positive result and next-state request

This intentionally severe, deterministic maximum-TV cyclic construction establishes oracle geometry within the tested bank. It does not establish unlabeled semantic-state estimation, a real sampled test-time distribution, or writer/federation performance. The inherited data remains the existing PFLlib CIFAR10 client partition; no new benchmark/data/checkpoint was created.

Return **T024-S** and request Lead review of a bounded *real sampled semantic-shift benchmark* task. Do not start that benchmark, an estimator, another SSL objective, a continuous writer or federated sharing without a later explicit Lead task.
