# T021 — T021-M with CTX+ for both frozen representations

2026-09-14T18:57:23+08:00. Lead `7b5bb34`. Scoring runtime `a5b74e10995e3a7cb846b553c70a68b39a736b11`, run `20260914-184639-ttfl-t021-score`; independent verification runtime `5296f57e0cc58058e8825ab8844f8ac4284d663a`, run `20260914-185130-ttfl-t021-verify-gpu1`. All five stages exit0. Final full suite **130 tests PASS**.

## Decision

**T021-M. L and H pass REP-MATCH-A but both fail REP-REAL-A.** L matched passes3/4shifts (Blur fails); H matched passes4/4. Both real paths pass onlyDark1/4. Both inherited real regression-safety checks pass. Neither representation passes REP-REGRET-A versus historical posterior CLS:0/4 shifts satisfy all bank/salt conditions.

**CTX+: L passes3/4 (Noise fails), H passes4/4.** Because both representations share the T021-M diagnosis and both context gates pass, the CTX+ flag is unambiguous without choosing a preferred representation. **REP-SRC-A is NOT_EXECUTED** for both, since neither real path qualifies. Source choices were frozen in Phase-A but not scored; no SRC success/failure is assigned.

T021 therefore does not establish that raw logits or512-D classifier-input features alone close the real K20 gap. The matched result shows that the fixed first-moment observer can be useful under its own class-conditional sample model. Context-specific prototypes outperform clean-image prototypes evaluated under the same frozen state, establishing a need for context-matched observation geometry within this tested model. That control improvement does not imply the80% real-capture target has been met. These results do not show that all task information is absent from H; only the specified raw Euclidean class-prototype observer was tested.

No architecture/operator change, learned projection/head, whitening, ridge, temperature, SSL/TTT, gradient step or federation was used. The BER branch remains closed.

## Inherited gain-capture evidence

Every range spans both banks/all four salts; each context needs every one of its8rows to reach80%. Matched columns use the median across the128 exact-count replicas. P is the frozen T018R2 posterior-CLS baseline; its observations/models were not recomputed or tuned.

| Context | P matched % | L matched % | H matched % | P real % | L real % | H real % |
|---|---:|---:|---:|---:|---:|---:|
| brightness_dark | 90.783–92.351 | 90.541–91.936 | 91.895–93.506 | 89.138–91.783 | 89.289–90.281 | 87.110–89.612 |
| contrast_low | 83.432–85.872 | 82.346–83.775 | 83.060–84.545 | 75.729–83.422 | 72.333–76.391 | 68.940–79.791 |
| gaussian_noise | 84.478–85.748 | 80.402–82.363 | 82.256–83.527 | 78.334–81.889 | 71.239–76.240 | 70.227–72.586 |
| gaussian_blur | 82.684–84.908 | 79.010–81.961 | 80.809–82.533 | 74.911–76.690 | 68.244–74.730 | 72.997–75.658 |


Gate implementation preserves T018's exact matched-median aggregation and real safety: every shifted oracle row must stay within0.5pp of BBSE-S-01, not a newly invented tolerance against P. The historical source clean rule would require no worse than zero−0.5pp, but source scoring was not authorized here. The same frozen P00 denominator, exact canonical tie order and opposite-half query counts are used throughout. Baseline P real integer numerators are independently reproduced exactly.

## Regret and context specificity

Positive reductions mean lower mean true-template regret. The gate additionally requires p90 regret no more than5% worse and the15% mean improvement in both banks/all salts for at least3/4shifts. Ranges below are effects across the8bank/salt rows; complete mean/p90, optimal-set agreement, prevalence L1 and client better/equal/worse counts are in the CSVs.

| Representation/context | Mean regret reduction vs P, % | Context vs clean@same-state reduction, % | Context paired better clients /100 |
|---|---:|---:|---:|
| L/brightness_dark | -3.386–11.591 | 52.709–60.985 | 16.000–24.000 |
| L/contrast_low | -32.603–9.132 | 68.726–73.582 | 50.000–52.000 |
| L/gaussian_noise | -56.242–-22.693 | 4.257–22.477 | 20.000–25.000 |
| L/gaussian_blur | -53.289–-33.420 | 56.139–60.475 | 37.000–39.000 |
| H/brightness_dark | -11.116–-1.152 | 43.653–50.121 | 13.000–14.000 |
| H/contrast_low | -81.358–-17.877 | 58.577–69.092 | 48.000–53.000 |
| H/gaussian_noise | -52.422–-33.064 | 74.013–77.756 | 46.000–49.000 |
| H/gaussian_blur | -27.063–-11.424 | 69.312–72.522 | 44.000–53.000 |


Clean@same-state control uses clean other-client images passed through state-t while the target remains shift-t passed through state-t. Only prototype calibration context changes. It is a diagnostic fixed policy, never a per-client prototype selector. Client comparisons aggregate each client's two held-out halves before counting better/equal/worse. Salts/halves reuse the same support and should not be interpreted as independent resampling trials. A zero baseline regret is not counted as a relative15% reduction.

## Execution and numerical validity

Frozen extraction: `20260914-183139-ttfl-t021-extract-gpu1`, runtime `e908116500c753ce7a7db4868776928934ba9f35`, 1807GPU support-batch forwards, max historical logit difference 0.0. Canonical H is512-D fc1/ReLU output; L is the exact10-D final linear output. No preprocessing/normalization changes. Model and fast-state hashes and support/calibration/query disjointness pass.

Phase-A: `20260914-183528-ttfl-t021-phase-a`, runtime `1849f39b3feb5f8bb8f5b2240b392c17c2385944`,4000 prototype cells,6000 actual CLS solutions and48000 exact state/argmax choices. The other99clients' labels calibrate each target's prototypes; target-i labels contribute zero to its own M or mean. Complete actual choices are frozen before target composition scoring. Matched stage: `20260914-183911-ttfl-t021-matched`, runtime `18ea37d77c30763cba49f671a941c5c9946532dc`,256000 certified solutions/2048000 choices, exactly128 draws preserving each target's20-label counts. Namespace is full big-endian SHA256(`T021|matched|representation|client|bank|context|replica`) into PCG64.

Both L/H prototype matrices have rank ranges [10, 10]/[10, 10]; condition ranges L=[861.896203599172, 3929.6110033156256], H=[170.05307555056856, 347.64115299537224] are diagnostic only. The existing solver forms a10×10 Gram matrix; no512×512 covariance is built. Maximum actual KKT: L=1.59872e-14, H=9.09495e-13; maximum matched KKT: L=1.07137e-14, H=6.8523e-13. Original acceptance tolerances hold.

Actual solves used zero fallbacks. Matched solves used9 cycle-only exhaustive-face fallbacks, all in L, listed in `matched/solver_fallbacks.json`. Each was independently re-enumerated with zero reference pi/objective difference; no fallback was added for any other error.

## Independent verification and artifacts

The separate verifier calls the original model forward with a pre-hook on the final classifier rather than the extractor helper: **2000 historical samples**, max logit and H difference0, same argmax, unchanged model/states. It reconstructs280 target-excluded prototype cells spanning both representations, all6000 target means and48000 Phase-A states/argmax masks; replays4000 matched CLS solutions with pi difference0 and32000 sampled template choices; verifies every matched draw's exact class counts/target exclusion and all9 fallbacks.

It reconstructs40000 exact true-template regrets,200 actual integer-count rows,10240 matched aggregate count rows, every reported gate, and all client paired counts. All runtime input hashes remain unchanged. Local cross-task manifests add **892 unchanged upstream entries**. Independent verification adds100 GPU audit forwards; these are replay only, with no new query inference. Scoring adds zero model forwards.

The inherited dataset is the existing PFLlib CIFAR10 client split (44961 train /15039 query after merged-source partitioning),100clients and historical10% participation checkpoint. It is not the official10000-example CIFAR10 test benchmark. No FedAvg retraining or new federation run occurred.

`extraction/`, `phase_a/`, `matched/` retain compact stage receipts; root contains final aggregate/paired/count tables, gates and independent verification. Large H/L, prototype, mean/pi/choice and matched NPZ arrays remain in the five canonical remote project run directories, with exact paths/sizes/hashes in `remote_artifact_manifest.json`; they are not claimed to have been copied into the local compact bundle. Logs and compact raw receipts are under `research_log/t021_receipts`. This keeps durable project evidence while avoiding another local D: space failure.

Return T021-M/CTX+ to Lead for the next bounded research decision. Do not infer authorization for whitening, learned features/heads, SSL writing or federation.
