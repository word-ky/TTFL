# T018R2: verified exact CLS, Outcome B

2026-09-14T13:38:04+08:00. Lead `c8aea5b`; frozen science `1744440`. Runtime `e4b83f09ea15357993d4d3fcb6562c4c45104b65`. Preflight `20260914-132551-ttfl-t018r2-preflight` and scientific run `20260914-132634-ttfl-t018r2-science` both exit0. Science took 84.08s. **100 tests PASS; zero new model forwards; no resampling.**

## Decision

**CLS-MATCH-A PASS (4/4); CLS-REAL-A FAIL (1/4); CLS-SRC-A FAIL (1/4): Outcome B.** Real regression safety and source clean safety both pass. The matched empirical K20 channel becomes task-useful when the simplex constraint is enforced in the measurement fit. This improvement does not transfer sufficiently to real support: output-space CLS alone has not solved real semantic estimation. This supports a remaining real-support/channel heterogeneity hypothesis, not a claim that the affine operator failed. No next phase has started.

Ranges below span both banks and all four salts. Passing a context requires every one of its eight bank×salt rows to reach 80%; no average-row substitution.

| Context | Matched median capture, % | Real oracle capture, % | Source capture, % | Matched / real / source |
|---|---:|---:|---:|---|
| brightness_dark | 90.783–92.351 | 89.138–91.783 | 89.138–91.783 | True / True / True |
| contrast_low | 83.432–85.872 | 75.729–83.422 | 75.729–83.422 | True / False / False |
| gaussian_noise | 84.478–85.748 | 78.334–81.889 | 78.334–81.889 | True / False / False |
| gaussian_blur | 82.684–84.908 | 74.911–76.690 | 57.308–62.253 | True / False / False |


## Paired estimator changes

Same q, checkpoint, states, context decisions, template lookups and P00 denominator; only the optimizer now realizes the unchanged CLS objective. Matched values are p50 of paired per-replica accuracy differences, not differences between independently resampled runs.

| Context | Matched paired median Δmacro, pp | Real Δmacro vs BBSE, pp | Source Δmacro vs BBSE, pp |
|---|---:|---:|---:|
| clean | 0.324–0.376 | 0.599–0.819 | 0.420–0.734 |
| brightness_dark | 0.750–0.915 | 0.180–0.673 | 0.180–0.673 |
| contrast_low | 0.657–0.970 | 0.368–1.547 | 0.368–1.547 |
| gaussian_noise | 0.378–0.446 | 0.772–0.892 | 0.772–0.892 |
| gaussian_blur | 0.368–0.501 | 0.609–0.970 | 0.747–1.752 |


The real oracle path has no shifted regression beyond 0.5pp. Source clean gain over zero is 4.162–4.668pp, passing -0.5pp safety. Each complete bank×context×salt row, its accuracy/capture and semantic diagnostics are in the CSVs; matched tables also include p05/p95 and paired L1, JS, dominant agreement, DU, exact optimal-set/canonical agreement and true-template regret.

| Context | Matched median L1 | Real L1 | Real dominant agreement, % |
|---|---:|---:|---:|
| clean | 0.666–0.667 | 0.718–0.718 | 70.000–70.000 |
| brightness_dark | 0.729–0.739 | 0.768–0.768 | 68.000–68.000 |
| contrast_low | 0.821–0.824 | 0.918–0.918 | 58.000–59.000 |
| gaussian_noise | 0.688–0.690 | 0.742–0.743 | 67.000–68.000 |
| gaussian_blur | 0.686–0.693 | 0.769–0.772 | 68.000–68.000 |


## Blur decomposition

Matched Blur median capture is 82.684–84.908%; its p05 is 73.445–76.765%. Real oracle Blur capture is 74.911–76.690%, and source Blur is 57.308–62.253%. Real oracle is below the matched p05 in 1/8 rows. Source composition adds -1.650–-1.255pp relative to oracle.

The old 8/8 below-p05 finding narrows to **1/8**, so the **extra-real-Blur residual** is now a row-specific observation, not a uniform anomaly across both banks and every salt. Distinguish this from the additional context-ID composition gap. The old T017 mismatch label was weak and was not redefined here; these task gaps alone do not establish a strong channel-mismatch label. Source context-correct/error subset sample counts and weighted accuracies are saved separately.

## Numerical certification, before metrics

T018R2 changes only the cached face application: cache KKT matrices; use `numpy.linalg.solve(K, actual_rhs)` on each visited face. All working-set logic and tolerances remain unchanged. No unconditional normalization, new dependency or new estimator was added. Old PGD and cached-inverse stops remain preserved in their receipts/history.

- Frozen blocker client35/B/Dark/replica7 passes with zero clipping; the full matched receipt explicitly records its repaired pi, KKT and sum error.
- Noise-free 1,000: max prevalence error 2.22412e-13, objective 6.13796e-30; 7,914 singleton and 86 exact-tie checks pass.
- Same 200/reference: max pi error 0; all 49 historical PGD caps remain repaired.
- All 1,000 real cells pass, max sum error 2.22045e-16, max KKT 2.22045e-16.
- **All 128,000 saved matched q solutions were certified and persisted before query counts opened**: max sum error 2.22045e-16, max KKT 2.22045e-16, max 13 updates. Arrays save pi, KKT, sum errors, update counts and objective improvements; p95/p99/max summaries are in `matched_solver_verification.json`.

Exact raw CLS prevalence selects the first rational-utility argmax without a true-label tie override. Oracle-context results are written and hashed before full-source composition.

## Independent replay and artifacts

Local independent replay reconstructs 5,120 matched class-count vectors from all 1,024,000 selected half-client contributions, checks all exact macro/capture fractions and paired baselines, all aggregate quantiles/deltas, 16,000 matched boundary-replica Fraction choices and all 16,000 actual Fraction choices plus 80 actual aggregate count vectors. All 85 upstream T015/T016/T017 manifest entries are byte-identical. The source q file hash is `108776cdab4d00e99076637353d1f19d0eccc7f6dee5eff8786ff058e5baf568`.

The science command's relative output directory initially placed artifacts under release `20260914-132535-ttfl-t018r2-preflight/artifacts/`; after successful completion they were copied unchanged into the canonical `20260914-132634-ttfl-t018r2-science/artifacts/` directory. No computation was repeated for this location correction. Raw copies remain remote and under `research_log/t018r2_receipts/`; large NPZ arrays are intentionally not in Git. Compact tables/JSON, hashes and this report are tracked.

Data limitation remains inherited: PFLlib 100-client/10%-participation baseline on the merged CIFAR train/test client split, not the official CIFAR-10 test benchmark. No new FL training occurred.

Return **Outcome B** to Lead. The next question is the matched-to-real observation/channel gap; do not interpret this as permission to start feature semantics, learned calibration, SSL/TTT or federation.
