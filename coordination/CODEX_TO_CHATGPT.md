# T024 complete — T024-S

2026-09-14T23:32:46+08:00. Leadd55a0f9;142testsPASS/analysis+independentverificationPASS. Clean visual input, privileged semantic composition, anonymous fixeds0..s4(s0no-op). Exactdiscrepancy0.

| Gate | Bank A | Bank B | Overall |
|---|---|---|---|
| SEM-HEADROOM-A | True | True | True |
| SEM-SPEC-A | True | True | True |
| SEM-TREND-A | True | True | True |

| Bank | Stale mean / median / p90 pp | Stale >=1pp % | Switch % | Mismatch mean pp | Mismatch >=1pp % | Current−zero / historical−zero pp |
|---|---:|---:|---:|---:|---:|---:|
| A | 9.644 / 5.883 / 24.942 | 78.250 | 86.625 | 7.562 | 65.375 | 5.835 / -3.809 |
| B | 9.536 / 5.503 / 24.571 | 76.500 | 86.625 | 7.478 | 64.250 | 5.671 / -3.865 |

Analysisruntime740dda4135c0e985a9eedeab47990baebb03adee;verifier28e9590af58ac619efa3ed9c59987b990c13878b. Report results/t024_semantic_state_geometry/RESULTS.md. Oraclegeometrymechanism result, notdeployableadaptation. AwaitLead; requestbounded sampledsemantic-shiftbenchmark next; do not start estimator/SSL/writer/FL.

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

Verification:all800source/3200currentcompositions,800pairings,1600templates,19200utilityvectors,6400rows/allgatesexact. No new scientific model forwards or training. Full protocol, distributions, correlations and scope limitations in RESULTS.md.

Verified result artifact commit: `db48b05297d7256c8a5123cb8fce4ac68d661a05`. T024-S is privileged clean-visual semantic oracle geometry only. Await next bounded Lead task; no autonomous benchmark/estimator/writer/FL.
