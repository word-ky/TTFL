# T019 COMPLETE — T019-X

2026-09-14T15:05:51+08:00. Lead d37bbc4. Prep6b5b2c0/run20260914-145145-ttfl-t019;finish15ad29f/run20260914-145757-ttfl-t019-finish;both exit0.107testsPASS;zero newforwards. TASK-MISMATCH-A0/3FAIL;CTX-SPEC-A0/3FAIL;LOCAL-AallcontextsPASS. Contrast/Noise/Blur task tails2–11%,pairedtails12–19%;cleanunion11/17%. Small-margin state errors frequent but upperhalf margins carry62.02%true regret: mixed, not forced S. T017 K20 already fixed class counts; T019 adds decomposition/paired/repair diagnostics. Read results/t019_real_channel_heterogeneity/RESULTS.md and independent_verification.json. Await Lead; no nextstage.

| Context | Task tail A, % | Task tail B, % | Paired tail A, % | Paired tail B, % | Single-class half-gap recovery A / B, % |
|---|---:|---:|---:|---:|---|
| brightness_dark | 8.00–8.00 | 9.00–10.00 | 16.00–16.00 | 17.00–17.00 | 76.47–79.41 / 70.59–76.47 |
| contrast_low | 2.00–2.00 | 6.00–8.00 | 17.00–17.00 | 15.00–16.00 | 65.00–71.79 / 61.90–66.67 |
| gaussian_noise | 8.00–9.00 | 6.00–7.00 | 14.00–14.00 | 12.00–12.00 | 73.68–87.88 / 79.41–85.29 |
| gaussian_blur | 11.00–11.00 | 7.00–9.00 | 17.00–18.00 | 18.00–19.00 | 59.18–64.00 / 65.22–70.21 |

| Context | Actual capture, % | T019 null median capture, % | Actual below null p05 rows / 8 |
|---|---:|---:|---:|
| brightness_dark | 89.14–91.78 | 91.18–92.28 | 0 |
| contrast_low | 75.73–83.42 | 83.94–85.93 | 3 |
| gaussian_noise | 78.33–81.89 | 83.79–85.43 | 0 |
| gaussian_blur | 74.91–76.69 | 83.15–84.01 | 6 |

Recommendation: bounded margin/regret-cost + robust neutral-state selection audit, without another semantic estimator or learned writer. No next task started.
