# T018R2 COMPLETE — Outcome B

2026-09-14T13:38:04+08:00. Leadc8aea5b/runtimee4b83f09ea15357993d4d3fcb6562c4c45104b65; preflight20260914-132551-ttfl-t018r2-preflight, science20260914-132634-ttfl-t018r2-science, bothexit0.100testsPASS; all128000matched numerical solves PASS, maxsum2.22045e-16,maxKKT2.22045e-16. CLS-MATCH-A4/4PASS; CLS-REAL-A1/4FAIL; CLS-SRC-A1/4FAIL (Darkonly). BothsafetychecksPASS. Zero newforwards/noresampling. Read results/t018_constrained_prevalence/t018r2/RESULTS.md and independent_verification.json. Previousstops preserved. ReturntoLead; no nextstage.

| Context | Matched median capture, % | Real oracle capture, % | Source capture, % | Matched / real / source |
|---|---:|---:|---:|---|
| brightness_dark | 90.783–92.351 | 89.138–91.783 | 89.138–91.783 | True / True / True |
| contrast_low | 83.432–85.872 | 75.729–83.422 | 75.729–83.422 | True / False / False |
| gaussian_noise | 84.478–85.748 | 78.334–81.889 | 78.334–81.889 | True / False / False |
| gaussian_blur | 82.684–84.908 | 74.911–76.690 | 57.308–62.253 | True / False / False |

| Context | Matched paired median Δmacro, pp | Real Δmacro vs BBSE, pp | Source Δmacro vs BBSE, pp |
|---|---:|---:|---:|
| clean | 0.324–0.376 | 0.599–0.819 | 0.420–0.734 |
| brightness_dark | 0.750–0.915 | 0.180–0.673 | 0.180–0.673 |
| contrast_low | 0.657–0.970 | 0.368–1.547 | 0.368–1.547 |
| gaussian_noise | 0.378–0.446 | 0.772–0.892 | 0.772–0.892 |
| gaussian_blur | 0.368–0.501 | 0.609–0.970 | 0.747–1.752 |

Blur real below matchedp05: 1/8. Do not rename old weak mismatch label. Full compact receipts and paired diagnostics tracked; rawNPZ local/remote.
