# CODEX → CHATGPT: T016 preflight stop

2026-09-14T05:47:38+08:00. Lead `6476c99`; runtime `6fd685e543f13303d201cdb7c9b810a395696a18`; release `20260914-054356-ttfl-t016`; run `20260914-054411-ttfl-t016-cached`. Exit 1 at historical preflight. **CAL-SEM-A / CAL-SRC-A: NOT EXECUTED.**

## What ran

Implemented the fixed target-excluded hard confusion / soft emission channel, NumPy default pseudoinverse and standard Euclidean simplex projection. All **63 tests passed**, including six new tests covering target exclusion, nonzero/normalized columns, deterministic projection, known-confusion recovery, target-label isolation, source-context indexing and raw posterior behavior.

The historical T015 replay checked saved mixture/logit/choice/freeze hashes, reproduced all **56,000** utility vectors/argmaxes, and reconstructed **280** macro-class metrics from exact integer receipts. All **220 quality-summary rows** and **96 historical gate rows** match field-for-field. T015 remains SUPPORT-COMP-A PASS and SEM-EST-A / SEM-SRC-A FAIL.

## Observed implementation blocker

The newly added comparison at `scripts/eval_t016_confusion_prevalence.py:60` compares entire parsed CSV rows as strings. In `mixture_quality_episodes.csv`, exactly **2 of 4,000 rows** differ only in JS:

| zero-based row | historical JS | replay JS | absolute difference |
|---|---|---|---|
| 3186 | 0.28094428886389666 | 0.2809442888638962 | 4.440892098500626e-16 |
| 3187 | 0.28094428886389666 | 0.2809442888638962 | 4.440892098500626e-16 |

Every other field is identical. This scale is consistent with floating-point numerical/serialization differences between the original Windows report and Linux replay; the exact runtime cause is not independently isolated. The comparison is stricter than scientific equivalence of this descriptive floating metric. No accuracy, mixture-quality aggregate, gate, state choice, checkpoint or frozen logit change was found. This is an implementation check issue introduced in T016, **not evidence against T015 or the scientific method**.

## Stop and scope

Lead section 1 says: “If any historical reconstruction/hash/exclusion check fails, stop and report implementation blocker instead of producing scientific conclusions.” The job stopped there. No calibration matrices were built, no T016 source mixtures/choices were frozen, and no T016 target evaluation ran. The later full-data disjointness check was not reached; only existing T015 receipts and focused tests support that portion so far. No new support or query inference, model updates, state fitting, estimator tuning, or outcome-based retry occurred.

The first deployment attempt had an SSH timeout before job launch; the second deployment succeeded. There was **one formal execution**, not a scientific rerun. Its full log and intermediate historical replay artifacts are retained under `research_log/t016_receipts/20260914-054411-ttfl-t016-cached`. Historical `phaseA_freeze.json` inside the `historical_preflight` subdirectory belongs to **T015** and must not be mistaken for a T016 freeze.

## Minimal repair proposed for the next bounded instruction

Keep exact comparisons for IDs, counts, hashes, utilities, argmaxes, gates and quality aggregates. For the observed per-episode floating JS comparison, use a predeclared absolute float64 tolerance (for example 1e-12) instead of decimal-string equality, and retain an explicit maximum-error receipt. Do not change the estimator, thresholds, support, candidate states or scientific gates. No repair/rerun was applied after this stop.

## Scientific status

Channel identifiability/calibration, semantic prevalence quality, context-ID coupling and fast-state utility are **not evaluated in T016**. Neither CAL gate is a scientific FAIL. No operator failure or success can be inferred. Calibration statistics, prevalence-quality outputs, policy metrics and source-choice artifacts do not exist because their stages were not reached; they are not populated with placeholders or fabricated values.

Await Lead instruction after this implementation-blocker report. Do not proceed to feature prototypes, learned heads, SSL/TTT or federation.
