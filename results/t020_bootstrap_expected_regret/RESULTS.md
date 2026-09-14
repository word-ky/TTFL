# T020 STOP — Monte Carlo instability before privileged evaluation

2026-09-14T15:54:10+08:00. Lead `5bb4d6e`; runtime `29d8443cf698014ec3f938e598e2633381941e77`; run `20260914-154818-ttfl-t020`, exit1 after 99.23s. **115 tests PASS; all 256,000 CLS bootstrap solutions satisfy unchanged invariants.**

## Decision

First128 versus all256 BER state agreement is **97.9500%**, below the frozen **99%** global requirement: **164/8000** template-slot decisions differ. This alone triggers the Lead's mandatory stop; the result does not depend on the additional predeclared per-bank/context cluster check. **BER-REGRET-A / BER-CAP-A / BER-SRC-A are NOT EXECUTED. No T020-R/C/F/X scientific diagnosis is assigned.**

| Context | Bank A agreement, % | Bank B agreement, % |
|---|---:|---:|
| clean | 97.750 | 98.500 |
| brightness_dark | 98.000 | 97.250 |
| contrast_low | 97.875 | 98.375 |
| gaussian_noise | 98.750 | 99.125 |
| gaussian_blur | 96.625 | 97.250 |


Differences cover 45 unique client IDs and 75 client×bank×context cells. The eight utility-template slots share support draws; they are not independent resampling trials. Exact per-slot mean/vote differences and both decisions are preserved.

Maximum per-state mean-utility difference is 0.0295579 (fractional utility units); maximum vote-frequency difference is 0.117188. Median full256 top-mean utility margin is 0.00221909 among changed slots versus 0.0390877 among unchanged slots. These are label-free diagnostics only; no claim about whether switching helps true regret or query accuracy is possible from this stopped run.

No R increase, seed retry, class stratification, confidence fallback, tolerance change or alternate state policy was applied. The original 256 draws are retained. Return to Lead for the next bounded decision; do not interpret MC instability as T020-F or permission to move to feature-level work.

## Label-blind implementation

Bootstrap positions 0..19 with replacement using full big-endian SHA256(`T020|client|bank|context|replica`) into PCG64. The preparation script lazily reads only T019 `probabilities` and label-free `actual_choices` array members, frozen soft channels and utility templates. It does not deserialize true support composition, true-template utilities or query counts. Upstream artifact hashing reads bytes solely for provenance. Target labels never enter positions, q, CLS or BER.

Utilities are computed from float64 prevalence as exact binary rationals and the existing rational templates. A shared denominator allows exact integer accumulation for first128/all256 means and expected regrets. Exact ties use the frozen candidate order; no rounding threshold is introduced. All five states' expected regrets share the same average-max term, so maximum mean utility exactly equals minimum expected regret. Both exact and floating mean receipts are saved.

The fixed-context dispatch helper was tested, but the source-context policy was never evaluated. The explicit T020 Section6 precomputed-context dispatch interpretation is recorded in `PROTOCOL_FREEZE.md`; it had no effect on this pre-context MC stop.

## Verification

- T019 posteriors replay byte-identically from frozen T015 logits; all 8,000 point-CLS choices replay the frozen T019/T018R2 baseline.
- 256,000 bootstrap CLS solves: max KKT 4.44089e-16, max simplex-sum error 4.44089e-16, maximum 14 active-set updates; objective dominance passes at unchanged tolerance.
- Local independent replay checks replica0 and replica255 positions and **2,000 CLS solutions**, max pi difference 2.49911e-13, objective difference 6.93889e-18.
- Independently average the exact binary-rational pi values, then evaluate the rational templates: all **8,000** first128/all256 mean-utility decisions agree with saved receipts. Recheck 16,000 replica argmax sets, all vote frequencies, BER/point expected-regret differences, and the global stability fraction.
- All 219 upstream manifest entries remain unchanged. Zero model/query forwards. No true-regret numerator or final query-count vector was computed because the stability condition failed.

`phaseB_choices_freeze.json` hashes all unlabeled candidate decisions/statistics. `bootstrap_stability.csv`, `bootstrap_uncertainty.csv`, `ber_choices.csv`, exact mean-utility gzip, seed receipts and the independent verification are tracked. Full position/q/pi/invariant/replica-mask arrays remain locally and remotely under `research_log/t020_receipts/20260914-154818-ttfl-t020/artifacts/t020_bootstrap_expected_regret/` (remote canonical run path). No scientific-table placeholders are fabricated.
