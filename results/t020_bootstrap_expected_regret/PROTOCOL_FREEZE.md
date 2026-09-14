# T020 protocol frozen before bootstrap

Lead `5bb4d6e`. K20, R256, unchanged direct-RHS CLS, soft channels, states, templates, and candidate order. No new model forwards. The preparation process reads only T019 `probabilities` and `actual_choices` NPZ members, frozen soft calibration arrays and rational utility templates. It never reads target composition, true utilities or query counts. File hashing is read-only byte provenance, not outcome analysis.

Bootstrap 20 positions with replacement using full big-endian SHA256 integer of UTF-8 `T020|client|bank|context|replica` with PCG64. No class stratification or count preservation. Save all 256 position vectors, q/pi, invariants, replica canonical choices and exact argmax masks.

Compute utility numerators exactly from float64 prevalence as binary rationals and frozen rational template coefficients. Within a cell, use a shared binary denominator for all 256 replicas. Sum integer utility numerators for the first128/all256; mean utilities and expected regrets have exact rational receipts. Canonical ties are exact equal numerators, never tolerance-based. No true composition is needed. Save floating summaries alongside exact means. This implements mean utility = minimum expected regret without summation-roundoff-induced ties.

Global 128-vs-256 BER agreement must be >=99%. Operationalize the requested absence of a systematic bank/context cluster as >=99% in each bank×context group too; report all groups, per-slot indicators, vote differences and maximum utility-mean changes before opening labels. If the global or group check fails, stop; do not increase R, change seeds or inspect privileged effects.

Point choices must equal T019's frozen label-free point-CLS choice array (which replays T018R2). For a future source evaluation, follow Section6's explicit dispatch to the precomputed BER state at the frozen chosen-context index; do not re-estimate the context detector. This literal dispatch differs from re-bootstrapping the actual shifted input under another state and must be identified in any source-path report.

No label or uncertainty-based fallback, alternate policy, confidence threshold, or later phase before the stability condition passes. Preserve negative/stopped results and all upstream artifacts.
