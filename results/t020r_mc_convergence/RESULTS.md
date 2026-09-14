# T020R implementation stop — reproducible CLS working-set cycles

2026-09-14T16:51:27+08:00. Lead `f0807dd`; runtime `0ec0480fc89f7dae48aa2e16a27bf5a87e1b49bf`; run `20260914-163857-ttfl-t020r`, exit1. **121 tests PASS, but the R4096 extension did not complete.**

The frozen `ActiveSetCLS` raises a working-set-cycle exception on deterministic replicas in clients64 and87. This is an observed solver limitation exposed by the longer stream. It is **not T020R-MC2**, because the complete convergence gate was never computed, and not T020-F or any scientific BER diagnosis. True support composition/utilities/query outcomes remain sealed. BER-REGRET-A, BER-CAP-A and BER-SRC-A remain NOT_EXECUTED.

## Reproduction

| Client | Bank | Context | Replica | Reference KKT |
|---:|---|---|---:|---:|
| 64 | A | contrast_low | 3071 | 3.11166e-17 |
| 87 | A | contrast_low | 2688 | 3.90313e-17 |


The local diagnostic replays only each missing client's stream until its first failure. Each case also fails with a fresh solver instance, so the failure does not require parallel scheduling or accumulated face-cache history. The recorded coordinate-removal/insertion trace revisits an identical active set; negative face coordinates are on the order of 1e-2, not a 1e-12 rounding violation. The independent exhaustive-face reference finds a feasible KKT-valid optimum for each same C/q. The convex CLS problem remains solvable; the current full-step face-removal/insertion procedure can cycle. Reference solutions are diagnostic only and were never substituted into a BER policy.

Exact C/q/positions/reference vectors and SHA256 are in the two `cycle_client_*.npz` diagnostic artifacts, retained locally/remotely with manifest hashes; human-readable traces and reference residuals are in `solver_cycle_diagnosis.json`. No solver code, tolerance, bootstrap seed, R, or policy was changed.

## Preserved partial computation

98/100 clients have complete per-client files (7840/8000 template slots), including 3,763,200 new CLS solutions plus their immutable first256 prefixes, or 4,014,080 logical replicas. These counts exclude partial work in failed clients and the later diagnostic replay. Preserved completed cells have max KKT 4.44089e-16, max sum error 4.44089e-16, maximum 17 updates. They do not establish that all 4,096,000 target solves passed.

The main process printed62 completed futures before the first exception. Python's executor shutdown waited for already-submitted work, which left98 complete client files on disk. Those files are preserved; no missing client was imputed or omitted to manufacture an agreement score. No complete phaseB choice freeze, full convergence table, or final uncertainty table exists. `partial_receipts_hashes.json` identifies partial evidence only.

Independent replay verified 3920 sampled CLS/position cases spanning all four quarters for the completed clients; max pi error 6.71456e-13, objective error 6.93889e-18. All 7840 completed-client legacy aggregate rows replay, with separately reconstructed binary-rational first128/all256 mean utilities and original replica canonical states/masks. Exact means, choices and other legacy fields remain exact; local Windows/Linux NumPy log differs by one ULP in diagnostic vote entropy, checked with absolute tolerance2e-15 only for this diagnostic. This changes neither solver nor policy tolerance. All 54880 saved block choices and 7840 partial ALL choices reconstruct from exact means and integer sums; quarter/half/ALL mean and vote composition agrees. All 245 upstream manifest entries remain unchanged. The complete4000-sample/8000-slot task verification is not claimed.

## Next bounded Lead decision

T020R explicitly freezes the T018R2 direct-RHS solver. Return the reproducible implementation blocker for a solver-specific repair instruction. A targeted next repair would need to remove the demonstrated active-set cycle while preserving the same convex objective/simplex, tolerances, seed stream and immutable first256 outputs, then replay these cases and the old baseline before resuming missing work. This report does not authorize or implement such a repair. No SSL/TTT, new semantic estimator/operator, context detector or FL work was started.
