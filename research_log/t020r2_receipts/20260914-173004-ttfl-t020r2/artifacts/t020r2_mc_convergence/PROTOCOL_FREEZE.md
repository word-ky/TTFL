# T020R2 cycle-only solver repair

Lead authorization: `43d2f85`. Parent runtime `0ec0480fc89f7dae48aa2e16a27bf5a87e1b49bf`; preserved run `20260914-163857-ttfl-t020r`.

Only the repeated-active-set branch changes: call the existing exhaustive `face_reference`, enumerate all 1023 nonempty faces in ascending binary-mask order with direct-RHS KKT solves, choose the feasible candidate with strictly minimum objective, and certify the full returned vector. No clipping or renormalization is added. Finite values, sum error<=1e-12, min>=-1e-12, direct KKT<=1e-10 and objective<=initial+1e-12 are mandatory. Other exceptions remain exceptions. Normal arithmetic is unchanged; receipts add `solver_path=active_set`. Cycle receipts record `cycle_face_fallback`, repeated set, trace, selected face and face count.

Reuse map: existing ActiveSetCLS owns normal solves; existing face_reference owns enumeration; existing direct_kkt owns acceptance; existing T020R owns bootstrap positions, prefix reuse and exact aggregation. Add only the cycle dispatch/certification, observed-cycle fixtures/tests, required regression, and missing-client resume. No new estimator or dependency.

Pre-resume checks: the two observed cycles; full1000 real observations; >=4000 non-cycle replicas across quarters; normal solutions/receipts identical to frozen implementation; preserved98 hashes unchanged. Full remote unit suite precedes resume.

Reuse exactly98 complete client accumulator/sample pairs after hash verification. Compute only clients64/87, preserving the immutable first256 and appending the same T020 namespace replicas256..4095. List every cycle fallback. Require exactly100 client IDs,1000 cells and8000 unique template slots before full aggregation. No omitted or substituted client.

R_FINAL4096, K20, channels, templates, states, seed namespace, exact tie rules and scientific gates remain unchanged. H1/H2>=99% global, H1/ALL andH2/ALL>=99.5%, every bank/context H1/H2>=98.5%, every pooled context H1/H2>=99%. Labels/true utilities/query remain sealed until full convergence PASS and choice freeze. Failure is T020R-MC2 and stops the task. No automatic R/seed/policy change or subsequent research stage.

Original T020R results remain immutable. New repair/results use `results/t020r2_mc_convergence` and a separate run receipt. Frozen solver source is copied verbatim from commit528672c for observational regression; its hash is recorded in the test receipt.
