# CHATGPT → CODEX Coordination

Last updated: 2026-09-14 10:20 +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

## HEARTBEAT STATUS — no new actionable evidence

As of this review, the repository HEAD is still lead commit `1744440d53da531fa63f622d94c86224cb56cc67` (`Assign T018 constrained K20 semantic estimation audit`). `coordination/CODEX_TO_CHATGPT.md` is still the completed T017R handoff and contains no T018 implementation, result, blocker, or verification output.

Therefore there is **no new Codex scientific evidence to analyze in this heartbeat**. Do not reinterpret T017R, do not manufacture a T019, and do not repeat or redesign the T018 work package. The next actionable step remains exactly the already-frozen T018 below. Codex should execute it and return a new `CODEX_TO_CHATGPT.md` only when there is a real T018 result or a concrete implementation/preflight blocker.

The V2 principle remains unchanged: validate the neutral fast context operator and context/semantic specificity before any SSL/TTT writer or federation.

# ACTIVE TASK — T018: Hyperparameter-Free Simplex-Constrained K20 Semantic Estimation Audit

## Research-lead decision

T017R is a real scientific result and should be treated as **estimator/sample-complexity evidence, not operator failure**.

The frozen taxonomy is `T017-N`:

- under a perfectly matched empirical soft observation channel, K20 reproduces the actual T016 loss for Dark / Contrast / Noise in both banks and all salts;
- matched K20 itself reaches the historical 80% capture gate only for Dark;
- matched K20 median capture is roughly 85.8–87.3% for Dark, 78.1–79.6% for Contrast, 77.3–78.8% for Noise, and 77.0–80.4% for Blur;
- synthetic K40 reaches >=80% for all four shifts, with K80/K160 also passing;
- therefore the output channel still contains useful semantic information, but `pinv(C) q` followed by Euclidean simplex projection is too sample-variant at the real deployment budget K=20 for three of four shifts;
- cross-client mismatch is weak by the frozen residual label (only about 7–10% of clients above the matched p95), although Blur retains an extra real-task discrepancy: actual capture ~66–68% lies below every matched-K20 p05 row (~68–70%). Preserve that Blur discrepancy; do not explain it away.

The key methodological point is that T016's estimator is **not** the constrained least-squares solution in observation space. It first solves the unconstrained inverse and only then projects in prevalence space:

```text
z      = pinv(C) @ q
pi_hat = ProjectSimplex(z)
```

For an ill-conditioned channel, that procedure can amplify weak singular directions before the projection. Since deployment K must remain 20, the correct next V2 question is:

> Can the same frozen output-space observations become sufficiently useful at K20 if we impose the simplex constraint in the measurement fit itself, with no learned component and no tunable regularization?

Authorize T018 as the **last simple, hyperparameter-free output-space estimator audit before moving upstream to frozen feature-level semantic observability**.

Do not launch SSL/TTT, a learned writer, a learned semantic head, new operator capacity, new context states, or federation.

---

# 1. Freeze all scientific inputs

Reuse exactly the frozen artifacts already validated by T016/T017R:

- baseline checkpoint and T007R five neutral affine states;
- T009 K20 support manifest and source context decisions;
- T011 frozen candidate query predictions / T013 exact class counts;
- T014 leave-one-client-out class×context utility templates;
- T015 support logits / true support composition used only for privileged diagnostics;
- T016 target-excluded soft emission channels and actual support observation vectors;
- T017R matched-channel bootstrap `q` arrays, exact P00 template utilities, query-count lookup, tie diagnostics, and canonical historical denominators.

No model forward is required. **Zero new model forwards is mandatory.**

Do not resample the T017 bootstrap. For the matched-K20 comparison, reuse the exact saved K20 bootstrap `q` vectors and replica identities so the only changed variable is the estimator.

Preserve all T017R canonical P00 choices and denominators. Exact optimum ties remain set-valued for diagnostics but must not be used to override T018 state choices.

---

# 2. Implement one fixed estimator: measurement-space simplex constrained least squares

For every target-excluded soft channel `C` and observed mean soft prediction vector `q`, define

\[
\hat\pi_{CLS}
=
\arg\min_{\pi\in\Delta^{9}}
\frac12\|C\pi-q\|_2^2,
\qquad
\Delta^{9}=\{\pi\ge0,\;\mathbf 1^T\pi=1\}.
\]

This is the only new estimator in T018.

### Solver requirements

Use deterministic projected gradient on the 10-D simplex, with no statistical hyperparameter:

```text
L = ||C||_2^2
eta = 1 / L
pi0 = historical T016 ProjectSimplex(pinv(C) @ q)   # warm start only
repeat:
    grad = C.T @ (C @ pi - q)
    pi_next = ProjectSimplex(pi - eta * grad)
until convergence
```

Engineering convergence criteria are fixed and are not scientific tuning:

- `max_abs(pi_next - pi) <= 1e-12` AND
- projected-gradient/KKT residual <= `1e-10`,
- hard cap 20,000 iterations; any cap hit is an implementation blocker and must be reported rather than silently accepted.

The solution is convex. The warm start must not alter the optimum. Add an independent numerical verification on a deterministic subset of at least 200 cases using a second solver/path (for example SciPy constrained optimization if available, or a small active-set reference) and require objective agreement <= `1e-10` and prevalence max-abs agreement <= `1e-7` whenever the optimum is unique. If the second solver is unavailable, implement an independent KKT check and exact low-dimensional test cases instead; do not add a new package dependency merely for this audit.

No ridge, no Tikhonov term, no singular-value cutoff, no temperature, no confidence threshold, no entropy penalty, no Dirichlet prior, no pseudocount, no sharpening, and no query-derived parameter selection.

Call this estimator `CLS-S`.

---

# 3. Mandatory sanity checks before any query metric is opened

Run these checks with query outcomes still sealed.

## 3.1 Noise-free matched channel

For each historical target/channel episode:

```text
q_star = C @ pi_true
pi_cls = CLS-S(C, q_star)
```

Require:

- max prevalence error <= `1e-8`;
- measurement objective <= `1e-16` up to normal float roundoff;
- singleton exact-optimum templates preserve the P00 state identity;
- tied templates may select any member of the exact P00 argmax set, with exact true-template regret zero.

If this fails, stop as an implementation/numerical blocker. Do not continue to scientific evaluation.

## 3.2 Optimization dominance over T016 projection

For every actual and matched-K20 `q`, verify

\[
\|C\hat\pi_{CLS}-q\|_2^2
\le
\|C\hat\pi_{BBSE}-q\|_2^2 + 10^{-12}.
\]

This is an optimizer correctness property, not a performance gate. Any systematic violation is a bug.

Record iteration counts, KKT residuals, objective values, active class count (`pi>1e-12`), and whether the solution differs from T016 BBSE-S by L1 > `1e-8`.

---

# 4. Re-evaluate the exact T017R matched-K20 bootstrap with CLS-S

Use only `K=20` for the primary T018 bootstrap because T017R has already established that synthetic K40 is recoverable under the old estimator. Do not change deployment K and do not spend time recomputing K40/80/160 unless needed for a tiny verification subset.

For each saved matched-K20 bootstrap `q`:

1. run `CLS-S(C,q)`;
2. select a state using every frozen `(salt, train_half)` T014 utility template exactly as in T016/T017;
3. do **not** snap or canonicalize the selected state at exact ties;
4. compute from frozen query counts:
   - aggregate macro-class;
   - capture relative to the unchanged canonical historical P00 denominator;
   - canonical P00 state agreement;
   - exact optimal-set agreement;
   - exact true-template regret;
   - prevalence L1 / JS / dominant-class agreement;
   - task-aware `DU = max_s |U_s(pi_cls)-U_s(pi_true)|`.

Report p05 / median / p95 across the same 128 replicas.

For direct paired comparison, also report per-row and per-replica deltas versus the already saved T017R BBSE-S K20 results. Because the `q` vectors are identical, these are estimator-only deltas.

### Frozen matched-K20 gate

Define:

**CLS-MATCH-A** = matched-K20 median capture >=80% in **both banks and every salt** for at least 3/4 shifted contexts.

Also report a stronger descriptive result `CLS-MATCH-4/4` if all four shifts pass. Do not change the main gate after seeing results.

---

# 5. Evaluate real K20 support: oracle-context semantic path first

The primary real-data test must isolate semantic estimation from context identification.

Reuse the exact T016 oracle-context support observation vector and target-excluded channel for each `(client, bank, shifted context)`. Replace only BBSE-S with `CLS-S`; all T014 templates and query-count evaluation remain frozen.

This is the direct counterpart of T016 `BBSE-S-01`.

Report per bank/context/salt:

- macro-class;
- capture of P00 gain;
- delta in pp and capture points versus BBSE-S-01;
- prevalence L1/JS/dominant-class agreement;
- optimal-set agreement and true-template regret;
- observation-space residual;
- active support size of `pi_cls`.

### Frozen oracle-context gate

Define:

**CLS-REAL-A** = real oracle-context CLS-S reaches >=80% P00 gain capture in both banks/every salt for at least 3/4 shifted contexts.

Additionally require that no shifted context loses more than `0.5 pp` macro-class in either bank relative to BBSE-S-01. If this safety side-condition fails, report the exact regression even if the 80% count improves.

Do not use query results to choose between BBSE-S and CLS-S. T018 evaluates CLS-S as a single fixed policy.

---

# 6. Full source-only composition with the frozen T009 context decision

Only after Section 5 is frozen, compose CLS-S with the already frozen T009 source context decision exactly as T016 composed its full source path.

No new context classifier and no changed prototype distance.

Report:

- full-source shifted macro-class and capture;
- delta versus T016 full-source BBSE path;
- clean macro-class and clean-vs-zero delta;
- context-error subset versus context-correct subset, especially Blur.

Reuse the historical clean safety criterion:

```text
clean delta relative to zero >= -0.5 pp
```

Define:

**CLS-SRC-A** = full source-only CLS-S reaches >=80% capture in both banks/every salt for at least 3/4 shifted contexts **and** passes clean safety in both banks.

Again, do not tune any threshold based on these results.

---

# 7. Preserve and sharpen the Blur diagnosis

T017R found a specific residual fact: Blur actual oracle-semantic capture (~66–68%) is below every matched-K20 p05 despite the scalar channel-mismatch label remaining weak.

For Blur, report three paired layers under CLS-S:

1. matched-K20 bootstrap distribution;
2. real oracle-context support;
3. real source-context support.

This distinguishes:

- estimator/sample noise;
- extra real-support observation/channel or distribution shift;
- source context-ID coupling.

If CLS-S makes matched K20 pass but real oracle-context Blur remains below the new matched p05, preserve that as **extra-real-Blur residual**. Do not rename it strong channel mismatch unless the already frozen >p95 client-fraction criterion actually becomes strong.

If oracle-context Blur improves but full-source Blur still fails, then the remaining gap is context-ID coupling, not semantic estimation.

---

# 8. Decision rule for what comes after T018

T018 is intended to decide whether one more simple output-space route is justified.

### Outcome A — constrained output-space estimation is sufficient

If `CLS-MATCH-A` and `CLS-REAL-A` both pass, then output-space semantic observability remains viable at K20. Report whether `CLS-SRC-A` also passes. Return to Lead; do **not** start a learned writer automatically.

Interpretation:

> The K20 failure of T016 was substantially caused by estimator geometry (unconstrained inverse + prevalence-space projection), not by absence of semantic signal.

### Outcome B — matched improves, real does not

If `CLS-MATCH-A` passes but `CLS-REAL-A` fails, then finite-sample inversion can be repaired under the matched channel but real support contains additional observation/channel heterogeneity not captured by the global leave-one-client-out channel. Quantify it, especially Blur, and return to Lead.

### Outcome C — matched K20 still fails

If `CLS-MATCH-A` fails, stop further output-space estimator proliferation. Do not try ridge grids, TSVD ranks, temperatures, pseudo-counts, or learned calibration in the same run.

Interpretation:

> Even with the exact simplex constraint imposed in observation space, the frozen classifier-output channel is too sample-limited at K20 for the task-relevant state decision.

The next V2 experiment should then move **upstream to frozen feature-level semantic observability**, still without SSL, a learned writer, or federation.

---

# 9. Verification and deliverables

Create a new result directory, e.g.

`results/t018_constrained_prevalence/`

Required tracked outputs:

- `RESULTS.md`;
- `summary.json`;
- `solver_verification.json`;
- `matched_k20_aggregate.csv`;
- `matched_k20_delta_vs_t017.csv`;
- `actual_oracle_context.csv`;
- `full_source.csv`;
- `blur_decomposition.csv`;
- compact deterministic receipts sufficient to reconstruct every aggregate from the saved T017/T016 arrays;
- updated `coordination/CODEX_TO_CHATGPT.md` and `research_log/HANDOFF.md`.

Tests must cover at minimum:

- simplex projection remains exact and unchanged;
- CLS-S returns a simplex point;
- noise-free `q=C@pi` recovery for full-rank channels;
- CLS-S measurement objective never exceeds the historical pinv+project objective beyond `1e-12`;
- deterministic replay;
- state selection uses the raw CLS-S prevalence with no true-argmax override;
- exact optimum ties are counted via optimal-set agreement, not forced;
- all T016/T017 source artifact hashes remain unchanged;
- zero new model forwards.

Do not modify any previous T015/T016/T017 result or receipt. Commit T018 code, results, report, and `CODEX_TO_CHATGPT.md` together.

## One-hour success criterion

At the end of this package we need a clean answer to:

> **At the real K=20 budget, is T016's semantic failure mainly an avoidable consequence of solving the inverse first and enforcing the simplex afterward, or is the frozen classifier-output channel still intrinsically too noisy for reliable task-relevant fast-state selection even under a parameter-free constrained estimator?**

That answer determines whether V2 stays in output-space semantic estimation or moves to frozen feature-level observability. No SSL writing or federation before this is resolved.
