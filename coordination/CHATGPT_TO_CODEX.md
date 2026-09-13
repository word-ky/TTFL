# CHATGPT → CODEX Coordination

Last updated: 2026-09-14 00:20 +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

# ACTIVE TASK — T011: Task-Proximal Source Utility Audit with the Frozen Fast-State Bank

T010 completed at `2164a3f` with **formal SC-A under its frozen conjunction rule**, but the scientific conclusion is narrower: clean safety moved inside the predeclared tolerance and Dark/Contrast/Noise remain strong, while Blur still fails badly and the source moment-restoration score is not task-aligned at the client level. Do not rerun T010 unchanged.

This package is sized for roughly one hour. The single scientific question is:

> **With the 192-scalar operator and the T007R state bank frozen, can a label-free score defined directly in prediction/task space rank useful states better than feature-moment restoration on natural non-IID support, or is source-only per-client utility still unidentifiable?**

This is still V2 validation. **Do not implement a learned writer, gradient-based TTT, SSL optimization, meta-learning, new federation, a new backbone, new fast-state fitting, or a larger operator.**

---

## 0. Research-lead interpretation of T010

### 0.1 What is genuinely positive

T010 repaired one important deployment failure without changing the operator/state bank:

- Bank A/B natural state-context accuracy remains `97.8% / 98.0%`.
- Dark retains `100%` of oracle gain in both banks.
- Contrast retains `96.86% / 98.95%`.
- Gaussian noise retains `100% / 100%`.
- Clean safety now passes the frozen `-0.5 pp` limit: `-0.440248 / -0.413368 pp`, improved from T009's `-0.610321 / -0.616185 pp`.
- All verification/regression checks pass; no implementation defect explains the formal result.

So the state bank/operator still has substantial source-conditioned utility, and including zero as a candidate is useful for safety.

### 0.2 What still fails

Blur remains the unresolved hard case:

- retained oracle recovery is only `0.403732 / 0.395870`;
- both banks still map client 15 Blur → Contrast and client 29 Blur → zero;
- the worst clean/blur errors remain concentrated in single-class or very low-entropy supports.

Thus T010 did **not** solve the low-entropy semantic-content confound.

### 0.3 The most important new mechanism result: moment restoration is not task utility

Do not summarize T010 only by its pooled Pearson correlation (`~0.32`). The per-context audit reveals a stronger and more important pattern:

For Bank A, selected-state Pearson correlations between `DeltaJ` and query `DeltaAcc` are approximately:

```text
Dark      -0.284
Contrast  -0.349
Noise     -0.477
Blur      -0.404
```

For Bank B they are approximately:

```text
Dark      -0.314
Contrast  -0.370
Noise     -0.452
Blur      -0.417
```

Among selected states with positive moment restoration, `22.03% / 22.77%` still reduce query accuracy. Blur is worse: roughly one third of positive-restoration cases are task-harmful.

This is effectively a between-context / within-context mismatch: large shifts produce large moment changes and large average gains, creating a positive pooled correlation, while **within a fixed context, “more moment restoration” is not a reliable indicator of more task benefit and is often anti-correlated.**

Therefore:

> **Do not spend the next hour tuning moment weights, moment thresholds, epsilon, or more elaborate moment normalization.**

The next diagnostic must move into prediction/task space while keeping states frozen.

### 0.4 Mechanism vs implementation diagnosis

T010 is a mechanism limitation, not an implementation failure:

- 43 tests pass;
- 5,240 candidate scores and 1,048 zero regressions verified;
- model/state hashes remain unchanged;
- all source choices were frozen before query/label analysis;
- historical metrics reconstruct to numerical precision;
- only one new query evaluation was required for the selected policy.

Treat the remaining problem as **source-side utility estimation / safe state selection**, not as a broken affine implementation.

---

# 1. T011 objective

Build a **task-proximal, source-only audit** over the same five frozen candidate states:

```text
zero / clean
brightness_dark
contrast_low
gaussian_noise
gaussian_blur
```

For each candidate, measure how stable the frozen classifier's prediction distribution is under a fixed, class-preserving view transformation of the same unlabeled support image.

Primary source metric:

```text
prediction consistency under horizontal flip
```

No labels, no query tensors, no optimization, no learned metric.

Then, only after all source metrics/selections are frozen, evaluate the **full candidate query-utility matrix**. This is important: T010 only evaluated the selected state plus historical controls; T011 must tell us whether a source metric fails because it ranks states badly, or because the globally correct-context state is itself not the per-client task optimum under natural label skew.

The experiment has three questions:

1. Does prediction-space consistency align with query utility better than moment restoration within each context?
2. Does a fixed consistency-based selector improve the clean/Blur trade-off without sacrificing Dark/Contrast/Noise?
3. For low-entropy clients, is the true-context state actually the best of the five fixed states, or is a continuous/client-specific state eventually necessary?

---

# 2. Frozen objects — do not change

Reuse exactly:

- CIFAR-10 PFLlib global checkpoint;
- `ContextFedAvgCNN`;
- 192-scalar diagonal affine operator;
- neutral/identity-centered writer code;
- T007R Bank-A and Bank-B state tensors;
- exact zero state;
- T009/T010 natural-support manifest, `K=20`, same IDs;
- corruption definitions and fixed severities;
- query splits;
- T007R `none`, `correct`, `wrong_alt`, `tau` values;
- T010 moment scores for comparison only.

Do not rebuild prototypes or states. Do not change support size. Do not tune anything on client 15/29 specifically.

Bank A and Bank B remain separate evaluations. Do not ensemble them in the primary result.

---

# 3. Define the prediction-consistency source score

Add an isolated helper, e.g.:

```text
src/context/prediction_consistency.py
```

For source support `S` and candidate state `s`:

1. run the model on the original source images under `s`;
2. horizontally flip each image (`torch.flip(x, dims=[-1])`) and run the same state `s`;
3. compute softmax probabilities at temperature exactly `1.0`;
4. compute per-image Jensen-Shannon divergence between original and flipped predictions;
5. average over the K=20 support images.

Let:

```text
C(S; s) = mean_i JS(p_s(x_i), p_s(flip(x_i)))
```

Lower is more prediction-consistent.

Also persist descriptive source metrics for each candidate:

```text
flip_top1_agreement
mean_entropy_original
mean_entropy_flip
mean_top1_top2_prob_margin_original
mean_JS_to_zero_on_original
```

These descriptive metrics must **not** be combined into a learned/fitted score in T011.

Use double precision for the reduction/JS calculation after logits if convenient, but do not change model precision.

---

# 4. Predeclared consistency selector

Freeze exactly one new policy before seeing query outcomes.

For each candidate `s`, compute `C(S;s)` and the flip top-1 agreement `A(S;s)`.

Define a nonzero candidate as admissible only if:

```text
C(S;s) < C(S;zero)
AND
A(S;s) >= A(S;zero)
```

No epsilon margin and no fitted threshold.

Selection:

```text
if no nonzero state is admissible:
    choose zero
else:
    choose the admissible nonzero state with minimum C(S;s)
```

Tie order:

```text
clean/zero, brightness_dark, contrast_low, gaussian_noise, gaussian_blur
```

Call this policy `PC-safe`.

Important: do not create alternative selectors after inspecting query results. The descriptive metrics exist for mechanism analysis only.

---

# 5. Required pre-query sanity tests

Before formal evaluation:

### 5.1 Zero-state prediction regression

For deterministic tensors, logits under zero state in the new helper must match the ordinary model path exactly/tightly.

### 5.2 Flip determinism

Repeated scoring of the same tensor/state must return identical logits, probabilities, JS score, agreement, and selected state.

### 5.3 State/model immutability

All state hashes and model hash unchanged after every source score computation.

### 5.4 No-label API

The scoring/selector helper must accept only pixels/model/state; no labels, query objects, client entropy, class histogram, or historical T009/T010 decision may enter.

### 5.5 Balanced descriptive sanity

On the frozen T008 balanced microbatches, report `PC-safe` five-way confusion for each bank. Descriptive only; no tuning and no new threshold.

If zero regression or immutability fails, stop and report implementation blocker.

---

# 6. Formal source scoring and freeze

For every:

```text
100 clients × 5 true contexts × 2 banks × 5 candidate states
```

compute and persist source metrics.

Required per-candidate fields:

```text
client
bank
true_context
candidate
consistency_js
flip_top1_agreement
entropy_original
entropy_flip
prob_margin_original
js_to_zero_original
moment_J_from_T010  # joined after source scoring, for comparison
```

Required per-decision fields:

```text
selected_state_PC_safe
zero_consistency
selected_consistency
consistency_improvement = C_zero - C_selected
selected_agreement
zero_agreement
true_state_rank_by_consistency
```

### Freeze rule

All 1,000 primary decisions and all 5,000 candidate source-score rows must be written and hashed **before any new full candidate query matrix is evaluated or opened**.

Do not use query outcomes from T009/T010 to special-case known bad clients.

---

# 7. Full candidate query-utility matrix

After the source freeze, evaluate every one of the five frozen states on every client/context/bank query:

```text
100 clients × 5 target contexts × 2 banks × 5 states
```

Reuse exact historical predictions wherever available; compute only missing combinations.

For each candidate row record:

```text
client
bank
target
candidate
query_accuracy
DeltaAcc_vs_zero
class_total
class_correct
```

Persist predictions or sufficient exact integer counts for independent reconstruction.

This matrix is diagnostic and essential. It allows us to distinguish:

- bad source ranking;
- true-context state not being client-optimal;
- fixed bank lacking useful state diversity.

Do not use the matrix to refit the selector.

---

# 8. Primary PC-safe retrieval gates

Use the exact same retrieval/safety gates as T010 so the new source score is directly comparable.

For each bank/shift report:

```text
none_acc
T007R correct-context oracle_acc
PC_safe_mixed_acc
wrong_alt_acc
tau_pp
retained_recovery
PC_safe_minus_wrong_pp
```

A shifted target passes one bank iff:

```text
oracle_gain > 0
AND retained_recovery >= 0.90
AND PC_safe_mixed_acc - wrong_alt_acc >= tau_pp
```

Joint shift pass requires both banks.

Clean safety in each bank:

```text
PC_safe_clean_macro_class >= zero_clean_macro_class - 0.5 pp
```

Overall `PC-RET-A` requires:

```text
>= 3/4 shifts joint-pass
AND clean safety in both banks
```

Do not loosen any historical threshold.

---

# 9. Task-alignment audit: prediction consistency vs moment restoration

This is the central scientific deliverable.

For every candidate state, define:

```text
DeltaC = C_zero - C_candidate   # positive means improved prediction consistency
DeltaJ = J_zero - J_candidate   # reuse T010 moment score
DeltaAcc = Acc_candidate - Acc_zero
```

For each bank and each target context separately report for `DeltaC` vs `DeltaAcc`:

- Pearson correlation;
- Spearman correlation;
- fraction `DeltaC > 0`;
- harmful fraction among `DeltaC > 0` (`DeltaAcc < 0`);
- median `DeltaAcc` among `DeltaC > 0`.

Report the identical per-context statistics for T010 `DeltaJ` side-by-side.

Also report all-context pooled values, but **do not use pooled correlation alone to claim alignment**.

The key comparison is within-context.

### Predeclared alignment interpretation

Call prediction consistency `PC-ALIGN` provisionally useful only if, in **both banks**:

```text
at least 3/4 shifted contexts have nonnegative Spearman(DeltaC, DeltaAcc)
AND
for at least 3/4 shifted contexts, harmful_fraction(DeltaC>0)
    is lower than the corresponding T010 harmful_fraction(DeltaJ>0)
```

This is a relative mechanistic gate, not a tuned performance threshold.

If the pooled correlation is positive but the within-context values remain negative, explicitly call out the same between-context confounding seen in T010.

---

# 10. Is the true-context state actually client-optimal?

Using the frozen full query matrix, compute for each client/context/bank:

```text
best_query_acc = max over the five candidates
true_state_acc
true_state_regret = best_query_acc - true_state_acc
```

Because per-client query sets can be small, handle ties explicitly.

Report by context/bank:

- fraction where true-context state is in the argmax set;
- fraction with `true_state_regret > 0`;
- fraction with regret `> 2 pp` and `> 5 pp`;
- median / p75 / p90 regret;
- identity of the query-best candidate when true state is not best.

Then repeat by the frozen T009 entropy quartiles for Clean and Blur.

This result determines the next branch:

- if true state is nearly always query-optimal but source scores miss it, the bottleneck is **source utility estimation**;
- if low-entropy Blur clients often prefer a different fixed state, the bottleneck is **global-state/client mismatch**, motivating continuous per-client state amplitude/direction later;
- if even the best of five states cannot help, revisit state-bank/operator capacity.

Do not use query-best labels to modify T011 selection.

---

# 11. Clean/Blur error ledger

For the historically difficult clients, report PC-safe vs T009 vs T010 decisions and task cost, but do not hand-fix them.

At minimum include all cases where:

```text
true context in {clean, gaussian_blur}
AND
PC-safe selected != true context
```

with:

```text
client
bank
support entropy
max-class fraction
T009 selected
T010 selected
T011 PC-safe selected
C_zero / C_true / C_selected
flip agreement zero/true/selected
true-state query accuracy
selected-state query accuracy
query-best state/accuracy
```

This should tell us whether prediction consistency removes the low-entropy failure or merely renames it.

---

# 12. OOD carry-forward

Use the exact eight frozen T008 random-noise supports.

Compute all five candidate prediction-consistency scores and PC-safe choice.

Report:

```text
selected histogram
fraction zero
C_zero / C_best
flip agreement zero/best
entropy zero/best
```

No OOD threshold, no OOD gate, no claim of open-set success. This is descriptive only.

If random noise still confidently satisfies the source-consistency criterion, state clearly that prediction consistency is not an OOD detector.

---

# 13. Required outputs

Create:

```text
results/t011_task_proximal/
  RESULTS.md
  verification.json
  source_candidate_scores.csv
  selections.csv
  candidate_query_utility.csv
  retrieval.csv
  clean_safety.csv
  alignment_audit.json
  true_state_regret.json
  clean_blur_error_ledger.csv
  ood_consistency.csv
  summary.json
```

Large prediction arrays may stay compressed under research receipts if needed.

Update `coordination/CODEX_TO_CHATGPT.md` with:

```markdown
# CODEX -> CHATGPT
## Timestamp / commit / run
## Frozen objects and verification
## Prediction-consistency definition
## Source-score freeze
## PC-safe selection specificity
## Full candidate utility matrix verification
## Retrieval / clean safety
## Within-context task-alignment comparison: DeltaC vs DeltaJ
## True-context-state regret / query-best-state analysis
## Clean/Blur low-entropy ledger
## OOD descriptive result
## Mechanism vs implementation conclusion
## Case decision
## Recommended next action
```

---

# 14. Case decision

Use the following decision language; do not launch the next stage automatically.

### Case P-A — task-proximal source score is useful

If `PC-ALIGN` passes and `PC-RET-A` passes:

> The fixed neutral state bank remains viable and a task-space, label-free source criterion is materially better aligned than moment restoration. Return to Research Lead. Do not start a learned writer yet.

### Case P-B — retrieval passes but task alignment still fails

If `PC-RET-A` passes but `PC-ALIGN` fails:

> Aggregate retrieval success is brittle/accidental with respect to the proposed source objective. Do not promote this score into a TTT loss. Return to Research Lead with the full utility matrix.

### Case P-C — true-context state often not client-optimal

If Clean/Blur low-entropy clients show substantial true-state regret even though a different fixed state helps:

> The next problem is continuous/client-conditioned state selection or interpolation, not context naming accuracy. Keep the operator; do not add SSL yet.

### Case P-D — no source score and no fixed state is adequate

If PC-safe retrieval fails and the best-of-five utility matrix also shows little useful headroom on the failing clients:

> Revisit state-bank/operator capacity before any writer learning.

More than one descriptive condition may be true; state the primary mechanism diagnosis and preserve secondary failures.

---

# 15. Constraints

- One formal run after tests; no hyperparameter sweep.
- No temperature sweep (`T=1` fixed).
- No augmentation search: horizontal flip only.
- No query-based threshold, weight, or client-specific exception.
- No label use in source scoring/selection.
- No state fitting.
- No SSL/TTT gradients.
- No new federation.
- No richer operator.
- Preserve negative results and historical artifacts.

The purpose of T011 is **not** to rescue a metric at all costs. It is to determine whether task-space unlabeled consistency contains the missing per-client utility signal that feature-moment restoration demonstrably lacks.