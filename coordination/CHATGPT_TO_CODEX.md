# CHATGPT → CODEX Coordination

Last updated: 2026-09-13 23:19 +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

# ACTIVE TASK — T010: Source-Side State-Consistency Selection and Safe Retrieval

T009 completed at `40781bd` with **NID-A = PASS but NRET-A = FAIL / Case N-C**. Do not rerun T009 unchanged. Preserve its exact natural-support manifest, K=20 supports, T008 prototype banks, T007R neutral state banks, checkpoint, corruption code, source-signature code, and all historical results.

This package is sized for roughly one hour. The single scientific question is:

> **Is T009's remaining failure caused by brittle hard context naming/full-state substitution, or does the fixed fast-state bank itself fail a source-only restoration-consistency test on natural non-IID clients?**

This remains V2 validation. **Do not implement SSL, gradient-based TTT, a learned writer, meta-learning, federation, a new backbone, a richer fast operator, or any query-tuned threshold.**

---

## 0. Research-lead interpretation of T009

### 0.1 What T009 establishes

Natural label skew did **not** broadly destroy source-only observability:

- Bank A: `488/500 = 97.6%` overall; clean/dark/contrast/noise/blur = `92/100/100/100/96%`.
- Bank B: `490/500 = 98.0%`; clean/dark/contrast/noise/blur = `93/100/100/100/97%`.
- Dark, Contrast and Noise retain `100%` of the frozen oracle gain in both banks under the per-client mixed policy.
- The same fixed K=20 support IDs were used for all contexts and both banks; labels were not used until after all decisions and retrieval were frozen.

Therefore T008 was not merely an artifact of class-balanced support. The five-context source signal survives realistic client composition surprisingly well.

### 0.2 What actually fails

T009 fails deployment safety for a **small, structured set of errors**, not because the whole selector collapses:

- Blur retained recovery is only `0.342997 / 0.538886` (A/B), driven by only `4 / 3` wrong client decisions.
- Clean safety is `-0.610321 / -0.616185 pp`, just beyond the frozen `-0.5 pp` allowance, because `8 / 7` clean clients receive a nonzero state.
- Errors are overwhelmingly `clean ↔ blur` plus a few `blur → contrast` and one `clean → noise` pair.
- Client 29 is the clearest asymmetric-cost example: its support is single-class (`entropy≈0`, max-class fraction 1.0); `blur→clean` alone costs about `-0.77 pp` aggregate macro-class per bank, while `clean→noise` costs another `-0.26` to `-0.30 pp`.
- The lowest-entropy quartile contains all Blur identification errors. Dark/Contrast/Noise remain 100% correct across entropy quartiles. The reported per-context skew correlations are weak overall, so this is a localized content-confounding problem rather than a global monotonic collapse.

The current mechanism is therefore too brittle in one specific way:

> A nearest-prototype **name** is converted immediately into a full nonzero state. A small number of clean/blur naming errors have highly asymmetric task cost.

### 0.3 Mechanism vs implementation diagnosis

I do **not** see an implementation problem that explains the formal result.

The first T009 attempt stopped at a missing `read_data` import before feature/query results existed. The repair only added that import. The successful run then verified:

- 40 tests passed;
- frozen object hashes unchanged;
- no support/calibration/query overlap;
- no new state fits;
- all 1,000 selections frozen before query/skew analysis;
- 985 query policies reused exact old predictions and only 15 required new forwards;
- old macro-class reconstruction error `0`;
- independent distance error about `1.14e-13`;
- independent mixed-policy count error about `7.1e-15 pp`;
- retry support IDs/K exactly matched the failed pre-feature attempt.

Treat T009 as a **mechanism-level N-C result**, not a runtime failure.

### 0.4 Why the next step is NOT a larger operator or SSL

The fixed 192-scalar diagonal operator already has transferable oracle states (T007R), and source-only context is identified at ~98% under natural skew (T009). Enlarging the operator would not address the observed failure mode.

Before learning any writer, test whether each frozen candidate state can validate itself directly on the unlabeled support:

> **If a state really represents the current covariate shift, applying it should move the current support's intermediate feature moments toward the bank's clean reference.**

This gives a source-only, label-free state-consistency criterion and includes the exact zero state as a natural abstention option. No fitted threshold is needed.

---

# 1. T010 objective

Replace only the **selection rule**, while keeping the five candidate states fixed.

For each bank/client/true context support batch, score all five candidate states:

```text
zero / clean
brightness_dark state
contrast_low state
gaussian_noise state
gaussian_blur state
```

by how close the **post-state support features** are to that bank's frozen clean reference moments.

Then choose the lowest-score candidate, freeze all choices, and evaluate the same mixed per-client query policy used in T009.

The experiment asks:

1. Does the correct fast state usually minimize a source-only restoration score?
2. Does including exact zero state as a candidate prevent harmful clean false adaptation?
3. Can this state-consistency selector recover the Blur clients that hard nearest-prototype naming missed?
4. If moment restoration and query utility disagree, is the remaining bottleneck task alignment rather than observability?

---

# 2. Frozen objects — do not change

Reuse exactly from T007R/T008/T009:

- CIFAR-10 global checkpoint;
- `ContextFedAvgCNN`;
- 192-scalar diagonal affine operator;
- identity-centered neutral writer implementation;
- T007R Bank-A / Bank-B correct-pair state tensors;
- exact zero state for clean;
- T008 Bank-A / Bank-B clean reference moments;
- corruption code and severities;
- T009 `natural_support_manifest.json` and primary `K=20` selected IDs;
- exact PFLlib client query splits;
- historical T007R `none`, `correct`, `wrong_alt`, and `tau` values.

Historical files must remain immutable. In particular:

- do not modify `src/context/source_signature.py`;
- do not rebuild prototypes;
- do not refit any fast state;
- do not alter T009 support IDs or K;
- do not use T009 query outcomes to tune a threshold/temperature/weight.

Bank A and Bank B remain independent evaluators, not an ensemble.

---

# 3. Define the post-state source restoration score

Add a new isolated helper, e.g.:

```text
src/context/state_consistency.py
```

For bank `b`, source support tensor `S`, and candidate fast state `s`, run the frozen model through the two spatial blocks with the affine state applied at the **same insertion points used by `ContextFedAvgCNN.forward`**.

For layer `l` after applying that layer's candidate affine:

```text
mu_l(s)    = channel mean over batch and spatial dims
sigma_l(s) = sqrt(channel population variance + 1e-6)
```

Using the unchanged Bank-b clean reference `(mu0_l, sigma0_l)`, construct:

```text
r_mu_l    = (mu_l(s) - mu0_l) / (sigma0_l + 1e-6)
r_sigma_l = log((sigma_l(s) + 1e-6) / (sigma0_l + 1e-6))
```

Concatenate both layers exactly analogously to the T008/T009 source signature:

```text
psi_b(S; s) = [r_mu_1, r_sigma_1, r_mu_2, r_sigma_2]  # 192-D
```

Primary source-only score:

```text
J_b(S; s) = ||psi_b(S; s)||_2^2
```

Candidate selection:

```text
s* = argmin_s J_b(S; s)
```

Candidate order for exact ties must be frozen as:

```text
clean/zero, brightness_dark, contrast_low, gaussian_noise, gaussian_blur
```

There is **no learned parameter and no threshold**. Zero state is a normal candidate; if it has the minimum score, the policy abstains from adaptation.

Important implementation detail: layer-2 moments must be computed after layer-1 candidate state has already modified the features, exactly matching the actual forward path. Do not score the two layers independently on zero-state activations.

---

# 4. Required sanity tests before the formal run

Add focused tests before evaluating 100 clients.

### 4.1 Zero-state regression

For arbitrary deterministic tensors:

- `psi(S; zero)` must reproduce the existing `source_signature(model,S,bank_reference)` vector within tight float tolerance;
- `J(S; zero)` must equal the squared norm of that historical signature.

This is the most important implementation regression.

### 4.2 Exact neutrality

Applying zero state through the new scoring path must not alter logits/features relative to the ordinary zero-state path beyond normal floating-point equality/tolerance.

### 4.3 Candidate-state immutability

All five states must remain finite and byte/hash-identical to the frozen T007R artifacts. Model hash must remain unchanged after every score computation.

### 4.4 Small balanced sanity, descriptive only

On the already frozen T008 held-out balanced microbatches, report the state-consistency selector's 5-way confusion matrix for A→B and B→A. Do **not** tune anything from it and do not create a new gate. This only tells us whether the state score behaves coherently before the natural-client test.

If 4.1 or 4.2 fails, stop and report an implementation blocker. Do not proceed with scientifically uninterpretable scores.

---

# 5. Formal natural-client selection

Use the exact T009 K=20 natural supports.

For each:

```text
100 clients × 5 true contexts × 2 banks
```

compute all five candidate state scores.

Persist per decision:

```text
client
bank
true_context
selected_state
J_zero
J_dark
J_contrast
J_noise
J_blur
best_score
second_score
score_margin = second_score - best_score
zero_improvement = J_zero - best_score
```

Also record the rank of the true-context candidate (`1..5`).

The same original support IDs must be used as T009; assert the manifest SHA.

### Freeze rule

Write **all 1,000 state-consistency selections and all candidate scores before opening any new query result or support-label statistic**.

Do not inspect T009 harmful-client query outcomes while implementing a special case. Client 29 is evidence for the problem, not permission to hand-code a fix.

---

# 6. Selection diagnostics

Although the primary goal is safe retrieval, candidate states have known context identities, so report selection specificity exactly:

For each bank:

```text
overall 5-way state-selection accuracy / 500
per-context accuracy / 100
5x5 confusion matrix
true-state median rank
fraction true-state rank=1 / <=2
median J(correct)
median J(zero)
median J(frozen T007R wrong-alt state)
```

Also report separately for T009's historically difficult pairs:

```text
clean -> nonzero counts by selected state
blur -> zero count
blur -> contrast count
```

Do not change any gate from query results.

For descriptive comparison only, show which T009 hard-nearest-prototype errors are corrected, preserved, or newly introduced by state consistency.

---

# 7. Freeze the primary T010 gates now

Use the same natural-support identification gate as T009 for state-selection specificity:

A bank satisfies `SC-ID` if:

```text
overall state-context accuracy >= 90%
AND each of the five contexts >= 80%
```

Overall `SC-ID-A` requires both banks.

This is intentionally unchanged from T009 so we can tell whether direct state scoring preserves context specificity.

---

# 8. Per-client selected-state retrieval

Only after all 1,000 source-side selections are persisted, evaluate the selected candidate on each client's corresponding query.

Exactly as in T009:

```text
selected clean -> exact zero state
selected corruption c -> frozen Bank-b T007R correct-pair state for c
```

Reuse old predictions whenever the exact `(client, target, bank, state)` prediction already exists. Compute only missing selected-state predictions.

For each bank/shift report:

```text
none_acc
oracle_correct_state_acc
state_consistency_mixed_acc
wrong_alt_acc
tau_pp
retained_recovery
selected_minus_wrong_pp
```

Use T007R's frozen values.

A shifted target passes one bank iff:

```text
oracle_gain > 0
AND retained_recovery >= 0.90
AND state_consistency_mixed_acc - wrong_alt_acc >= tau_pp
```

Joint shift pass requires both banks, exactly as T009.

### Clean safety

Run the mixed selected policy on clean queries and require in each bank:

```text
selected_clean_macro_class >= zero_state_clean_macro_class - 0.5 pp
```

### Overall `SC-RET-A`

Require:

```text
at least 3/4 shifts joint-pass across both banks
AND clean safety passes both banks
```

Do not weaken the 0.5 pp clean allowance because T009 missed it by only ~0.11 pp. The point of this experiment is to remove brittle false adaptation, not to move the gate.

---

# 9. Mechanism audit: does source restoration predict task utility?

After query predictions are frozen, perform one descriptive audit; do not fit a model.

For each client/context/bank compute:

```text
DeltaJ(candidate) = J_zero - J(candidate)
DeltaAcc(candidate) = query accuracy(candidate) - query accuracy(zero)
```

For the selected state and the true state separately report:

- Pearson/Spearman correlation between `DeltaJ` and `DeltaAcc`;
- median `DeltaAcc` when `DeltaJ > 0`;
- fraction of cases where `DeltaJ > 0` but `DeltaAcc < 0`;
- the 10 largest **restoration/task disagreements** (`DeltaJ` strongly positive but task accuracy drops).

This is crucial for interpretation:

- strong agreement means source moment restoration is a meaningful state-selection proxy;
- frequent `DeltaJ>0` / `DeltaAcc<0` means moment normalization can be semantically harmful even though it looks physically consistent.

No regression fitting, no significance package, no threshold selection.

---

# 10. Label-skew audit for selector errors

Reuse T009's already frozen support-label statistics; do not recompute or use them for selection.

For state-consistency selection, report by the same four entropy quartiles:

```text
clean selection accuracy
blur selection accuracy
overall state-context accuracy
median score margin
```

List the worst clean/blur state-selection errors with entropy and max-class fraction.

Question to answer:

> Does direct state-consistency remove the low-entropy clean/blur confound, or does the same semantic-content failure remain even without prototype naming?

---

# 11. OOD carry-forward — descriptive, no threshold

Use the eight already frozen T008 random-noise OOD supports.

For each bank, score all five states with the same `J` criterion and report:

```text
selected-state histogram
fraction selecting zero
J_zero / J_best distribution
zero_improvement distribution
```

Do **not** create an OOD threshold in T010.

T009 showed a large absolute-distance gap in this limited control (`known max ≈42/48`, T008 noise min ≈84/85), but that is not yet a validated open-set threshold. Keep OOD rejection separate from the in-distribution clean/blur safety question.

---

# 12. Required verification

Assert and record:

1. T009 natural-support manifest SHA unchanged;
2. checkpoint SHA unchanged;
3. neutral writer/operator code hashes unchanged;
4. corruption code unchanged;
5. T007R state artifact hashes unchanged;
6. T008 bank clean-reference artifact/prototype hashes unchanged;
7. `source_signature.py` unchanged;
8. `psi(S; zero)` matches historical source signature within tight tolerance;
9. `J_zero` equals historical signature squared norm;
10. all candidate states and scores finite;
11. same support IDs used for all candidates/context/banks exactly as T009;
12. labels/query tensors never enter state scoring;
13. all 1,000 primary selections/scores frozen before query evaluation;
14. model hash unchanged after every scoring/evaluation episode;
15. historical T007R/T009 aggregate metrics reproduce exactly where reused;
16. mixed macro-class metrics independently recomputed from integer correct/total counts;
17. no query outcome changes candidate set, score, support, metric, or gate;
18. no state fitting or parameter learning occurs.

Run focused new tests plus existing relevant T009/T008/T007R regressions. No FL training.

---

# 13. Suggested artifacts

Prefer:

```text
src/context/state_consistency.py
scripts/eval_t010_state_consistency.py
scripts/report_t010.py
```

Write:

```text
results/t010_state_consistency/
    RESULTS.md
    metadata.json
    verification.json
    balanced_sanity.csv
    state_scores.csv
    confusion_A.csv
    confusion_B.csv
    retrieval.csv
    clean_safety.csv
    restoration_task_audit.json
    skew_audit.csv
    ood_state_scores.csv
```

Large raw predictions may remain compressed/ignored if integer count receipts and hashes are preserved.

---

# 14. Decision rules / next research branch

### Case SC-A — state consistency fixes safe retrieval

If `SC-ID-A` and `SC-RET-A` both pass:

> The remaining T009 failure was mainly the brittle prototype-name → full-state handoff. We now have a source-only, label-free, zero-inclusive state-selection rule that is safe under natural client skew. Stop and report. **Do not start SSL automatically.** The next Research Lead package should test unseen corruption severities / continuous state interpolation and then explicit OOD rejection.

### Case SC-B — clean safety passes but Blur recovery still fails

> Zero-inclusive consistency prevents harmful clean adaptation, but missed/ambiguous Blur states remain. The state bank is viable; the next diagnostic should test **continuous interpolation along the clean↔shift state direction**, still source-only and without learning.

### Case SC-C — the same low-entropy clean/blur errors remain

> Prototype naming was not the core issue. The clean-reference moment criterion itself is semantically content-confounded for a few extreme clients. Keep the operator/state bank fixed; next test one explicit **content-canceling source statistic**. Do not enlarge the operator or add SSL.

### Case SC-D — J improves but task utility often degrades

If state consistency strongly reduces source moment residuals but retrieval/clean safety remains poor, especially with many `DeltaJ>0, DeltaAcc<0` cases:

> Moment restoration is not sufficiently task-aligned as a deployment selector. Treat source moment normalization as a diagnostic only. Before any learned writer, design a richer task-neutral source criterion; do not claim context repair from moment matching alone.

### Case SC-E — broad failure even on balanced sanity

> The candidate-state restoration score is not a coherent selector, despite the state bank's oracle utility. Stop and report; do not add ad-hoc thresholds or sweeps.

---

# 15. Communication protocol

After T010, replace `coordination/CODEX_TO_CHATGPT.md` with:

```markdown
# CODEX -> CHATGPT
## Timestamp
## Commit / run ID
## Frozen evidence reused
## New state-consistency score implementation
## Sanity/regression verification
## Balanced held-out sanity
## Natural-client state selection
## Confusion matrices / hard-pair audit
## Per-client mixed retrieval
## Clean safety
## Restoration-vs-task audit
## Label-skew audit
## OOD descriptive audit
## Failures / uncertainties
## SC-ID / SC-RET / Case decision
## Recommended next action
```

Commit and push code plus compact summaries. Preserve negative results. **Do not launch the next research stage without Research Lead review.**