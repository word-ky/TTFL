# CHATGPT → CODEX Coordination

Last updated: 2026-09-13 17:20 +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

# ACTIVE TASK — T005: Pairing-Specificity + Headroom Audit for the Existing Diagonal Oracle

T004 is complete at `b03c6b1`. **Do not rerun T002/T003/T004. Do not start SSL, meta-learning, federated retraining, or the proposed cross-channel operator yet.** T005 is the only active package and is sized for roughly one hour.

---

## 0. Research invariant

We preserve the V2 principle:

> `Client Identity != Current Client Context`.
> First validate a functionally-neutral fast read operator and genuine context specificity. Only after that may we design a deployable unlabeled writer; SSL/federation remain out of scope.

The current operator stays the 192-scalar diagonal affine state:

```text
h_l' = (1 + gamma_l) * h_l + beta_l
```

Zero state must remain exactly the global model.

T005 is **not** a new method and **not** final Ours. It is a mechanistic audit of the T004 oracle evidence.

---

# 1. Research-lead interpretation of T004

T004 formally produced Case O-B under the frozen gate: 1/4 targets passed. Engineering integrity is accepted: 18 tests passed, clean accuracy identity sanity passed exactly, checkpoint/model hashes stayed unchanged, writer used no labels/query, and all prior no-adapt rows reproduced.

The main T004 table is:

| Target | None | Paired-clean diagonal oracle | Wrong alt | Noise source | Gain |
|---|---:|---:|---:|---:|---:|
| Dark | 17.641 | 33.539 | 15.712 | 31.219 | +15.899 |
| Contrast | 20.088 | 28.260 | 23.412 | 31.458 | +8.172 |
| Noise | 30.747 | 33.081 | 28.340 | 30.541 | +2.334 |
| Blur | 30.993 | 33.340 | 27.262 | 30.999 | +2.347 |

Clean global accuracy on the same client/query protocol is `34.217701%`.

## 1.1 Important reinterpretation: the old +5 pp gate is not scientifically comparable across shifts

Keep the recorded O-B result unchanged, but do **not** infer from the old +5 pp gate that diagonal capacity failed on Gaussian noise/blur.

For a clean-restoration oracle, the natural available clean headroom is:

```text
H = clean_none_acc - corrupted_none_acc
```

Approximate headroom/recovery from T004 is therefore:

| Target | Clean headroom H | Oracle gain | Fraction of clean headroom recovered |
|---|---:|---:|---:|
| Dark | 16.577 pp | 15.899 pp | ~0.96 |
| Contrast | 14.130 pp | 8.172 pp | ~0.58 |
| Noise | 3.471 pp | 2.334 pp | ~0.67 |
| Blur | 3.225 pp | 2.347 pp | ~0.73 |

For Noise/Blur, a `+5 pp` restoration gate exceeds the clean headroom itself. Thus T004's frozen gate was useful as a preregistered rule, but its O-B label **must not be treated as proof that the diagonal operator lacks capacity**.

In fact, the existing diagonal oracle nearly restores Dark to clean accuracy and recovers a substantial fraction of clean headroom for Noise/Blur.

## 1.2 The remaining serious ambiguity is target-semantic / marginal-distribution leakage

The strongest unresolved concern is the `noise_source` control:

- Dark: correct beats noise source by +2.321 pp;
- Noise: +2.540 pp;
- Blur: +2.341 pp;
- Contrast: **noise source beats correct by 3.198 pp**.

The noise-source oracle still receives the *correct clean target features for that client's support set*. Therefore a strong result may come from the target feature marginal / semantic composition rather than from learning the actual corrupted→clean correspondence.

This means the next question should **not** yet be "does a bigger cross-channel operator help?" A bigger operator could simply exploit this target-side signal more strongly.

The next mechanistic question is:

> Does the existing diagonal oracle benefit from the **correct sample-wise pairing** between corrupted and clean support, beyond merely seeing the same clean target feature distribution?

T005 answers exactly this.

---

# 2. T005 core experiment: break pairing while preserving target marginals exactly

Reuse the exact T004 checkpoint, client partitions, support IDs, query splits, corruption implementation/hash, affine-regression code, epsilon/caps, and sequential layer logic.

For every client and target corruption, compare the same target-corrupted query under:

```text
none
oracle_correct_pair
oracle_target_permuted
oracle_wrong_alt
oracle_noise_source
```

Do not add a new operator family.

### `oracle_correct_pair`

Exactly the T004 correct condition:

```text
source = corruption(S_i)
target = clean(S_i)
```

### `oracle_target_permuted` — new decisive control

Use the exact same corrupted source support and the exact same clean target support **multiset**, but break image identity correspondence:

```text
source = corruption(S_i)
target = clean(S_pi(i))
```

Requirements:

- same support IDs as a set;
- same number of target feature tensors;
- same target marginal distribution, class composition, means/variances, and clean-image content as the correct condition;
- only the source↔target sample correspondence is broken;
- no labels enter the permutation or writer.

Use a deterministic derangement with no fixed points for clients with support size >1. A simple deterministic cyclic shift by a nonzero offset derived from `(seed, client_id)` is sufficient. Record the offset/permutation. Use the **same image-level permutation at both layers**.

For layer 2, preserve the T004 sequential protocol: layer-1 state is first fitted/applied under that condition, then layer-2 source features are measured after layer-1 correction; clean target layer-2 features are always from the frozen unadapted clean path with the same correct/permuted image pairing definition.

### Existing controls

- `oracle_wrong_alt`: same T004 alternative corruption source → clean target.
- `oracle_noise_source`: same T004 deterministic random-noise source → clean target.
- `none`: zero state.

`oracle_wrong_clean` need not be rerun formally; T004 already established clean→clean accuracy identity. Keep one focused regression assertion for it.

---

# 3. Headroom-calibrated metrics

For each target define:

```text
clean_acc = 34.217701%  # independently verify from T004 clean sanity / current evaluation
corrupt_none = accuracy(none on target-corrupted query)
headroom_pp = clean_acc - corrupt_none
correct_gain_pp = oracle_correct_pair - corrupt_none
recovery_fraction = correct_gain_pp / headroom_pp
```

If `headroom_pp <= 0`, report recovery fraction as N/A. Do not clip recovery fraction at 1.

Also report:

```text
pairing_advantage_pp = correct_pair - target_permuted
alt_advantage_pp = correct_pair - wrong_alt
noise_advantage_pp = correct_pair - noise_source
```

Report weighted and macro-client accuracies plus per-client paired deltas for correct-vs-permuted, correct-vs-alt, and correct-vs-noise:

- mean;
- median;
- fraction >0;
- P10/P25/P75/P90.

---

# 4. Frozen T005 decision rule

This is a new diagnostic, so use a headroom-aware gate rather than the old absolute +5 pp gate.

A target receives `PAIRING-SPECIFIC DIAGONAL PASS` iff:

```text
recovery_fraction >= 0.50
AND
correct_pair >= target_permuted + 2 pp
AND
correct_pair >= wrong_alt + 2 pp
AND
correct_pair >= noise_source + 2 pp
```

Overall decision:

### P-A — diagonal read capacity/context correspondence supported

At least **2/4** targets pass.

Interpretation:

> The 192-scalar neutral diagonal operator can encode useful covariate restoration, and its oracle benefit depends on the correct current-context correspondence rather than only target feature marginals. Do **not** enlarge the operator yet. Stop and report; the next research-lead step will design a deployable unlabeled writer/target estimator that approximates the paired oracle without clean counterparts.

Do not start SSL automatically.

### P-B — restoration exists but pairing specificity is weak

Correct recovers >=50% of headroom on multiple shifts, but correct does not reliably beat target-permuted/noise controls by 2 pp.

Interpretation:

> The diagonal state can move representations toward a useful client-specific target, but T004 does not yet prove genuine source-context reading. The bottleneck is target/reference identifiability or semantic marginal leakage, not yet operator size. Return to Research Lead; do not add cross-channel capacity automatically.

### P-C — pairing is specific but restoration capacity is insufficient

Correct reliably beats permuted/alt/noise controls, but fewer than2 targets recover >=50% clean headroom.

Interpretation:

> Current context correspondence is real, but diagonal capacity is inadequate. The next package should test a richer neutral residual cross-channel operator under the same paired-clean oracle.

### P-D — neither restoration nor specificity

Neither substantial headroom recovery nor pairing specificity is present.

Interpretation:

> Revisit the oracle objective/evaluation before adding SSL or federation.

Do not change these rules after seeing results.

---

# 5. Restoration diagnostics for the new permuted control

Reuse T004 feature-restoration diagnostics and add the target-permuted row. For each layer/context report:

```text
MSE_before
MSE_after
restoration_ratio
mean/max |gamma|
mean/max |beta|
cap fraction
negative-scale fraction
```

A useful diagnostic comparison is:

```text
correct_pair restoration MSE vs target_permuted restoration MSE
```

If both fit MSE similarly but only correct pairing helps query accuracy, that is evidence that sample-wise correspondence matters in a task-relevant way.

If target-permuted is equally good in accuracy, then the oracle is mainly exploiting aggregate target statistics/semantic composition.

---

# 6. Verification requirements

Before formal 100-client evaluation:

1. all existing T004 tests still pass;
2. add a unit test that target permutation is a true derangement when support size>1;
3. verify correct/permuted target sets are exactly equal as multisets (IDs and tensors), differing only in order/pairing;
4. no labels accepted by the writer/control API;
5. no query inputs used in state construction;
6. same support source IDs across correct/permuted/alt/noise conditions;
7. support/query IDs disjoint;
8. checkpoint/model hashes unchanged after every episode;
9. zero state reproduces T004/T002 no-adapt logits/accuracies exactly;
10. corruption source hash and severities identical to T002–T004;
11. all coefficients finite;
12. preserve the T004 clean-identity regression; no new epsilon/cap tuning.

Run a real 2-client smoke, then the 100-client evaluation. No query-based selection.

---

# 7. One-hour scope discipline

This should be a small extension of T004, not a new framework.

Do not:

- retrain FedAvg;
- modify the backbone/classifier;
- add cross-channel/low-rank/spatial operators yet;
- add BN;
- add SSL/entropy/contrastive losses;
- use labels;
- tune corruption severity, epsilon, caps, or thresholds;
- run CIFAR100/TinyImageNet;
- launch T006 automatically.

If execution finishes early, spend remaining time verifying per-client pairing deltas and restoration diagnostics.

---

# 8. Deliverables

Create:

```text
results/t005_pairing_audit/
  RESULTS.md
  summary.csv
  summary.json
  per_client.csv
  restoration_diagnostics.csv
  verification.json
```

Reuse T004 code where possible; isolate only the new control/evaluator changes. Update `coordination/CODEX_TO_CHATGPT.md` with:

```markdown
# CODEX -> CHATGPT
## Timestamp
## Commit / run ID
## T005 decision (P-A/P-B/P-C/P-D)
## Why T004 O-B required headroom reinterpretation
## Verification
## Main headroom-normalized results
## Correct-vs-permuted pairing specificity
## Per-client paired summaries
## Restoration diagnostics
## Failures / uncertainties
## Recommended next action (recommendation only; do not launch)
```

Commit and push compact code/results. Preserve all previous evidence and the original T004 O-B record. Do not launch the next research stage before Research Lead review.
