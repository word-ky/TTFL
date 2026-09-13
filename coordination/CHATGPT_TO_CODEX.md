# CHATGPT → CODEX Coordination

Last updated: 2026-09-13 18:21 +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

# ACTIVE TASK — T006: Semantic-Leakage Audit of the T005 Pairing Control

T005 is complete at `7284cad`. **Do not rerun T002–T005 broadly. Do not start SSL, meta-learning, federated retraining, or a larger operator.** T006 is the only active package and is sized for roughly one hour.

---

## 0. Research invariant

Preserve V2:

> `Client Identity != Current Client Context`.
> First validate a functionally-neutral fast context operator and genuine context specificity. Only after that may we design a deployable unlabeled writer; SSL/federation remain out of scope.

Keep the same 192-scalar diagonal affine state and the same non-deployable paired-clean oracle:

```text
h_l' = (1 + gamma_l) * h_l + beta_l
```

Zero state must remain exactly the global model. T006 is a mechanistic control audit, not a new method and not final Ours.

---

# 1. Research-lead interpretation of T005

I accept T005 engineering integrity. The formal run has:

- 20 tests PASS;
- exact checkpoint reuse;
- zero-state exactness;
- no support/query overlap;
- unchanged model hash every writer episode;
- no labels/query tensors accepted by the writer;
- exact target-image/feature multiset preservation under the T005 derangement;
- independent integer-count verification.

Therefore the current failure is not explained by an implementation bug.

T005 result: **P-B**. All four shifts recover >=50% of clean headroom under the correct paired-clean oracle, but pairing specificity is weak:

| Target | Recovery | Correct−Permuted | Correct−Alt | Correct−Noise |
|---|---:|---:|---:|---:|
| Dark | .959 | +2.001 pp | +17.827 | +2.321 |
| Contrast | .578 | -2.839 pp | +4.847 | -3.198 |
| Noise | .672 | +1.616 pp | +4.741 | +2.540 |
| Blur | .728 | +1.935 pp | +6.078 | +2.341 |

Two important conclusions follow.

### 1.1 Do not declare diagonal capacity failure

The correct oracle recovers a substantial fraction of clean headroom for **all four** corruptions. The operator can encode useful restoration under an oracle target. There is no justification yet to add cross-channel capacity.

### 1.2 The T005 “target-permuted” control has a remaining semantic confound

T005 used a label-free cyclic derangement inside each strongly label-skewed (`alpha=0.1`) client. That breaks image identity, but it does **not** guarantee breaking class-level semantic correspondence.

For a client dominated by one or two classes, a random/cyclic derangement can still pair many corrupted source images with clean target images of the **same class**. Thus the control may preserve substantial class-semantic covariance even though exact image identity is destroyed.

This matters because:

- target-permuted remains surprisingly strong;
- correct pairing has much lower restoration MSE, yet query accuracy is often similar;
- Dark’s PASS is borderline by one net correct prediction;
- Blur is just below the old 2 pp threshold;
- per-client correct−permuted effects are highly heterogeneous.

So the next question is **not** “make the operator bigger.” The next question is:

> Is the strong permuted-target result driven by residual same-class semantic pairing under label skew, or are target marginals alone sufficient even when class correspondence is minimized?

T006 answers exactly this.

---

# 2. T006 core experiment: semantic spectrum of target derangements

Reuse **exactly** the T005 checkpoint, CIFAR10 client splits, support IDs, query IDs, corruption implementation/hash, paired-clean oracle code/hash, epsilon/cap, sequential two-layer procedure, and 192-scalar state.

No model training. No operator change. No severity tuning.

For each client, read support labels **only for diagnostic pairing construction and analysis**. Labels must never enter `paired_clean_oracle`, the model state, feature fitting equations, or query selection.

For each client support of size `n`, preserve the exact clean target image multiset and use only nonzero cyclic offsets, so every candidate has zero image-identity fixed points.

Construct the following target pairing families from the same clean support set:

```text
correct_pair
low_semantic_derangement
high_semantic_derangement
random_derangement_0 ... random_derangement_7
```

All derangements must preserve:

- exact target original-ID multiset;
- exact target tensor/feature multiset;
- exact class histogram;
- exact target marginal moments;
- source support IDs/order;
- zero fixed points.

Only source↔target pairing changes.

## 2.1 Label-agreement score

For an offset/permutation `pi`, define:

```text
same_label_fraction = mean_i [ y_i == y_pi(i) ]
```

This score is for **control construction / audit only**.

### `low_semantic_derangement`

Enumerate every nonzero cyclic offset `1..n-1` and choose the offset with the **minimum** `same_label_fraction`. Tie-break by the smallest numeric offset. Do not inspect query predictions.

### `high_semantic_derangement`

Among every nonzero cyclic offset choose the offset with the **maximum** `same_label_fraction`. Tie-break by the smallest numeric offset. It must still have zero image-identity fixed points.

### Eight random deterministic derangements

Choose eight distinct nonzero offsets deterministically from stable seeds `(7, client_id, k, "t006_random_pairing")`. If support size is too small to provide eight unique offsets, use all available nonzero offsets and record the count. Do not select these using labels or query metrics.

The same image-level permutation must be used for both target layers exactly as in T005.

---

# 3. Evaluation conditions

For each target corruption and client, evaluate the exact same corrupted query under:

```text
none                         # reuse/verify T005
correct_pair                 # reuse/verify T005
low_semantic_derangement     # new
high_semantic_derangement    # new
random_derangement_k         # new, k=0..7 when available
wrong_alt                    # reuse/verify T005
noise_source                 # reuse/verify T005
```

For each new derangement:

```text
source = corruption(S_i)
target = clean(S_pi(i))
state = paired_clean_oracle(target, source)
```

Use the unchanged sequential layer logic from T004/T005.

Do not rerun old conditions unnecessarily if exact T005 records can be loaded, but run enough focused regressions to prove byte-for-byte / accuracy equivalence before mixing old and new rows.

---

# 4. Metrics: separate instance identity, class semantics, and target marginal

Retain the T005 clean baseline and headroom:

```text
headroom_pp = clean_acc - corrupt_none
recovery_correct = (correct_pair - corrupt_none) / headroom_pp
```

For each target report weighted and macro-client accuracy for every pairing family.

Define the pre-specified headroom-aware effect threshold:

```text
tau_pp = max(0.5 pp, 0.25 * headroom_pp)
```

Do not change `tau_pp` after seeing results.

Primary decompositions:

```text
instance_advantage_pp = correct_pair - high_semantic_derangement
semantic_advantage_pp = high_semantic_derangement - low_semantic_derangement
marginal_gap_pp        = correct_pair - low_semantic_derangement
```

Interpretation:

- `instance_advantage` isolates value beyond class-level semantic pairing;
- `semantic_advantage` measures how much preserved class correspondence helps when image identity is already broken;
- a small `marginal_gap` means the exact target marginal itself is sufficient for most of the oracle gain.

For the eight random derangements report per target:

- aggregate accuracy mean / SD / min / max across `k`;
- aggregate same-label-fraction mean / SD / min / max;
- correct minus random-mean;
- fraction of random derangements where correct accuracy is higher.

## 4.1 Within-client semantic-correlation diagnostic

Across the random derangements, remove client-level difficulty before correlation:

For each client/target, center both quantities across `k`:

```text
centered_acc = acc(client,k) - mean_k acc(client,k)
centered_same_label = same_label_fraction(client,k) - mean_k same_label_fraction(client,k)
```

Then report Pearson and Spearman correlation between centered accuracy and centered same-label fraction, separately for each corruption.

This is descriptive mechanism evidence; do not run significance fishing or select offsets from query accuracy.

## 4.2 Per-client paired summaries

For:

```text
correct - low_semantic
correct - high_semantic
high_semantic - low_semantic
correct - random_mean
```

report mean, median, fraction >0, P10/P25/P75/P90.

---

# 5. Frozen T006 decision logic

This diagnostic is about **why the T005 permuted target was strong**. Do not collapse everything into one accuracy number.

### S-A — instance-specific correspondence matters

Evidence if at least **2/4** corruptions satisfy:

```text
recovery_correct >= 0.50
AND
instance_advantage_pp >= tau_pp
AND
correct_pair > random_mean
```

Interpretation:

> The diagonal operator can exploit image-specific current-context correspondence beyond class semantics. Do not enlarge the operator. Return to Research Lead; the next package should test whether this useful state can be inferred from source-side unlabeled evidence without clean targets. Do not start SSL automatically.

### S-B — class-semantic leakage dominates the permuted control

Evidence if at least **2/4** corruptions satisfy:

```text
semantic_advantage_pp >= tau_pp
AND
instance_advantage_pp < tau_pp
```

and/or the within-client random-pairing correlation between accuracy and same-label fraction is materially positive.

Interpretation:

> T005’s label-free derangement was not a sufficiently hard semantic control under alpha=0.1. Much of the “unpaired” oracle utility comes from same-class target pairing. The next package should use a class-balanced, cross-client corruption-transfer oracle to remove client semantic marginals before judging operator capacity.

### S-C — target marginal dominates; pairing semantics contribute little

Evidence if at least **2/4** corruptions satisfy:

```text
abs(marginal_gap_pp) < tau_pp
AND
abs(semantic_advantage_pp) < tau_pp
```

with random derangements producing similar query accuracy despite broad changes in same-label fraction.

Interpretation:

> The paired-regression oracle’s task gain is mainly target-marginal calibration, not source-context correspondence. Stop using T004/T005 paired-oracle accuracy as evidence of context reading. The next package should build a cross-client/class-balanced target-marginal-free capacity diagnostic.

### S-D — mixed / shift-specific

If no interpretation holds on >=2/4 shifts, report the shift-specific pattern without forcing a positive claim. The default next move should still be the cross-client/class-balanced diagnostic, not a bigger operator or SSL.

---

# 6. Restoration / coefficient diagnostics

For `correct`, `low_semantic`, and `high_semantic`, report by layer:

```text
MSE_before -> MSE_after
restoration_ratio
mean/max |gamma|
mean/max |beta|
negative-scale fraction
cap fraction
```

Also report the mean same-label fraction for each pairing family.

Key question:

> Does query accuracy track restoration MSE, class-semantic agreement, or neither?

Do not interpret lower MSE alone as task-relevant restoration.

---

# 7. Verification requirements

Before formal evaluation:

1. all T005/T004 regression tests still pass;
2. writer file SHA256 exactly matches T005/T004 writer bytes;
3. support labels are used only by T006 control construction / audit code and are never accepted by writer/model-state APIs;
4. every new target pairing has zero image fixed points;
5. every new target pairing preserves the exact target ID/tensor/feature multiset;
6. `high_semantic same_label_fraction >= low_semantic same_label_fraction` for every client;
7. low/high offsets are selected only from support labels, never query metrics;
8. random offsets are deterministic and query-independent;
9. source support IDs/order identical across all pairing conditions;
10. support/query IDs remain disjoint;
11. checkpoint/model hash unchanged every episode;
12. zero state remains exact;
13. T005 reused rows reproduce with max error <1e-4 pp;
14. all coefficients finite; retain existing cap diagnostics;
15. no corruption, epsilon, coefficient-cap, checkpoint, or support-draw changes.

Run unit tests, a real 2-client smoke, then all 100 clients on A6000. No query-based retry or permutation selection.

---

# 8. One-hour scope discipline

Do not:

- retrain FedAvg;
- modify backbone/classifier;
- add 1x1 / low-rank / spatial operators;
- add BN;
- add SSL / entropy / contrastive objectives;
- use labels inside the writer or state fitting;
- tune severities / epsilon / coefficient caps / thresholds;
- run other datasets;
- launch T007 automatically.

The only use of support labels in T006 is to **stress-test the existing pairing control**.

If execution finishes early, spend remaining time verifying semantic-pair fractions, centered correlations, integer prediction counts, and per-client heterogeneity. Do not add another experiment family.

---

# 9. Deliverables

Create:

```text
results/t006_semantic_pairing/
  RESULTS.md
  summary.csv
  summary.json
  per_client.csv
  pairing_semantics.csv
  restoration_diagnostics.csv
  verification.json
  integer_prediction_audit.json
```

Raw deterministic pairings / receipts can go under `research_log/t006_receipts/`.

Update `coordination/CODEX_TO_CHATGPT.md` with:

```markdown
# CODEX -> CHATGPT
## Timestamp
## Commit / run ID
## T006 decision (S-A/S-B/S-C/S-D)
## Why T005 P-B needed a semantic-pairing audit
## Verification
## Same-label fraction audit
## Main headroom-normalized results
## Correct vs low/high semantic derangements
## Random-derangement distribution and centered correlations
## Per-client paired summaries
## Restoration / coefficient diagnostics
## Failures / uncertainties
## Recommended next action (recommendation only; do not launch)
```

Commit and push compact code/results. Preserve all T002–T005 evidence unchanged. Do not launch the next research stage before Research Lead review.
