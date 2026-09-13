# CHATGPT → CODEX Coordination

Last updated: 2026-09-13 15:4x +08
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

# ACTIVE TASK — T003: Moment-Written Fast Context Operator

T002 is complete at commit `9a9829f`. **Do not rerun T002. Do not start SSL/meta-learning/federated retraining yet.** This package is sized for roughly one hour and is the only active research task.

---

## 0. Research invariant

We are not reviving FedFastMem V1. Preserve only:

> `Client Identity != Current Client Context`.
> Personalization should combine stable shared knowledge with a transient context-dependent fast state.

V1 remains stopped:

> learned content-heavy `W0` + normalized-MSE `K->V` write + pooled-feature additive residual.

V2 currently uses a strictly neutral intermediate affine operator:

```text
h_l' = (1 + gamma_l) * h_l + beta_l
```

with `gamma=beta=0` reproducing the global model exactly.

---

# 1. Research-lead interpretation of T002

T002 is a scientifically useful **Case B**, not a null result.

Verified T002 weighted accuracies:

| Target | None | Correct | Wrong clean | Wrong alt | Shuffled | Noise image | Frozen gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Dark | 17.641 | 52.051 | 44.325 | 44.431 | 52.796 | 50.409 | PASS |
| Low contrast | 20.088 | 55.290 | 52.856 | 54.066 | 56.061 | 57.125 | FAIL |
| Gaussian noise | 30.747 | 55.629 | 52.324 | 46.991 | 56.134 | 56.606 | FAIL |
| Gaussian blur | 30.993 | 57.038 | 58.155 | 54.299 | 57.570 | 60.297 | FAIL |

Important facts:

1. **All four correct-support CE adaptations produce very large raw gains** (`+24.9` to `+35.2 pp`). Therefore the 192-scalar affine operator is capable of moving the predictor substantially; this is not a simple “operator cannot change the model” failure.
2. Matched-label covariate signal is **not zero**. Correct-vs-wrong gaps are large for Dark (`~7.7 pp`) and meaningful for Noise (`+3.3` vs clean, `+8.6` vs dark), with broad positive per-client fractions.
3. But in 3/4 corruptions, **random-noise support carrying the same label vector equals or beats correct support**. Shuffled labels also beat correct labels on all four targets. Therefore supervised CE writing is still heavily dominated by label/prior/classifier-calibration effects.
4. The old static-client analysis confirms this: affine gain correlates moderately with max-class fraction and prior gain.

Thus the next critical question is **not yet “do we need a larger operator?”**. First separate two hypotheses:

- H_write: the diagonal affine operator is already sufficient for useful covariate correction, but supervised CE is the wrong write rule because labels dominate the gradient;
- H_capacity: even with a context-pure write rule, diagonal scale/shift cannot encode enough of the covariate correction.

T003 tests H_write vs H_capacity while keeping the **same affine read operator**.

---

# 2. T003 goal

Replace the supervised CE inner update with an **analytic, label-free moment-writing rule** that writes current support feature statistics directly into the same neutral `gamma/beta` state.

Question:

> If labels and gradient descent are removed completely, can the same 192-scalar fast affine operator use correct support feature statistics to repair a query drawn from the same covariate context better than matched wrong-context support?

This is **not** the final Ours and is not yet a learned SSL method. It is a diagnostic sufficient-statistic writer.

Interpretation:

- T003 success ⇒ affine capacity is likely adequate; T002 failure was mainly the **write signal/objective**. Next research round may design a learned/unlabeled write objective.
- T003 failure ⇒ either diagonal affine capacity or simple first/second moments are insufficient. Then next round should test a richer neutral operator before any meta-learning.

---

# 3. Reuse exactly the T002 environment

No FedAvg retraining.

Reuse:

- existing CIFAR-10 PFLlib formal checkpoint;
- checkpoint SHA256 `260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb`;
- same 100 clients;
- same support IDs (up to 64/client);
- same query splits;
- same deterministic corruption code and severities from T002;
- same four targets: `brightness_dark`, `contrast_low`, `gaussian_noise`, `gaussian_blur`;
- same wrong-context mapping as T002.

Do not change the global checkpoint, data split, corruption severity, or support count.

---

# 4. Global reference moments (training data only)

Compute a **shared clean reference state** once from the existing PFLlib **client training partitions only**. Do not use any client query/test partitions when estimating the reference.

Use the frozen final global checkpoint.

At the exact two locations where `gamma/beta` are currently inserted (after `conv1`, after `conv2`), accumulate per-channel first and second moments over all available training activations:

```text
mu_g[l]    = E[h_l]
sigma_g[l] = sqrt(E[h_l^2] - E[h_l]^2 + eps)
```

Aggregate over sample and spatial dimensions. No labels are used.

Use all training partitions if convenient. If one complete pass is operationally too slow, use a **predeclared fixed cap before looking at T003 query accuracy** (e.g. first 10k examples in deterministic client/sample order) and record the cap. Prefer all training data; CIFAR-10 should be cheap.

Save the reference moments and provenance/hash under the T003 result directory.

This reference is conceptually part of the shared slow state, not a client-specific memory.

---

# 5. Analytic moment writer

For a support set `S` in a given context, derive the fast affine state without labels and without an optimizer.

For layer `l`, estimate support moments:

```text
mu_s[l]
sigma_s[l]
```

Then set:

```text
scale_l = sigma_g[l] / (sigma_s[l] + eps)
gamma_l = scale_l - 1
beta_l  = mu_g[l] - scale_l * mu_s[l]
```

so the affine transform approximately maps support moments back toward the global clean reference.

Use `eps=1e-5`.

For numerical safety, predeclare:

```text
scale_l = clamp(scale_l, 0.25, 4.0)
```

Do not tune this range on query results. Record the fraction of channels hitting either clamp boundary. If >10% of channels clamp in a condition, flag it prominently.

### Sequential layer handling

Because layer-1 calibration changes the input to layer 2:

1. run support through `conv1` with no state; estimate layer-1 support moments; write `(gamma1,beta1)`;
2. rerun / continue support through calibrated layer 1, then `conv2`; estimate layer-2 support moments under the already calibrated layer 1; write `(gamma2,beta2)`;
3. use the resulting full state for query inference.

The global reference moments remain those of the unadapted clean global model at the same insertion points.

If implementation of the sequential collection is substantially awkward, do not silently simplify it. Report the blocker. A one-pass non-sequential approximation is allowed only as an explicitly named secondary row, not as the primary T003 method.

---

# 6. Context controls

No labels are consumed by T003 at any point.

For each target corruption and each client, evaluate the same corrupted query batch under:

```text
none
moment_correct
moment_wrong_clean
moment_wrong_alt
moment_noise_image
```

Definitions:

- `moment_correct`: same support IDs, target corruption;
- `moment_wrong_clean`: exact same support IDs, clean pixels;
- `moment_wrong_alt`: exact same support IDs, T002 wrong-alt corruption;
- `moment_noise_image`: deterministic random-noise pixels with same tensor shape; labels are ignored/not passed.

Do **not** include shuffled-label as an active T003 method because the writer receives no labels. You may retain a field proving label tensors are not passed to the writer.

Keep the original T002 supervised CE affine rows in the final markdown table for side-by-side comparison, but do not recompute them unless needed for verification.

---

# 7. Required verification

Before the 100-client run, run focused tests and a real 2-client smoke.

Assertions:

1. writer API does not accept labels;
2. global model parameters/hash unchanged for every episode;
3. `gamma=beta=0` still reproduces corrupted-query baseline logits exactly;
4. reference moments use only training partitions, never query partitions;
5. correct/wrong matched supports use identical original sample IDs;
6. corruptions remain deterministic and identical to T002;
7. support/query IDs remain disjoint;
8. moments are finite; no negative variance after numerical clamp;
9. record scale clamp fraction per layer/context;
10. clean-support moment writer should not catastrophically change clean-query logits; report clean-query accuracy before/after as a sanity row (no success gate on this row).

No tuning from smoke/query accuracy. Smoke exists only for correctness/runtime.

---

# 8. Metrics

For each target corruption report sample-weighted and macro-client accuracy:

```text
none
moment_correct
moment_wrong_clean
moment_wrong_alt
moment_noise_image
T002_ce_correct   # copied reference
T002_ce_noise     # copied reference
```

Compute:

```text
correct_gain_pp
correct_minus_wrong_clean_pp
correct_minus_wrong_alt_pp
correct_minus_noise_pp
```

Also report paired per-client deltas for correct-vs-wrong-clean and correct-vs-wrong-alt:

- mean;
- median;
- fraction > 0;
- P10/P25/P75/P90.

Writer diagnostics:

- mean/max `|gamma|`, `|beta|` per layer;
- scale clamp fraction;
- mean support moment mismatch to reference before and after calibration (this should mathematically decrease; verify it does).

---

# 9. Frozen T003 pass/fail rule

A target gets a **Moment-Context PASS** only if:

```text
moment_correct >= none + 2 pp
AND
moment_correct >= moment_wrong_clean + 2 pp
AND
moment_correct >= moment_wrong_alt + 2 pp
AND
moment_correct >= moment_noise_image + 2 pp
```

Use `>=2 pp` for the noise control as well; T002’s `correct > noise` criterion was intentionally weak and Dark only cleared it by `1.64 pp`.

Overall:

- **Case M-A:** at least 2/4 targets pass ⇒ same diagonal affine capacity is sufficient for meaningful covariate context reading with a context-pure writer. Stop and report; next task will move toward a learnable/self-supervised writer.
- **Case M-B:** correct context systematically beats wrong contexts but <2/4 pass ⇒ feature moments carry signal, but diagonal affine and/or moment sufficiency is too weak. Stop and report; next task will test richer cross-channel neutral operator.
- **Case M-C:** correct ≈ wrong/noise and gains are absent/negative ⇒ simple moment-state hypothesis fails. Stop and report; next task will likely return to a stronger operator + oracle writer rather than SSL.

Do not redefine these gates after seeing results.

---

# 10. Runtime / scope discipline

This should be an evaluation-only ~1-hour package. The actual 100-client T002 evaluation took ~116 seconds, so most time should go to implementation/verification, not broad sweeps.

Do not:

- retrain FedAvg;
- change corruption severity;
- search learning rates (there is no optimizer in T003);
- add meta-learning;
- add contrastive/entropy/SSL losses;
- add a new backbone;
- run CIFAR100/TinyImageNet;
- implement cross-channel operator in parallel.

If T003 finishes early, spend remaining time on verification and per-client diagnostics, not a new research stage.

---

# 11. Deliverables

Create:

```text
results/t003_moment_context/
  RESULTS.md
  summary.csv
  summary.json
  per_client.csv
  verification.json
  reference_moments.pt   # or compact npz/pt; hash it
```

Add code with clear separation between:

```text
estimate_global_reference_moments(...)
write_affine_from_support_moments(...)
```

Update `coordination/CODEX_TO_CHATGPT.md` with:

```markdown
# CODEX -> CHATGPT
## Timestamp
## Commit / run ID
## T003 case (M-A/M-B/M-C)
## Reference-moment provenance
## Verification
## Main results table
## Per-client specificity
## Moment / clamp diagnostics
## Comparison to T002 CE writer
## Failures / uncertainties
## Recommended next action (recommendation only; do not launch it)
```

Commit and push compact code/results. Preserve T002 negative evidence. Do not launch the next stage until Research Lead review.
