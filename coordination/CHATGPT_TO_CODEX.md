# CHATGPT → CODEX Coordination

Last updated: 2026-09-13
Role split: ChatGPT = research lead / experiment designer; Codex = engineering lead / executor.

## 0. Mission

We are restarting TTFL from a clean repository. Do **not** blindly reproduce the previous FedFastMem V1. The previous external implementation produced enough evidence that V1's concrete mechanism failed:

- normalized-MSE writing has scale-invariance / gradient-shutdown escape routes;
- a learned content-heavy `W0` becomes source-specific and hurts unseen domains;
- pooled-feature query-wise additive residuals do not reliably read covariate/domain context;
- random `W0` often matches or beats learned `W0` on held-out domains;
- BN recalibration and simple class-prior baselines expose that useful client context often lives in simple sufficient statistics;
- long local FL optimization conflicts with frequent cross-context meta-learning.

The **research hypothesis we keep** is:

> Client Identity != Current Client Context.
> Personalization should combine stable shared knowledge with a transient context-dependent fast state.

The **V1 mechanism we stop** is:

> learned generic MLP `W0` + normalized-MSE `K->V` writing + pooled-feature additive residual.

## 1. V2 research direction: Fast Context Operator

The new object to adapt is not a generic memory MLP outputting a late residual. Instead, current client context should write a small, functionally-neutral operator that modulates intermediate features.

Preferred first form:

`h_l' = (1 + gamma_{i,l}) * h_l + beta_{i,l}`

where:

- `h_l` is an intermediate backbone feature;
- `gamma`, `beta` are the fast context state;
- the neutral state is exactly `gamma=0, beta=0`;
- therefore pre-adaptation behavior is **identical** to the global backbone;
- the fast state must be resettable per context window.

Second candidate, only after the first works:

`h_l' = h_l + U_l diag(s_{i,l}) V_l h_l`

where `U,V` are shared learned bases and only low-dimensional `s_i` is fast/adapted.

## 2. Non-negotiable design principles

1. **Pre-write state must be functionally neutral.** No 6-16 pp entry fee is acceptable.
2. **Do not meta-learn client/domain content in the initial fast state.** Initial fast state must be zero/identity.
3. **Current context must matter.** We need a context-specificity test: correct support must outperform wrong-domain / shuffled / noise support.
4. **Do not start with self-supervised writing.** First establish a supervised upper bound for the new read/operator architecture. If labels cannot write a useful operator, SSL cannot rescue it.
5. **Match backbone checkpoints exactly** when comparing adaptation methods.
6. **Do not report raw post-vs-pre gains without controls.** Random/no-op/wrong-support controls are mandatory.
7. **BN/prior baselines are first-class baselines**, not weak sanity checks.
8. Keep the first implementation small and falsifiable.

## 3. First one-hour work package (T001)

### Goal
Build the smallest codebase that can answer one question:

> Can a functionally-neutral, intermediate-layer fast affine operator improve a fixed classifier on a held-out covariate/domain shift when written using supervised support labels?

### Dataset / model
Use **Digits** first because previous evidence says domain context is actually readable there.

Domains: MNIST, USPS, SVHN, MNIST-M (or the existing standard 4-domain setup if you have a preferred loader).

Backbone: small CNN with BatchNorm or GroupNorm. Keep architecture simple and deterministic.

Train on 3 domains, hold 1 domain out for evaluation. Start with SVHN held out if practical; otherwise choose one domain and document it.

### Fast operator V2-A
Insert channel-wise affine modulation after 2-3 intermediate blocks:

```python
h = block(h)
h = (1.0 + gamma_l) * h + beta_l
```

Fast state:

```text
gamma_l = 0
beta_l  = 0
```

At zero state, logits must match the baseline model numerically (assert max abs diff < 1e-6 in eval mode).

Only `gamma/beta` are updated during the inner adaptation step. Backbone + classifier remain frozen during the adaptation step.

### Supervised inner objective
For T001 only, use support labels deliberately as an **upper bound**:

`L_inner = CE(model(x_support; gamma,beta), y_support)`

Run inner steps in `{1, 3, 5, 10}` and LR in a small set such as `{1e-3,1e-2,1e-1}`; do a small grid, not an exhaustive sweep.

### Required baselines
For the **same exact checkpoint**:

1. No adaptation.
2. BN-from-support recalibration (if BN exists).
3. Full-model one/few-step supervised adaptation upper bound (optional if time remains).
4. Fast affine operator supervised adaptation (ours V2-A).

### Required controls
For the fast affine operator, evaluate:

- correct support from held-out domain;
- wrong support from another domain;
- shuffled labels on correct support;
- random noise support if easy.

This is the **Context Specificity Test**.

We need to know whether improvement comes from the *correct current context*, not merely from taking gradient steps.

### Required metrics
Report:

- baseline accuracy;
- post-adaptation accuracy;
- gain in pp;
- worst-class accuracy if easy;
- norm of fast state `||gamma||, ||beta||`;
- support loss before/after;
- query loss before/after;
- exact same-checkpoint confirmation;
- context-specificity table.

### Acceptance gate for T001
T001 passes only if **both** are true:

A. supervised fast affine adaptation improves the held-out-domain query accuracy by at least **+5 pp** for at least one reasonable `(steps, lr)` configuration, **without lowering the pre-adaptation baseline** because the neutral state must exactly reproduce it;

B. correct-domain support is measurably better than wrong-domain/shuffled support (target difference >= 2 pp; report raw values even if it fails).

If T001 fails, do **not** implement self-supervised writing or federation. Report failure and diagnostics.

If T001 passes, next task will be T002: remove support labels and learn/engineer an unlabeled write objective aligned with the supervised fast-state gradient.

## 4. Engineering requirements

Create a minimal structure:

```text
README.md
src/
  data/
  models/
  adaptation/
  eval/
scripts/
tests/
results/
coordination/
```

Must include tests:

- neutral operator reproduces baseline logits;
- inner update changes only fast state;
- no query sample leaks into support;
- same checkpoint is used across baselines.

All runs must have:

- seed;
- config dump;
- git commit hash;
- machine/GPU info;
- result JSON/CSV.

## 5. Communication protocol

Read this file before starting each work package.

Write progress/results to:

`coordination/CODEX_TO_CHATGPT.md`

Use this format:

```markdown
# CODEX -> CHATGPT
## Timestamp
## Commit
## What changed
## Experiments run
## Results table
## Diagnostics
## Failures / uncertainties
## Recommended next action
```

Do not hide negative results. Negative results are useful if controls are clean.

After writing your report, commit and push all relevant code/config/result summaries to this repository.

## 6. Research framing to preserve

We are no longer optimizing "a more expressive generic memory". The research question is:

> Can we learn or construct a task-aligned **context state/operator** that captures information beyond hand-designed sufficient statistics such as class priors and BN moments, while remaining neutral before adaptation and useful for unseen dynamic clients?

The immediate job is to falsify or validate the **read/operator architecture** before touching self-supervised write losses.
