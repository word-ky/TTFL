# CODEX -> CHATGPT

## Timestamp
2026-09-13T13:08:00.3869572+08:00 — USER-DIRECTED PFLlib benchmark COMPLETE

## User scope and research status
The user explicitly requested PFLlib, MNIST/CIFAR-10/CIFAR-100/TinyImageNet,100 clients,10% participation and GPU execution. This authorized the FL benchmark migration. The earlier T001 result remains FAIL. Lead5647af7's old-SVHN T001B assignment is retained but was not executed during this migration. No SSL/meta-learning or further sweep was started.

## Commit
- Unchanged upstream PFLlib subset:0169ba7e412c9856a08bb3faefab1e35f538a3c1, Apache2.0, provenance/license retained.
- MNIST/CIFAR10 experimental runtime:af29ce6235c4afa63db67e004942ad87eab937c8.
- CIFAR100/Tiny experimental runtime:5db9d154450d4a0fdbe7608b132318801c3df894. This fixes support statistics to use the true class count; training and adaptation update rules are unchanged.
- Final evidence commit is the commit containing this report; delivery pointer is research_log/pfllib_delivery.json after push.

## What changed
Imported PFLlib's FedAvg sampler, client training, weighted aggregation, CNN and data reader without semantic modifications. A thin entrypoint supplies the requested configuration and records final model/metrics and the sampled IDs. Exactly10 distinct clients participate per round, not a random participation ratio. PFLlib uses an inclusive loop, so its global_rounds argument is99 for exactly100 aggregation updates. Final scoring broadcasts the last aggregate, avoiding pre-update final-model confusion.

Configuration: one seed7, Dirichlet alpha0.1,100 rounds,1 local epoch, SGD0.005,batch10, original FedAvgCNN. Default budget and partition were stated to the user before execution; no alternative was supplied. All actual model training/inference ran on A6000 CUDA. Data preparation was CPU/I/O.

PFLlib standard data protocol: merge original labeled splits, partition, then split each client75/25. TinyImageNet uses labeled train+val and excludes unlabeled test. Saved IDs cover exactly70000/60000/60000/110000 samples with no duplication. These are not official held-out image benchmark test scores.

The final global checkpoint also receives a supervised static-client diagnostic:192 zero-initialized affine scalars after two spatial CNN blocks; backbone frozen; LR0.1,10 full-support steps; up to64 support examples per client. Correct, cyclic wrong client, shuffled labels, noise, random/no-write and prior controls are all reported. Prior uses Laplace1 support-label frequencies versus uniform classes. BN is N/A because the upstream CNN has no BN. No BN layer was silently added.

## Experiments run
| Dataset | Run | Physical GPU | Finish |
|---|---|---:|---|
| MNIST + CIFAR10 |20260913-123718-ttfl-pfl-gpu0|0|12:47:10 +08, exit0|
| CIFAR100 |20260913-124228-ttfl-pfl-c100-fixed|1|12:47:37 +08, exit0|
| TinyImageNet |20260913-124728-ttfl-pfl-tiny-gpu0|0|13:03:22 +08, exit0|

Date2026-09-13. Each dataset had a real2-round smoke before its100-round formal run. Initial unmodified upstream synthetic100-client/10-participant health test passed. Focused suite9 tests passed locally and remotely, covering all four shapes/classes, high-numbered-only support classes, exact neutral logits, frozen updates and T001 regressions.

Reported train/final-eval times after server construction: MNIST226.10s,CIFAR10238.48s,CIFAR100250.02s,Tiny715.72s. Peak allocated GPU memory:.460/.684/.719/4.517GiB. These timings exclude server construction, context evaluation and data preparation; they are run receipts, not a controlled efficiency comparison.

## Results table
Sample-weighted accuracy across all100 clients; one fixed seed/budget, no tuning.

| Dataset | FedAvg | Affine correct | Gain pp | Affine wrong | Correct−wrong pp | Shuffled | Noise | Prior correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST |92.498|94.755|+2.258|92.019|+2.736|94.687|90.879|98.244|
| CIFAR-10 |34.218|57.278|+23.060|25.833|+31.445|58.049|59.937|79.194|
| CIFAR-100 |11.718|13.008|+1.290|11.319|+1.689|13.327|11.931|36.590|
| TinyImageNet |9.645|10.425|+0.781|8.820|+1.605|8.758|4.020|24.957|

Random/no-write accuracies:92.486/34.184/11.758/9.645. All exact values, macro-client accuracies and additional context/prior rows are in the raw receipts. Summary JSON/CSV/Markdown: results/pfllib_100c.

## Diagnostics and interpretation
The requested benchmark migration is complete; a distinct fast-operator advantage is **not established**. Prior beats affine on every dataset. MNIST correct-vs-shuffled is only0.068pp; CIFAR10 and CIFAR100 shuffled-label affine outperform correct-label affine. CIFAR10 noise also outperforms correct affine. Therefore large correct-minus-wrong margins under label skew cannot, alone, establish semantic or dynamic-context reading. Tiny affine gain is small despite better correct than shuffled/noise. No claim of successful T001, T001B, dynamic-context disentanglement or novel method superiority follows.

This is a single-seed preliminary budget using stock CNN/SGD. Low CIFAR/Tiny global accuracies are reported as measured and do not establish converged/optimal FedAvg performance. No learning-rate, epoch or checkpoint selection was changed after inspecting results.

## Verification
PASS for all4 datasets:100 updates;10 unique client IDs in every round; CUDA models;1000 context records each(4000 total); all train/test sample IDs unique within the entire partition; correct/wrong support IDs disjoint from current query; checkpoint hash shared across every comparison. Neutral maximum absolute logit difference0.0. No nonfinite loss/norm diagnostic values. Independent prediction-derived accuracy errors <7.7e-6pp, consistent with float32 rounding. Final remote checkpoint SHA256 matches the recorded hashes. Transfer manifests:36/36 first3 files and12/12 Tiny files match.

See results/pfllib_100c/verification.json and numerical_diagnostics.json. Raw records, sampling histories, supports, config, split IDs, environment and original logs are committed under research_log/pfllib_receipts. Prediction NPZ arrays are retained locally/remotely but ignored in Git. Full models remain on the remote F drive; paths and SHA hashes are recoverable from summary/run IDs and verification.

## Failures / uncertainties
- Tiny data preparation first reached ZIP before SCP completed: BadZipFile at open. After complete transfer, only Tiny preparation was rerun; no dataset protocol change.
- CIFAR100 smoke baseline completed, then support statistics defaulted to10 classes and failed on labels>=10. Fixed the two calls, added a focused regression, resumed from the saved smoke checkpoint, and ran the full experiment once.
- Local aggregation initially confused data/MNIST with a run folder; fixed that exact path-selection collision. Scientific code/results were unchanged.
- NVML driver/library mismatch remains on the machine, but actual CUDA tensors/models and completed experiments work. No global driver change. Missing PFLlib dependencies were installed in the TTFL venv and frozen in the environment receipt.

All original failure logs are retained. No duplicate full training, broad sweep or unrelated project modification occurred.

## Artifact locations and recommended next action
Code: /home/wenchang/asdasdsad/wjq/TTFL. Large data/models/results: /media/wenchang/F/wjq/TTFL. Local project root remains D:/work/fightccfa-agin/CVPR2027/TTT-FL. All TTFL run sessions have exited; unrelated TOVD session was left running.

Await the user's/Research Lead's next instruction. A useful next research decision is how to separate label-prior information from semantic/current-context evidence before adding more machinery. This is a recommendation only; no new experiment or research stage has been launched.
