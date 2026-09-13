# CODEX -> CHATGPT

## Timestamp
2026-09-13T12:54:49.5781765+08:00 — PFLlib benchmark: three datasets complete, TinyImageNet running

## Active user instruction
The user explicitly requested PFLlib with MNIST, CIFAR-10, CIFAR-100 and TinyImageNet,100 clients,10% fixed participation, and GPU execution. This authorizes the FL benchmark migration despite the older pause. Lead5647af7's old-SVHN T001B remains retained but unexecuted. T001 remains FAIL. No SSL/meta-learning is implemented.

## Configuration and implementation
- PFLlib upstream0169ba7, Apache2.0; unchanged FedAvg sampler, client updates, aggregation and CNN.
- Default configuration stated to user: Dirichlet alpha0.1, seed7,100 aggregation rounds,1 local epoch, SGD0.005, batch10. Exactly10 distinct clients per round; random_join_ratio=False. PFLlib inclusive argument99 gives100 updates.
- Standard PFLlib labeled-split merge then75/25 client split, with saved sample IDs. TinyImageNet uses train+val, excludes unlabeled test. These are not official held-out benchmark accuracies.
- Posthoc supervised affine:192 fast scalars after two spatial blocks, zero reset, frozen backbone, LR0.1/10steps, up to64 support examples. Correct/wrong/shuffled/noise/random/prior controls on the same final checkpoint. BN N/A: upstream CNN has no BN. Prior uses Laplace1 label counts against uniform classes.

## Runs and tests
- MNIST+CIFAR10:20260913-123718-ttfl-pfl-gpu0, physical A6000 GPU0, exit0 at12:47:10.
- CIFAR100:20260913-124228-ttfl-pfl-c100-fixed, physical GPU1, exit0 at12:47:37.
- TinyImageNet:20260913-124728-ttfl-pfl-tiny-gpu0, physical GPU0, active after real2round smoke passed; last inspected31/100 formal rounds.
- Original upstream synthetic100-client health passed. Focused suite9 tests passed locally and on repaired remote execution. All four tensor shapes, high-numbered classes, zero-state equivalence, frozen adaptation and T001 regressions covered.
- MNIST/CIFAR10 runtime af29ce6; CIFAR100/Tiny runtime5db9d15. The latter changes support statistics to pass the actual class count; it does not change training or adaptation updates. Initial CIFAR100 smoke failed before full training; resumed evaluation from its saved checkpoint, then ran full training once.

## Completed results
Sample-weighted accuracy across all100 clients, one seed, fixed budget.

| Dataset | FedAvg | Affine correct | Affine wrong | Shuffled | Prior correct |
|---|---:|---:|---:|---:|---:|
| MNIST |92.50|94.76|92.02|94.69|98.24|
| CIFAR-10 |34.22|57.28|25.83|58.05|79.19|
| CIFAR-100 |11.72|13.01|11.32|13.33|36.59|
| TinyImageNet |running|running|running|running|running|

These results do not establish a distinctive fast-operator advantage: prior is stronger on every completed dataset and shuffled-label affine nearly matches or exceeds correct-label affine. Static label-skew personalization is not dynamic-context proof. No settings were changed after inspecting results.

## Evidence and operational issues
First3 complete JSON/config/prediction/split receipts fetched to research_log/pfllib_receipts; committed metadata at306fcd5. Prediction arrays retained locally and remotely (ignored in Git). Large data/checkpoints are under /media/wenchang/F/wjq/TTFL, code/logs under /home/wenchang/asdasdsad/wjq/TTFL. Existing NVML mismatch persists but actual model parameters/training are CUDA.

Data transfer ordering caused Tiny's first ZIP open to fail before upload completed; only its preparation was rerun after complete transfer. CIFAR100 high-label support metric failure was repaired as described above. Original failure logs retained. No broad sweep or duplicate full training.

## Next action
Finish the existing Tiny run, export its complete receipt, run scripts/summarize_pfllib.py for all4, publish final result/verification and update recovery logs. No further research stage inferred.
