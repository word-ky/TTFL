# PFLlib benchmark handoff — COMPLETE

The direct-user task is complete: PFLlib MNIST/CIFAR10/CIFAR100/TinyImageNet,100 clients,10 fixed participants per round,100 rounds,1 local epoch,CUDA. Config: configs/pfllib_100c.json. Read coordination/CODEX_TO_CHATGPT.md and results/pfllib_100c/RESULTS.md for the final result and limitations. Do not rerun or start further research without a new instruction.

Runs: MNIST+CIFAR10=20260913-123718-ttfl-pfl-gpu0 (af29ce6); CIFAR100=20260913-124228-ttfl-pfl-c100-fixed (5db9d15); Tiny=20260913-124728-ttfl-pfl-tiny-gpu0 (5db9d15). All exit0. Tiny finished2026-09-13T13:03:22+08. No active TTFL tmux remains.

FedAvg/affine/prior accuracies: MNIST92.50/94.76/98.24; CIFAR1034.22/57.28/79.19; CIFAR10011.72/13.01/36.59; Tiny9.64/10.43/24.96. No distinctive affine advantage or dynamic-context proof. Single-seed stock-CNN budget, static label-skew, PFLlib merged labeled splits with client75/25 split.

Verification:9focusedtests;4datasets x100rounds x10participants;4000context records; zero support/query ID overlap; neutraldiff0; finite losses/norms; prediction-derived accuracies match. Exact hashes and paths are in results/pfllib_100c/verification.json and research_log/pfllib_receipts manifests. Initial data-ordering, CIFAR100 support-metric and summary-folder failures retained; repaired without extra full training.

Remote code/logs: /home/wenchang/asdasdsad/wjq/TTFL. Large data and runs: /media/wenchang/F/wjq/TTFL. Final model for each dataset: runs/<run-id>/<dataset>/global_state.pt. Full prediction arrays remain there and locally in research_log/pfllib_receipts (ignored by Git). Small JSON/config/log/split evidence is committed. Use existing workflow with this project's .autodl/config.json for future operations; do not use another project's last-run pointer.

Earlier T001 remains FAIL, with results/t001 and T001_HANDOFF.md preserved. Lead5647af7 old-SVHN T001B remains retained and unexecuted during this user-directed migration. No SSL/meta-learning or new sweep. Final delivery SHA is recorded in research_log/pfllib_delivery.json after evidence push.
