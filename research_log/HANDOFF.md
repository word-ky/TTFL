# TTFL active handoff

Current task: the user's PFLlib migration to MNIST, CIFAR-10, CIFAR-100 and TinyImageNet,100 clients and10% fixed participation, CUDA preferred. Read research_log/PFLLIB_HANDOFF.md, configs/pfllib_100c.json and the latest progress entries.

MNIST/CIFAR10/CIFAR100 formal100round baseline and context evaluations are complete. TinyImageNet run20260913-124728-ttfl-pfl-tiny-gpu0 is active on physical A6000 GPU0; its real2round smoke passed. Do not launch duplicate jobs. Its output is /media/wenchang/F/wjq/TTFL/runs/20260913-124728-ttfl-pfl-tiny-gpu0/TinyImagenet and log is /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-124728-ttfl-pfl-tiny-gpu0/train.log.

First3 receipts fetched to research_log/pfllib_receipts. Once Tiny completes, export its receipts using scripts/export_pfllib_receipts.py, extract into the same directory, run scripts/summarize_pfllib.py for all4, finalize coordination/CODEX_TO_CHATGPT.md and logs, commit/push/mirror. Model checkpoints remain on remote F drive; prediction arrays retained locally/remotely. Preserve all original failure receipts.

Prior T001 remains FAIL; its handoff is research_log/T001_HANDOFF.md. Lead commit5647af7 assigned old-SVHN T001B, retained but not run during the direct-user benchmark migration. No SSL/meta-learning or further experiment sweep authorized/inferred.
