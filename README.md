# TTFL: PFLlib benchmark and neutral context diagnostics

The current direct-user setup uses **PFLlib,100clients,10%fixed participation** on
**MNIST,CIFAR-10,CIFAR-100,TinyImageNet**, with CUDA training and inference.
The earlier T001 Digits diagnostic is preserved in results/t001 and remains FAIL.
No SSL or meta-learning is implemented. The separately assigned old-SVHN T001B
has not been run during this user-directed benchmark migration.

## Current fixed configuration

See configs/pfllib_100c.json: Dirichlet alpha0.1, seed7,100aggregation rounds,
1local epoch, SGD0.005, batch10, FedAvgCNN. Every round chooses exactly10distinct
clients; random_join_ratio is false. Evaluate all100clients every20rounds and
again after the final aggregation. PFLlib's inclusive round argument is99.

PFLlib's standard dataset protocol merges the original labeled splits and then
partitions/splits client data75/25. TinyImageNet uses labeled train+val, excluding
unlabeled test. Exact sample IDs and per-client counts are retained. These scores
are not the official held-out image benchmark test scores.

Unchanged upstream FedAvg server/client/model/data modules are imported from
PFLlib0169ba7, with Apache2.0 license/provenance in third_party/PFLlib.
The TTFL entrypoint supplies configuration and records selections/results.

The final checkpoint is also evaluated with a supervised, zero-initialized
intermediate affine state after the two CNN convolution blocks:192fast scalars.
Only this state updates; backbone remains frozen. FixedLR0.1/10steps/support64
(or all available if fewer). Controls: correct client, cyclic wrong client,
shuffled labels, noise, random/no-write, and support-label prior against uniform
classes. The upstream CNN has no BN, so BN is N/A. This is a static-client
personalization diagnostic, not proof of dynamic-context reading.

## Run

```bash
python -m unittest discover -s tests -v
python scripts/prepare_pfllib.py --dataset MNIST --root /media/wenchang/F/wjq/TTFL --mnist-root /home/wenchang/asdasdsad/wjq/TTFL/shared/digits/MNIST
CUDA_VISIBLE_DEVICES=0 python scripts/run_pfllib.py --dataset MNIST --data-root /media/wenchang/F/wjq/TTFL/dataset --output /path/to/new-run/MNIST --commit COMMIT_SHA --smoke
python scripts/eval_pfllib_context.py --baseline /path/to/new-run/MNIST
# Remove --smoke for100rounds; each output directory is a new run.
```

The existing AutoDL workflow controls deployment/execution. Set
AUTODL_CONFIG_PATH to this project's .autodl/config.json and deploy first.
Then use CUDA_VISIBLE_DEVICES and TTFL_COMMIT with scripts/run_pfllib_gpu.sh.
That script runs a2round smoke then100rounds plus context evaluation per dataset.
Large data/checkpoints live under /media/wenchang/F/wjq/TTFL; code and run logs
remain under /home/wenchang/asdasdsad/wjq/TTFL. Read research_log/PFLLIB_HANDOFF.md
before resuming; do not duplicate active runs. Original acquisition/failure logs
are retained and fixes are recorded in research_log/progress.md.

## Earlier T001

The same-checkpoint SVHN affine screen is documented in research_log/T001_plan.md,
configs/t001.json, scripts/run_t001.py and results/t001. The3-block CNN uses448fast
scalars and was run in a separate recorded environment. Pre-adaptation logits
matched exactly; its best gain+1.30pp did not pass the required+5pp threshold.
Datasets: [MNIST CVDF](https://github.com/cvdfoundation/mnist),
[torchvision datasets](https://github.com/pytorch/vision/tree/v0.19.0/torchvision/datasets),
[MNIST-M image mirror](https://github.com/mashaan14/MNIST-M).
No dataset images are committed or relicensed.
