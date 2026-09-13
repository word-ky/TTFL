# TTFL: T001 supervised Fast Context Operator

Research assignment: `coordination/CHATGPT_TO_CODEX.md`. T001 only: fixed-source CNN,
zero-initialized intermediate affine state, supervised support adaptation and paired
context controls. No SSL, meta-learning or federation is implemented.

The affine operator has 448 scalars after three spatial blocks. The backbone stays
in eval mode and frozen; each adaptation allocates fresh zero state. All comparisons
start from one saved/reloaded checkpoint. BN uses a single full support batch with
momentum1. Prior adjustment uses Laplace-smoothed support/source training histograms.
See `research_log/T001_plan.md` and `configs/t001.json` for the complete fixed design.

Environment used: Python3.12 / PyTorch2.4.0+cu121 / torchvision0.19.0, numpy, scipy,
Pillow. Dataset loaders reuse torchvision's public interfaces. MNIST uses the
[CVDF mirror](https://github.com/cvdfoundation/mnist); USPS and SVHN retain
[torchvision0.19 dataset provenance](https://github.com/pytorch/vision/tree/v0.19.0/torchvision/datasets).
MNIST-M uses [this image mirror](https://github.com/mashaan14/MNIST-M), LFS SHA256
`bd899e2c286f1267955a73f6a374f31fc98764380bb8928dbfee37d8ab0393d6`.
No dataset images are committed or relicensed.

```bash
python -m unittest discover -s tests -v
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt python scripts/prepare_data.py --root /path/to/digits
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_t001 --data-root /path/to/digits --mnistm-train /path/to/digits/MNISTM/MNIST-M/training --output results/t001 --commit "$(git rev-parse HEAD)" --smoke
# Remove --smoke for the fixed full experiment.
```

A6000 project root: `/home/wenchang/asdasdsad/wjq/TTFL`. Use the existing local
AutoDL workflow with `AUTODL_CONFIG_PATH` set to this project's local `.autodl/config.json`.
Deploy project source, then invoke `TTFL_COMMIT=<sha> bash scripts/run_a6000.sh [--smoke]`.
Results contain JSON/CSV, sample IDs, predictions and global checkpoint. Keep large
files remotely and copy receipts into project-local `research_log/remote_runs`.
The 12-cell screen is exploratory; full raw results and controls must be reported.
