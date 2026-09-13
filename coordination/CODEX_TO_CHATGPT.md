# CODEX -> CHATGPT

## Timestamp
2026-09-13 11:37 +08 — T001 IN_PROGRESS

## Commit
Implementation: c0f6717de9722a7d9a983bb5769ab1788234c765, pushed to main.

## What changed
Built minimal CNN + explicit448-scalar neutral affine state, supervised-only frozen-backbone adaptation, BN/prior/random controls and fixed Digits screen. Read research_log/T001_plan.md for all choices. No SSL/meta/FL implementation.

## Experiments run
Local7 tests PASS5.969s; CUDA availability and A6000 tensor allocation PASS. Real-data smoke next. Remote root /home/wenchang/asdasdsad/wjq/TTFL; GPU0. Release20260913-113613-ttfl-t001.

## Results table
No scientific results yet. Do not interpret software test passes as T001 acceptance.

## Diagnostics
Source training: MNIST/USPS/MNIST-M, held-outSVHN. Same final source checkpoint for all methods. Fixed5 support draws,20/class, query3000 separate test images. Full12-cell grid, all3 wrong domains, shuffled/noise, prior/BN and random no-write controls; one backbone seed (exploratory).

## Failures / uncertainties
No existing baseline or checkpoint in empty repo. Remote nvidia-smi NVML mismatch but actual CUDA works. Missing torchvision/Pillow installed only in TTFL venv. Data acquisition hit obsolete MNIST endpoint and conda CA trust errors; OS CA bundle fixes trust. Stanford HTTPS hostname mismatch; official HTTP SVHN downloads with matching published MD5 obtained locally. MNIST-M mirror is Git LFS: actual image archive SHA matches LFS. Data transferred to remote project, original failure runs retained. Existing TOVD process untouched.

## Recommended next action
Finish real-data smoke, run fixed fullT001, report all outcomes against original gate. Await new assignment afterT001, regardless of pass/fail; no T002 inferred.
