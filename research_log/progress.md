# T001 execution log

## 2026-09-13 11:25 +08 — takeover and environment

Local directory was empty. Cloned word-ky/TTFL main at 757c51f; read both coordination files. No baseline implementation/checkpoint or project logs existed. No applicable parent AGENTS.md found by project inventory; user global rules apply.

Located existing workflow at D:/work/claude-autodl/autodl-workflow-clean. Created isolated remote root /home/wenchang/asdasdsad/wjq/TTFL on wenchang-PR4904W1 (liujianhua). Both GPUs report NVIDIA RTX A6000 through PyTorch with ~50.6GB free. NVML fails due kernel580.173.02/library580.178 mismatch; actual CUDA tensor allocation succeeds. No driver change. Use physical GPU0. Existing TOVD job left running.

Created TTFL/.venv with system site packages (Python3.12.12 / torch2.4.0+cu121), added torchvision0.19.0 and Pillow because imports demonstrated those were absent. No existing Digits data found in bounded wjq search. Dataset acquisition is required.

Implementation mode: Complete new small baseline (not V1 reproduction). Reuse torchvision dataset loaders and torch layers/SGD/Adam/evaluation primitives. No donor research code/checkpoint found or copied. Build baseline and exercise training/checkpoint roundtrip before adding affine adaptation. Then test neutrality/frozen weights/split disjointness and paired context runner. No SSL/meta/FL or T002 in this work package.

## 2026-09-13 11:29 +08 — first increments
Baseline train/save/load/metric test passed. Added explicit external gamma/beta state, supervised SGD with frozen eval-mode backbone, and BN recalibration. All 5 focused tests passed locally (3.993s). Initial data run failed: original MNIST endpoint404 and conda CA trust error; public CVDF mirror alone also hits CA trust error. Retrying with OS CA bundle, retaining certificate verification. No scientific run yet.

## 2026-09-13 11:37 +08 — acquisition and implementation
Code c0f6717 pushed. Local 7 tests PASS. Source release20260913-113613-ttfl-t001. SVHN train/test downloaded through official HTTP locally: MD5e26dedcc434d2e4c54c9b2d4a06d8373 / eb5a983be6a315427106f1b164d9cef3. MNIST-M LFS SHA256 verified. All uploaded through workflow Copy-ToAutodl. Corrected attempted Stanford HTTPS after hostname mismatch; no verification disabled. Source train/test URLs and raw archive retained. Preparing real smoke.

## 2026-09-13 11:39 +08 — full experiment launched
Smoke20260913-113707-ttfl-t001-smoke exit0 at11:37:33; remote7 testsPASS0.973s, all21 miniature real-data records completed. Full20260913-113757-ttfl-t001-full launched onGPU0 with unchanged c0f6717 code/config.446 expected records,20 epochs; no target-driven selection. Remote receipt/handoff contains exact IDs. No T001 scientific conclusion until complete.

## 2026-09-13T11:44:13.1676514+08:00 — T001 complete, acceptance FAIL
Full run exit0 at11:39:01+08, remote7testsPASS0.956s,446records. Downloaded complete receipt. Best affine42.933% vs41.633 baseline(+1.300pp), wrong-context margins.787/1.240/1.760pp, shuffled margin8.413pp. All12cellsFAIL; largest individualgain1.90pp. BNcorrect45.200%, queryCEworse. Fullmodel limited10step comparison42.913%. Neutraldiff0, checkpoint hashverified, zero IDoverlap, prediction accuracyrecomputed within1.9e-6pp. No implementation failure observed; limited protocol fails, does not prove universal affine impossibility. No SSL/meta/FL, no extra grid. Reporting and final evidence sync only.
