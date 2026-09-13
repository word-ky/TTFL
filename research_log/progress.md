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

## 2026-09-13T12:34:59.5154541+08:00 — user-directed PFLlib migration
User requests4datasets(MNIST/Cifar10/Cifar100/TinyImagenet),100clients,10percentGPU. Read fetchedLead5647af7 T001B; currentuserbenchmarktakespriority, no oldSVHNrunstarted. PFLlib official0169ba7 subset imported; codeb365411 baselinehealthrun20260913-123218-ttfl-pfl-health exit0,100clients10selected2rounds onCUDA. Smoke command metadata used pending-baseline-import before final receipt: exactsourceb365411, no scientificresult. PFLdata on/media/wenchang/F/wjq/TTFL due/home19GB. Missing sklearn/matplotlib/ujson installedprojectvenv; resultingnumpy2.5.3/imports and actualFedAvghealthPASS. Added PFLneutraladapter; local8testsPASS3.511s(allfourshapes+multiclassmetrics), priorT001regressionPASS. Defaultalpha.1/100rounds/1localepoch stated; no user override received.

## 2026-09-13T12:38:40.9086069+08:00 — real-data dispatch and acquisition ordering failure
MNIST/Cifar10/Cifar100 partitions complete. GPU0 run20260913-123718-ttfl-pfl-gpu0 launched with testedaf29ce6. Tiny preparation reachedarchivebeforeSCPfinished:BadZipFile duringopen, noextractionortraining. This was executionordering error; uploadnowcomplete248100043bytes. RerunonlyTinydata after transfercompletion; no changes to partition/experiments or extra validators. Preserveoriginal20260913-123506 data log.

## 2026-09-13T12:40:16.4418505+08:00 — two GPU datasets active
GPU0 run20260913-123718-ttfl-pfl-gpu0 passed MNIST2round+2clientcontext smoke, main100rounds reachedround40. GPU1 run20260913-123913-ttfl-pfl-c100-gpu1 launched Cifar100 smoke/main onphysicalGPU1. Tinydata retry20260913-123833-ttfl-pfl-tinydata writingclientdata; Tinytrainingnotyetlaunched. CIFAR archivesmatchofficialMD5 c58f30108f718f92721af3b95e74349a / eb9058c3a382ffc7106e4002c42a8d85.

## 2026-09-13T12:42:32.5327952+08:00 — observed CIFAR100 smoke evaluator repair
Run20260913-123913-ttfl-pfl-c100-gpu1 completed2roundbaseline thencontextfailed beforemain: supportstatistics callstilldefault10classes; supportonlyclassIDs>=10 led min(empty). Fixedtwo calls to passactualnum_classes, added highnumbered-only regression. Local9testsPASS3.733s. Commit5db9d15. No baseline/datachange. Resumecontextfromexistingsmokecheckpoint; formalCifar100runnotpreviouslystarted. MNIST/Cifar10have10classesandcontinueoriginalrun unaffected.

## 2026-09-13T12:44:53.6275462+08:00 — repaired CIFAR100 and completed MNIST
Cifar100 fixedrun20260913-124228-ttfl-pfl-c100-fixed active onGPU1,9remotetests andsaved-smoke evaluationpassed thenformal100roundtraining. GPU0MNIST formalbaseline/contextcomplete; Cifar10active. Tinydataretrytest94/100, noTinytrainingyet. Analysis/export scripts added forcomplete receipts, not changingactive scientificcode.
