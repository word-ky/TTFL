# T002 — COMPLETE / Case B

2026-09-13: resumed on user request; fetched and fast-forwarded to Lead 94b02d5 (T002 issued in 37ac0de). Prior completed PFLlib receipt is not a reason to wait. A 15-minute thread heartbeat `ttfl-chatgpt` is now ACTIVE, reads latest GitHub instructions and resumes unfinished work without duplicate jobs.

Reuse: PFLlib read_client_data, ContextFedAvgCNN, affine.adapt, evaluate/state_hash. No training changes. Formal CIFAR10 checkpoint /media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10/global_state.pt, SHA256 260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb. Existing support_indices.json and dataset/Cifar10/split_manifest.json are authoritative.

Baseline health: remote 9 tests PASS before edits; actual saved client0 train tensor observed [-1,1], shape (533,3,32,32). PyTorch CUDA available, NVIDIA RTX A6000 GPU0 ~50.6GB free. Existing unrelated TOVD job untouched. NVML mismatch remains, actual CUDA works.

Fixed protocol before any new evaluation: 100 clients, up to64 saved support IDs, fixed labels, original query IDs; full-batch SGD .1/10 steps. Convert z=(x+1)/2; dark z*.45; contrast (z-per-image per-channel spatial mean)*.35+mean; noise clamp(z+N(0,.15^2),0,1); blur torchvision Gaussian5/sigma1.5 reflect padding; map back 2z-1. Noise uses CPU generator seeded by SHA256(global7,client,original_sample_id,transform), independent of ordering. Noise-image uses uniform[0,1] with same IDs/labels. Wrong contexts exactly match Lead mapping. Per-corruption gate uses weighted accuracies, macro/paired distributions also reported; >=2/4 PASS required.

Increment1: add deterministic transforms and numerical tests; increment2: evaluation-only runner reusing above modules and original files, 2-client real smoke then100-client evaluation. No severity or optimizer tuning based on results. All requested assertions retained.

## 2026-09-13T15:20:04.2242029+08:00 — CUDA job started
Run 20260913-151900-ttfl-t002-gpu0; release20260913-151830-ttfl-t002; runtime codee09a4a8; GPU0. Output /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-151900-ttfl-t002-gpu0/artifacts/{t002_smoke,t002_covariate}. Remote12 tests PASS; 2-client smoke passed required engineering checks; formal100-client evaluation running (17clients completed at latest check). Monitor this run; never launch a duplicate. Lead4455f46 RESEARCH_PROTOCOL.md read/merged viaeda9c6e, no T002 parameter changes.


## 2026-09-13T15:25:32.5325815+08:00 — FINAL COMPLETE
Formal100-client run exit0 at15:21:13+08. Onlybrightness passed;1/4below2/4 gate =>CaseB. Compact outputs results/t002_covariate; raw/logs/smoke research_log/t002_receipts/20260913-151900-ttfl-t002-gpu0. 2400records/2000adaptations, IDs/labels matched, no overlap, zero neutral diff, old clean baseline exact, unchanged checkpoint/model. Local12tests and remote12tests PASS. Optional prior correlations complete. Read final CODEX_TO_CHATGPT.md. No next stage launched; heartbeat remains enabled to receive new task. Do not duplicate completedT002.
