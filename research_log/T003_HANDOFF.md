# T003 ACTIVE — Moment-Written Fast Context Operator

Lead2b5e7a8 fetched/read and fast-forwarded on2026-09-13. No duplicate TTFL job active; previousT002 complete. Keep15-minute heartbeat; no next stage without instruction.

Reuse existing ContextFedAvgCNN, T002 corruption/seed code, support_indices, query partitions, read_data/read_client_data, evaluate, state_hash. New code only analytic writer and evaluation runner. Same192 diagonal affine state; no optimizer/labels passed to writer. Same CIFAR10 checkpointSHA260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb. All100train partitions in deterministic client/index order used once for reference, no cap. Exclude every query ID from reference. Batch256, double first/second-moment accumulation overN,H,W; varianceclamp>=0, sigma=sqrt(var+1e-5).

Before query results: sequential layer1 calibrated before layer2 stats; reference remains unadapted clean model. eps1e-5, scaleclamp[.25,4]. Mismatch=channelmean(abs(mu-mug)+abs(sigma-sigmag)), measured from actual calibrated float32 activations; verify nonincrease with1e-5 absolute numerical tolerance. Report all >10%clamp conditions prominently. Clean support/clean query sanity row is descriptive, no success gate. Four2pp gates includingnoise,>=2/4 overallM-A. If partial evidence, report full metrics before case interpretation.

Increment1: moment reference/writer unit tests including direct-stream equivalence, exact sequential equation,192state, frozen/no-grad and clamp handling. Increment2: reference training-only provenance, real2-client smoke then100-client inference using same saved reference; compare copied T002 none row, save raw predictions/diagnostics and perclient. No T002 CE recomputation. Current implementation in progress.

## 2026-09-13T15:38:20.3798113+08:00 — RUNNING
A6000GPU0 run20260913-153752-ttfl-t003-gpu0, release20260913-153748-ttfl-t003, code7bc31ded5e2a0dc6ad7e20b7688240c34c24c366. Artifacts /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-153752-ttfl-t003-gpu0/artifacts/{t003_moment_context,t003_smoke}. Reference collected once at formaloutput/reference_moments.pt, then reused bothsmoke/formal. Monitor this run; no duplicates.

## 2026-09-13T15:40:19.7832584+08:00 — SSH connection timeout
Two consecutive log reads to existinghost202.101.162.22:8220 timed out duringSSHconnection (exit255). Last successful observation formalT003 completedclient21; reference andsmoke already finished. Remote tmux job may continue; completion unverified. Reconnect to20260913-153752-ttfl-t003-gpu0 and fetch its artifacts; do not launch duplicate/reference recomputation.
