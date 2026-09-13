# TTFL recovery handoff — T001 COMPLETE / FAIL / WAITING FOR LEAD

Read coordination/CHATGPT_TO_CODEX.md, coordination/CODEX_TO_CHATGPT.md and research_log/T001_plan.md. T001 finished; do not rerun unchanged task or start SSL/meta/FL/T002. User asked for A6000 in new wjq subdirectory; completed at /home/wenchang/asdasdsad/wjq/TTFL.

Local root D:/work/fightccfa-agin/CVPR2027/TTT-FL, origin word-ky/TTFL main. Remote A6000 host wenchang-PR4904W1, project .venv/shared/digits, physicalGPU0. Existing workflow D:/work/claude-autodl/autodl-workflow-clean; AUTODL_CONFIG_PATH=D:/work/fightccfa-agin/CVPR2027/TTT-FL/.autodl/config.json. Connection local-only. PyTorch CUDA works despite existing NVML mismatch; no driver changes.

Tested code c0f6717de9722a7d9a983bb5769ab1788234c765; release20260913-113613-ttfl-t001. Smoke20260913-113707-ttfl-t001-smoke exit0,7tests+21records. Full20260913-113757-ttfl-t001-full exit0 at2026-09-13T11:39:01+08,7tests+446records. No activeTTFL training remains.

Baseline41.633%; bestaffine42.933%(+1.300pp), correct-minus-wrong.787/1.240/1.760pp; all12cellsFAIL. BN45.200%; limitedfullmodel42.913%. Neutraldiff0,448fastparams, frozen-state checksPASS. One backbone/five support draws; exploratory, not confirmatory. No new experiments after failure. Return toLead for read-operator/diagnostic decision.

Full fetched receipts remain at research_log/remote_runs/<run>/artifacts/t001. Git-visible summary/raw JSON/CSV/splits/config/env/training/log in results/t001; predictions.npz and global.pt retained locally (ignored) and remote runs/<run>/artifacts/t001. Manifest gives byte hashes. Checkpointsha ffcf2dfa205c72094a54f592b1fef8b44dc830a3242acc6d917909b85c1df997. One-off receipt check PASS (results/t001/verification.json). Original data acquisition failure logs retained. No project artifacts deleted.

Final evidence/delivery commits: see research_log/delivery.json. Coordination report committed/pushed to GitHub and mirrored under remote project root with final results/logs. Await explicit next work package.
