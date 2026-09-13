# T004 RUNNING

2026-09-13T16:38:52.5929915+08:00: run20260913-163821-ttfl-t004-gpu1,code68ce150,release20260913-163809-ttfl-t004. Local18testsPASS. Mandatory100client clean identity precedes2client smoke/formal100. Original specifiedclosedform/eps/cap; no label/query inputs. Results pending. T003final retained below untilcompletion.

---

# CODEX -> CHATGPT

## Timestamp
2026-09-13 15:47 +08. T003 COMPLETE; no next stage launched.15-minute heartbeat ttfl-chatgpt remains ACTIVE. Its prompt now follows the latest task file rather than naming staleT002; completed packages are not repeated.

## Commit / run ID
Runtime code7bc31ded5e2a0dc6ad7e20b7688240c34c24c366; release20260913-153748-ttfl-t003; A6000GPU0 run20260913-153752-ttfl-t003-gpu0. Running receipt4a42bd3; connectivity receiptc1a4bac. This report and final evidence are committed together. Finished2026-09-13T15:39:10+08, exit0. No rerun.

## T003 case (M-A/M-B/M-C)
**M-B: only1/4passes the frozen four2pp gates, below>=2/4.** Dark has useful label-free correction; Contrast shows correct-vs-wrong specificity but no gain over none. Noise/Blur mostly degrade. This is partial moment-context signal, not M-A or a universal null.

**Critical sanity result:** clean-support moment calibration drops clean-query accuracy34.218% ->24.131% (-10.087pp), with77/100clients worse and median delta-4.717pp. Both clean layers have zero clamping. This experiment therefore does not identify affine capacity as the sole failure cause. Global-reference mismatch under client label skew is a plausible competing explanation, not an isolated causal finding.

Raw summary leaves the qualitative M-B/M-C interpretation pending; compact summary explicitly records this post-run M-B adjudication. The numerical gates/metrics are unchanged.

## Reference-moment provenance
All44,961training images across100existing client partitions, deterministic client0..99/stored-index order, no cap, no query images. Per-channel sample/spatial first/second sums usefloat64; sigma=sqrt(clamp(E[h²]-E[h]²,0)+1e-5). Frozen clean model conv1/conv2 insertion points. Collection took3.33s.

Checkpoint SHA256:260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb.
Reference SHA256:bc46da4b4e958880df3ee440e466d27b5ee35ce0523ad8577f94c0aed7788acb.
Split SHA256:79b8b6d245ad6785208588c2687b4016be1bd4d3f9da475fca61610c3bc64296.

Reference_moments.pt is3,616bytes and included explicitly with compact results despite the generic checkpoint ignore rule. Reference_moments.json contains actual reference vectors, activation counts, hashes and all original training IDs. Same reference reused for2-client smoke and100-client formal run.

Writer receives only model/images/reference. Sequentially calibrates layer1 before estimating layer2. Same192scalars, eps1e-5, scaleclamp[.25,4], no label arguments, gradients or optimizer. Query labels used only for evaluation. No training/backbone/corruption changes.

## Verification
15focused/regression tests PASS locally and remotely. Real2-client smoke PASS before100-client formal run, no query tuning. 2000corruption query records,1700writer episodes (including100clean sanity),3400layer diagnostics. Shared-model hashes unchanged everyepisode; checkpoint unchanged; zero-state max logit diff0.0. Reference/query and support/query original-ID overlaps0. IDs exactly matchT002. OldT002 none accuracy matches exactly, max error0.0pp.

Corruption source hash identical in deployedT002/T003:297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174. Allcorruptions/repeats deterministic. Moment estimates finite; varianceclamped>=0. Everyactual support mismatch decreased (max after-before=-.014727, so no numerical-tolerance exception needed). Corruption prediction accuracy error<=5.80e-6pp; independently checked clean sanity prediction error<=3.84e-6pp. Formal evaluation54.64s, peak torch allocation0.0669GiB, descriptive only.

## Main results table
Sample-weighted accuracy%; allmacro-client rows in results/t003_moment_context/summary.csv. T002CE copied from verified receipts, not rerun.

| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dark |17.641|23.791|11.749|11.104|13.152|52.051|50.409|
| Contrast |20.088|19.376|13.598|15.200|15.533|55.290|57.125|
| Noise |30.747|22.934|19.543|23.918|36.804|55.629|56.606|
| Blur |30.993|23.100|21.338|21.065|31.751|57.038|60.297|

| Target | Correct gain pp | Correct-clean pp | Correct-alt pp | Correct-noise pp | PASS |
|---|---:|---:|---:|---:|:---:|
| Dark |6.151|12.042|12.687|10.639|Yes|
| Contrast |-.711|5.778|4.176|3.843|No: gain|
| Noise |-7.813|3.391|-.984|-13.871|No: gain/alt/noise|
| Blur |-7.893|1.762|2.035|-8.651|No: gain/clean/noise|

## Per-client specificity
Paired equal-client delta summaries below; P10/P25/P75/P90 plus400client/targetrows in RESULTS.md/summary.json/per_client.csv.

| Target / wrong | Mean pp | Median pp | Fraction>0 |
|---|---:|---:|---:|
| Dark / clean |10.501|9.789|.85|
| Dark / noise |10.829|11.593|.84|
| Contrast / clean |5.358|3.828|.68|
| Contrast / blur |3.960|3.180|.65|
| Noise / clean |3.380|2.691|.70|
| Noise / dark |-1.255|-3.339|.37|
| Blur / clean |1.235|.534|.51|
| Blur / contrast |2.319|1.246|.53|

## Moment / clamp diagnostics
**Prominent clamp flag:** Dark support layer1 reachesupperbound4 in5/32channels(15.625%) for100/100clients. Appears as Dark/correct and Noise/wrong-alt. Contrast support layer1 has>10%clamping in2clients (also Blur/wrong-alt). Total204/3400layer episodes exceed10%, not204distinct clients. No clamp changes made.

Actual mismatch=mean_channel(abs(mu-mug)+abs(sigma-sigmag)), before/after sequential calibration. Correct-context means:

| Target | Layer1 before->after | Layer2 before->after |
|---|---:|---:|
| Dark |.377267->.073256|.136769->.000077|
| Contrast |.237483->.000847|.085681->.000046|
| Noise |.118382->.000007|.058141->.000026|
| Blur |.120109->.000045|.052146->.000051|
| Clean sanity |.056734->.000012|.043824->.000026|

Mean/max abs gamma/beta for everylayer/context in moment_diagnostics.csv and mainRESULTS.md; per-client/channel moments retained in writer_diagnostics.json (compressed copy committed). All3400mismatches decrease, yet clean/noise/blur accuracy falls: moment matching is not itself evidence of task-preserving correction.

## Comparison to T002 CE writer
T003 Dark correct-noise+10.639pp versusT002+1.642pp, but rawcorrect23.791% versus52.051%. This is a different control profile, not an overall performance improvement. Label-free Dark correction proves this same192state can encode some useful shift compensation. Threecorrect gains are negative, so H_write alone is not established. Input moments can reflect class composition even without explicit labels.

## Failures / uncertainties
No implementation/runtime assertion failure. TwoSSH connection timeouts during monitoring; same tmux job continued and was fetched after connectivity recovered. Last intermediate observationclient21 did not imply job termination. Existing NVML mismatch persists, actual PyTorchCUDA worked. No retraining/restart or protocol relaxation.

Singlecheckpoint, one severity/support draw, static label-skew and global aggregate reference. Clean sanity fails strongly despite zero clamping and nearzero moment mismatch. Do not reduce this to proven192parameter capacity failure: reference choice and moment sufficiency remain intertwined. Clamping limits Dark layer1 correction; no sweep was performed.

## Recommended next action (recommendation only; do not launch it)
Research Lead should account explicitly for the clean-reference degradation when designing the next richer-operator/oracle diagnostic. Distinguish reference mismatch from insufficient correction capacity before assigning causality. No SSL/meta/cross-channel/FL stage launched;15-minute heartbeat will read the next explicit work package.

Compact results: results/t003_moment_context/ including requestedRESULTS.md,summary.csv/json,per_client.csv,verification.json,reference_moments.pt/json,clean_sanity.csv,moment_diagnostics.csv.
Raw/smoke/logs/meta: research_log/t003_receipts/20260913-153752-ttfl-t003-gpu0/; remote originals at /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-153752-ttfl-t003-gpu0/. RawpredictionsNPZ stay local/remote; large writer diagnostics are also committed as gzip. T002 evidence preserved unchanged.

