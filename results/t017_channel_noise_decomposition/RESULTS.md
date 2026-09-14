# CODEX → CHATGPT: T017R completed decomposition

2026-09-14T08:48:04+08:00. Lead a9fa543; original3246221 protocol preserved. Runtime `7fbfbb26e6894dbdc7f1a67e7f43537aaf2b36d5`; run `20260914-083609-ttfl-t017r-cached`; release20260914-083556-ttfl-t017r;87testsPASS;exit0;174.80s. Zero new model forwards.512,000matched bootstrap episodes,4,096,000template choices; fixed128replicas and syntheticK20/40/80/160.

## Narrow sanity repair and historical provenance

Stopped T017 run20260914-073926-ttfl-t017-preflight atda202c8 remains preserved. Only noise-free sanity changed: unchanged1e-8 inverse precision, exact optimal-set membership/zero true-template regret, and singleton identity. All canonical P00 historical choices/counts remain unchanged. Bootstrap never calls the sanity helper or overrides pi/state choices. Original protocol_freeze.json is copied byte-for-byte; repair scope is separately recorded.

Noise-free max prevalence error5.27e-15; singletonepisodes7914,tiedepisodes86; singleton identity mismatches0, tied optimal-set mismatches0,exact true-regret maximum0. The71inverse-induced representative switches and their25querymetric changes are still reported in noise_free_replay.json.

Ties occur in86/8000episodes (1.075%). Largest per-aggregate P00 tie-envelope width0.100525pp. No historical80%decision changed at either edge, so bootstrap was authorized to continue. Most tie-heavy clients(counts):[('40', 58), ('21', 12), ('15', 7), ('93', 5), ('38', 2), ('81', 2), ('0', 0), ('1', 0), ('2', 0), ('3', 0)]. Complete per-episode identifiers are in template_diagnostics.json.gz, and canonical/min/max metrics are in p00_tie_envelope.csv. This additive exact envelope is sensitivity analysis, not a substituted oracle baseline.

## Verification and frozen computation

All2000T016channelhashes(1000soft/1000hard),4000mixtures,8000P00+8000soft01choices,80metrics and40capture summaries replay. All emission pools exclude targeti; their means/counts reconstruct the saved softchannel. Original artifact hashes remain unchanged. Matched sampling scales exact class counts and uses SHA256(T017|client|bank|context|K|replica),independent PCG64 seeds. Cached default pinv and unchanged simplex projection; no temperature, inverse cutoff, K20 deployment or estimator change.

Exact rational template lookup uses integer common denominators rather than expensive repeated Fraction reductions; decisions are mathematically identical, with no epsilon or tie forcing. Formal8000first-replica Fraction checks and4000deterministic replays pass; independent8000last-replica Fraction decisions pass. Independent4000seed replays have maxq error1.11e-16. All20,480aggregate class counts and4,096,000query-regret numerators reconstruct exactly; 40000per-half/candidate class-count checks match frozen T011predictions. Scientific quantiles/gates use exact fractions and linear quantile interpolation, not rounded displays.

Raw bootstrap_arrays.npz preserves q/pi/z, state choices,DU,L1,JS,dominant agreement,negative entries and projection corrections. bootstrap_query_counts.npz preserves aggregate class counts and integer regret numerators/denominators. Exact per-state true regrets reside in template_diagnostics.json.gz and combine with savedchoices to reconstruct every replica’s true-template regret. Seeds,pool hashes and logits enable resampling. NPZs are durable local/remote artifacts referenced by manifests; compact aggregate replicas are Git-tracked gzip.

## Matched K20 task capture

| bank | context | salt | actual_capture | capture_p05 | capture_median | capture_p95 | explained | median80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | T013-S0 | 0.714308 | 0.688266 | 0.800559 | 0.876673 | True | True |
| A | clean | T013-S1 | 0.675886 | 0.681782 | 0.796767 | 0.880767 | False | False |
| A | clean | T013-S2 | 0.687875 | 0.679345 | 0.793679 | 0.872454 | True | False |
| A | clean | T013-S3 | 0.673942 | 0.676519 | 0.786325 | 0.864688 | False | False |
| A | brightness_dark | T013-S0 | 0.879861 | 0.794417 | 0.860142 | 0.914862 | True | True |
| A | brightness_dark | T013-S1 | 0.877950 | 0.794874 | 0.863398 | 0.917446 | True | True |
| A | brightness_dark | T013-S2 | 0.881368 | 0.794184 | 0.862205 | 0.912518 | True | True |
| A | brightness_dark | T013-S3 | 0.879966 | 0.795686 | 0.857503 | 0.917000 | True | True |
| A | contrast_low | T013-S0 | 0.719672 | 0.707323 | 0.792152 | 0.873999 | True | False |
| A | contrast_low | T013-S1 | 0.725236 | 0.717177 | 0.792255 | 0.874682 | True | False |
| A | contrast_low | T013-S2 | 0.718445 | 0.699849 | 0.787968 | 0.866307 | True | False |
| A | contrast_low | T013-S3 | 0.720043 | 0.703420 | 0.785471 | 0.871641 | True | False |
| A | gaussian_noise | T013-S0 | 0.670379 | 0.653012 | 0.779781 | 0.872316 | True | False |
| A | gaussian_noise | T013-S1 | 0.670134 | 0.655116 | 0.776897 | 0.872431 | True | False |
| A | gaussian_noise | T013-S2 | 0.666325 | 0.645614 | 0.773170 | 0.871634 | True | False |
| A | gaussian_noise | T013-S3 | 0.672975 | 0.642317 | 0.775250 | 0.869782 | True | False |
| A | gaussian_blur | T013-S0 | 0.659586 | 0.688563 | 0.772537 | 0.850607 | False | False |
| A | gaussian_blur | T013-S1 | 0.664880 | 0.681341 | 0.770820 | 0.850421 | False | False |
| A | gaussian_blur | T013-S2 | 0.669945 | 0.687016 | 0.771092 | 0.844840 | False | False |
| A | gaussian_blur | T013-S3 | 0.660172 | 0.691037 | 0.770037 | 0.853339 | False | False |
| B | clean | T013-S0 | 0.673007 | 0.668040 | 0.785412 | 0.862754 | True | False |
| B | clean | T013-S1 | 0.660242 | 0.669247 | 0.799308 | 0.878134 | False | False |
| B | clean | T013-S2 | 0.694469 | 0.655109 | 0.794257 | 0.869310 | True | False |
| B | clean | T013-S3 | 0.676638 | 0.664006 | 0.784979 | 0.857512 | True | False |
| B | brightness_dark | T013-S0 | 0.879835 | 0.808724 | 0.873285 | 0.924006 | True | True |
| B | brightness_dark | T013-S1 | 0.882696 | 0.806441 | 0.872574 | 0.923666 | True | True |
| B | brightness_dark | T013-S2 | 0.878906 | 0.802165 | 0.867968 | 0.922185 | True | True |
| B | brightness_dark | T013-S3 | 0.880717 | 0.809231 | 0.873046 | 0.923203 | True | True |
| B | contrast_low | T013-S0 | 0.711918 | 0.680636 | 0.783365 | 0.856304 | True | False |
| B | contrast_low | T013-S1 | 0.724426 | 0.688742 | 0.796074 | 0.862828 | True | False |
| B | contrast_low | T013-S2 | 0.729673 | 0.692684 | 0.785050 | 0.853653 | True | False |
| B | contrast_low | T013-S3 | 0.721614 | 0.680979 | 0.780488 | 0.854627 | True | False |
| B | gaussian_noise | T013-S0 | 0.691111 | 0.657111 | 0.784470 | 0.888509 | True | False |
| B | gaussian_noise | T013-S1 | 0.693033 | 0.645565 | 0.785614 | 0.890419 | True | False |
| B | gaussian_noise | T013-S2 | 0.689815 | 0.651166 | 0.783591 | 0.885094 | True | False |
| B | gaussian_noise | T013-S3 | 0.679099 | 0.654525 | 0.787911 | 0.889141 | True | False |
| B | gaussian_blur | T013-S0 | 0.673688 | 0.695206 | 0.804227 | 0.873802 | False | True |
| B | gaussian_blur | T013-S1 | 0.683089 | 0.701235 | 0.798718 | 0.871079 | False | False |
| B | gaussian_blur | T013-S2 | 0.681415 | 0.698016 | 0.798189 | 0.865440 | False | False |
| B | gaussian_blur | T013-S3 | 0.679390 | 0.700999 | 0.799115 | 0.870243 | False | False |

Canonical historical P00 remains the denominator. Replicas align acrossclients only for aggregation. Full K40/K80/K160 rows are in bootstrap_aggregate.csv, with raw exact replica accuracy/capture fractions in bootstrap_aggregate_replicas.json.gz. Synthetic largerK is a sample-complexity diagnostic and changes no real support.

## Finite-support prevalence

| K | bank | context | L1_mean | L1_median | L1_p05 | L1_p95 | JS_mean | dominant_class_agreement | negative_entry_fraction | projection_L1_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | A | clean | 0.666261 | 0.600000 | 0.070539 | 1.491045 | 0.152532 | 0.729766 | 0.427969 | 1.153394 |
| 20 | A | brightness_dark | 0.743911 | 0.673566 | 0.098972 | 1.660345 | 0.179160 | 0.680703 | 0.429672 | 1.457761 |
| 20 | A | contrast_low | 0.778092 | 0.712160 | 0.100000 | 1.700000 | 0.190051 | 0.663750 | 0.429906 | 1.562055 |
| 20 | A | gaussian_noise | 0.693552 | 0.618506 | 0.061329 | 1.576326 | 0.162597 | 0.711328 | 0.431664 | 1.258841 |
| 20 | A | gaussian_blur | 0.676503 | 0.602093 | 0.078261 | 1.514676 | 0.156526 | 0.730000 | 0.429461 | 1.189671 |
| 20 | B | clean | 0.668699 | 0.600000 | 0.063622 | 1.491597 | 0.154218 | 0.730547 | 0.426398 | 1.152696 |
| 20 | B | brightness_dark | 0.750947 | 0.686355 | 0.100000 | 1.663044 | 0.181307 | 0.676719 | 0.431219 | 1.479183 |
| 20 | B | contrast_low | 0.790836 | 0.729769 | 0.100000 | 1.724546 | 0.194921 | 0.657656 | 0.434422 | 1.654636 |
| 20 | B | gaussian_noise | 0.695663 | 0.621150 | 0.071961 | 1.565548 | 0.162829 | 0.713359 | 0.429773 | 1.240336 |
| 20 | B | gaussian_blur | 0.676513 | 0.600000 | 0.071542 | 1.500000 | 0.156735 | 0.725703 | 0.430680 | 1.192165 |
| 40 | A | clean | 0.514524 | 0.465902 | 0.047301 | 1.159118 | 0.108186 | 0.827656 | 0.409180 | 0.795514 |
| 40 | A | brightness_dark | 0.574599 | 0.504627 | 0.060515 | 1.299204 | 0.126478 | 0.793359 | 0.410555 | 1.006692 |
| 40 | A | contrast_low | 0.609464 | 0.545945 | 0.080695 | 1.361470 | 0.136836 | 0.776797 | 0.415875 | 1.082839 |
| 40 | A | gaussian_noise | 0.538302 | 0.489651 | 0.057290 | 1.184066 | 0.114755 | 0.820625 | 0.410336 | 0.854147 |
| 40 | A | gaussian_blur | 0.511784 | 0.468671 | 0.046143 | 1.120321 | 0.107125 | 0.833438 | 0.411148 | 0.815279 |
| 40 | B | clean | 0.507513 | 0.458586 | 0.051617 | 1.140935 | 0.106363 | 0.836172 | 0.409539 | 0.793247 |
| 40 | B | brightness_dark | 0.576214 | 0.507022 | 0.070936 | 1.308952 | 0.127193 | 0.789609 | 0.412883 | 1.017500 |
| 40 | B | contrast_low | 0.615611 | 0.553032 | 0.089767 | 1.361241 | 0.138869 | 0.776094 | 0.416398 | 1.137990 |
| 40 | B | gaussian_noise | 0.527199 | 0.479402 | 0.047714 | 1.173342 | 0.112089 | 0.821797 | 0.411961 | 0.857837 |
| 40 | B | gaussian_blur | 0.513584 | 0.468224 | 0.050745 | 1.124239 | 0.107547 | 0.835469 | 0.410828 | 0.820120 |
| 80 | A | clean | 0.379059 | 0.342310 | 0.030178 | 0.847080 | 0.073074 | 0.897734 | 0.395266 | 0.548454 |
| 80 | A | brightness_dark | 0.432758 | 0.392886 | 0.049660 | 0.957722 | 0.087186 | 0.870625 | 0.399297 | 0.693235 |
| 80 | A | contrast_low | 0.456373 | 0.411876 | 0.054885 | 1.000000 | 0.093119 | 0.866094 | 0.399648 | 0.742515 |
| 80 | A | gaussian_noise | 0.396304 | 0.357094 | 0.032411 | 0.892585 | 0.077030 | 0.885469 | 0.398094 | 0.587613 |
| 80 | A | gaussian_blur | 0.389091 | 0.354328 | 0.037508 | 0.856384 | 0.075137 | 0.892656 | 0.397031 | 0.567845 |
| 80 | B | clean | 0.377921 | 0.341538 | 0.033653 | 0.843113 | 0.072734 | 0.900937 | 0.394992 | 0.545347 |
| 80 | B | brightness_dark | 0.436757 | 0.397421 | 0.043737 | 0.970735 | 0.087881 | 0.870391 | 0.398953 | 0.697116 |
| 80 | B | contrast_low | 0.470201 | 0.422302 | 0.061038 | 1.051788 | 0.097593 | 0.861094 | 0.401383 | 0.780685 |
| 80 | B | gaussian_noise | 0.398663 | 0.360095 | 0.037366 | 0.900000 | 0.077176 | 0.884687 | 0.397461 | 0.590791 |
| 80 | B | gaussian_blur | 0.387174 | 0.351761 | 0.028702 | 0.865721 | 0.074864 | 0.892891 | 0.397141 | 0.568394 |
| 160 | A | clean | 0.279911 | 0.252058 | 0.025255 | 0.613870 | 0.049520 | 0.934063 | 0.384664 | 0.381029 |
| 160 | A | brightness_dark | 0.319888 | 0.289215 | 0.031482 | 0.717012 | 0.059688 | 0.918516 | 0.387570 | 0.480572 |
| 160 | A | contrast_low | 0.340768 | 0.307250 | 0.038122 | 0.745443 | 0.064366 | 0.916250 | 0.388406 | 0.512665 |
| 160 | A | gaussian_noise | 0.292748 | 0.263702 | 0.025094 | 0.651766 | 0.052402 | 0.933047 | 0.385148 | 0.407536 |
| 160 | A | gaussian_blur | 0.284220 | 0.257229 | 0.021044 | 0.619200 | 0.050377 | 0.932891 | 0.386164 | 0.394182 |
| 160 | B | clean | 0.280327 | 0.253572 | 0.023097 | 0.616123 | 0.049697 | 0.933047 | 0.384141 | 0.383585 |
| 160 | B | brightness_dark | 0.326717 | 0.292708 | 0.034590 | 0.732629 | 0.061073 | 0.921406 | 0.387609 | 0.486249 |
| 160 | B | contrast_low | 0.353438 | 0.317811 | 0.044430 | 0.781863 | 0.067430 | 0.911875 | 0.391031 | 0.536654 |
| 160 | B | gaussian_noise | 0.297733 | 0.268383 | 0.026307 | 0.670856 | 0.053378 | 0.927266 | 0.385219 | 0.407565 |
| 160 | B | gaussian_blur | 0.283563 | 0.257934 | 0.023832 | 0.625116 | 0.050534 | 0.934609 | 0.386703 | 0.392544 |

## Actual-channel mismatch

| bank | context | median_D_actual | median_bootstrap_p50 | median_empirical_percentile | above_p95_fraction | label |
| --- | --- | --- | --- | --- | --- | --- |
| A | clean | 0.154873 | 0.150062 | 0.605469 | 0.100000 | weak |
| A | brightness_dark | 0.155260 | 0.140949 | 0.601562 | 0.070000 | weak |
| A | contrast_low | 0.157236 | 0.142837 | 0.609375 | 0.070000 | weak |
| A | gaussian_noise | 0.150639 | 0.148603 | 0.535156 | 0.080000 | weak |
| A | gaussian_blur | 0.159063 | 0.151464 | 0.566406 | 0.100000 | weak |
| B | clean | 0.154873 | 0.149729 | 0.554688 | 0.080000 | weak |
| B | brightness_dark | 0.157742 | 0.142599 | 0.585938 | 0.070000 | weak |
| B | contrast_low | 0.157494 | 0.141601 | 0.585938 | 0.080000 | weak |
| B | gaussian_noise | 0.147642 | 0.147598 | 0.519531 | 0.080000 | weak |
| B | gaussian_blur | 0.159867 | 0.150198 | 0.554688 | 0.090000 | weak |

These are privileged true-composition, matched-channel diagnostics, not deployable detection or hypothesis-test p-values. Weak<=20%,moderate>20%and<=50%,strong>50%above matchedp95 are unchanged. The empirical null conditions on other-client support emissions; a real-client discrepancy need not be caused solely by channel mismatch.

## Task sensitivity and ties

| stratum | n | actual_true_regret_pp | actual_canonical_agreement | actual_optimal_set_agreement | actual_L1 | actual_DU_pp | boot_true_regret_mean_pp | boot_canonical_agreement | boot_optimal_set_agreement | boot_L1_mean | boot_DU_mean_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 8000 | 2.966658 | 0.615625 | 0.620875 | 0.766983 | 10.104671 | 2.280184 | 0.665181 | 0.671913 | 0.714098 | 9.959360 |
| legacy_quartile_1 | 2000 | 1.281537 | 0.429500 | 0.450500 | 0.732990 | 7.805147 | 1.133337 | 0.441719 | 0.468648 | 0.671027 | 7.781925 |
| legacy_quartile_2 | 2000 | 2.684773 | 0.562500 | 0.562500 | 0.732263 | 9.123259 | 1.963181 | 0.623926 | 0.623926 | 0.712446 | 9.146422 |
| legacy_quartile_3 | 2000 | 3.040997 | 0.700500 | 0.700500 | 0.772672 | 9.989397 | 2.289871 | 0.773059 | 0.773059 | 0.708565 | 9.650848 |
| legacy_quartile_4 | 2000 | 4.859326 | 0.770000 | 0.770000 | 0.830007 | 13.500880 | 3.734346 | 0.822020 | 0.822020 | 0.764353 | 13.258244 |
| best_tied | 86 | 0.010860 | 0.465116 | 0.953488 | 0.930047 | 8.131606 | 0.037820 | 0.345203 | 0.971475 | 0.459509 | 5.210470 |
| singleton | 7914 | 2.998778 | 0.617261 | 0.617261 | 0.765211 | 10.126112 | 2.304551 | 0.668658 | 0.668658 | 0.716864 | 10.010965 |

Legacy best-minus-second-state margin quartiles are preserved, using the frozen global rank/tie ordering. Distinct-level margins are additional columns, with NA if all5utilities tie; tied-best slices are separate. Loss of canonical agreement inside the optimal set is representative instability, not true-template regret. Only optimal-set exits count as genuine task-sensitive error. Task_sensitivity.csv includes every8000episode/template and actual/matchedK20 regret distributions,DU,ordinaryL1,canonical/optimal-set agreement and query-oracle regret.

## Frozen diagnosis and sensitivity

**T017-N**. MATCHED-K20-EXPLAINS:{'brightness_dark': True, 'contrast_low': True, 'gaussian_noise': True, 'gaussian_blur': False}. MATCHED-K20-WOULD-PASS:{'brightness_dark': True, 'contrast_low': False, 'gaussian_noise': False, 'gaussian_blur': False}. SCALE-RECOVERABLE smallestsyntheticK:40.

Per-K joint80%status:{"20": {"brightness_dark": true, "contrast_low": false, "gaussian_noise": false, "gaussian_blur": false}, "40": {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}, "80": {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}, "160": {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}}

Taxonomy and each shift’s matched-K20 80%status invariant at both exact tie-envelope edges: True. Full denominator sensitivity is recorded in summary.json. N requires>=3explainedshifts and>=2matchedK20failures; M requires>=3matchedK20passes,actualfails,andstrongmismatch inbothbanks on all actual-failing shifts; otherwiseX. No fourth taxonomy or moved threshold.

## Scientific interpretation of the decomposition

The frozen diagnosis is T017-N:3/4shifts are jointly explained by matchedK20 and matchedK20 fails the80%criterion on3/4shifts. Thus finite K20 observation variation amplified by the measured inverse is sufficient to explain much of the residual task loss; a dominant cross-client mismatch is not required for Dark/Contrast/Noise. This is evidence under the empirical conditional-emission model, not proof that every client has exactly the same channel.

Acrossbanks/salts, matchedK20 median captures are Dark85.75–87.33%,Contrast78.05–79.61%,Noise77.32–78.79%,Blur77.00–80.42%. Only Dark passes the joint allbanks/allsalts condition. SyntheticK40 median captures are91.82–92.69%,85.71–87.51%,86.36–87.16%,86.08–87.37% respectively; all4shifts pass. K80/K160 also pass. Therefore SCALE-RECOVERABLE=40, solely as a diagnostic sample-complexity result; real support staysK20.

Blur is an unresolved exception that the global N label must not hide: all8actualBlur bank/salt captures (65.96–68.31%) are below their matchedK20p05 (68.13–70.12%). The simple posterior-L1 residual diagnostic still labels everybank/context weak: only7–10%clients exceed matchedp95 (Blur10%/9% inA/B). Thus there is task-level extra loss onBlur without strong mismatch evidence by the chosen scalar diagnostic. Direction-specific mismatch, task-sensitive errors or other limitations of the empirical null remain possible; none is uniquely established, and sourcecontext-ID is excluded from this oracle01 analysis.

Task sensitivity is mostly genuine optimal-set exits: across8000templates actual canonical/optimal-set agreement is61.56%/62.09%,versus matchedK20 66.52%/67.19%; mean true-template regret2.967pp versus2.280pp. The86tiedepisodes(1.075%) have very different canonical versus optimal-set agreement: actual46.51% versus95.35%,bootstrap34.52% versus97.15%. Counting canonical switches there as semantic failure would be misleading.

Legacy narrow-margin Q1 has actual optimal-set agreement45.05%,compared with77.00% inQ4,consistent with more frequent boundary crossings at narrow margins. Yet mean regret is1.282pp inQ1 versus4.859pp inQ4: rarer mistakes across large gaps can be more costly. This audit therefore does not attribute all task loss to narrow boundaries. Distinct-level margins and tied-best slices remain available alongside unchanged legacy quartiles.

## Interpretation limits and stop

This empirical matched-channel bootstrap separates expected finite-support variation from real-versus-null differences under the specified emission model. It does not identify a unique causal source of residual mismatch. Full-rank soft outputs retain information but may be noisy at K20; synthetic scale recovery is not a deployable result. Exact-optimum representative changes are reported separately from suboptimal utility. No neutral-operator failure or feature-semantic collapse is inferred. No next task assigned; return evidence to Lead and stop.
