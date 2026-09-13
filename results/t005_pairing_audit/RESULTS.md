# CODEX -> CHATGPT

## Timestamp
T005 COMPLETE, 2026-09-13. Formal run finished17:35:35+08 exit0.15-minute heartbeat remains ACTIVE for next explicit Lead task; noT006/newoperator/SSL started.

## Commit / run ID
Runtime17b55fcebc9e80bfca78f8f3f0ecb9c45ee85912; release20260913-173454-ttfl-t005; run20260913-173459-ttfl-t005-gpu1 onA6000GPU1. Running receipte466053. Final report/code/compact evidence are committed together; enclosing commit is delivery. OriginalT004O-B remains unchanged in results/t004_paired_oracle and Git.

## T005 decision (P-A/P-B/P-C/P-D)
**P-B: restoration exists but pairing specificity is weak.** All4targets recover>=50%clean headroom, yet onlyDark passes everyfrozenT005criterion (1/4, required>=2/4). Do not enlarge the operator automatically. The target/reference identifiability and target marginal contribution remain unresolved; this does not establish pure source-context reading or inadequate diagonal capacity.

**Borderline Dark PASS:** exact net correct-count difference301/15039=2.001462863pp againsttarget-permuted. The margin above2pp is only.001462863pp; reducing the netcount difference by1 would fail. We retain the frozenPASS, but do not call this robust pairing evidence. Blur1.934969pp is below2pp and remainsFAIL; no rounding-based reinterpretation or newpermutation search.

## Why T004 O-B required headroom reinterpretation
Clean accuracy independently verified from current clean evaluation and originalT004predictions:5146/15039=34.217701%. Available headroom is16.577/14.130/3.471/3.225pp. The old5ppgate exceedsNoise/Blur's clean headroom, so originalO-B cannot by itself establish capacity failure. T005uses gain/headroom without clipping; zero/negativeheadroom would beN/A (nonehere). AlloldT004records/gatespreserved.

## Verification
20unit/regression tests PASS locally/remotely, includingallT004tests and deterministic derangement/2-layer targetfeaturemultiset tests. Real2-client smoke preceded100-client formal, with no query selection. Same checkpoint, support IDs, query IDs, corruptions,192state, eps1e-5/cap[-8,8], sequentialclosedform. No labels or query tensors accepted by writer/controlAPI.

Derangement: nonzero cyclic offset=1+sample_seed(7,client,0,target_pairing)%(n-1); same image permutation used forbothclean target layers. Actualsupport sizes30..64; all100clients have0fixedpoints. TargetoriginalIDsets and tensors/features match exactly underinversepermutation atbothlayers; means/variances/classcomposition are therefore preserved by construction, without inspectinglabels. SourceIDorder unchanged. Pairings.json recordsoffset/permutation/source-target-queryIDs.

2000query records,1600writer episodes plusfocusedcleanidentity regression; sharedmodelhash unchanged eachwriter,0support/queryoverlap, coefficientsfinite. Zero-state logits exact. OldT002none and allreusedT004conditions reproduced perclient withmaxaccuracyerror0.0pp. Focusedcleanidentitydelta0.0pp; original100-clientidentity reportretained, including itslowvariancecoefficientcaveat. Independent integerprediction/headroom auditmatchesallfourPASSdecisions and confirmscleanbaseline; maxpredictionmetricdiscrepancy6.06e-6pp.

Checkpoint SHA256:260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb.
Corruption SHA256:297faeb32abbb7d26d9c083d463fe9e65e9d8606151ac7b7eb5cb995e53d7174.
Writer SHA256:99824a64af46ad500c8be742c0775435bf7566188935eb59d39d7acca86a11dd; independently matchesT004commit68ce150writerbytes. No equation/refactor-driven methodchange.

## Main headroom-normalized results
Weighted accuracy%; macro-client rows in summary.csv. Currentcleanbaseline34.217701%.

| Target | None | Correct pair | Target permuted | Wrong alt | Noise source |
|---|---:|---:|---:|---:|---:|
| Dark |17.640801|33.539464|31.538001|15.712481|31.218831|
| Contrast |20.087772|28.259858|31.099143|23.412461|31.458209|
| Noise |30.746725|33.080657|31.464858|28.339650|30.540595|
| Blur |30.992752|33.339983|31.405014|27.262451|30.999402|

| Target | Headroom pp | Gain pp | Recovery fraction | Pair advantage pp | Alt advantage pp | Noise advantage pp | PASS |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Dark |16.576900|15.898664|.959085|2.001463|17.826984|2.320633|Yes|
| Contrast |14.129929|8.172086|.578353|-2.839285|4.847397|-3.198351|No|
| Noise |3.470976|2.333932|.672414|1.615799|4.741007|2.540062|No|
| Blur |3.224948|2.347230|.727835|1.934969|6.077532|2.340581|No|

Frozenrule:recovery>=.5 AND eachpair/alt/noisegap>=2pp, >=2/4targets forP-A. Evaluated atfullprecision and independently fromintegercorrectcounts. All4recoverycriteria pass; PairgapfailsContrast/Noise/Blur, noisegapalsofailsContrast. SupplementalT002CE/T003moment rows are copiedreferences, not rerun.

## Correct-vs-permuted pairing specificity
Pairingadvantage weighted:Dark+2.001463,Contrast-2.839285,Noise+1.615799,Blur+1.934969pp. Target-permuted remainsstrong despitebreakingallimageidentitycorrespondences. Correct pairing does affectoutputs and often helpsclients, but the requested2ppmargin isnotreliableacrossshifts. This supportsP-B and continued investigation oftarget/referenceidentifiability, notanoverallpositiveclaim.

## Per-client paired summaries
Equal-client delta mean/median/fraction>0 below. All12comparisons includingP10/P25/P75/P90 are in summary.json; full400client/targetrows inper_client.csv.

| Target / control | Mean pp | Median pp | Fraction>0 |
|---|---:|---:|---:|
| Dark / permuted |3.404|5.468|.66|
| Dark / alt |15.780|11.614|.86|
| Dark / noise |3.514|4.826|.64|
| Contrast / permuted |-1.758|1.800|.54|
| Contrast / alt |4.543|1.887|.65|
| Contrast / noise |-1.363|2.237|.54|
| Noise / permuted |2.710|5.000|.62|
| Noise / alt |4.393|3.570|.61|
| Noise / noise-source |4.028|6.869|.63|
| Blur / permuted |3.153|5.058|.64|
| Blur / alt |6.008|2.294|.62|
| Blur / noise-source |4.076|4.393|.62|

Substantial heterogeneity remains: correct-minus-permutedP10 isabout-20to-27pp acrossshifts; positivefractions alone do not establishtheweightedmargin.

## Restoration diagnostics
MSEcomparedwiththeassignedclean/permutedtarget ofeachcondition. Layer2sourcesaremeasuredaftertheircondition'slayer1correction. Thesearemeansoverclients;meanratioisnotratioofpooledmeans.

| Target / layer | Correct MSE before->after | Correct ratio | Permuted MSE before->after | Permuted ratio |
|---|---:|---:|---:|---:|
| Dark /1 |.123326->.043158|.347076|.183376->.100574|.547983|
| Dark /2 |.059320->.052217|.881210|.197110->.168155|.853623|
| Contrast /1 |.049407->.025376|.522143|.144992->.100311|.691636|
| Contrast /2 |.051811->.044337|.857275|.196874->.167636|.852101|
| Noise /1 |.023557->.013247|.563888|.213787->.100277|.467598|
| Noise /2 |.011959->.009780|.819257|.196089->.167930|.856756|
| Blur /1 |.012844->.008183|.639348|.172514->.100212|.580305|
| Blur /2 |.009660->.008549|.885130|.196152->.167784|.855768|

CorrectpairhasmuchlowerabsolutefinalMSE;permuted neverthelessremainscompetitiveinqueryaccuracy. Thusrestorationlossqualityandtaskaccuracyarenotinterchangeablemechanismevidence. Permutedlayer1negative-scale fractions30.5–40.1%, layer2about17.1–19.3%; correctscalesmostlypositive. Everycontext/layermean/maxabs gamma/beta,caps,negative-scalefractionsinrestoration_diagnostics.csv; perclientcoefficientsinrawrestorationreceipt.

**Cap flag:** oneepisodeexceeds10%:client16,Dark/target-permuted,layer2,7/64channels=10.9375%. Boundsunchanged; no cap sweep or discardedcondition. Otherlayerepisodesnotabove10%.

## Failures / uncertainties
No runtime/assertionfailure, rerun orparameterchange. ExistingNVMLwarning remains, actualA6000CUDA succeeds. Formal20.20s, peak torch allocation.0684GiB (descriptive, notcontrolledspeedclaim). Singlecheckpoint/supportdraw/derangementperclient/severity. No repeatedpermutation studywasperformed; Dark'sborderlinePASSmustnotbeoverinterpreted.

Theoracleisnon-deployable:itreceivesclean counterparts. Targetmarginalscontainclientcontentevenwithoutclasslabels; thecurrentauditpreservesthesemarginalsexactly. P-Bdoesnotproveallgainissemanticmarginalnorzero correspondence; itshowsinsufficientmarginunderfrozencontrols. OriginalT004O-Bisretainedaspreregisteredhistory, notcapacityrejection. PFLlibmergedoriginalsplitswithclient75/25diagnostic, notofficialCIFAR10testaccuracy.

## Recommended next action (recommendation only; do not launch)
ReturntoResearchLead todesignthenexttarget/referenceidentifiabilitydiagnostic. Do notincreaseoperatorcapacityautomatically; fourheadroomrecoveriesarealready>=.5butpairing-specificmarginsareweak. NoT006/SSL/meta/FLimplemented. Heartbeatwillreadthenext explicitboundedworkpackage.

Outputs:results/t005_pairing_audit/{RESULTS.md,summary.csv,summary.json,per_client.csv,restoration_diagnostics.csv,verification.json,integer_prediction_audit.json}. Rawpairings/restoration/predictions/smoke/logs/meta underresearch_log/t005_receipts/20260913-173459-ttfl-t005-gpu1/; remoteoriginals /home/wenchang/asdasdsad/wjq/TTFL/runs/20260913-173459-ttfl-t005-gpu1/. Large rawrestorationalso compressedforGit; originalJSONandpredictionNPZremainlocally/remotely. Priornegativeevidencepreserved.
