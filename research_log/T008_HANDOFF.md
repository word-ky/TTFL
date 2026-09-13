# T008 active — source-only signature and fixed state retrieval

Lead e20ef8f read in full. Baseline31remote tests PASS. Source signature3focused tests PASS: bankreference/formula/zeroidentity, Euclideannearest/margin/tieorder, exact4x20microbatches2perclass. Reuse existing T007R pools, neutralwriter, checkpoint/corruptions, evaluate/hash/CSV utilities. Nooperator/writer edits. New code only signature/microbatchselection and cachedstate retrieval evaluation.

Prototype bank: five contexts fixed order clean,dark,contrast,noise,blur; float64 populationmoments (sqrt(var+1e-6)); relative ONLYbankcleanmoments. SquaredEuclidean192D; tiesfixedprototypeorder. Four20image heldoutmicrobatches via committedclasspositions[0:2],[2:4],[4:6],[6:8], IDs persistedbeforeclassification. Bothbanksprototypes+references/hashes writtenbeforeheldoutdecisions. All40ID+8OOD decisions persistedbeforeopeningquerymetrics/labels. OODnotinprimarybank. Bank8corruptionstates reconstructed then exactlycomparedT007Rstored; wrongaltstates exactcorrespondencechecked.

Retrievecached T007Rnone/correct/wrongalt predictions where choice matches; independentcountrecompute +first2client exactpredictionregression perbank/target/control. Onlyunavailable selected-state/targetpairs get100client newevaluation. No perclientstatefits. PrimaryIDboth>=18/20andeach>=3/4; RETbothjoint>=3/4 withretained>=.90 andselectedwrong>=frozenT007Rtau. T007RNoisefailureunchanged. Full36tests beforeformal small40decisiondiagnostic. NoSSL/nextstage.

## 2026-09-13T21:31:31.4667998+08:00 — launched
Run20260913-213117-ttfl-t008-gpu1 release20260913-213110-ttfl-t008 runtime4a71d7d. A6000GPU1;36tests beforefull40decision identification andcachedretrieval.

## 2026-09-13T21:36:30.1204310+08:00 — COMPLETE ID-A + RET-A
Run20260913-213117-ttfl-t008-gpu1 exit0.36tests PASS. Bothdirections20/20 all5contexts4/4;4/4jointretrievalpass retained1.0. Eightbankstates exactT007R,48focusedquerypredictionregressions exact,0new100clientqueryevaluations. Independentdistanceerror5.68e-14 andcountmetricerror7.1e-15pp. Prototypes/vector/decisionfreeze hashesverified. OOD8/8noise_image→gaussian_noise withmargin66.73..69.41: noOODrejectioncapability claimed. Allresults persisted, noSSL/nextstage.
