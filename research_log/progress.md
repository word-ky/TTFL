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

## 2026-09-13T12:46:51.1029787+08:00 — first completed dataset evidence
MNISTfull100rounds:FedAvg92.497577,affinecorrect94.755145,wrong92.018699,shuffled94.686734,priorcorrect98.244114. ThesearecompleteMNISTresults, nootherdataconfigtuning. Prior/shuffled explainmuchofgain; noindependentoperatoradvantageclaimed. Tinydataretryexit0at12:44:19+08;4datasetsready. AwaitonefreeGPUforTiny100rounds.

## 2026-09-13T12:47:35.9322716+08:00 — CIFAR10 complete, TinyImageNet dispatch
GPU0 MNIST+Cifar10 run20260913-123718-ttfl-pfl-gpu0 exit0at12:47:10. CIFAR10FedAvg34.217701,affine57.277745,prior79.194095,shuffled58.049072,noise59.937496:largeaffinegainisnotsemanticcontextproof; simplerpriorandshuffled/noiseoutperform. Cifar100formaltrainingcomplete,clientcontextevalinprogress. TinyreadyandGPU0free:launchTinyImageNet smoke+100roundsonGPU0withclassfix5db9d15.

## 2026-09-13T12:49:44.2299219+08:00 — CIFAR100 complete, first3 export
Cifar100run20260913-124228-ttfl-pfl-c100-fixed exit0at12:47:37. FedAvg11.717763,affine13.007914,prior36.589746,shuffled13.327127. First3complete receipts exportedtoF/wjq/TTFL/pfllib_first3.tar.gz(36smallartifactfilesplusdata/logs). Tiny20260913-124728 activeonGPU0, no configtuning. Full training remainsstockPFLlibCNN64x64 forTiny,100rounds10participants.

## 2026-09-13T12:51:46.3444444+08:00 — Tiny formal training and recovery state
Tinyreal2roundsmoke+contextpassed. Formal100rounds active onGPU0,~5secondsperround. First3downloaded baseline receipts each confirm100rounds andCUDA. PrimaryHANDOFFredirects to activePFLscope; oldT001handoffretainedinT001_HANDOFF.md. No furthercodechanges/parameterchanges.

## 2026-09-13T13:06:06.3290000+08:00 — all four completed and verified
Tiny20260913-124728 exit0at13:03:22. Full100rounds plus100clientcontext done. FinalscoreTiny9.644504/affine10.425215/prior24.957333. Exportedpfllib_tiny.tar.gz, unpacked alongsidefirst3. Summaryhelperinitiallymatcheddata/MNISTasrunfolder; fixedonlythatobservedpathselectioncollision, no experimentalchanges. Finalscripts/summarize_pfllib.pyPASS4datasets,4000records,100updates/10uniqueclientseachround,zerotrainsupport/queryIDoverlap,independentpredictionaccuracyerror<7.7e-6pp. Alloutputs/results inresults/pfllib_100c. No further experiments.

## 2026-09-13T15:18:16.9552178+08:00 — T002 resumed and heartbeat enabled
Fetched Lead37ac0de/94b02d5; T002 only. Created ACTIVE 15-minute heartbeat ttfl-chatgpt on this thread. Remote baseline9 tests passed; actual saved input[-1,1], A6000 CUDA available. Reused existing loader/model/affine/evaluate, added fixed deterministic covariate transforms and evaluation-only paired runner. Transform tests passed locally; full old+new suite11 passed before additional hand-computed summary test. Full run planned after2-client real smoke with exactly same checkpoint/IDs/labels; no FedAvg retraining. See T002_HANDOFF.md.

## 2026-09-13T15:20:04.2242029+08:00 — T002 GPU launch
Local tests passed. Deployed release20260913-151830-ttfl-t002; launched20260913-151900-ttfl-t002-gpu0 codee09a4a8 onA6000GPU0. Remote12 tests and2-client real smoke passed, formal100-client evaluation running. Lead4455f46 merged/pushed aseda9c6e; T002 remains active. Initial push rejected due concurrent Lead protocol commit, resolved with normal merge. A local log-update command had PowerShell interpolation syntax error before any write; corrected here.

## 2026-09-13T15:25:32.5325815+08:00 — T002 COMPLETE / Case B
Run20260913-151900-ttfl-t002-gpu0 exit0. All100clients and2400records verified; brightnessPASS only(1/4), overallCaseB. Correct gains34.411/35.202/24.882/26.046pp; correct-noise1.642/-1.835/-.977/-3.258pp. Shuffled exceedscorrect onall4. Optional descriptive prior correlation analysis completed. Retrieved full artifacts underresearch_log/t002_receipts; compact results underresults/t002_covariate. Final12tests passed locally/remotely. No evaluation failure; report patch syntax failure corrected with direct file write. Finalreport replaces priorPFL report; previousreport preserved inGit/results. Heartbeat remainsACTIVE for nextexplicitLead task.

## 2026-09-13T15:37:29.7969790+08:00 — T003 implementation
Fetched/read Lead2b5e7a8, no activeTTFL job. Kept same192state/checkpoint/corruptions. Added double-moment train-only reference estimator and sequential label-free/no-gradient writer. Three moment tests PASS; local full15 tests PASS. Predeclared alltrain reference(no cap), epsilon1e-5, clamp[.25,4], mismatch channelmean abs(mu diff)+abs(sigma diff), actual postcalibration measurement with1e-5numeric tolerance. Eval runner uses sameT002 support/query IDs, copiesCE rows, records clean sanity, clamp/moment diagnostics. T003_HANDOFF.md persists contract.

## 2026-09-13T15:38:20.3798113+08:00 — T003 A6000 launch
Deployed20260913-153748-ttfl-t003 and launched20260913-153752-ttfl-t003-gpu0. Reference+smoke+formal scheduled sequentially in one job.

## 2026-09-13T15:40:19.7832584+08:00 — observed remote connectivity failure
Two SSH connecttimeouts while readingT003 existingrun. Last observedformalclient21. No execution result inferred from timeout; retry same run.

## 2026-09-13T15:45:32.1551416+08:00 — T003 COMPLETE / M-B
Samejobexit0 aftermonitorSSHtimeouts. Allrawfetched; corruptionfileshashsameT002/T003. Ref44961trainonly,15testslocal/remotePASS,2000queryrecords/1700writers. Darkgain6.151pp andgaps12.042/12.687/10.639 =>PASS; othersfail,1/4overallM-B. Clean sanity-10.087pp with77clientsworse despitezero clamp; reportreference mismatch ascompetingexplanation, notcapacityconclusion.204layerepisodes>10%clamp; allmomentsmismatchdecrease. Cleanpredictionsrecalculatedmaxerror3.84e-6pp. Finalreport/compactresults andcompressedrawdiagnostics prepared. Updatedheartbeatgenericlatesttaskprompt toavoidstaleT002 naming; no nextexperiment.

## 2026-09-13T16:37:46.3549953+08:00 — T004 implementation
Heartbeat fetchedLead ec2b54a. Pairedclean oracle originalformula covariance/(variance+1e-5),[-8,8]cap,sequential192state.3focusedtestsPASS; full18testsPASS. Mandatory100clientidentity before2clientsmoke/formal. No label/query writerarguments; exactT002IDs/corruptions. No duplicateTTFLjob. OtherTOVD/TAISPjobs untouched.

## 2026-09-13T16:38:52.5929915+08:00 — T004 launched
Run20260913-163821-ttfl-t004-gpu1 code68ce150 onA6000GPU1.18testsPASSlocal. Sequentialidentity/smoke/formal, noT002/T003rerun.

## 2026-09-13T16:43:56.6900389+08:00 — T004 COMPLETE / O-B
Samejobexit0; formal20.29sidentity4.64s.18testsPASSlocal/remote; cleanaccuracyunchangedall100. Originalepsregression lowvariancechannelgamma-.998002 butlogitdiffmax.005154; targetedauditconfirmedvariance1.9682e-8, nohiddenidentityshortcut.2000records/1600writers, frozenhash,zerononfinite/IDoverlap,1/4PASS(Dark),O-B. Resultsandfullreportprepared, predictionrecalculationerrors<6.06e-6pp. Nohyperparameterchange,newoperatororT005.

## 2026-09-13T17:34:34.1617728+08:00 — T005 implemented
Lead a53c57f read. Unchangedpairedwriter; addeddeterministicderangementcontrol and --pairing-audit extensiontoT004 evaluator. ExacttargetID/tensor/featuremultisetassertions, sharedconditionscompareT004, cleanbaselineheadroomindependentverification. Full20testsPASSincluding2newpairingtests; defaultT004formulaunchanged. FrozenT005gates/prerunchoices inT005_HANDOFF.md.

## 2026-09-13T17:35:25.7565618+08:00 — T005 A6000 launch
Deployed20260913-173454-ttfl-t005 andlaunched20260913-173459-ttfl-t005-gpu1 code17b55fc.20testslocalPASS, originalT004writerunchanged.

## 2026-09-13T17:40:17.0028085+08:00 — T005 COMPLETE / P-B
Formalexit0,20.20s,20testslocal/remotePASS. Fullrecordandintegerpredictionauditpassed; all4recoveryfractions.9591/.5784/.6724/.7278 butpairgaps2.00146/-2.83928/1.61580/1.93497;1/4PASS=>P-B. Darkjust301netcorrect/15039gateborderline; explicitlyreported. Unchangedwriterhash99824a.. andoldconditionperclientaccuracyerror0.1capwarningclient16Darkpermutedlayer2. Noquerytuning/rerun/newstage. Results/rawreceipts/derangements savedprojectlocally, compressedlargeJSON,finalcoordinationreportwritten.

## 2026-09-13T18:32:48.6218929+08:00 — T006 implementation
Leadf93c0ff read; noTTFLactivejob.2newsemanticconstructiontests andfull22testspassed. Unchangedwriter; new10derangementsperclient/target. ReuseT0054oldconditions withfirst2clientprediction-exactregression; allpairingsID/image/featuremultisetverified. Labelsrestrictedtoconstruction/analysis. Metricsdefinition frozeninT006_HANDOFF.md.

## 2026-09-13T18:34:23.1566888+08:00 — T006 launch
Launched20260913-183307-ttfl-t006-gpu1 code c53c9be;22localtestsPASS. Onlynewsemanticderangementfamiliesrun; T005oldconditionsreusedafterfocusedpredictionregression.

## 2026-09-13T18:42:47+08:00 — T006 complete

Run 20260913-183307-ttfl-t006-gpu1; S-B on Noise/Blur. S-A0/4; Dark/Contrast numerical S-C only, random semantic variation narrow. Local OpenMP analysis conflict resolved by remote analysis (no override). Report, raw receipts, integer audit and manifest persisted. Await Lead, no T007.

## 2026-09-13T18:43:52+08:00 — T006 complete

Run 20260913-183307-ttfl-t006-gpu1; S-B on Noise/Blur. S-A0/4; Dark/Contrast numerical S-C only, random semantic variation narrow. Local OpenMP analysis conflict resolved by remote analysis (no override). Report, raw receipts, integer audit and manifest persisted. Await Lead, no T007.

## 2026-09-13T19:34:52.7667668+08:00 — T007 accepted
Lead3394d2c. Baseline24remote tests PASS; pool2local tests PASS. Reusing frozenwriter/checkpoint; implement real identity prerequisite first. No activeTTFLjob.

## 2026-09-13T19:35:51.6740444+08:00 — T007 preflight run
20260913-193525-ttfl-t007-preflight-gpu1 started on A6000GPU1, codebc07059.

## 2026-09-13T19:39:26.8270275+08:00 — T007 identity stop reported
26remote tests PASS. Two pooledidentitystates violate nonidentity-state prerequisite (layer2scale0.324/0.469); cleanmacroclass34.13869 unchanged; 1changedprediction perpool, no changedcorrectcounts. FrozenEPS/caps unchanged. Noformaltransfer, noC-A/B/C/Dassignment. AwaitLead.

## 2026-09-13T20:31:41.5631369+08:00 — T007R implementation
Lead7553405; baseline26remotePASS; neutralwriter3remotePASS. Originalpools preserved;6client smoke allclasses, then100formal afteridentityexact100querycheck.

## 2026-09-13T20:33:23.6337037+08:00 — T007R run
20260913-203200-ttfl-t007r-gpu1 onGPU1. Code5794dbf. Frozenpools/corruptions/checkpoint.

## 2026-09-13T20:38:02.8249020+08:00 — T007R COMPLETE C-A
Residualwriter exactidentity;31tests passed. A6000run20260913-203200-ttfl-t007r-gpu1,37.17sformal.3/4jointPASS; NoiseB0.499741262FAIL. Independentcount/Fractionauditpasses. 42globalstates,5600records. Capmax6.25percent notzero; no>10percent flags. Results/receipts/oldpreflight persisted. AwaitLead; noSSL/nextstage.

## 2026-09-13T21:30:57.5551574+08:00 — T008 implementation
Leade20ef8f. Baseline31remotePASS; signature/microbatch3remotePASS. No activeTTFLjob. Source-only bankreference192D; frozen40ID+8OOD decisions beforequery. ReuseT007Rexactstates/predictions wherepossible.

## 2026-09-13T21:31:31.4973933+08:00 — T008 launch
20260913-213117-ttfl-t008-gpu1, code4a71d7d. ExistingT007Rbankstates/predictions reused afterequalitychecks.

## 2026-09-13T21:36:30.1842543+08:00 — T008 COMPLETE
ID-A+RET-A.40/40source-onlycontext IDs correct,4/4jointretrieval with1.0oracle-gain retention.36tests andindependentdistance/countauditsPASS;48predictionregressions exact. OODnoise8/8mapsGaussianNoise despitehighmargin; retainlimitation. T007RNoiseoriginalFAIL unchanged. AwaitLead.

## 2026-09-13T22:45:41.9414191+08:00 — T009 implementation
Lead25e6ea7;36baseline remotePASS;2natural-support tests localPASS. Evaluationonly; unchangedsignature/prototypes/states. Kchosenfromcounts, labelauditafterdecisions/retrieval.

## 2026-09-13T22:46:19.7567206+08:00 — T009 run
20260913-224559-ttfl-t009-gpu1 runtime d136746. Frozen objectsunchanged; manifest/K beforefeatures.

## 2026-09-13T22:46:58.8541622+08:00 — T009 import failure fixed
First runfailedbeforeanydecisions atread_data missingimport. Addedexistingutility import, noformula/protocolchange. Failedrun224559preserved; nextsameIDsK verified.

## 2026-09-13T22:53:51.1824888+08:00 — T009 COMPLETE N-C
NIDpasses97.6/98.0; NRETfailsclean-.61pp andBlurretention.343/.539. K20count-onlychosen.40tests independentinteger/Fraction/distancechecksPASS. Missingimportfirstattemptfixedbeforeanydecisions; manifestsKIDsidentical. Actualruntimefab17bd sourceSHAverifiedrawaliaspreserved. Harmfulconfusionledger andskewaudit persisted. No nextstage.

## 2026-09-13T23:33:50.4743696+08:00 — T010 implementation
Lead50f324a;40baseline remotePASS,3poststate scoringtests remotePASS. Poststate sequentialmoments/Jzero exact; frozenstates/IDs. Allchoices beforequery. No nextstage.

## 2026-09-13T23:34:48.6968520+08:00 — T010 launch
20260913-233408-ttfl-t010-gpu1 onA6000GPU1 runtime1feaec9. Poststatecriteria unchangedforwholefixedcandidatebank.

## 2026-09-13T23:42:52.6679887+08:00 — T010 delivered analysis
A6000 run 20260913-233408-ttfl-t010-gpu1 exit0; runtime1feaec9. ID97.8/98.0, 3/4 joint targets, clean -.440248/-.413368pp PASS; Blur retention .403732/.395870 FAIL. Positive-moment-improvement harmful fractions22.03/22.77%; OOD8/8noise state. Raw receipts and compact report persisted. Existing15minuteheartbeat verified ACTIVE; awaitLead.

## 2026-09-14T00:34:28.2406804+08:00 — T011 implementation
Lead941731c; fixed prediction consistency helper and fullcandidate evaluator.3focusedtests PASS; no formalrun yet. No changed frozen code or states.

## 2026-09-14T00:35:01.1360148+08:00 — T011 launch
A6000GPU1 run20260914-003444-ttfl-t011-gpu1 runtimef323674; full46tests before formal scoring.

## 2026-09-14T00:41:37.4304104+08:00 — T011 COMPLETE P-C
Run003521-lf finished; initial003444launchCRLFfailurebeforetests corrected withLF only.46tests,5000candidateprediction/countchecks, sourcefreezeSHAandhistoricalaggregate checksPASS. PC-RET0/4FAIL; PC-ALIGNFAILbothbanks. True-state regret and best-of-fiveheadroom rejectcapacity-only diagnosis. Results andfullmatrixpersisted; no nextstage.

## 2026-09-14T01:30:37.1097860+08:00 — T012 started
Lead7d42fe2; PhaseI historicalmatrix decomposition, PhaseIIscalinghelpertests pending.

## 2026-09-14T01:33:11.1440504+08:00 — T012 launch
A6000GPU1 run20260914-013254-ttfl-t012-gpu1;48tests+frozenstateendpoint preflight before source.

## 2026-09-14T01:38:12.6620619+08:00 — T012 COMPLETE T012-B / AMP-B
Runtime87f1bde run20260914-013254-ttfl-t012-gpu1. Strongclientlock28pp overlapexcess;amplitudeNoise/Blur>=2.34ppgainbutcapture56-61/48-49percent. J-rayandPC-ray-safe0/4jointretention.48tests/count/endpoints/freezeauditsPASS. Rawresultsandreportpersisted. No nextstage.

## 2026-09-14T02:26:44.4896662+08:00 — T013 started
Existing T011NPZlocated; deterministichalves andFractionleaveoneouthelperfocusedtests, no newforwards planned.

## 2026-09-14T02:34:27.1485796+08:00 — T013 COMPLETE FACTOR-B
Existingprediction-onlyCPUrun089c918. Disjointexampleslocksurvivesall4saltsbothbanks. AdditivecrossfithelpsDark/Contrastbutnot>=80percentoracleon3/4contexts.2focusedtests,5000oldrows/T012overlap and600newmetricindependentcountchecksPASS. Allchoicesfreezebeforecomposition. No newstate/writer/forwards.

## 2026-09-14T03:23:09.7752943+08:00 — T014 started
Lead1462732;classconditionalutilityhelperthreefocusedtestsPASS, savedprediction-only analysis planned.

## 2026-09-14T03:30:20.3705312+08:00 — T014 COMPLETE COMP-A + CLASS-INT-A
Savedprediction-onlyrun78de03c; compositionexplainscleanpreference andclasscontextclosesContrast/Blur gate;Noiseincrementweak−.082/+.018pp preserved. Residualpositiveclientlockvanishesunderprescribedsubtraction. Allgatesexactcountverified; nolearnedwriter/newstage.

## 2026-09-14T04:36:02.9826656+08:00 — T015 started
Leadffb85ee;sourceposteriorhelper, stagedfreeze evaluation planned. Existing noTTFLremotejobs; T014templatespresentremote.

## 2026-09-14T04:40:17.9474007+08:00 — T015 launch
A6000GPU1 run20260914-043941-ttfl-t015-gpu1 runtime099c5f1; stagedlabel-free/privilegedfreeze; noqueryforwards.

## 2026-09-14T04:46:45.7346289+08:00 — T015 COMPLETE
Runtime099c5f1 run20260914-043941-ttfl-t015-gpu1. Supportsampleadequatebutfrozenposteriorsemanticmixturefailswithoraclecontext. Allfreezes/hash/model/57tests/count/utilitychecksPASS. ReportnegativeobservabilityandsecondaryBlurloss; no newwriter/stage.

2026-09-14 05:36 +08 T016 Lead 6476c99 synchronized; implementing fixed confusion audit, reusing T015 logits (no inference needed).

2026-09-14 05:44 +08 Initial T016 deployment SSH timeout before launch; no experiment ran. Retrying transport. Preserve per-example calibration labels after freeze for independent soft-channel reconstruction.

2026-09-14 05:44 +08 T016 formal run 20260914-054411-ttfl-t016-cached started at runtime 6fd685e; full suite precedes preflight/calibration/freeze/evaluation.

T016 STOPPED AT PREFLIGHT

2026-09-14T05:47:38+08:00. Run 20260914-054411-ttfl-t016-cached, runtime6fd685e, 63 tests PASS. Historical aggregate/gate/count/choice replay exact; two episode JS strings differ by4.44e-16. Added overstrict comparison stopped before calibration. CAL-SEM-A/CAL-SRC-A NOT EXECUTED, not scientific FAIL. Read results/t016_confusion_debiased_semantics/RESULTS.md. No automatic retry; await Lead. Existing T015 conclusions unchanged.

2026-09-14 06:38 +08 Lead2bad2a0 received; narrow JS-only repair implemented, eight tests added. Old stopped run checked; no active TTFL session. Resume fixed T016 after full suite.

2026-09-14 06:38 +08 T016R formal run20260914-063758-ttfl-t016r-cached launched, runtimee94adc6, fixed original protocol.

# T016R COMPLETE

2026-09-14T06:43:54+08:00. Lead2bad2a0; runtimee94adc67af83b553a3673aa4fb1971ec68caf4dc; run20260914-063758-ttfl-t016r-cachedexit0.71testsPASS;2000matrices,4000mixtures,32000choicesfrozen;0newforwards. CAL-SEM-A=False,CAL-SRC-A=False. Read results/t016_confusion_debiased_semantics/RESULTS.md. Preserve stopped2e51d81provenance. No nextstage; awaitLead.

# T016R COMPLETE

2026-09-14T06:46:17+08:00. Lead2bad2a0; runtimee94adc67af83b553a3673aa4fb1971ec68caf4dc; run20260914-063758-ttfl-t016r-cachedexit0.71testsPASS;2000matrices,4000mixtures,32000choicesfrozen;0newforwards. CAL-SEM-A=False,CAL-SRC-A=False. Read results/t016_confusion_debiased_semantics/RESULTS.md. Preserve stopped2e51d81provenance. No nextstage; awaitLead.

2026-09-14 07:36 +08 Lead3246221 synchronized. Start T017 with fixed protocol and historical/noise-free checks; prior T016R exit0 confirmed.

2026-09-14 07:39 +08 T017 preflight launched run20260914-073926-ttfl-t017-preflight, runtime2f7b88b. Protocol/seeds saved before diagnostics; bootstrap contingent on exact P00 sanity.

# T017 STOP — NOISE-FREE EXACT-P00 TIES

2026-09-14T07:43:02+08:00. Lead3246221;runtime2f7b88b0bee6f3b587d42f21837f86a8356536b9;run20260914-073926-ttfl-t017-preflight;80testsPASS;historical replayPASS; inverseerror5.27e-15 but71/8000state changes all exactP00ties,zero trueutilityregret.25/40querycountrows change−.067to+.020pp. MandatoryexactP00invariantfails;bootstrap0. Read results/t017_channel_noise_decomposition/RESULTS.md. AwaitLead; no automaticretry or scientifictaxonomy.

2026-09-14 08:34 +08 Leada9fa543 synchronized. Narrow tie-aware sanity implemented; exact lookup accelerates identical rational decisions. Preparing envelope and frozen bootstrap.

2026-09-14 08:36 +08 T017R run20260914-083609-ttfl-t017r-cached launched at7fbfbb2. Raw bootstrap decisions use exact rational utility arithmetic without overrides.

2026-09-14 08:43 +08 T017R experiment finished exit0,174.97s. Initial SFTP download stalled with template_diagnostics.json.gz at32768bytes across repeated checks; stopped only matching local scp PID20768 to trigger existing workflow legacy-SCP fallback. No experiment rerun; formalrun20260914-083609-ttfl-t017r-cached remains source.

# T017R COMPLETE

2026-09-14T08:45:24+08:00. Runtime7fbfbb26e6894dbdc7f1a67e7f43537aaf2b36d5;run20260914-083609-ttfl-t017r-cached;87testsPASS;512000bootstrapepisodes,4096000exacttemplatechoices;0newforwards. DiagnosisT017-N;scaleK40;tie-envelope robustTrue. Read results/t017_channel_noise_decomposition/RESULTS.md. Priorstopda202c8preserved. No nextstage;awaitLead.

# T017R COMPLETE

2026-09-14T08:48:04+08:00. Runtime7fbfbb26e6894dbdc7f1a67e7f43537aaf2b36d5;run20260914-083609-ttfl-t017r-cached;87testsPASS;512000bootstrapepisodes,4096000exacttemplatechoices;0newforwards. DiagnosisT017-N;scaleK40;tie-envelope robustTrue. Read results/t017_channel_noise_decomposition/RESULTS.md. Priorstopda202c8preserved. No nextstage;awaitLead.

2026-09-14T11:37:43.7247025+08:00 T0181744440 synchronized viae3e9c70; fixedCLS-S and8focusedtests implemented. Preparing95test+200caseindependent numericalpreflight before queryevaluation.

2026-09-14T11:39:23.7310672+08:00 T018preflight launched run20260914-113842-ttfl-t018-preflight at8718519; numericalsubset fixed200cases, no metricaccess.

# T018 STOP — FIXED PGD CAP

2026-09-14T11:44:07+08:00. Lead1744440;runtime87185194a087f09da634a6f88d81e2fd2cfee59c;run20260914-113842-ttfl-t018-preflight;95testsPASS;noise-free1000PASS;49/200actualcap20000,41referencefailures;151convergedallreferenceagree. Worstpierror.04343. Noquery/bootstrapevaluation;CLSgatesNOT_EXECUTED. Read results/t018_constrained_prevalence/RESULTS.md. No solverchange ornextstage;awaitLead.

2026-09-14T12:40:48.3202788+08:00 T018R lead8d12997 synchronized. Added specified deterministic active-set solver with per-channel lazy face maps, old PGD preserved. Three focused tests passed (12 synthetic reference comparisons, condition3/50/100/200, insertion/removal, boundary, deterministic replay). Preparing remote full suite and sealed numerical preflight. Previous run exit1 confirmed; no running tmux session.

2026-09-14T12:41:58.0813363+08:00 T018R preflight launched: run20260914-124138-ttfl-t018r-preflight, release20260914-124118-ttfl-t018r-preflight, runtime5f5e1a5a845f743646aea7f67c1d47463f1b9832. Full98tests then sealed noise-free1000/same200/all1000 numerical checks.

2026-09-14T12:46:06.8190226+08:00 T018R preflight PASS:98tests,1000noise-free,same200reference incl49oldcaps,all1000actual KKT/dominance. Raw receipt fetched. Implemented unchanged cached K20 paired evaluation, oracle freeze before source composition.

2026-09-14T12:47:31.3048309+08:00 T018R science launch run20260914-124709-ttfl-t018r-science runtime315a984ee4ba5c26495b48f3a73c081c5f0340bc release20260914-124625-ttfl-t018r-science. Deploy SSH255 occurred at current-symlink update; confirmed release present/current old/no job, repeated only symlink+last-release update then launched once. Numeric certification of128000matched q before metrics.

# T018R STOP — matched simplex sum tolerance

2026-09-14T12:50:42+08:00. Lead8d12997; preflight runtime5f5e1a5/run20260914-124138-ttfl-t018r-preflight exit0:98tests,1000noise-free,same200reference incl49oldcaps,all1000real audit pass. Science runtime315a984/run20260914-124709-ttfl-t018r-science exit1 before metrics:45,575matched solves accepted then client35/B/Dark/replica7 sum error1.0480505352461478e-12 >1e-12. Same-runtime reproduced; independent face error0, maxpi difference4.77e-13. No scientific gates/outcome; zero new forwards. Read results/t018_constrained_prevalence/t018r/RESULTS.md. No changes to tolerances or solver after stop; await Lead engineering instruction.

2026-09-14T13:25:28.4643111+08:00 T018R2 leadc8aea5b synchronized; previous run exit1/no active tmux confirmed. Changed only cached inverse application to direct KKT RHS solve; added exact frozen blocker fixture regression and many-RHS cache regression. Five focused tests pass; thresholds/working-set logic unchanged. Preparing full100tests and sealed recertification.

2026-09-14T13:26:16.6709323+08:00 T018R2 preflight launched run20260914-132551-ttfl-t018r2-preflight release20260914-132535-ttfl-t018r2-preflight runtimee4b83f09ea15357993d4d3fcb6562c4c45104b65. Full100tests then exact blocker/noise1000/reference200/allactual1000 checks.

2026-09-14T13:27:11.7230756+08:00 T018R2 preflight exit0/100testsPASS. Science launched once as run20260914-132634-ttfl-t018r2-science at samee4b83f0 release20260914-132535-ttfl-t018r2-preflight. Command used --output artifacts/t018r2_constrained_prevalence relative to release cwd; actual output is releases/20260914-132535-ttfl-t018r2-preflight/artifacts/t018r2_constrained_prevalence. After completion copy to canonical run/artifacts for receipts; do not rerun experiment for output location.

2026-09-14T13:37:45.7158461+08:00 Science completed exit0/84.08s: OutcomeB, matched4/4 real1/4 source1/4. Artifacts copied from release output into canonical run and fetched. Local report replay was slow due repeated NPZ decompression inside Fraction-check loop; stopped only own report process42548, cached its two read-only arrays, restarting report only (no experiment rerun).

# T018R2 COMPLETE — Outcome B

2026-09-14T13:38:04+08:00. Leadc8aea5b/runtimee4b83f09ea15357993d4d3fcb6562c4c45104b65; preflight20260914-132551-ttfl-t018r2-preflight, science20260914-132634-ttfl-t018r2-science, bothexit0.100testsPASS; all128000matched numerical solves PASS, maxsum2.22045e-16,maxKKT2.22045e-16. CLS-MATCH-A4/4PASS; CLS-REAL-A1/4FAIL; CLS-SRC-A1/4FAIL (Darkonly). BothsafetychecksPASS. Zero newforwards/noresampling. Read results/t018_constrained_prevalence/t018r2/RESULTS.md and independent_verification.json. Previousstops preserved. ReturntoLead; no nextstage.

2026-09-14T14:51:15.9241303+08:00 T019 d37bbc4 synchronized. Reuse unchanged posterior, CLS and exact-template code. New residual/seed/paired-ID helpers pass7focused tests. Protocol frozen before run: exact counts, SHA256 perclass PCG64 replacement, same-source-ID paired clean, per-salt two-half averages and all-salt stability, positive individual capture denominators only. Prepared PhaseA membership/identity replay then no-query null/repair/paired choices.

2026-09-14T14:55:47.5463052+08:00 T019 launched run20260914-145145-ttfl-t019 release20260914-145126-ttfl-t019 runtime6b5b2c0162ae46d8b4db18d06393d932147ade15; tests then posterior/utility preparation only. Implemented downstream per-cell null percentiles, exact utility-ranked class repair, all-salt gates and final query layer after mechanism hashes.

2026-09-14T14:56:47.6824238+08:00 T019 preparation exit0/180.53s;107testsPASS; PhaseA and all posterior/null/repair/paired utility choices frozen. Scientific clarification from existing bootstrap_observation: T017 K20 already conditions on exact counts. T019 adds per-class seed streams, decomposition, task-null percentiles, repairs and paired IDs; do not claim newly removed composition randomness relative to T017.

2026-09-14T14:59:17.4447596+08:00 T019 finish runtime15ad29fb9195240b290231a75c84ccddf05c9c5d run20260914-145757-ttfl-t019-finish writes into original20260914-145145 artifact directory. First finish deployment timed out before creating release; confirmed original run exit0/current old, then redeployed as20260914-145739-ttfl-t019-finish. No preparation rerun.

# T019 COMPLETE — T019-X

2026-09-14T15:05:51+08:00. Lead d37bbc4. Prep6b5b2c0/run20260914-145145-ttfl-t019;finish15ad29f/run20260914-145757-ttfl-t019-finish;both exit0.107testsPASS;zero newforwards. TASK-MISMATCH-A0/3FAIL;CTX-SPEC-A0/3FAIL;LOCAL-AallcontextsPASS. Contrast/Noise/Blur task tails2–11%,pairedtails12–19%;cleanunion11/17%. Small-margin state errors frequent but upperhalf margins carry62.02%true regret: mixed, not forced S. T017 K20 already fixed class counts; T019 adds decomposition/paired/repair diagnostics. Read results/t019_real_channel_heterogeneity/RESULTS.md and independent_verification.json. Await Lead; no nextstage.

2026-09-14T15:47:29.2771517+08:00 T020 lead5bb4d6e synchronized; previous T019 finish exit0/no running tmux. Added label-blind fixedR256 position bootstrap and exact integer/rational utility aggregation. Eight focused tests pass incl label blindness/ties/BER-regret equivalence/fixed-context dispatch. Protocol frozen global>=99% and perbankcontext>=99%; no label/query unsealing if stability fails.

2026-09-14T15:49:08.2390111+08:00 T020 run20260914-154818-ttfl-t020 launched at29d8443cf698014ec3f938e598e2633381941e77 release20260914-154738-ttfl-t020. Deploy SSH255 at symlink update; confirmed new release present/current old/no job; repaired only symlink+last-release, then launched once. Full115tests and256000label-blind bootstrap cells before any privileged metrics.

2026-09-14T15:53:03.4084889+08:00 T020 numerical bootstrap completed115tests/256000CLS pass, exit1 mandated MC stop:97.95%128vs256 agreement (164/8000 changed), global<99% and9/10bank-context groups<99%. 99.23s, no true composition/utilities/querycount parsing. Fetch same completed run; do not increaseR or launch evaluation.

# T020 STOP — MC instability

2026-09-14T15:54:10+08:00. Lead5bb4d6e/runtime29d8443cf698014ec3f938e598e2633381941e77/run20260914-154818-ttfl-t020,exit1.115testsPASS;256000CLSsolvesPASS. BER128-vs256 agreement97.9500%<99%;164/8000slots differ. Mandatory MCstop before true labels/utilities/querycounts. All scientific gates NOT_EXECUTED; no T020-R/C/F/X. Independent2000CLS+8000exactmeans+16000replicaargsets replayPASS;219upstream hashes unchanged. Read results/t020_bootstrap_expected_regret/RESULTS.md. No increasedR/newseeds/alternatepolicy;awaitLead.

2026-09-14T16:38:18.8827352+08:00 T020R Lead f0807dd: same-stream fixed R4096 extension implemented, first256 immutable; 6 focused tests PASS. Four independent client processes with 2 BLAS threads each. No privileged labels/query opened. Preparing runtime commit and remote run.

2026-09-14T16:39:31.8816668+08:00 T020R run 20260914-163857-ttfl-t020r launched once at runtime 0ec0480fc89f7dae48aa2e16a27bf5a87e1b49bf, release 20260914-163835-ttfl-t020r. Running 121 tests then fixed4096 sealed preparation.

2026-09-14T16:49:30.4888994+08:00 T020R remote exit1 due ActiveSetCLS working-set cycle;98client receipts retained (missing64,87). Main logged62futures before exception, executor drained submitted work. Local deterministic diagnosis:64/A/contrast_low/3071 and87/A/contrast_low/2688 cycle with fresh solver; exhaustive-face reference KKT3.11e-17/3.90e-17. No solver changes or policy fallback; not MC2; complete convergence/scientific gates unexecuted. Independent partial receipt verification running.

# T020R implementation stop — CLS cycles, still sealed

2026-09-14T16:50:42+08:00. Lead f0807dd; runtime 0ec0480fc89f7dae48aa2e16a27bf5a87e1b49bf; run 20260914-163857-ttfl-t020r, exit1. 121testsPASS but only98/100clients complete. Missing64/87 reproduce deterministic ActiveSetCLS cycles; see results/t020r_mc_convergence/solver_cycle_diagnosis.json and RESULTS.md. Complete MC gate NOT_EXECUTED: not MC2, no scientific diagnosis. Labels/true utilities/query remain sealed. Independent3920sampledCLS/7840legacy/54880block checks PASS; 245upstream manifests unchanged. Frozen solver unchanged. Await Lead solver-specific repair; do not rerun full task or silently substitute reference solver. Preserve all partial98clients and original T020 prefix.

# T020R implementation stop — CLS cycles, still sealed

2026-09-14T16:51:27+08:00. Lead f0807dd; runtime 0ec0480fc89f7dae48aa2e16a27bf5a87e1b49bf; run 20260914-163857-ttfl-t020r, exit1. 121testsPASS but only98/100clients complete. Missing64/87 reproduce deterministic ActiveSetCLS cycles; see results/t020r_mc_convergence/solver_cycle_diagnosis.json and RESULTS.md. Complete MC gate NOT_EXECUTED: not MC2, no scientific diagnosis. Labels/true utilities/query remain sealed. Independent3920sampledCLS/7840legacy/54880block checks PASS; 245upstream manifests unchanged. Frozen solver unchanged. Await Lead solver-specific repair; do not rerun full task or silently substitute reference solver. Preserve all partial98clients and original T020 prefix.

2026-09-14T17:29:32.4536996+08:00 T020R2 Lead43d2f85. Existing5solver testsPASS; cycle-only fallback3testsPASS; normal1000real+4020bootstrap bitwise identical vs frozen solver with0fallback,8000point template decisions same;553upstream manifests and98client196files hash verified. Resume implementation reuses98 and computes only64/87, unchangedR4096 gate;6aggregation testsPASS.

2026-09-14T17:30:52.1374855+08:00 T020R2 runtime1d7b6b8f7ca3e58fea1ae486cf7505192fdc05a1/run20260914-173004-ttfl-t020r2/release20260914-172956-ttfl-t020r2 launched once. Remote124testsPASS; client64 resumed. No previous98 re-solved; no privileged data opened.

2026-09-14T17:37:35.8320391+08:00 T020R2 complete100clients;124testsPASS;2cycle-only fallbacks. R4096 MC gateFAIL:98.825%H1H2,99.25%H1ALL,99.575%H2ALL;94slots differ. Scientific gatesNOT_EXECUTED, labels/querysealed. Local D:filled during artifact extraction; same archive SHA aae4ebed93f1ed54559ad0ae84a4820806f34aef7369168867228d2decb038db verified remote/local. Removed only redundant t020r_delivery_bundle.tar, re-extracted same completedrun successfully, removed temporary local receipt.tar. Final independent replay is running.

# T020R2 repair complete; sealed T020R-MC2 stop

2026-09-14T17:39:11+08:00. Lead43d2f85/runtime1d7b6b8f7ca3e58fea1ae486cf7505192fdc05a1/run20260914-173004-ttfl-t020r2.124testsPASS;98clients reused byte-identically;64/87 resumed with exactly2cycle fallbacks;complete100clients/8000slots. H1/H2=98.8250%<99%; H1/ALL=99.2500%<99.5%; H2/ALL=99.5750%.94disagreements/38clients/59cells. Independent4000CLS/8000legacy/56000blocks/allfallbacks/553upstreamentriesPASS. T020R-MC2; labels/true utilities/query sealed, all scientific gatesNOT_EXECUTED; no T020-R/C/F/X. Read results/t020r2_mc_convergence/RESULTS.md. AwaitLead; no furtherR/seeds/alternatepolicy/nextstage.

2026-09-14T18:31:13.7072993+08:00 T021 Lead7b5bb34 synced. Remote no active experiment; CUDA torch2.4.0cu121 available2devices despite NVML mismatch. Implemented experiment-local H/L extractor and frozen historical-equivalence preflight; no global model changes. Protocol fixed. Next remote125tests and GPU1 exact historical replay before prototypes.

2026-09-14T18:35:04.1900036+08:00 T021 extraction runtimee908116/run20260914-183139-ttfl-t021-extract-gpu1 exit0;1807frozenforwardcalls;historical equivalencePASS. Prototype helper tests2PASS (target exclusion and matched exact counts). Implemented Phase-A fixedoracle/source/clean-same-state L/H prototypes with existing10x10Gram CLS and exact choices; no target scores.
