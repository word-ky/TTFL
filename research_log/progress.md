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
