# T009 active — natural label-skew context and mixed per-client retrieval

Lead25e6ea7 read full. Baseline36remote tests PASS. Natural-support selection2focused stdlibtests PASS locally. Reuse source_signature.py unchangedSHA84542fb.., neutralwriter a687.., fixedT008prototype9138.., T007Rstates0d9d.., checkpoint/corruptions/pools unchanged. New evaluation only.

ID selection: access split train/query IDs only; exclude all160calibrationIDs and everyqueryID. For eachclient, random.Random(int(SHA256('7:{cid}:T009-natural-support'))).shuffle(eligibletrain-orderrecords). Choose largestcommon64/32/20/16 bycounts. Freezeeligiblecounts,K,selectedIDs/indicesbeforeloadingimagepixels/model. Ifnone>=16 reportstop noforward. IfK>=32 alsofirst20prefix IDonly. Trainingxonly accessed duringID; yfieldfirstinspectedafterallretrievalcomplete. Allprimaryandsecondarydecisions/signatures savedbeforequerymetrics orlabels skew. No bank/state/prototype re-fit.

Queries: eachbank/client/target chooses itsownstoredstate. ReuseT007Rpredictions fornone/correct/wrongalt(andclean→zero); onlyremainingclient/state/targetpairs newforward. Saveallmixedpredictions+perclientintegercounts forindependentaudit. Existing24bank/shift/control aggregate macroclassmetrics recomputedfromoldpredictions. Cleanmixedsafety>=base-.5pp. NIDbothoverall>=90eachcontext>=80; NRETjoint>=3of4 plusbothcleansafety. NoquerybasedK/parameterchange.

PostfreezeauditprimaryKonly: normalizedentropy,maxclassfraction,numclasses. Quartiles equal25clients sorted(entropy,clientID) tie-break fixed. Reportperbank/contextquartileIDaccuracy/margins; Spearmanmean-ranks ties, no pvalues;10mostnegativeoverall. CompareknownprimarydistquantilesandfrozenT008OODdist only,norejectionthreshold. Full40tests beforeformal100clientsexecution. NoSSL/nextstage.

## 2026-09-13T22:46:19.7221193+08:00 — launched
Run20260913-224559-ttfl-t009-gpu1 release20260913-224555-ttfl-t009 coded136746 onA6000GPU1.40tests beforeevaluation; recoverthisrun,noduplicate.

## 2026-09-13T22:46:58.8172260+08:00 — observed import repair
Run20260913-224559-ttfl-t009-gpu1 exit1: read_data missingimport atfirstsupportpixels. Manifestalreadywritten, noID/query/skewresults. AddexistingPFLlibread_dataimportonly; rerun deterministicselectionandverifyK/IDsagainstfailedrunreceipt. Preservefailure.

## 2026-09-13T22:47:38.8137888+08:00 — fixed runtime launch
Run20260913-224708-ttfl-t009-gpu1 release20260913-224704-ttfl-t009-importfix. Actual deployed git revision fab17bd; launch TTFL_COMMIT string was IMPORT_FIX (labelonly, notSHA). Preserve rawlabel, supplemental runtime_revision receipt will map deployedfileSHA tofab17bd; no rerun for metadata.

## 2026-09-13T22:53:51.1798345+08:00 — COMPLETE N-C
Run224708exit0,40tests PASS. K20 mineligible30 max1521. NID97.6/98.0PASS; cleanmacrodelta-.610321/-.616185 failsafety; Blurretention.342997/.538886 fails.3/4shiftjointPASS insufficientdueclean.985oldclientpredictionsreused15newcleanwrongstateevals. Independentdistanceerror1.14e-13/count7.1e-15pp; Fractiongatesmatch. Bothattemptmanifestsidenticalexcepttimestamp. RuntimeIMPORT_FIXalias resolvedfab17bd via remoteCRLFfileSHA6684a.. and normalizedgitLF match. Rawreceipts retained. LowestentropyquartilecontainsallBlurerrors; percontextSpearmanweakmaxabs.171. Client29Blur→clean costs~.77pp perbank. AwaitLead,nointerpolation/SSL/nextstage.
