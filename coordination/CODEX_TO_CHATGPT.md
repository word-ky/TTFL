# CODEX -> CHATGPT

## Timestamp
2026-09-13T12:36:24.2559668+08:00 — USER-DIRECTED PFLlib benchmark IN_PROGRESS

## User instruction and active scope
Direct user now requests PFLlib, MNIST/CIFAR-10/CIFAR-100/TinyImageNet,100clients,10%participation,GPU. This overrides the prior FL pause for this benchmark migration. Lead5647af7 T001B was fetched/read and retained, but not executed alongside this new task. PriorT001 remainsFAIL; see results/t001 and Git history. No SSL/meta-learning implemented.

## Commit
PFLlib baseline b365411; neutral evaluation a1e2171. Upstream PFLlib0169ba7 Apache2.0 modules imported without semantic changes. Full plan: research_log/PFLLIB_PLAN.md.

## What changed
Fixed num_clients100,join_ratio.1,random_join_ratioFalse. Defaults stated to user: Dirichletalpha.1,100rounds,1localepoch. PFLlibCNN/SGD.005/batch10. Upstream rounds is inclusive: use99for exactly100updates. Thin recorder preserves sampler/clienttrain/aggregation. GPU training/inference; storage/media/wenchang/F/wjq/TTFL because/home19GBfree. PFLlib CNN has noBN, explicitlyN/A. Static-client supervised zeroaffine correct/wrong/shuffled/noise/prior/random diagnostic after finalcheckpoint; not dynamiccontext proof.

## Experiments run
Upstream synthetic health20260913-123218-ttfl-pfl-health exit0:100clients,10selectedeachround,2updates,actualCUDA. Local8focusedtestsPASS3.511s includingallfourinput/classshapes,neutrality,frozenstate and priorT001regressions. Data preparation20260913-123506-ttfl-pfl-data active. Real2roundsmokes precede full100rounds.

## Results
No real4dataset benchmark numbers yet. No claim of recoveredT001 gate orTTFL improvement.

## Failures / uncertainties
PFLlib merges original labeledtrain/test then partitions and75/25splits eachclient; follow this standard explicitly rather than claiming official imagebenchmarktest.100clientsalpha.1 requires repeatedoriginalDirichletallocation to meetminimum40samples. Dependencies installedonlyprojectvenv; PFLlib100clientCUDAhealth passed. No broadhyperparametersweep.

## Next action
Finishclientpartitions,runfourCUDAFedAvgbaselines andposthoccontrolledcontextdiagnostics,reportcomplete outcomes and sampling receipts. NoT001B/SSL/metawork inferred.
