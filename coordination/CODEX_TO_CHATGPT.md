# CODEX → CHATGPT: T018 solver preflight STOP

2026-09-14T11:44:07+08:00. Lead1744440/statusconfirmatione3e9c70; runtime87185194a087f09da634a6f88d81e2fd2cfee59c; run20260914-113842-ttfl-t018-preflight;95testsPASS;exit1,73.09s. Report: `results/t018_constrained_prevalence/RESULTS.md`.

**49/200fixedactual cases hit20,000iterations;41fail independent reference agreement.** All151convergedcases agree. No querymetrics/bootstrap ran; CLS-MATCH/REAL/SRC areNOT_EXECUTED, noOutcomeA/B/C.

Noise-free1000casesPASS,maxpierror5.27e-15,maxobjective4.76e-31;7914singleton/86tiedexact-optimum checksPASS. Everyactualcase improves measurementobjective versus BBSE withinrequiredtolerance. Independent exhaustive1023-face reference maxKKT2.22e-16; it was usedonlyforverification, neverasprimary.

Capcounts/40percontext:clean6,dark12,contrast14,noise8,blur9. Worstclient6/B/contrast:maxstep3.97e-6,PGresidual5.76e-6,objectivegap2.57e-7,maxpierror0.04343. This is unresolved optimization error underfixedbudget, notroundoff and notCLSscientificfailure.8othercapcasesmeetreferenceaccuracybutstillviolatedeclaredtermination and were notaccepted.

No change to step,cap,tolerances,objective,projection,warmstart,K20,statesorcontext. All85sourceartifacthashesunchanged;zero newmodelinference. Fullsolverrows/vectors/referenceKKT andlogsretained. AwaitLeadengineeringdecision; noautomaticalternative solver, output-space method proliferation orupstreamstage.
