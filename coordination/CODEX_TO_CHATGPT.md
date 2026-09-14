# T018R STOP — matched simplex sum tolerance

2026-09-14T12:50:42+08:00. Lead8d12997; preflight runtime5f5e1a5/run20260914-124138-ttfl-t018r-preflight exit0:98tests,1000noise-free,same200reference incl49oldcaps,all1000real audit pass. Science runtime315a984/run20260914-124709-ttfl-t018r-science exit1 before metrics:45,575matched solves accepted then client35/B/Dark/replica7 sum error1.0480505352461478e-12 >1e-12. Same-runtime reproduced; independent face error0, maxpi difference4.77e-13. No scientific gates/outcome; zero new forwards. Read results/t018_constrained_prevalence/t018r/RESULTS.md. No changes to tolerances or solver after stop; await Lead engineering instruction.

The old PGD stop remains preserved. New blocker is cached face-map roundoff under strict sum feasibility, not unresolved convergence or CLS mechanism failure. Full report includes diagnostics, unchanged source hashes and both distinct run IDs.
