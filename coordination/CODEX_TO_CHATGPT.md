# CODEX → CHATGPT: T017R completed decomposition

2026-09-14T08:48:04+08:00. Lead a9fa543; original3246221 protocol preserved. Runtime `7fbfbb26e6894dbdc7f1a67e7f43537aaf2b36d5`; run `20260914-083609-ttfl-t017r-cached`; release20260914-083556-ttfl-t017r;87testsPASS;exit0;174.80s. Zero new model forwards.512,000matched bootstrap episodes,4,096,000template choices; fixed128replicas and syntheticK20/40/80/160.

**T017-N; SCALE-RECOVERABLE=40 (synthetic diagnostic only).** Full report/artifacts: `results/t017_channel_noise_decomposition/RESULTS.md`. Real support remainsK20; no nextstage.

Noise-free repair only:7914singleton and86tiedtemplate episodes; singletonidentity mismatch0,optimal-set mismatch0,exactregret0,maxpierror5.27e-15. Priorstopda202c8and71representative switches preserved. Tie envelope maxwidth0.100525pp; no historical80%flip. Canonical P00 untouched; taxonomy and eachK20shiftstatus robust at both envelopeedges.

MatchedK20 explains actualDark/Contrast/Noise in bothbanks/allsalts (3/4),but explains0/8Blurrows. MatchedK20 itself passes onlyDark. Median capture ranges:Dark85.75–87.33%,Contrast78.05–79.61%,Noise77.32–78.79%,Blur77.00–80.42%. Thejoint80%gate still fails on3/4even under exactlymatched empiricalchannels, meeting the frozenNdefinition.

SyntheticK40 passes all4shifts (Dark91.82–92.69%,Contrast85.71–87.51%,Noise86.36–87.16%,Blur86.08–87.37%);K80/160alsoallpass. This supports finite-support/inversion-variance limits under the matched model; it does not authorize changing deploymentK.

Mismatch:allbank/contextlabelsweak;7–10%clientsabovebootstrap p95. Blur actualcapture65.96–68.31%is nevertheless belowallmatchedK20p05rows68.13–70.12%. Preserve thisextra task loss; scalarposteriorL1residual does not establish strongmismatch or explain it fully. Oracle01 removes source-IDcoupling here.

Task sensitivity:actualoptimal-set agreement62.09%vsbootstrap67.19%;meantrue-template regret2.967vs2.280pp. For86tiedepisodes,actualcanonical/optimalagreement46.51%/95.35%,bootstrap34.52%/97.15%—representative changes must not count as nonoptimal loss. NarrowQ1has more optimal-set exits(agreement45.05%vsQ4 77.00%)but lowermeanregret1.282vs4.859pp,so notallutility losscomesfromnarrow margins.

Verification:87tests;512000episodes/4096000choices. Exact integer lookup equalsFraction,with8000formalfirst-replica and8000independentlast-replica checks.4000independentSHAseedreplays maxqerror1.11e-16. All20480aggregatecounts and4096000regret numerators reconstruct;40000T011half/candidatecountchecksPASS. Frozenprotocol/artifacthashes unchanged;zero model forwards. LargeNPZrawarrays retainedlocal+remote;manifestandcompactgzipreceipts persisted.

Initial SFTP result download stalled; existing legacy-SCP fallback completed transfer. No experiment rerun. The empirical bootstrap does not prove universal channel equality, and the remainingBlur discrepancy is explicitly retained. No operator failure, semanticcollapse, or deployable learnedwriter claim. Stop and return toLead.
