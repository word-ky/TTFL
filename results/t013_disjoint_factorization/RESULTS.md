# CODEX -> CHATGPT

## Timestamp / commit / run

2026-09-14T02:33:54+08:00. Lead9d6ada0; runtime 089c918d38273c06b85074c6e51224b5b4c0bf28; local run 20260914-t013-local; 42.59s; CPU analysis of saved predictions only. No new model forward pass or GPU job was needed. Enclosing delivery commit contains this report.

## Frozen evidence / implementation verification

The formal T011 candidate_predictions.npz exists and its SHA matches its original query-freeze receipt. All5,000 per-client/state class-correct/total rows reconstruct exactly; every prediction/label vector length matches the frozen original query-ID length. All T012 full-query pair overlaps and eight-offset mean controls reproduce before splitting. No fallback prediction regeneration, state/model modification or training.

Two focused tests pass: label-blind complete/disjoint deterministic splitting and exact equal-client leave-one-out utilities with ties. Policy utilities use Fraction arithmetic. Test changes the target client’s shifted training counts and confirms its leave-one-out context term is unaffected. The API accepts only training-half counts and denominators. Historical model tests were not rerun because this task has no model code or forward pass.

## Label-blind disjoint halves

Four fixed salts T013-S0..S3. SHA256(T013|salt|client|original_id) order with index tiebreak; alternating ranks become H0/H1. Every index appears once, no overlap, size difference<=1. Membership is identical across contexts/banks. query_halves.json frozen at 2026-09-13T18:30:10.417117+00:00, SHA 38af1dd67b0125d032a06ab3b40e991b5239eef5f5c0c1795312467fb9cf6fc4, before T013 half-derived analysis. Required historical full-query count regression accessed historical labels first; it did not influence the split.

## Disjoint-example CLIENT-LOCK

| salt | bank | mean_same_overlap | mean_control_overlap | difference_pp | passed |
| --- | --- | --- | --- | --- | --- |
| T013-S0 | A | 0.613500 | 0.373375 | 24.012500 | True |
| T013-S0 | B | 0.614500 | 0.380875 | 23.362500 | True |
| T013-S1 | A | 0.605500 | 0.377312 | 22.818750 | True |
| T013-S1 | B | 0.613000 | 0.387875 | 22.512500 | True |
| T013-S2 | A | 0.623000 | 0.385875 | 23.712500 | True |
| T013-S2 | B | 0.609000 | 0.382312 | 22.668750 | True |
| T013-S3 | A | 0.617500 | 0.387062 | 23.043750 | True |
| T013-S3 | B | 0.605500 | 0.383375 | 22.212500 | True |

**DISJOINT-LOCK-STRONG passes all8 salt/bank combinations.** Margins22.2125–24.0125pp exceed the unchanged15pp gate. These margins are smaller than T012’s28.0–28.2pp, but remain substantial on disjoint underlying images. Repeated-example coupling therefore cannot explain away persistent client preference. Class-composition coupling is still possible because disjoint samples from a client share its distribution.

disjoint_lock.csv retains every pair, orientation and offset summary. orientation_averaged_pair_overlap.csv supplies per-pair values; disjoint_utility_similarity.csv supplies valid counts, same-client mean/median Spearman and control mean/range. The JSON retains all individual offset statistics. Utility ranks use exact half correct counts, equivalent to half-specific DeltaAcc ranks.

## Cross-fit factorization and policy freeze

For each target half, persistent P is the same client’s clean utility on the opposite half. Context C is the equal-client mean of shifted-minus-clean utilities from the other99 clients on that training orientation. two_factor chooses argmax(P+C). client_only uses P; context_only uses C; eight mismatches replace P with client(i+k)%100 while keeping the same C(-i). All predicted argmax sets are retained; the first frozen candidate wins policy ties.

All120,000 half-policy choices and8,000 exact half-argmax sets were saved before composition analysis. Policy freeze 2026-09-13T18:30:39.425519+00:00; choices SHA 4d2b1bf8091f14b36690ba33e89c9874d12e7c4e9013655f37419402ca3206cd. Every example receives a state selected using its opposite-half labels, never its own evaluation-half labels, for the cross-fit diagnostic policies. Half-oracle and historical full-query oracle are explicitly privileged comparison policies and do use evaluated labels.

**This is supervised diagnostic factorization with a known context and other-client label access. It is not an unlabeled writer or deployable adaptation method.** Concatenating the swapped evaluation halves reconstructs each full-query policy.

## Cross-fit metrics and oracle granularity

Main macro-class accuracy, averaged across all four salts (all per-salt metrics remain in crossfit_metrics.csv):

| bank | target | zero | true_state | full_query_oracle | half_oracle | client_only | context_only | two_factor | mismatch_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | 34.138690 | 34.138690 | 40.723160 | 40.995225 | 39.857358 | 34.138690 | 39.857358 | 32.719017 |
| A | brightness_dark | 17.546026 | 32.296536 | 35.712121 | 35.879543 | 25.613593 | 24.233087 | 33.043927 | 27.725936 |
| A | contrast_low | 19.955264 | 28.561473 | 33.993166 | 34.175340 | 28.366562 | 28.561473 | 29.885422 | 25.213045 |
| A | gaussian_noise | 30.676644 | 32.499558 | 37.825842 | 38.082635 | 36.912367 | 31.861084 | 37.130695 | 30.376176 |
| A | gaussian_blur | 30.901579 | 33.048500 | 40.343614 | 40.665880 | 38.464626 | 29.798137 | 38.556368 | 30.720294 |
| B | clean | 34.138690 | 34.138690 | 40.572038 | 40.878239 | 39.650134 | 34.138690 | 39.650134 | 32.693556 |
| B | brightness_dark | 17.546026 | 32.424808 | 35.686070 | 35.859442 | 25.490595 | 20.155433 | 33.047307 | 27.652445 |
| B | contrast_low | 19.955264 | 28.111626 | 33.862340 | 34.037413 | 28.198676 | 28.111626 | 29.803546 | 25.036584 |
| B | gaussian_noise | 30.676644 | 32.406771 | 37.894827 | 38.159925 | 36.779287 | 31.432144 | 36.999457 | 29.947931 |
| B | gaussian_blur | 30.901579 | 33.009919 | 40.374218 | 40.711123 | 38.478654 | 28.354235 | 38.366666 | 30.234160 |

The split-matched half-oracle chooses a fixed state separately per evaluation half and is the prescribed gain denominator. Both oracles maximize on the same outcomes used to report them, so they are optimistic diagnostics. All exact ties are preserved; display/policy order remains clean,dark,contrast,noise,blur. State-frequency counts sum to200 decisions per salt/bank/context/policy; regret and hit rate are unweighted over those200 client halves.

crossfit_metrics.csv contains all600 salt/bank/context/policy rows, including every mismatch offset: macro-class, sample-weighted, macro-client accuracy, selection frequencies, half-oracle hit rate, regret median/p75/p90 and >2/>5pp fractions. Raw per-half regrets are in crossfit_regret.json.gz. Integer_count_receipts.json plus the formal half-count artifact allow reconstruction. An independent per-example concatenation path exactly matches every policy’s class counts; historical full-query best-of-five matches T011.

## Factor capture and mismatch controls

| salt | bank | target | factor_capture | two_minus_client | two_minus_context | two_minus_mean_mismatch | mismatch_mean | mismatch_min | mismatch_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T013-S0 | A | clean | 0.8407814275959522 | 0.0 | 5.772385024505814 | 7.295535858997333 | 32.61553908527702 | 31.51476411152267 | 33.692104728892495 |
| T013-S0 | A | brightness_dark | 0.8511620566559701 | 7.761774723577536 | 8.35401211724269 | 5.454635466866205 | 27.66667442859193 | 26.213552503233068 | 29.366974033427216 |
| T013-S0 | A | contrast_low | 0.7011411307135998 | 1.5429979566165122 | 1.3433748722925731 | 4.59897188993306 | 25.30587640286553 | 23.15459230540695 | 26.968040547085508 |
| T013-S0 | A | gaussian_noise | 0.846550169407927 | 0.08502760040914234 | 5.459382956565222 | 6.670674848427449 | 30.288734795843297 | 28.844701380600306 | 31.534295580564223 |
| T013-S0 | A | gaussian_blur | 0.8092038129842752 | 0.25881570245812413 | 8.981592087686208 | 8.178772557410392 | 30.60095670596814 | 29.358545990881318 | 32.08628069580212 |
| T013-S0 | B | clean | 0.8343843888725231 | 0.0 | 5.614275925269238 | 6.9863734880622435 | 32.766592356975536 | 31.176062023722803 | 33.633765845212395 |
| T013-S0 | B | brightness_dark | 0.8601997418852331 | 7.783378032458467 | 13.099056219200062 | 5.280358145817598 | 27.97413113835403 | 26.059173025840888 | 29.116909731260737 |
| T013-S0 | B | contrast_low | 0.7008023855183328 | 1.594375033852829 | 1.7295186294162088 | 4.810629697786023 | 25.030514441454386 | 22.601433554531365 | 26.764442665085102 |
| T013-S0 | B | gaussian_noise | 0.8315768620485834 | 0.11297709940640503 | 5.430793507343744 | 6.974544369257906 | 29.934231160591995 | 27.924867638211214 | 30.824912500099337 |
| T013-S0 | B | gaussian_blur | 0.7840005364330102 | 0.051956012600215054 | 10.213171766057302 | 8.445618559765219 | 30.12178819144958 | 29.049815493383445 | 31.581213840969713 |
| T013-S1 | A | clean | 0.8311610930193293 | 0.0 | 5.683542616938382 | 7.076731865110514 | 32.74550067159641 | 31.71251564044934 | 33.772679904801414 |
| T013-S1 | A | brightness_dark | 0.8149490729968152 | 6.88060215216742 | 5.72024624320601 | 4.953488430443308 | 27.537297856364354 | 26.241513395214792 | 29.559426962777273 |
| T013-S1 | A | contrast_low | 0.6990480679612018 | 1.8217472669848458 | 1.3536261724958607 | 4.59175896841146 | 25.32334062459042 | 23.760242264788616 | 26.923164808627934 |
| T013-S1 | A | gaussian_noise | 0.8654847306224718 | 0.2982599909843245 | 5.057074390213513 | 6.570493451017103 | 30.573763451611125 | 29.777364997620296 | 31.480607302300914 |
| T013-S1 | A | gaussian_blur | 0.7779132508302802 | 0.22049104340432815 | 8.753525831487352 | 7.762782363386395 | 30.78888064379328 | 29.511544086221505 | 32.06314377030927 |
| T013-S1 | B | clean | 0.791394995981064 | 0.0 | 5.341655786001336 | 6.903167922208263 | 32.57717778356161 | 30.499693119852616 | 34.016052704580126 |
| T013-S1 | B | brightness_dark | 0.8135019764087258 | 6.749296546987826 | 12.282756270084109 | 4.90639657713788 | 27.531792757917795 | 25.326244490380247 | 29.056677570797092 |
| T013-S1 | B | contrast_low | 0.7052585341021192 | 1.671681192886168 | 1.8010995948889974 | 4.855608200023245 | 25.057116904689952 | 22.928956613302834 | 26.241123188884877 |
| T013-S1 | B | gaussian_noise | 0.8145588899749421 | 0.3155443292744911 | 5.566470037681851 | 6.841645055576781 | 29.977802488234328 | 28.45861074523629 | 30.79847896262415 |
| T013-S1 | B | gaussian_blur | 0.7542441535980053 | -0.12018507781346931 | 9.961152093136292 | 8.082863942801202 | 30.23252313549258 | 28.875566834421207 | 31.23426731754606 |
| T013-S2 | A | clean | 0.8562457063897874 | 0.0 | 5.837041401649558 | 7.204925106441637 | 32.77080621497646 | 31.69946957890279 | 33.9164451409764 |
| T013-S2 | A | brightness_dark | 0.8550594707509253 | 7.315884928861457 | 10.767865872887686 | 5.359536856373597 | 27.888428586230823 | 25.85699759395081 | 29.950205869241522 |
| T013-S2 | A | contrast_low | 0.6918737985948574 | 1.0896206897798106 | 1.2558954212863211 | 4.660656717028768 | 25.15671212476357 | 23.227542031135226 | 26.941113504923067 |
| T013-S2 | A | gaussian_noise | 0.884939748979879 | 0.16577388048595496 | 4.718165991039013 | 6.720124864361618 | 30.48445362231398 | 29.465548049315668 | 31.98519340664388 |
| T013-S2 | A | gaussian_blur | 0.7708065265034073 | -0.22314699451101927 | 8.657058467253474 | 7.534187499314009 | 30.92100814363179 | 29.559116465622143 | 32.05164541393109 |
| T013-S2 | B | clean | 0.8284023780111666 | 0.0 | 5.565886602053769 | 7.0435944527847685 | 32.66098206903754 | 30.903795067131195 | 33.577515474401615 |
| T013-S2 | B | brightness_dark | 0.8604563230517165 | 7.642162706682912 | 13.172559645440833 | 5.752422956715322 | 27.575569753697074 | 25.324578219540793 | 28.93622928270607 |
| T013-S2 | B | contrast_low | 0.6943610986852492 | 1.269154267349267 | 1.6182210919519897 | 4.701704268432742 | 25.028142333343446 | 23.069553837771128 | 26.7516142732859 |
| T013-S2 | B | gaussian_noise | 0.8613456255490337 | 0.246329830792336 | 5.614194890915407 | 7.121625154698895 | 29.96503308000674 | 28.406847182161297 | 31.20464682299374 |
| T013-S2 | B | gaussian_blur | 0.7509187561810896 | -0.31781535992712723 | 9.955589513875285 | 7.944792434802138 | 30.36503206423064 | 29.317839505246287 | 31.342836183106456 |
| T013-S3 | A | clean | 0.8082926137462065 | 0.0 | 5.58170240373355 | 6.976169944425593 | 32.7442223790765 | 31.93848660820517 | 33.83913628667903 |
| T013-S3 | A | brightness_dark | 0.8601587184580783 | 7.763071490151628 | 10.401234646824262 | 5.504300012088925 | 27.81134472916165 | 25.66974047627749 | 29.392279843445515 |
| T013-S3 | A | contrast_low | 0.7012420232721792 | 1.6210749503531616 | 1.342899034904976 | 4.838122542353751 | 25.06624991305724 | 23.016770826008038 | 27.49587076086516 |
| T013-S3 | A | gaussian_noise | 0.8891663518582318 | 0.3242530043955714 | 5.843822939485405 | 7.056785212922004 | 30.157750246742825 | 29.112224669251095 | 31.38674919748659 |
| T013-S3 | A | gaussian_blur | 0.7780213490494479 | 0.11080769604661458 | 8.640746304189433 | 7.868552119923953 | 30.570331359957805 | 29.176827947340318 | 31.797217636523975 |
| T013-S3 | B | clean | 0.8170254395228569 | 0.0 | 5.523958162687494 | 6.893175669988456 | 32.76947241246758 | 31.441426956513855 | 33.87844935534021 |
| T013-S3 | B | brightness_dark | 0.851613838660861 | 8.052010474660689 | 13.013122368919413 | 5.640269950485671 | 27.52828548340531 | 25.510883661650272 | 28.46387938847843 |
| T013-S3 | B | contrast_low | 0.6969289905597436 | 1.8842677487230521 | 1.6188410337144743 | 4.699903040618292 | 25.030563502920383 | 22.756219507554075 | 26.99247096800551 |
| T013-S3 | B | gaussian_noise | 0.872667805280545 | 0.2058282563237362 | 5.657793541190831 | 7.268285811637568 | 29.914659077846114 | 28.450345205967462 | 30.89584880313063 |
| T013-S3 | B | gaussian_blur | 0.75496472249597 | -0.061906516044868444 | 9.919810339699648 | 8.056747570747591 | 30.217297754109552 | 28.891219081470787 | 31.266655294129166 |

Dark and Noise exceed80% split-oracle capture in both banks under every salt. Contrast remains about69–71%; Blur about75–81% and fails under multiple salts. Thus only2/4 shifted targets pass the all-salt/all-bank requirement. No salt or mismatch offset is selected for favorable performance.

## Clean calibration safety

| salt | bank | delta_pp |
| --- | --- | --- |
| T013-S0 | A | 5.772385024505814 |
| T013-S0 | B | 5.614275925269238 |
| T013-S1 | A | 5.683542616938382 |
| T013-S1 | B | 5.341655786001336 |
| T013-S2 | A | 5.837041401649558 |
| T013-S2 | B | 5.565886602053769 |
| T013-S3 | A | 5.58170240373355 |
| T013-S3 | B | 5.523958162687494 |

| bank | mean_delta_pp |
| --- | --- |
| A | 5.718668 |
| B | 5.511444 |

Clean two_factor equals client_only because its transient term is exactly0. Cross-fit clean gain averages+5.718668 /+5.511444pp, passing the allowed−.5pp limit. This is supervised persistent calibration gain, not evidence of clean zero neutrality being violated or of label-free context adaptation.

## FACTOR decision

**FACTOR-A FAIL; FACTOR-B is the primary diagnosis.** Disjoint lock survives every salt/bank, and on Dark and Contrast the two-factor policy beats both single-factor controls by>=.5pp after averaging all salts, in each bank:

| bank | target | mean_two_minus_client | mean_two_minus_context | passed |
| --- | --- | --- | --- | --- |
| A | brightness_dark | 7.430333 | 8.810840 | True |
| A | contrast_low | 1.518860 | 1.323949 | True |
| A | gaussian_noise | 0.218329 | 5.269612 | False |
| A | gaussian_blur | 0.091742 | 8.758231 | False |
| B | brightness_dark | 7.556712 | 12.891874 | True |
| B | contrast_low | 1.604870 | 1.691920 | True |
| B | gaussian_noise | 0.220170 | 5.567313 | False |
| B | gaussian_blur | -0.111988 | 10.012431 | False |

On Noise, two_factor improves only~.22pp over client_only. On Blur the gain over client_only is+.092pp in A and−.112pp in B. Persistent calibration accounts for much of their gain; the simple global additive transient residual adds little. Both persistent and transient effects have predictive value, but additive utility is not sufficient under the frozen gate. Residual client×context interaction, estimation noise and finite-half oracle optimism remain possible explanations; this audit does not isolate their individual contributions.

## Post-freeze label-composition audit

label_composition_audit.csv records all800 client/salt/half rows: normalized entropy, maximum class fraction, represented classes and H0/H1 histogram L1/JS. Only after all policies and argmax sets were frozen were entropy/histogram-distance statistics computed. Continuous descriptive associations were used; no threshold, label-prior correction or matching rule was fitted.

| feature | overlap_rho_min | overlap_rho_max | regret_rho_min | regret_rho_max |
| --- | --- | --- | --- | --- |
| normalized_entropy | -0.124845 | -0.078822 | 0.021904 | 0.139950 |
| max_class_fraction | 0.110503 | 0.136226 | -0.138975 | 0.031370 |
| halves_histogram_L1 | -0.082986 | 0.049842 | 0.026454 | 0.199884 |
| halves_histogram_JS | -0.031242 | 0.089973 | 0.048513 | 0.225329 |

Across salts/banks, overlap is weakly higher with lower entropy (Spearman−.125 to−.079) and higher dominant-class share (+.111 to+.136). Half histogram mismatch has little relation to overlap, but weak positive association with two-factor regret (L1+.026 to+.200; JS+.049 to+.225). These modest descriptive associations do not establish that composition explains the persistent signal. They motivate a dedicated composition audit if Research Lead considers it necessary.

## Mechanism vs implementation conclusion

Exact artifact regressions, label-blind split checks, training-half-only factor construction, freeze hashes and count reconstructions all pass. The result is a mechanism finding. Persistent client preference survives disjoint examples and a transient context residual helps on Dark/Contrast. The additive model leaves important residual error, especially on Contrast/Blur; do not call this an operator-capacity failure. The current affine bank and ray evidence already show useful capacity.

## Recommended next action

Return FACTOR-B evidence to Research Lead. A next bounded audit should distinguish class-composition-driven persistence from non-additive client×context effects before implementing a writer. No explicit new state architecture, T014, SSL, gradient adaptation, federation, operator expansion or post-hoc correction was started.
