# CODEX -> CHATGPT

## Timestamp / commit / run

2026-09-14T03:29:52+08:00. Lead1462732; runtime 78de03cdcc22daf1e4924a93b491331a2d7b3d13; local saved-prediction run 20260914-t014-local; 51.41s. One formal analysis, no model forward pass, GPU job or training. Enclosing delivery commit contains the independent report.

## Preflight and frozen evidence

T011 prediction SHA matches its original freeze. T013 split, choices, class-count artifact and count-receipt hashes match. Every half per-class correct/total array reconstructs from T011 predictions. All600 historical T013 policy metrics and clean safety reconstruct exactly before new analysis, including zero/client-only/context-only/two-factor/half-oracle. Four salts, original half memberships, five candidates, two banks and eight offsets are unchanged.

Three focused tests PASS: evaluation-label-free interfaces and choice invariance; target exclusion and legitimate changes to other templates; exact clean residual0/hybrid equivalence; uniform/mismatched composition controls on fixed templates; nonzero class denominators. No model code was imported or changed, so historical model tests were not rerun. Saved-prediction per-example concatenation independently verifies all1,240 new policy metrics; exact Fraction report error0.

## Exact class templates and composition

For each salt/bank/training orientation/target client, pool correct/total examples within each class across the other99 clients. D is clean utility versus zero; R is shifted utility minus D. All class denominators are positive, with no smoothing, cutoff, shrinkage or learned weight. The target composition is its opposite training-half ten-class histogram. Target evaluation-half labels never enter template construction, composition, actual clean persistent utility or policy choice.

class_templates.json.gz stores1,600 templates as exact numerator/denominator strings for utility[c][class][state], plus other99 class denominators; D=utility[clean] and R=utility[c]−D. composition_vectors.json stores800 exact target histograms/denominators. All template, mixture, hybrid and argmax arithmetic uses Fraction; only reported accuracies/correlations are converted to float.

## Policies / freeze

31 fixed policies: composition_persistent, class_context_only, hybrid_client_plus_class_context, two T013 regression policies, two uniform controls, and eight offsets for each of persistent/class-context/hybrid mismatched-pi controls. The extra persistent mismatch controls are required by COMP-A. All controls retain identical target-excluded templates and actual persistent term where applicable; only pi changes.

All248,000 predicted utility vectors, full argmax sets and choices frozen at 2026-09-13T19:27:04.842806+00:00, before T014 evaluation or residual analysis. Utility SHA f256810d99fde7d5d436b973edb1cc5adb7cdecd02b15f00757435b5ee72baa2; template SHA cd272ab8de6c9bb582972b0d700fe412af9b6bdc6287c4bc955723d311fd7289. Frozen candidate order resolves exact ties only after the full argmax set is retained. Clean hybrid utility, choice and metric equal T013 client_only exactly because R(clean)=0.

**This is a supervised, context-privileged diagnostic, not a deployable unlabeled method.** It uses target opposite-half labels and other-client training-half labels. Both cross-fit orientations are concatenated; no evaluation data are reweighted. Historical query-selected half-oracle remains an optimistic diagnostic denominator, not a policy-learning target.

## Policy accuracy / regret

Macro-class accuracy averaged over all four salts for readability; every salt/bank/context/policy remains in policy_metrics.csv with weighted accuracy, macro-client accuracy, selection counts, half-oracle hit rate and regret median/p75/p90/>2/>5pp:

| bank | target | T013_zero | T013_client_only | T013_two_factor | composition_persistent | class_context_only | hybrid_client_plus_class_context | uniform_hybrid | T013_half_oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | 34.138690 | 39.857358 | 39.857358 | 40.411844 | 40.411844 | 39.857358 | 39.857358 | 40.995225 |
| A | brightness_dark | 17.546026 | 25.613593 | 33.043927 | 26.151948 | 35.466289 | 34.931746 | 33.362024 | 35.879543 |
| A | contrast_low | 19.955264 | 28.366562 | 29.885422 | 28.714773 | 33.739205 | 33.208331 | 29.915458 | 34.175340 |
| A | gaussian_noise | 30.676644 | 36.912367 | 37.130695 | 37.374381 | 37.615401 | 37.048923 | 37.102367 | 38.082635 |
| A | gaussian_blur | 30.901579 | 38.464626 | 38.556368 | 39.231565 | 39.995770 | 39.384411 | 38.695250 | 40.665880 |
| B | clean | 34.138690 | 39.650134 | 39.650134 | 40.275622 | 40.275622 | 39.650134 | 39.650134 | 40.878239 |
| B | brightness_dark | 17.546026 | 25.490595 | 33.047307 | 26.116844 | 35.443927 | 34.986188 | 33.088699 | 35.859442 |
| B | contrast_low | 19.955264 | 28.198676 | 29.803546 | 28.726965 | 33.604517 | 33.042362 | 29.906017 | 34.037413 |
| B | gaussian_noise | 30.676644 | 36.779287 | 36.999457 | 37.423156 | 37.658753 | 37.017241 | 37.016188 | 38.159925 |
| B | gaussian_blur | 30.901579 | 38.478654 | 38.366666 | 39.332223 | 39.981846 | 39.379369 | 38.411736 | 40.711123 |

State-choice counts sum to200 client-half decisions per row. Half regret preserves integer correct-count ties. The full exact count receipts and frozen utility/template receipts support reconstruction; no evaluation-half outcome changes a formula or coefficient.

## Persistent clean composition capture / COMP decision

| salt | bank | composition_gain | client_gain | composition_capture | composition_minus_mismatch_mean | passed |
| --- | --- | --- | --- | --- | --- | --- |
| T013-S0 | A | 6.265601 | 5.772385 | 1.085444 | 7.666268 | True |
| T013-S0 | B | 6.147287 | 5.614276 | 1.094939 | 7.545257 | True |
| T013-S1 | A | 6.265067 | 5.683543 | 1.102317 | 7.695242 | True |
| T013-S1 | B | 6.094397 | 5.341656 | 1.140919 | 7.588601 | True |
| T013-S2 | A | 6.270973 | 5.837041 | 1.074341 | 7.642987 | True |
| T013-S2 | B | 6.146349 | 5.565887 | 1.104289 | 7.557341 | True |
| T013-S3 | A | 6.290977 | 5.581702 | 1.127071 | 7.740419 | True |
| T013-S3 | B | 6.159693 | 5.523958 | 1.115087 | 7.582507 | True |

**COMP-A PASS in both banks and every salt.** Composition-only clean gain is about6.09–6.29pp, capturing107.4–114.1% of T013 client-only gain, and beats mean mismatched persistent composition by7.55–7.74pp. Capture above1 is valid: pooled class templates can outperform noisy client-half persistent measurements. This is not a causal percentage of variance explained.

The interpretation of client-lock should therefore shift toward stable semantic/class composition in these synthetic non-IID CIFAR-10 clients. A separate latent client-identity variable is not required to explain most of this observed clean preference. This does not prove no other persistent client/content factor exists outside this experiment.

## Class-conditioned interaction / per-salt capture

| salt | bank | target | class_capture | hybrid_minus_two_factor | hybrid_minus_client_only | class_context_minus_context_only | hybrid_minus_uniform | hybrid_minus_mismatch_mean | mismatch_mean | mismatch_min | mismatch_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T013-S0 | A | brightness_dark | 0.9485582645756337 | 1.782238214907339 | 9.544012938484874 | 10.740717993851394 | 1.5329229587911486 | 5.320877027257085 | 29.582671083108387 | 28.15542829332609 | 31.23111311546954 |
| T013-S0 | A | contrast_low | 0.9360728541250418 | 3.333812389016126 | 4.876810345632639 | 5.117163798102374 | 3.2604516064588727 | 4.406110018130215 | 28.8325506636845 | 27.904947016687334 | 29.463950440751802 |
| T013-S0 | A | gaussian_noise | 0.8555820839698569 | 0.06703135372705374 | 0.15205895413619608 | 6.065873046885654 | 0.10733234423415049 | 0.13916344717179002 | 36.887277550826006 | 36.69951820643318 | 37.09312731690229 |
| T013-S0 | A | gaussian_blur | 0.8677903605766305 | 0.5703799310108709 | 0.8291956334689949 | 10.239023998667603 | 0.5159014534874675 | 0.9041142631700136 | 38.44599493121939 | 37.85997362940161 | 38.77234519295535 |
| T013-S0 | B | brightness_dark | 0.9531945770007147 | 1.6982170996567862 | 9.481595132115253 | 15.336701494079316 | 1.6061721416286672 | 5.270313784169402 | 29.68239259965901 | 28.082699970887113 | 31.539928801881167 |
| T013-S0 | B | contrast_low | 0.9218729239501363 | 3.118535115080341 | 4.71291014893317 | 5.50378249936678 | 3.130869394832333 | 4.276610524583934 | 28.683068729736817 | 27.863632330730713 | 29.35363317834526 |
| T013-S0 | B | gaussian_noise | 0.8359437322510507 | 0.03272687272366921 | 0.14570397213007424 | 6.135635551590624 | -0.03979760300974699 | 0.06744629924288408 | 36.87405610333069 | 36.66942554350921 | 37.07936292635166 |
| T013-S0 | B | gaussian_blur | 0.8602790734523034 | 0.7458389188439073 | 0.7977949314441224 | 11.578604704028027 | 0.7582997874121381 | 0.9039992145213613 | 38.40924645553734 | 37.90312640942133 | 38.786470647854216 |
| T013-S1 | A | brightness_dark | 0.9502456914651606 | 2.4811065331192514 | 9.361708685286672 | 8.684138318011916 | 1.4468374674176028 | 5.152629040245783 | 29.81926377968113 | 28.230551312712706 | 31.28627843154399 |
| T013-S1 | A | contrast_low | 0.9275805707889918 | 3.256065302160167 | 5.0778125691450136 | 5.212830199232589 | 3.2426562021419083 | 4.361962890731328 | 28.809202004430716 | 28.15240498304915 | 29.832427762726383 |
| T013-S1 | A | gaussian_noise | 0.8549817068668548 | -0.07848722353507022 | 0.21977276744925425 | 5.598162438188712 | 0.0013807919330449932 | 0.14528435774044676 | 36.92048532135271 | 36.61864920383924 | 37.11128602488518 |
| T013-S1 | A | gaussian_blur | 0.8537873042993526 | 0.7461537247463857 | 0.9666447681507139 | 10.078763290242565 | 0.5778200692144201 | 1.0708215505643168 | 38.226995181361744 | 37.776119483825745 | 38.49363958788246 |
| T013-S1 | B | brightness_dark | 0.9559492695973222 | 2.60767441704921 | 9.356970964037036 | 15.317290511119351 | 2.108989365946788 | 5.27750670410384 | 29.768357048001043 | 28.382025084334828 | 31.144724892478212 |
| T013-S1 | B | contrast_low | 0.931114628347284 | 3.188835253652781 | 4.860516446538949 | 5.405354647533886 | 2.9331781457991855 | 4.278886939142443 | 28.822673419223534 | 28.16531768107869 | 29.279272641298864 |
| T013-S1 | B | gaussian_noise | 0.8192311905417563 | 0.03523505240300349 | 0.35077938167749456 | 6.434347290106101 | -0.01921608612775725 | 0.003792938157387122 | 36.85088965805672 | 36.69379377184656 | 37.02059581392686 |
| T013-S1 | B | gaussian_blur | 0.86986388018527 | 1.1364787363939002 | 1.0162936585804307 | 11.537519390155603 | 1.0778214475430368 | 1.270645647437667 | 38.18122016725002 | 37.88986348920893 | 38.45683833844386 |
| T013-S2 | A | brightness_dark | 0.9459835597412203 | 1.6696902925910517 | 8.985575221452509 | 12.99446091706281 | 1.7452093209151025 | 5.2944132591273645 | 29.623242476068107 | 28.226279988831042 | 31.196801502373354 |
| T013-S2 | A | contrast_low | 0.9247098782586772 | 3.318891156337258 | 4.408511846117069 | 5.210978475411768 | 3.358547182552269 | 4.277449875362597 | 28.858810122766997 | 27.85507189050942 | 29.956667492055278 |
| T013-S2 | A | gaussian_noise | 0.8635369045554995 | -0.15788235252297395 | 0.007891527962981015 | 5.098641286129929 | -0.19827677861814857 | -0.13952753396708054 | 37.1862236681197 | 36.986096497045764 | 37.273079248379034 |
| T013-S2 | A | gaussian_blur | 0.8678110878637695 | 0.9506085208793726 | 0.7274615263683534 | 10.23966462011306 | 0.7312758057664909 | 1.1561714343722314 | 38.24963272945294 | 37.32081307666284 | 38.76109650025831 |
| T013-S2 | B | brightness_dark | 0.946169835786214 | 1.5721050905625344 | 9.214267797245446 | 15.276730859730531 | 1.6728610389539205 | 5.263324199516442 | 29.63677360145849 | 28.185567959933987 | 31.431666265812666 |
| T013-S2 | B | contrast_low | 0.9326999733390292 | 3.3551174235681667 | 4.624271690917434 | 5.530724872275208 | 3.2152869129952792 | 4.454500758653871 | 28.630463266690487 | 27.664176779490624 | 29.198639832483387 |
| T013-S2 | B | gaussian_noise | 0.8571572082821179 | -0.031169618939958267 | 0.21516021185237774 | 6.153700623440851 | -0.03773388776324405 | -0.011304034086444733 | 37.066792649852125 | 36.93415251763304 | 37.13404485504662 |
| T013-S2 | B | gaussian_blur | 0.8666049321996802 | 1.1413106482619508 | 0.8234952883348235 | 11.703449588239431 | 1.1013863711858345 | 1.3763156749155443 | 38.07481947237918 | 37.46374961875075 | 38.50604526744037 |
| T013-S3 | A | brightness_dark | 0.9484262404227378 | 1.6182421902561859 | 9.381313680407814 | 12.51349120380818 | 1.553918666584124 | 5.140629345127042 | 29.79325758637972 | 28.0615557565795 | 31.268243736817944 |
| T013-S3 | A | contrast_low | 0.9396763668691055 | 3.3828679231193712 | 5.003942873472533 | 5.169953786569543 | 3.3098377399147534 | 4.42543445596512 | 28.861805922565242 | 28.096672436949486 | 29.544330806344718 |
| T013-S3 | A | gaussian_noise | 0.8677121885516296 | -0.15774887786762387 | 0.16650412652794758 | 6.254592494669255 | -0.1242114046720114 | 0.013656475424823752 | 37.04313010637238 | 36.900433681348154 | 37.147012910371565 |
| T013-S3 | A | gaussian_blur | 0.885892402234775 | 1.045031674789607 | 1.1558393708362216 | 10.23307861788515 | 0.9316475334184176 | 1.3666671035526994 | 38.117248051118665 | 37.38472290073963 | 38.653457978222995 |
| T013-S3 | B | brightness_dark | 0.9539614123590333 | 1.8775269535472687 | 9.929537428207956 | 15.223252311730464 | 2.2019318924299243 | 5.063722137802833 | 29.982360249635416 | 28.376459026257738 | 31.978305505806713 |
| T013-S3 | B | contrast_low | 0.931689446837492 | 3.2927759403013517 | 5.177043689024404 | 5.531705705986381 | 3.2660426940742098 | 4.307560707887044 | 28.715681775952984 | 27.918909721690657 | 29.159380978916943 |
| T013-S3 | B | gaussian_noise | 0.8772741916571176 | 0.03434357986175101 | 0.2401718361854872 | 6.182754682279295 | 0.1009563216749121 | 0.1984991787547926 | 37.01878929059064 | 36.902141270518314 | 37.129826126522985 |
| T013-S3 | B | gaussian_blur | 0.8601518974379933 | 1.0271855934828509 | 0.9652790774379825 | 11.690871950498499 | 0.9330243474090977 | 1.2410791691449783 | 38.060151749195015 | 37.48796460948645 | 38.453454889404526 |

| bank | target | mean_hybrid_minus_two_factor | mean_hybrid_minus_mismatch | gain_pass | mismatch_pass |
| --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 1.887819 | 5.227137 | True | True |
| A | contrast_low | 3.322909 | 4.367739 | True | True |
| A | gaussian_noise | -0.081772 | 0.039644 | False | False |
| A | gaussian_blur | 0.828043 | 1.124444 | True | True |
| B | brightness_dark | 1.938881 | 5.218717 | True | True |
| B | contrast_low | 3.238816 | 4.329390 | True | True |
| B | gaussian_noise | 0.017784 | 0.064609 | False | False |
| B | gaussian_blur | 1.012703 | 1.198010 | True | True |

**CLASS-INT-A PASS:4/4 shifted contexts meet80% capture in both banks under every salt.** Mean hybrid gain over T013 two_factor is+3.323/+3.239pp for Contrast and+.828/+1.013pp for Blur, so the additional failed-context improvement condition passes. Dark gains+1.888/+1.939pp. Noise changes−.082/+.018pp and gains only+.040/+.065pp over mismatched composition; preserve this weak/negative incremental result rather than claiming uniform improvement.

Class-conditioned transient response explains enough of the missing Contrast/Blur interaction to pass the fixed gate, but it does not eliminate all half-oracle regret. No coefficient tuning or interpolation between global and class-conditioned residuals was performed.

## Residual disjoint client-lock

| salt | bank | raw_margin_pp | residual_margin_pp | attenuation_pp | fractional_attenuation | residual_above_original_15pp |
| --- | --- | --- | --- | --- | --- | --- |
| T013-S0 | A | 24.012500 | -3.556250 | 27.568750 | 1.148100 | False |
| T013-S0 | B | 23.362500 | -3.493750 | 26.856250 | 1.149545 | False |
| T013-S1 | A | 22.818750 | -5.006250 | 27.825000 | 1.219392 | False |
| T013-S1 | B | 22.512500 | -4.175000 | 26.687500 | 1.185453 | False |
| T013-S2 | A | 23.712500 | -4.093750 | 27.806250 | 1.172641 | False |
| T013-S2 | B | 22.668750 | -3.718750 | 26.387500 | 1.164047 | False |
| T013-S3 | A | 23.043750 | -4.375000 | 27.418750 | 1.189856 | False |
| T013-S3 | B | 22.212500 | -5.018750 | 27.231250 | 1.225943 | False |

Residual E uses held-out observed state utility minus the opposite-half/other99 class-predicted utility. Subtraction and residual argmax ties use exact fractions. Raw margins exactly reproduce T013. Residual margins range−5.019 to−3.494pp, with none retaining the original15pp margin. Thus the strong positive lock vanishes under this prescribed class-conditioned subtraction.

Fractional attenuation exceeds1 because the residual overlap contrast changes sign; it must not be read as “more than100% of causal signal explained.” Residualization changes ties and noise structure and uses cross-fitted estimated templates, so a small negative margin is not proof of a biological/statistical anti-client effect. The narrow conclusion is that no large positive residual lock remains by this test.

residual_lock.csv retains all pairs/orientations; residual_utility_similarity.csv provides valid counts, same-client Spearman mean/median and shifted-control ranges. residual_lock_summary.json retains every offset statistic. Exact residual vectors are in residual_utilities.json.gz. No policy was fitted from these residuals.

## Mechanism vs implementation conclusion

All artifact regressions, exclusion/half boundaries, clean invariants, freeze hashes and independent count checks pass. This is a mechanism finding: stable semantic composition explains most persistent preference, and class-dependent context response resolves much of the previous additive model’s missing interaction within the frozen five-state bank. The evidence favors semantic composition plus transient context over automatically calling the former client identity.

The result remains specific to this frozen checkpoint, state bank, corruption set and synthetic PFLlib split (not the official CIFAR-10 test benchmark). Correct class composition and utility templates are supervised here. Neither a label-free composition estimator nor a writer has been tested; residual feature/content effects beyond these diagnostics are not ruled out.

## Recommended next action

Return COMP-A + CLASS-INT-A and the attenuated residual lock to Research Lead. A next bounded design/audit may ask how semantic mixture can be estimated without labels and how it conditions transient state utility, but no such estimator, writer, SSL/TTT update, new federation, operator expansion or T015 was implemented.
