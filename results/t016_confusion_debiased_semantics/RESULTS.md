# CODEX → CHATGPT: T016R / completed T016

## Execution and implementation repair

2026-09-14T06:46:17+08:00. Lead T016R2bad2a0 resumes original T0166476c99. Runtime `e94adc67af83b553a3673aa4fb1971ec68caf4dc`; release20260914-063742-ttfl-t016r; run `20260914-063758-ttfl-t016r-cached`, exit0, 28.26s. All71 tests PASS. Zero new support/query forwards; frozen A6000 logits reused. NumPy2.5.3, default pinv; no model/state/temperature/support/context/solver/gate changes.

Stopped run20260914-054411-ttfl-t016-cached at2e51d81 remains preserved as implementation-blocker provenance. It never built a T016 channel or evaluated a policy. The authorized repair permits only finite episodeJS absolute difference<=1e-12,rtol0. All other episodefields, schema/order/count, aggregates and gates remain exact. Eight comparator tests plus63 prior tests pass. The repaired receipt records4000rows,2nonzeroJSdeltas,max4.440892098500626e-16; these observed counts are not hard-coded acceptance rules.

Historical T015 hashes,56,000utilities/argmaxes,280count-derived macroclass rows,220quality aggregate rows and96gate rows reproduce. Original calibration-pool/support/query IDs remain disjoint. No estimator/outcome-driven retry occurred.

## Calibration, exclusion and freeze semantics

Hard channels use other99clients’ T011 query predictions under the exact modeled context and matching frozen state; soft primary channels use other99clients’ exact K20 support logits under the modeled context/state. Columns condition on true class; hard integer numerators/denominators and soft float64 sums/counts are persisted. The classifier’s clean state is the unchanged zero affine. No new inference or learned calibration parameters.

Target exclusion is per episode: the builder skips i before calling the calibration loader. Since all100clients rotate as targets, the process necessarily reads all clients’ labels in their roles as other-client calibration data. Historical replay is also explicitly privileged. Thus “all target labels globally unopened” would be inaccurate; the enforced invariant is that target i labels/outcomes never enter its own calibration, prevalence estimate or source choice. The receipt states this scope explicitly. This is supervised offline, transductive cross-client calibration, not fully unsupervised training or evidence for an independently trained deployable writer.

Phase A freeze2026-09-13T22:38:30.516713+00:00:2000target-excluded matrices,4000corrected source mixtures and32,000complete utility/argmax/state choices. ChoiceSHA`812f7972c6de3453cd4de5252bb1b6d76bce284989cf72441ed82fe9bbb55bf3`; mixtureSHA`9df0f672f04fe01d3d3e53a3cc2cdfaecdc4824929cbdba13286abead103e57b`; calibrationNPZSHA`c3b399a691ce25fe53b10cc6a1ee5b99ddbe47d144d9ff68a05fa6aba16e71c5`. Only after freeze are own-support prevalence quality and selected-policy target metrics computed. Source11 indexes both channel and utility template by frozen T009 predicted context; oracle01 uses privileged true context.

Independent local replay reconstructs all hard integer matrices exactly and soft numerators from saved ordered labels/logits with max error1.42e-14; pinv+projection max prevalence difference1.19e-13. All32,000utility vectors/argmaxes match exact Fraction arithmetic. All160newmetrics reconstruct from integer counts; the formal evaluation additionally checks per-example T011 predictions. All1600T014target-excluded templates reconstruct from unchanged half counts. Numeric BLAS replay tolerances do not enter CAL gates.

## Observation-channel identifiability

| bank | context | family | count_min | count_median | rank_min | rank_median | rank_max | condition_min | condition_median | condition_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| A | clean | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 72.676272 | 74.865611 | 80.635778 |
| A | brightness_dark | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| A | brightness_dark | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 186.945864 | 196.499136 | 205.966229 |
| A | contrast_low | H | 1041 | 1500.000000 | 8 | 8.000000 | 8 | inf | inf | inf |
| A | contrast_low | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 163.511187 | 167.833481 | 181.228955 |
| A | gaussian_noise | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| A | gaussian_noise | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 79.191880 | 83.413598 | 89.852231 |
| A | gaussian_blur | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| A | gaussian_blur | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 76.777423 | 77.630679 | 85.512209 |
| B | clean | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| B | clean | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 72.676272 | 74.865611 | 80.635778 |
| B | brightness_dark | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| B | brightness_dark | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 190.124797 | 199.823204 | 209.690299 |
| B | contrast_low | H | 1041 | 1500.000000 | 8 | 8.000000 | 8 | inf | inf | inf |
| B | contrast_low | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 176.889596 | 184.837463 | 194.578958 |
| B | gaussian_noise | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| B | gaussian_noise | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 81.103924 | 86.472893 | 93.357807 |
| B | gaussian_blur | H | 1041 | 1500.000000 | 9 | 9.000000 | 9 | inf | inf | inf |
| B | gaussian_blur | S | 143 | 188.000000 | 10 | 10.000000 | 10 | 77.928149 | 79.194636 | 87.632053 |

Full per-matrix singular values, ranks, counts and hashes are in calibration_matrix_stats.csv. H/S differ in calibration sample source/size as specified; their comparison is not a controlled test of hard-vs-soft observations alone. Rank is NumPy numerical matrix_rank; condition numbers are reported without an invented pass threshold.

| bank | target | family | mode | negative_entry_fraction | episode_negative_fraction | projection_L1_mean | projection_L1_median | projection_L1_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | H | 01 | 0.440000 | 1.000000 | 1.365344 | 1.139026 | 6.068671 |
| A | clean | H | 11 | 0.438000 | 1.000000 | 1.350894 | 1.117571 | 6.068671 |
| A | clean | S | 01 | 0.418000 | 1.000000 | 1.269500 | 1.074012 | 3.523726 |
| A | clean | S | 11 | 0.408000 | 1.000000 | 1.264496 | 1.074012 | 3.523726 |
| A | brightness_dark | H | 01 | 0.466000 | 1.000000 | 1.578405 | 1.156097 | 5.281716 |
| A | brightness_dark | H | 11 | 0.466000 | 1.000000 | 1.578405 | 1.156097 | 5.281716 |
| A | brightness_dark | S | 01 | 0.425000 | 1.000000 | 1.626617 | 1.504401 | 5.665000 |
| A | brightness_dark | S | 11 | 0.425000 | 1.000000 | 1.626617 | 1.504401 | 5.665000 |
| A | contrast_low | H | 01 | 0.446000 | 1.000000 | 1.585013 | 1.155196 | 7.128246 |
| A | contrast_low | H | 11 | 0.446000 | 1.000000 | 1.585013 | 1.155196 | 7.128246 |
| A | contrast_low | S | 01 | 0.444000 | 1.000000 | 1.692949 | 1.644134 | 4.287153 |
| A | contrast_low | S | 11 | 0.444000 | 1.000000 | 1.692949 | 1.644134 | 4.287153 |
| A | gaussian_noise | H | 01 | 0.478000 | 1.000000 | 1.367881 | 1.167268 | 4.482421 |
| A | gaussian_noise | H | 11 | 0.478000 | 1.000000 | 1.367881 | 1.167268 | 4.482421 |
| A | gaussian_noise | S | 01 | 0.424000 | 1.000000 | 1.349315 | 1.241178 | 3.735271 |
| A | gaussian_noise | S | 11 | 0.424000 | 1.000000 | 1.349315 | 1.241178 | 3.735271 |
| A | gaussian_blur | H | 01 | 0.469000 | 1.000000 | 1.108898 | 0.904031 | 3.506912 |
| A | gaussian_blur | H | 11 | 0.465000 | 1.000000 | 1.165143 | 0.947569 | 4.258298 |
| A | gaussian_blur | S | 01 | 0.433000 | 1.000000 | 1.313522 | 1.110326 | 3.701031 |
| A | gaussian_blur | S | 11 | 0.428000 | 1.000000 | 1.349189 | 1.110326 | 4.070112 |
| B | clean | H | 01 | 0.440000 | 1.000000 | 1.365344 | 1.139026 | 6.068671 |
| B | clean | H | 11 | 0.439000 | 1.000000 | 1.355726 | 1.117571 | 6.068671 |
| B | clean | S | 01 | 0.418000 | 1.000000 | 1.269500 | 1.074012 | 3.523726 |
| B | clean | S | 11 | 0.409000 | 1.000000 | 1.283999 | 1.082164 | 3.523726 |
| B | brightness_dark | H | 01 | 0.476000 | 1.000000 | 1.651192 | 1.234999 | 6.095593 |
| B | brightness_dark | H | 11 | 0.476000 | 1.000000 | 1.651192 | 1.234999 | 6.095593 |
| B | brightness_dark | S | 01 | 0.427000 | 1.000000 | 1.647145 | 1.519028 | 5.763591 |
| B | brightness_dark | S | 11 | 0.427000 | 1.000000 | 1.647145 | 1.519028 | 5.763591 |
| B | contrast_low | H | 01 | 0.450000 | 1.000000 | 1.688067 | 1.210873 | 8.273910 |
| B | contrast_low | H | 11 | 0.450000 | 1.000000 | 1.688067 | 1.210873 | 8.273910 |
| B | contrast_low | S | 01 | 0.440000 | 1.000000 | 1.765655 | 1.731268 | 4.518201 |
| B | contrast_low | S | 11 | 0.440000 | 1.000000 | 1.765655 | 1.731268 | 4.518201 |
| B | gaussian_noise | H | 01 | 0.471000 | 1.000000 | 1.309604 | 1.095854 | 4.071234 |
| B | gaussian_noise | H | 11 | 0.471000 | 1.000000 | 1.309604 | 1.095854 | 4.071234 |
| B | gaussian_noise | S | 01 | 0.423000 | 1.000000 | 1.356016 | 1.244482 | 3.792592 |
| B | gaussian_noise | S | 11 | 0.423000 | 1.000000 | 1.356016 | 1.244482 | 3.792592 |
| B | gaussian_blur | H | 01 | 0.467000 | 1.000000 | 1.116609 | 0.896605 | 3.430700 |
| B | gaussian_blur | H | 11 | 0.466000 | 1.000000 | 1.152104 | 0.902666 | 3.547908 |
| B | gaussian_blur | S | 01 | 0.434000 | 1.000000 | 1.313886 | 1.099959 | 3.722433 |
| B | gaussian_blur | S | 11 | 0.427000 | 1.000000 | 1.324112 | 1.099959 | 3.722433 |

## Semantic prevalence quality

| bank | target | estimator | L1_mean | L1_median | JS_mean | JS_median | top_class_agreement | spearman_mean | spearman_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | BBSE-H-01 | 1.086164 | 1.000000 | 0.304398 | 0.249645 | 0.460000 | 0.356873 | 100 |
| A | clean | BBSE-H-11 | 1.089202 | 1.019000 | 0.306332 | 0.253090 | 0.450000 | 0.350253 | 100 |
| A | clean | BBSE-S-01 | 0.732250 | 0.707791 | 0.172738 | 0.151145 | 0.680000 | 0.551492 | 100 |
| A | clean | BBSE-S-11 | 0.768371 | 0.753779 | 0.181897 | 0.164936 | 0.670000 | 0.530953 | 100 |
| A | clean | raw_soft_01 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 100 |
| A | clean | raw_hard_01 | 1.264000 | 1.300000 | 0.360963 | 0.343529 | 0.290000 | 0.206758 | 100 |
| A | clean | raw_soft_11 | 1.295678 | 1.331933 | 0.335741 | 0.332573 | 0.270000 | 0.249138 | 100 |
| A | clean | raw_hard_11 | 1.271000 | 1.300000 | 0.363035 | 0.353700 | 0.290000 | 0.198973 | 100 |
| A | brightness_dark | BBSE-H-01 | 1.119819 | 1.126235 | 0.313932 | 0.276473 | 0.440000 | 0.317037 | 100 |
| A | brightness_dark | BBSE-H-11 | 1.119819 | 1.126235 | 0.313932 | 0.276473 | 0.440000 | 0.317037 | 100 |
| A | brightness_dark | BBSE-S-01 | 0.746077 | 0.739688 | 0.182846 | 0.170117 | 0.670000 | 0.526162 | 100 |
| A | brightness_dark | BBSE-S-11 | 0.746077 | 0.739688 | 0.182846 | 0.170117 | 0.670000 | 0.526162 | 100 |
| A | brightness_dark | raw_soft_01 | 1.310229 | 1.303101 | 0.341273 | 0.327241 | 0.290000 | 0.237988 | 100 |
| A | brightness_dark | raw_hard_01 | 1.298000 | 1.400000 | 0.383189 | 0.399183 | 0.330000 | 0.162394 | 100 |
| A | brightness_dark | raw_soft_11 | 1.310229 | 1.303101 | 0.341273 | 0.327241 | 0.290000 | 0.237988 | 100 |
| A | brightness_dark | raw_hard_11 | 1.298000 | 1.400000 | 0.383189 | 0.399183 | 0.330000 | 0.162394 | 100 |
| A | contrast_low | BBSE-H-01 | 1.180589 | 1.246503 | 0.329355 | 0.328198 | 0.370000 | 0.332962 | 100 |
| A | contrast_low | BBSE-H-11 | 1.180589 | 1.246503 | 0.329355 | 0.328198 | 0.370000 | 0.332962 | 100 |
| A | contrast_low | BBSE-S-01 | 0.839997 | 0.812892 | 0.211763 | 0.184963 | 0.690000 | 0.476676 | 100 |
| A | contrast_low | BBSE-S-11 | 0.839997 | 0.812892 | 0.211763 | 0.184963 | 0.690000 | 0.476676 | 100 |
| A | contrast_low | raw_soft_01 | 1.345680 | 1.349090 | 0.356656 | 0.349470 | 0.270000 | 0.186676 | 100 |
| A | contrast_low | raw_hard_01 | 1.345000 | 1.500000 | 0.403251 | 0.420310 | 0.300000 | 0.143626 | 100 |
| A | contrast_low | raw_soft_11 | 1.345680 | 1.349090 | 0.356656 | 0.349470 | 0.270000 | 0.186676 | 100 |
| A | contrast_low | raw_hard_11 | 1.345000 | 1.500000 | 0.403251 | 0.420310 | 0.300000 | 0.143626 | 100 |
| A | gaussian_noise | BBSE-H-01 | 1.023352 | 0.978350 | 0.283746 | 0.251578 | 0.540000 | 0.386583 | 100 |
| A | gaussian_noise | BBSE-H-11 | 1.023352 | 0.978350 | 0.283746 | 0.251578 | 0.540000 | 0.386583 | 100 |
| A | gaussian_noise | BBSE-S-01 | 0.762790 | 0.697161 | 0.182096 | 0.150331 | 0.660000 | 0.528703 | 100 |
| A | gaussian_noise | BBSE-S-11 | 0.762790 | 0.697161 | 0.182096 | 0.150331 | 0.660000 | 0.528703 | 100 |
| A | gaussian_noise | raw_soft_01 | 1.302194 | 1.331811 | 0.337775 | 0.331711 | 0.270000 | 0.243293 | 100 |
| A | gaussian_noise | raw_hard_01 | 1.270000 | 1.300000 | 0.364713 | 0.371204 | 0.320000 | 0.202145 | 100 |
| A | gaussian_noise | raw_soft_11 | 1.302194 | 1.331811 | 0.337775 | 0.331711 | 0.270000 | 0.243293 | 100 |
| A | gaussian_noise | raw_hard_11 | 1.270000 | 1.300000 | 0.364713 | 0.371204 | 0.320000 | 0.202145 | 100 |
| A | gaussian_blur | BBSE-H-01 | 1.026421 | 0.991219 | 0.279276 | 0.219241 | 0.490000 | 0.385838 | 100 |
| A | gaussian_blur | BBSE-H-11 | 1.032995 | 0.999768 | 0.280918 | 0.223221 | 0.480000 | 0.383334 | 100 |
| A | gaussian_blur | BBSE-S-01 | 0.743443 | 0.702465 | 0.175157 | 0.148328 | 0.720000 | 0.540995 | 100 |
| A | gaussian_blur | BBSE-S-11 | 0.788269 | 0.715490 | 0.191198 | 0.152716 | 0.700000 | 0.516579 | 100 |
| A | gaussian_blur | raw_soft_01 | 1.302992 | 1.328046 | 0.338005 | 0.337836 | 0.270000 | 0.245529 | 100 |
| A | gaussian_blur | raw_hard_01 | 1.274000 | 1.300000 | 0.366391 | 0.369912 | 0.300000 | 0.198602 | 100 |
| A | gaussian_blur | raw_soft_11 | 1.311582 | 1.328046 | 0.340862 | 0.337836 | 0.270000 | 0.242048 | 100 |
| A | gaussian_blur | raw_hard_11 | 1.281000 | 1.300000 | 0.367808 | 0.369912 | 0.300000 | 0.192783 | 100 |
| B | clean | BBSE-H-01 | 1.086164 | 1.000000 | 0.304398 | 0.249645 | 0.460000 | 0.356873 | 100 |
| B | clean | BBSE-H-11 | 1.087702 | 1.019000 | 0.305947 | 0.253090 | 0.450000 | 0.349054 | 100 |
| B | clean | BBSE-S-01 | 0.732250 | 0.707791 | 0.172738 | 0.151145 | 0.680000 | 0.551492 | 100 |
| B | clean | BBSE-S-11 | 0.763324 | 0.753779 | 0.180979 | 0.162198 | 0.670000 | 0.532827 | 100 |
| B | clean | raw_soft_01 | 1.286992 | 1.296740 | 0.332231 | 0.325306 | 0.270000 | 0.266209 | 100 |
| B | clean | raw_hard_01 | 1.264000 | 1.300000 | 0.360963 | 0.343529 | 0.290000 | 0.206758 | 100 |
| B | clean | raw_soft_11 | 1.295334 | 1.331933 | 0.335466 | 0.334570 | 0.270000 | 0.252023 | 100 |
| B | clean | raw_hard_11 | 1.271000 | 1.300000 | 0.363120 | 0.357133 | 0.290000 | 0.198436 | 100 |
| B | brightness_dark | BBSE-H-01 | 1.129981 | 1.095362 | 0.318571 | 0.275713 | 0.430000 | 0.322266 | 100 |
| B | brightness_dark | BBSE-H-11 | 1.129981 | 1.095362 | 0.318571 | 0.275713 | 0.430000 | 0.322266 | 100 |
| B | brightness_dark | BBSE-S-01 | 0.751893 | 0.742737 | 0.184813 | 0.181028 | 0.670000 | 0.522113 | 100 |
| B | brightness_dark | BBSE-S-11 | 0.751893 | 0.742737 | 0.184813 | 0.181028 | 0.670000 | 0.522113 | 100 |
| B | brightness_dark | raw_soft_01 | 1.308314 | 1.298764 | 0.340602 | 0.326238 | 0.290000 | 0.238952 | 100 |
| B | brightness_dark | raw_hard_01 | 1.297000 | 1.350000 | 0.381618 | 0.391894 | 0.330000 | 0.169921 | 100 |
| B | brightness_dark | raw_soft_11 | 1.308314 | 1.298764 | 0.340602 | 0.326238 | 0.290000 | 0.238952 | 100 |
| B | brightness_dark | raw_hard_11 | 1.297000 | 1.350000 | 0.381618 | 0.391894 | 0.330000 | 0.169921 | 100 |
| B | contrast_low | BBSE-H-01 | 1.228853 | 1.300000 | 0.342746 | 0.323369 | 0.320000 | 0.324860 | 100 |
| B | contrast_low | BBSE-H-11 | 1.228853 | 1.300000 | 0.342746 | 0.323369 | 0.320000 | 0.324860 | 100 |
| B | contrast_low | BBSE-S-01 | 0.853261 | 0.841652 | 0.214547 | 0.204336 | 0.690000 | 0.493707 | 100 |
| B | contrast_low | BBSE-S-11 | 0.853261 | 0.841652 | 0.214547 | 0.204336 | 0.690000 | 0.493707 | 100 |
| B | contrast_low | raw_soft_01 | 1.347697 | 1.361463 | 0.358087 | 0.352300 | 0.260000 | 0.183474 | 100 |
| B | contrast_low | raw_hard_01 | 1.362000 | 1.500000 | 0.410668 | 0.423947 | 0.270000 | 0.134479 | 100 |
| B | contrast_low | raw_soft_11 | 1.347697 | 1.361463 | 0.358087 | 0.352300 | 0.260000 | 0.183474 | 100 |
| B | contrast_low | raw_hard_11 | 1.362000 | 1.500000 | 0.410668 | 0.423947 | 0.270000 | 0.134479 | 100 |
| B | gaussian_noise | BBSE-H-01 | 1.025235 | 1.000000 | 0.284258 | 0.253843 | 0.540000 | 0.378847 | 100 |
| B | gaussian_noise | BBSE-H-11 | 1.025235 | 1.000000 | 0.284258 | 0.253843 | 0.540000 | 0.378847 | 100 |
| B | gaussian_noise | BBSE-S-01 | 0.764503 | 0.696733 | 0.182812 | 0.153533 | 0.640000 | 0.525821 | 100 |
| B | gaussian_noise | BBSE-S-11 | 0.764503 | 0.696733 | 0.182812 | 0.153533 | 0.640000 | 0.525821 | 100 |
| B | gaussian_noise | raw_soft_01 | 1.304167 | 1.335631 | 0.338605 | 0.333953 | 0.270000 | 0.240809 | 100 |
| B | gaussian_noise | raw_hard_01 | 1.276000 | 1.300000 | 0.366779 | 0.371204 | 0.310000 | 0.198763 | 100 |
| B | gaussian_noise | raw_soft_11 | 1.304167 | 1.335631 | 0.338605 | 0.333953 | 0.270000 | 0.240809 | 100 |
| B | gaussian_noise | raw_hard_11 | 1.276000 | 1.300000 | 0.366779 | 0.371204 | 0.310000 | 0.198763 | 100 |
| B | gaussian_blur | BBSE-H-01 | 1.024209 | 0.963544 | 0.277138 | 0.220117 | 0.500000 | 0.389669 | 100 |
| B | gaussian_blur | BBSE-H-11 | 1.030363 | 0.943601 | 0.278324 | 0.220117 | 0.500000 | 0.386327 | 100 |
| B | gaussian_blur | BBSE-S-01 | 0.743365 | 0.704015 | 0.174992 | 0.150415 | 0.720000 | 0.544149 | 100 |
| B | gaussian_blur | BBSE-S-11 | 0.781296 | 0.717577 | 0.187314 | 0.153238 | 0.700000 | 0.523002 | 100 |
| B | gaussian_blur | raw_soft_01 | 1.303075 | 1.328956 | 0.338155 | 0.339476 | 0.280000 | 0.249099 | 100 |
| B | gaussian_blur | raw_hard_01 | 1.275000 | 1.300000 | 0.365583 | 0.369912 | 0.300000 | 0.199948 | 100 |
| B | gaussian_blur | raw_soft_11 | 1.310458 | 1.328956 | 0.340605 | 0.339476 | 0.280000 | 0.245617 | 100 |
| B | gaussian_blur | raw_hard_11 | 1.283000 | 1.300000 | 0.367222 | 0.369912 | 0.300000 | 0.194892 | 100 |

Quality compares to the same true K20 histogram only after freeze. All entropy quartiles, medians, per-episode Spearman and raw-soft/raw-hard improvement deltas are retained in prevalence_quality*.csv and prevalence_improvement.csv. Raw_hard01/11 are histograms under matching provisional states; T015_zero_hard additionally preserves the original zero-state hard baseline. Dominant-class agreement uses deterministic first argmax; constant-vector Spearman is NA and valid counts are reported.

## Retrieval utility: exact prescribed policy chain

| bank | target | zero | class_context_only | P00 | P01 | P11 | BBSE-H-01 | BBSE-H-11 | BBSE-S-01 | BBSE-S-11 | uniform | zero_hard | soft_source_minus_oracle_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | 34.138690 | 40.411844 | 40.311665 | 36.245939 | 35.403716 | 38.687848 | 38.340642 | 38.385190 | 38.051445 | 34.320986 | 35.802859 | -0.333745 |
| A | brightness_dark | 17.546026 | 35.466289 | 35.477773 | 32.296536 | 32.296536 | 31.796440 | 31.796440 | 33.322129 | 33.322129 | 32.296536 | 22.223445 | 0.000000 |
| A | contrast_low | 19.955264 | 33.739205 | 33.519650 | 28.585856 | 28.585856 | 29.300720 | 29.300720 | 29.733069 | 29.733069 | 28.561473 | 23.155531 | 0.000000 |
| A | gaussian_noise | 30.676644 | 37.615401 | 37.528393 | 33.032858 | 33.032858 | 35.544336 | 35.544336 | 35.267016 | 35.267016 | 32.499558 | 32.793003 | 0.000000 |
| A | gaussian_blur | 30.901579 | 39.995770 | 40.019018 | 34.079180 | 32.220486 | 37.145052 | 36.298925 | 36.952266 | 34.863465 | 31.263385 | 33.854137 | -2.088801 |
| B | clean | 34.138690 | 40.275622 | 40.268839 | 36.015444 | 35.125628 | 38.500626 | 37.883949 | 38.283368 | 37.816902 | 34.202018 | 35.589144 | -0.466466 |
| B | brightness_dark | 17.546026 | 35.443927 | 35.292164 | 32.424808 | 32.424808 | 32.068405 | 32.068405 | 33.172169 | 33.172169 | 32.424808 | 22.152906 | 0.000000 |
| B | contrast_low | 19.955264 | 33.604517 | 33.311370 | 28.209162 | 28.209162 | 29.232719 | 29.232719 | 29.596863 | 29.596863 | 28.111626 | 22.943772 | 0.000000 |
| B | gaussian_noise | 30.676644 | 37.658753 | 37.580895 | 32.922221 | 32.922221 | 35.542852 | 35.542852 | 35.428570 | 35.428570 | 32.406771 | 32.589541 | 0.000000 |
| B | gaussian_blur | 30.901579 | 39.981846 | 39.879851 | 34.103375 | 32.649585 | 36.887638 | 36.452949 | 37.001387 | 35.370848 | 31.716664 | 33.873781 | -1.630539 |

Displayed accuracies are macro-class averaged across four salts, solely for readability. policy_metrics.csv contains all per-salt/bank/context policies including weighted/macro-client accuracy, state frequencies, half-oracle hit, regret quantiles and>2/>5pp fractions, gain capture vsP00/T014 and clean deltas. Decisions use exact per-salt integer ratios, not rounded/averaged captures. CIFAR pool follows the inherited PFLlib merged train/test client split, not official CIFAR10-test benchmark.

## Unchanged CAL gates

**CAL-SEM-A: FAIL. CAL-SRC-A: FAIL.** Primary remains soft_emission_pinv.

Joint80%capture in bothbanks/every salt: {"BBSE-H-01": {"brightness_dark": false, "contrast_low": false, "gaussian_noise": false, "gaussian_blur": false}, "BBSE-H-11": {"brightness_dark": false, "contrast_low": false, "gaussian_noise": false, "gaussian_blur": false}, "BBSE-S-01": {"brightness_dark": true, "contrast_low": false, "gaussian_noise": false, "gaussian_blur": false}, "BBSE-S-11": {"brightness_dark": true, "contrast_low": false, "gaussian_noise": false, "gaussian_blur": false}}

Clean>=zero−.5pp: {"BBSE-H-01": true, "BBSE-H-11": true, "BBSE-S-01": true, "BBSE-S-11": true}; primaryoracle meanL1improvement in bothbanks: {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}

| bank | target | control | mean_gain_pp | passed |
| --- | --- | --- | --- | --- |
| A | brightness_dark | P11 | 1.025593 | True |
| A | brightness_dark | uniform | 1.025593 | True |
| A | contrast_low | P11 | 1.147213 | True |
| A | contrast_low | uniform | 1.171596 | True |
| A | gaussian_noise | P11 | 2.234158 | True |
| A | gaussian_noise | uniform | 2.767458 | True |
| A | gaussian_blur | P11 | 2.642979 | True |
| A | gaussian_blur | uniform | 3.600080 | True |
| B | brightness_dark | P11 | 0.747361 | True |
| B | brightness_dark | uniform | 0.747361 | True |
| B | contrast_low | P11 | 1.387701 | True |
| B | contrast_low | uniform | 1.485238 | True |
| B | gaussian_noise | P11 | 2.506349 | True |
| B | gaussian_noise | uniform | 3.021798 | True |
| B | gaussian_blur | P11 | 2.721263 | True |
| B | gaussian_blur | uniform | 3.654184 | True |

Source>.5pp advantage requirements against rawP11 and uniform remain separate prescribed tests: {"P11": {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}, "uniform": {"brightness_dark": true, "contrast_low": true, "gaussian_noise": true, "gaussian_blur": true}}

All exact gate rows appear in scientific_gates.csv; all clean rows in clean_safety.csv. The1e-12JS replay tolerance is never used in any gate.

## Scientific diagnosis

Both CAL gates FAIL on the gain-capture condition: only Dark passes jointly. Soft oracle01 captures87.8–88.3% of P00 gain on Dark,71.2–73.0% on Contrast,66.6–69.3% on Noise and66.0–68.3% on Blur across banks/salts. Source11 preserves the first three but Blur falls to42.9–50.1%. Every clean-safety check passes. Source11 exceeds both rawP11 and uniform by>=.5pp on all4shifts in bothbanks averaged acrosssalts; those positive subconditions do not override the failed80%capture criterion.

This is not the “little prevalence improvement” case. On shifted contexts, softoracle L1 falls from1.302–1.348 to0.743–0.853 (roughly37–43% reduction), and dominant-class agreement rises from26–29% to64–72%. Utility also improves: source11 versus rawP11 gains A/B Dark+1.026/+0.747pp,Contrast+1.147/+1.388pp,Noise+2.234/+2.506pp,Blur+2.643/+2.721pp. Thus the fixed observation model recovers substantial signal, but not enough for the predeclared retrieval requirement; semantic estimation and task-weighted sensitivity are both residual issues rather than a zero-signal conclusion.

Hard channels are rank9 in clean/dark/noise/blur and rank8 in contrast, with infinite condition numbers. This establishes lost directions in the hard argmax observation channel. In contrast all1000softchannels have rank10 and condition numbers72.7–209.7. There is no numerical rank-collapse evidence for the soft primary channel; full rank alone does not guarantee a stable K20 estimate. All oracle-soft episodes have negative unprojected entries (about42–44% of entries); simplex projection changes L1 by means1.27–1.77. Finite-support variation, channel mismatch and task-sensitive residual error are compatible explanations, not separately identified causes. No claim that frozen feature semantics are intrinsically unobservable follows.

Blur retains a secondary coupling loss: softsource11 is2.089/1.631pp below oracle01 for A/B, while Dark/Contrast/Noise branches coincide. Oracle01 already fails on three shifts, so this is not the taxonomy case of a successful oracle estimator blocked only by deployment context ID. Clean source11 remains+3.913/+3.678pp abovezero.

## Scope of interpretation

The implementation repair is separate from channel identifiability, prevalence estimation and retrieval utility. The numerical results above test the specified fixed classifier-output observation channel. Selector failure alone does not establish failure of the neutral192-scalar operator or absence of semantic information in frozen features. No next task is assigned; stop here and return evidence to Lead.
