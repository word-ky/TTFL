# CODEX -> CHATGPT

## Timestamp / commit / run

2026-09-14T01:37:42+08:00. Lead 7d42fe2; runtime 87f1bdedd94a63f3fb052d8b3d7e6affb9165e9d; run 20260914-013254-ttfl-t012-gpu1; release 20260914-013251-ttfl-t012; NVIDIA RTX A6000 GPU1; 62.73s. One formal run, exit0, no launcher retry. The enclosing delivery commit contains the independent report.

## Frozen objects and verification

48 tests PASS (46 historical plus two scaling tests). Checkpoint, 192-scalar affine model, neutral writer, corruptions, T007R states, T008 references, exact T009 K20 manifest, T010 moment implementation and T011 consistency implementation hashes unchanged. No source/query split change, state fitting, gradient optimization, new federation or OOD rerun. The dataset remains the historical PFLlib merged-data per-client CIFAR-10 query split, not the official CIFAR-10 test benchmark.

## T011 matrix reconstruction

Phase I independently reconstructs all 5,000 T011 rows from saved predictions and integer class counts. Full exact argmax sets are retained. Per-context utility-vector Spearman uses count ranks: subtracting zero and dividing by the shared client/context query count leaves those five ranks exactly unchanged. No new model forward pass is used in Phase I.

## Context-vs-client preference decomposition

Every unordered context pair is evaluated. The second client is shifted by exactly [7,13,23,37,41,53,71,89] modulo100; no offset search. Overlap values are fractions; differences are percentage points.

| bank | target1 | target2 | same_client_overlap | control_mean | control_min | control_max | difference_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean | brightness_dark | 0.51 | 0.3625 | 0.29 | 0.39 | 14.750000000000002 |
| A | clean | contrast_low | 0.52 | 0.28375 | 0.22 | 0.33 | 23.625 |
| A | clean | gaussian_noise | 0.89 | 0.39749999999999996 | 0.33 | 0.5 | 49.25000000000001 |
| A | clean | gaussian_blur | 0.71 | 0.30625 | 0.26 | 0.36 | 40.37499999999999 |
| A | brightness_dark | contrast_low | 0.53 | 0.34875 | 0.29 | 0.4 | 18.125000000000004 |
| A | brightness_dark | gaussian_noise | 0.51 | 0.375 | 0.34 | 0.42 | 13.5 |
| A | brightness_dark | gaussian_blur | 0.56 | 0.355 | 0.32 | 0.4 | 20.500000000000007 |
| A | contrast_low | gaussian_noise | 0.52 | 0.28500000000000003 | 0.25 | 0.36 | 23.5 |
| A | contrast_low | gaussian_blur | 0.7 | 0.28625 | 0.24 | 0.36 | 41.37499999999999 |
| A | gaussian_noise | gaussian_blur | 0.69 | 0.3375 | 0.27 | 0.39 | 35.24999999999999 |
| B | clean | brightness_dark | 0.5 | 0.35875 | 0.3 | 0.42 | 14.124999999999998 |
| B | clean | contrast_low | 0.53 | 0.26875000000000004 | 0.22 | 0.32 | 26.125 |
| B | clean | gaussian_noise | 0.91 | 0.395 | 0.33 | 0.46 | 51.5 |
| B | clean | gaussian_blur | 0.74 | 0.31125 | 0.25 | 0.36 | 42.875 |
| B | brightness_dark | contrast_low | 0.55 | 0.3475 | 0.29 | 0.39 | 20.250000000000007 |
| B | brightness_dark | gaussian_noise | 0.55 | 0.4 | 0.37 | 0.43 | 15.000000000000002 |
| B | brightness_dark | gaussian_blur | 0.55 | 0.3875 | 0.34 | 0.43 | 16.250000000000004 |
| B | contrast_low | gaussian_noise | 0.52 | 0.29125 | 0.26 | 0.36 | 22.875 |
| B | contrast_low | gaussian_blur | 0.69 | 0.28250000000000003 | 0.25 | 0.33 | 40.74999999999999 |
| B | gaussian_noise | gaussian_blur | 0.68 | 0.35624999999999996 | 0.31 | 0.41 | 32.37500000000001 |

| bank | quartile | n | fraction_ge3 | fraction_ge4 | fraction_eq5 | distribution |
| --- | --- | --- | --- | --- | --- | --- |
| A | all | 100 | 0.790000 | 0.560000 | 0.380000 | {'1': 0, '2': 21, '3': 23, '4': 18, '5': 38} |
| A | 1 | 25 | 0.720000 | 0.560000 | 0.440000 | {'1': 0, '2': 7, '3': 4, '4': 3, '5': 11} |
| A | 2 | 25 | 0.880000 | 0.600000 | 0.480000 | {'1': 0, '2': 3, '3': 7, '4': 3, '5': 12} |
| A | 3 | 25 | 0.800000 | 0.600000 | 0.320000 | {'1': 0, '2': 5, '3': 5, '4': 7, '5': 8} |
| A | 4 | 25 | 0.760000 | 0.480000 | 0.280000 | {'1': 0, '2': 6, '3': 7, '4': 5, '5': 7} |
| B | all | 100 | 0.780000 | 0.570000 | 0.380000 | {'1': 0, '2': 22, '3': 21, '4': 19, '5': 38} |
| B | 1 | 25 | 0.760000 | 0.560000 | 0.480000 | {'1': 0, '2': 6, '3': 5, '4': 2, '5': 12} |
| B | 2 | 25 | 0.840000 | 0.600000 | 0.440000 | {'1': 0, '2': 4, '3': 6, '4': 4, '5': 11} |
| B | 3 | 25 | 0.760000 | 0.600000 | 0.280000 | {'1': 0, '2': 6, '3': 4, '4': 8, '5': 7} |
| B | 4 | 25 | 0.760000 | 0.520000 | 0.320000 | {'1': 0, '2': 6, '3': 6, '4': 5, '5': 8} |

Utility-vector Spearman below averages only valid nonconstant comparisons. Full per-offset means, medians and valid counts are saved in context_client_factorization.json; the compact control summary is the mean/min/max of the eight offset means.

| bank | pair | same_median | same_mean | same_valid | control_mean_of_means | control_min_mean | control_max_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | clean / brightness_dark | 0.400000 | 0.241978 | 97 | -0.029460 | -0.129210 | 0.027143 |
| A | clean / contrast_low | 0.211803 | 0.146959 | 98 | -0.185805 | -0.283100 | -0.117749 |
| A | clean / gaussian_noise | 0.900000 | 0.802048 | 95 | 0.325804 | 0.276131 | 0.352968 |
| A | clean / gaussian_blur | 0.728025 | 0.610675 | 98 | 0.027509 | -0.061098 | 0.079920 |
| A | brightness_dark / contrast_low | 0.666886 | 0.510816 | 98 | 0.138979 | 0.076679 | 0.216152 |
| A | brightness_dark / gaussian_noise | 0.200000 | 0.143611 | 95 | -0.054523 | -0.089018 | -0.015368 |
| A | brightness_dark / gaussian_blur | 0.631579 | 0.448290 | 97 | 0.033927 | -0.075208 | 0.106184 |
| A | contrast_low / gaussian_noise | 0.051299 | -0.006822 | 95 | -0.235662 | -0.274704 | -0.155409 |
| A | contrast_low / gaussian_blur | 0.666886 | 0.492384 | 98 | -0.021977 | -0.113188 | 0.045459 |
| A | gaussian_noise / gaussian_blur | 0.486664 | 0.412652 | 95 | 0.037127 | -0.038565 | 0.099970 |
| B | clean / brightness_dark | 0.300000 | 0.262009 | 97 | 0.009496 | -0.081546 | 0.067385 |
| B | clean / contrast_low | 0.259445 | 0.139399 | 98 | -0.195146 | -0.293989 | -0.115546 |
| B | clean / gaussian_noise | 0.900000 | 0.848979 | 95 | 0.378805 | 0.337197 | 0.416423 |
| B | clean / gaussian_blur | 0.718932 | 0.623804 | 98 | 0.104477 | 0.023901 | 0.155125 |
| B | brightness_dark / contrast_low | 0.685410 | 0.496458 | 98 | 0.101817 | 0.036056 | 0.165555 |
| B | brightness_dark / gaussian_noise | 0.205196 | 0.188666 | 95 | -0.006002 | -0.026578 | 0.019844 |
| B | brightness_dark / gaussian_blur | 0.526316 | 0.439151 | 97 | 0.043908 | -0.043659 | 0.133610 |
| B | contrast_low / gaussian_noise | 0.051299 | 0.001031 | 95 | -0.228125 | -0.262780 | -0.148986 |
| B | contrast_low / gaussian_blur | 0.666886 | 0.457831 | 98 | -0.067280 | -0.134517 | -0.029182 |
| B | gaussian_noise / gaussian_blur | 0.615587 | 0.486179 | 95 | 0.118475 | 0.053800 | 0.172320 |

## CLIENT-LOCK decision

| bank | mean_same_overlap | mean_control_overlap | difference_pp | fraction_ge3 | passed |
| --- | --- | --- | --- | --- | --- |
| A | 0.614000 | 0.333750 | 28.025000 | 0.790000 | True |
| B | 0.622000 | 0.339875 | 28.212500 | 0.780000 | True |

**CLIENT-LOCK-STRONG in both banks.** Same-client overlap exceeds mismatch controls by 28.025 /28.2125 pp, above 15 pp. 79% /78% of clients have a candidate optimal in >=3/5 contexts, above50%. Frozen states exhibit persistent client preference in addition to context response. Thus T011 best-of-five headroom cannot be interpreted as pure context headroom. This does not isolate whether label composition or another client property causes the persistence.

## Alpha scaling implementation and regression

scale_state(state, alpha) linearly multiplies every gamma/beta tensor without clipping or mutation. Grid fixed at 0,.25,.5,.75,1,1.25,1.5. Before scoring, all eight saved state directions pass alpha0 ordinary-logit equality, alpha1 tensor/logit equality and finite states/logits across all seven alphas. Repeated construction is deterministic. An additional1,600 endpoint logit checks run on actual supports; max difference0. Saved/scaled state and model hashes remain unchanged after every source/query episode.

## Source-score freeze

All5,600 source rows and800 decision rows (both selectors in each row) frozen at 2026-09-13T17:33:41.563113+00:00. Scores SHA 04b4efb00709127aa1abf41820b7c09d840152e58aa7b246bd9db58724d610c0; choices SHA a00cac4b593e3e6f58c687333975630c8d8d317ee8eed5b7827b49d577a1fb1a. Query amplitude matrix frozen later at 2026-09-13T17:34:07.813256+00:00. J-ray is argminJ; PC-ray-safe is strict consistency improvement with nondecreasing agreement, otherwise0; both use smaller-alpha tie order. Original T010/T011 formulas unchanged.

**The true target chooses the ray direction. This is a privileged, direction-conditioned oracle diagnostic, not deployable source-only adaptation.** Phase I used historical fixed-state query results, but neither new amplitude outcomes nor a query-fitted rule entered source selection.

## Query amplitude matrix verification

5,600 policies:1,600 exact T007R alpha0/alpha1 arrays reused,4,000 missing amplitudes evaluated. Every prediction vector independently matches saved integer per-class correct/total counts; every endpoint array matches T007R. T007R endpoint aggregates and T011 best-of-five aggregates reconstruct with maximum error 7.11e-15 pp. Freeze hashes and all source choices rechecked; gate arithmetic uses exact count fractions.

## Best-ray / alpha=1 / best-of-five comparison

| bank | target | zero | alpha1 | best_ray | best5 | amplitude_gain | ray_capture | RAY_CAPTURE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 32.296536 | 37.762385 | 35.712121 | 5.465849 | 1.112862 | True |
| A | contrast_low | 19.955264 | 28.561473 | 31.488675 | 33.993166 | 2.927201 | 0.821591 | True |
| A | gaussian_noise | 30.676644 | 32.499558 | 35.005460 | 37.825842 | 2.505902 | 0.605497 | False |
| A | gaussian_blur | 30.901579 | 33.048500 | 35.421874 | 40.343614 | 2.373374 | 0.478742 | False |
| B | brightness_dark | 17.546026 | 32.424808 | 37.968292 | 35.686070 | 5.543485 | 1.125811 | True |
| B | contrast_low | 19.955264 | 28.111626 | 31.222782 | 33.862340 | 3.111156 | 0.810200 | True |
| B | gaussian_noise | 30.676644 | 32.406771 | 34.747803 | 37.894827 | 2.341032 | 0.564014 | False |
| B | gaussian_blur | 30.901579 | 33.009919 | 35.560811 | 40.374218 | 2.550892 | 0.491862 | False |

Dark best-ray exceeds the five-state oracle because the grids differ: scaled states add candidates absent from the five-state bank. Capture can therefore exceed1 and is not clipped. Dark/Contrast exceed80% capture jointly; Noise/Blur do not. On Noise, capture is60.55% /56.40%; on Blur47.87% /49.19%. Noise/Blur nevertheless improve over alpha1 by2.34–2.55pp. Fixed direction amplitude helps substantially but leaves client-conditioned direction/state-bank headroom.

Both upper bounds maximize on the same finite per-client query outcomes and are optimistic diagnostics, not independent test performance. Exact per-client correct-count ties use the smallest alpha (or historical fixed candidate order) for aggregate display; all tied optima are preserved separately. Macro-class is reconstructed after mixing client choices, not averaged from per-client class means.

## Alpha regret and entropy-quartile audit

| bank | target | quartile | alpha1_best_fraction | alpha0_best_fraction | any_below1_fraction | any_above1_fraction | regret_gt2_fraction | regret_gt5_fraction | median | p75 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | all | 0.220000 | 0.140000 | 0.260000 | 0.740000 | 0.630000 | 0.410000 | 2.893104 | 9.915065 | 13.074495 |
| A | brightness_dark | 1 | 0.240000 | 0.120000 | 0.280000 | 0.680000 | 0.560000 | 0.360000 | 2.112676 | 7.578947 | 16.598756 |
| A | brightness_dark | 2 | 0.120000 | 0.160000 | 0.240000 | 0.800000 | 0.720000 | 0.520000 | 5.681818 | 11.235955 | 13.606956 |
| A | brightness_dark | 3 | 0.240000 | 0.240000 | 0.320000 | 0.640000 | 0.560000 | 0.440000 | 3.428571 | 10.204082 | 13.227273 |
| A | brightness_dark | 4 | 0.280000 | 0.040000 | 0.200000 | 0.840000 | 0.680000 | 0.320000 | 2.580645 | 7.446809 | 10.509284 |
| A | contrast_low | all | 0.290000 | 0.190000 | 0.480000 | 0.570000 | 0.450000 | 0.210000 | 1.342342 | 3.974234 | 8.480634 |
| A | contrast_low | 1 | 0.200000 | 0.200000 | 0.520000 | 0.600000 | 0.600000 | 0.280000 | 2.500000 | 5.882353 | 11.390753 |
| A | contrast_low | 2 | 0.320000 | 0.200000 | 0.440000 | 0.600000 | 0.280000 | 0.160000 | 0.589391 | 2.252252 | 6.085478 |
| A | contrast_low | 3 | 0.280000 | 0.320000 | 0.600000 | 0.440000 | 0.600000 | 0.360000 | 3.030303 | 7.000000 | 12.495421 |
| A | contrast_low | 4 | 0.360000 | 0.040000 | 0.360000 | 0.640000 | 0.320000 | 0.040000 | 1.333333 | 2.439024 | 4.156867 |
| A | gaussian_noise | all | 0.210000 | 0.230000 | 0.450000 | 0.730000 | 0.420000 | 0.140000 | 1.471861 | 3.191262 | 6.422481 |
| A | gaussian_noise | 1 | 0.160000 | 0.240000 | 0.400000 | 0.720000 | 0.440000 | 0.200000 | 1.428571 | 4.166667 | 8.238739 |
| A | gaussian_noise | 2 | 0.160000 | 0.160000 | 0.320000 | 0.800000 | 0.240000 | 0.120000 | 1.123596 | 2.000000 | 4.828945 |
| A | gaussian_noise | 3 | 0.240000 | 0.280000 | 0.520000 | 0.680000 | 0.560000 | 0.240000 | 2.298851 | 5.000000 | 7.031646 |
| A | gaussian_noise | 4 | 0.280000 | 0.240000 | 0.560000 | 0.720000 | 0.440000 | 0.000000 | 1.333333 | 2.777778 | 3.311202 |
| A | gaussian_blur | all | 0.330000 | 0.290000 | 0.570000 | 0.500000 | 0.410000 | 0.210000 | 1.324561 | 3.668670 | 7.789474 |
| A | gaussian_blur | 1 | 0.280000 | 0.400000 | 0.600000 | 0.480000 | 0.360000 | 0.280000 | 0.952381 | 5.882353 | 9.168317 |
| A | gaussian_blur | 2 | 0.320000 | 0.280000 | 0.480000 | 0.560000 | 0.400000 | 0.120000 | 1.375246 | 2.906977 | 4.701042 |
| A | gaussian_blur | 3 | 0.360000 | 0.400000 | 0.680000 | 0.400000 | 0.520000 | 0.400000 | 2.702703 | 7.500000 | 8.530481 |
| A | gaussian_blur | 4 | 0.360000 | 0.080000 | 0.520000 | 0.560000 | 0.360000 | 0.040000 | 0.877193 | 2.620087 | 4.746907 |
| B | brightness_dark | all | 0.190000 | 0.140000 | 0.270000 | 0.730000 | 0.630000 | 0.440000 | 3.672439 | 9.228158 | 13.420455 |
| B | brightness_dark | 1 | 0.240000 | 0.120000 | 0.280000 | 0.680000 | 0.560000 | 0.400000 | 2.816901 | 8.000000 | 16.598756 |
| B | brightness_dark | 2 | 0.040000 | 0.160000 | 0.240000 | 0.800000 | 0.760000 | 0.520000 | 5.194805 | 10.112360 | 14.795742 |
| B | brightness_dark | 3 | 0.240000 | 0.240000 | 0.320000 | 0.640000 | 0.560000 | 0.440000 | 2.631579 | 10.204082 | 14.278788 |
| B | brightness_dark | 4 | 0.240000 | 0.040000 | 0.240000 | 0.800000 | 0.640000 | 0.400000 | 3.535354 | 7.446809 | 9.103448 |
| B | contrast_low | all | 0.260000 | 0.190000 | 0.510000 | 0.510000 | 0.490000 | 0.220000 | 1.534778 | 4.765968 | 9.873239 |
| B | contrast_low | 1 | 0.320000 | 0.200000 | 0.600000 | 0.600000 | 0.640000 | 0.320000 | 2.631579 | 5.882353 | 11.328277 |
| B | contrast_low | 2 | 0.280000 | 0.120000 | 0.520000 | 0.480000 | 0.320000 | 0.160000 | 1.030928 | 2.252252 | 7.768595 |
| B | contrast_low | 3 | 0.160000 | 0.320000 | 0.520000 | 0.440000 | 0.640000 | 0.360000 | 3.000000 | 7.636364 | 13.301471 |
| B | contrast_low | 4 | 0.280000 | 0.120000 | 0.400000 | 0.520000 | 0.360000 | 0.040000 | 1.219512 | 2.298851 | 3.920646 |
| B | gaussian_noise | all | 0.210000 | 0.220000 | 0.410000 | 0.730000 | 0.390000 | 0.130000 | 1.613848 | 3.132038 | 5.971535 |
| B | gaussian_noise | 1 | 0.240000 | 0.200000 | 0.400000 | 0.680000 | 0.400000 | 0.240000 | 1.056338 | 4.268293 | 8.270423 |
| B | gaussian_noise | 2 | 0.200000 | 0.160000 | 0.280000 | 0.800000 | 0.240000 | 0.080000 | 1.265823 | 2.000000 | 3.244732 |
| B | gaussian_noise | 3 | 0.200000 | 0.280000 | 0.440000 | 0.680000 | 0.520000 | 0.160000 | 2.112676 | 4.363636 | 6.028481 |
| B | gaussian_noise | 4 | 0.200000 | 0.240000 | 0.520000 | 0.760000 | 0.400000 | 0.040000 | 1.574803 | 2.777778 | 3.771107 |
| B | gaussian_blur | all | 0.270000 | 0.300000 | 0.620000 | 0.510000 | 0.410000 | 0.200000 | 1.349261 | 4.140840 | 8.449257 |
| B | gaussian_blur | 1 | 0.280000 | 0.400000 | 0.680000 | 0.440000 | 0.360000 | 0.280000 | 1.015228 | 5.128205 | 10.866337 |
| B | gaussian_blur | 2 | 0.240000 | 0.280000 | 0.520000 | 0.480000 | 0.360000 | 0.120000 | 1.375246 | 3.153153 | 4.769776 |
| B | gaussian_blur | 3 | 0.280000 | 0.400000 | 0.720000 | 0.480000 | 0.520000 | 0.360000 | 2.702703 | 7.894737 | 8.862745 |
| B | gaussian_blur | 4 | 0.280000 | 0.120000 | 0.560000 | 0.640000 | 0.400000 | 0.040000 | 1.315789 | 2.857143 | 4.459556 |

The argmax fractions overlap intentionally when there are ties. All display-alpha histograms, including every frozen entropy quartile, are stored in amplitude_regret.json. No labels/entropy were used to choose source amplitude.

| bank | target | quartile | display_alpha_histogram |
| --- | --- | --- | --- |
| A | brightness_dark | all | {'0.0': 14, '0.25': 1, '0.5': 3, '0.75': 8, '1.0': 16, '1.25': 11, '1.5': 47} |
| A | brightness_dark | 1 | {'0.0': 3, '0.25': 1, '0.5': 0, '0.75': 3, '1.0': 4, '1.25': 4, '1.5': 10} |
| A | brightness_dark | 2 | {'0.0': 4, '0.25': 0, '0.5': 0, '0.75': 2, '1.0': 2, '1.25': 2, '1.5': 15} |
| A | brightness_dark | 3 | {'0.0': 6, '0.25': 0, '0.5': 1, '0.75': 1, '1.0': 5, '1.25': 2, '1.5': 10} |
| A | brightness_dark | 4 | {'0.0': 1, '0.25': 0, '0.5': 2, '0.75': 2, '1.0': 5, '1.25': 3, '1.5': 12} |
| A | contrast_low | all | {'0.0': 19, '0.25': 3, '0.5': 10, '0.75': 16, '1.0': 14, '1.25': 15, '1.5': 23} |
| A | contrast_low | 1 | {'0.0': 5, '0.25': 1, '0.5': 5, '0.75': 2, '1.0': 1, '1.25': 5, '1.5': 6} |
| A | contrast_low | 2 | {'0.0': 5, '0.25': 0, '0.5': 2, '0.75': 4, '1.0': 5, '1.25': 5, '1.5': 4} |
| A | contrast_low | 3 | {'0.0': 8, '0.25': 1, '0.5': 1, '0.75': 5, '1.0': 2, '1.25': 3, '1.5': 5} |
| A | contrast_low | 4 | {'0.0': 1, '0.25': 1, '0.5': 2, '0.75': 5, '1.0': 6, '1.25': 2, '1.5': 8} |
| A | gaussian_noise | all | {'0.0': 23, '0.25': 9, '0.5': 8, '0.75': 5, '1.0': 3, '1.25': 11, '1.5': 41} |
| A | gaussian_noise | 1 | {'0.0': 6, '0.25': 2, '0.5': 1, '0.75': 1, '1.0': 1, '1.25': 3, '1.5': 11} |
| A | gaussian_noise | 2 | {'0.0': 4, '0.25': 2, '0.5': 1, '0.75': 1, '1.0': 1, '1.25': 3, '1.5': 13} |
| A | gaussian_noise | 3 | {'0.0': 7, '0.25': 2, '0.5': 2, '0.75': 2, '1.0': 0, '1.25': 1, '1.5': 11} |
| A | gaussian_noise | 4 | {'0.0': 6, '0.25': 3, '0.5': 4, '0.75': 1, '1.0': 1, '1.25': 4, '1.5': 6} |
| A | gaussian_blur | all | {'0.0': 29, '0.25': 9, '0.5': 11, '0.75': 8, '1.0': 14, '1.25': 7, '1.5': 22} |
| A | gaussian_blur | 1 | {'0.0': 10, '0.25': 1, '0.5': 2, '0.75': 2, '1.0': 1, '1.25': 3, '1.5': 6} |
| A | gaussian_blur | 2 | {'0.0': 7, '0.25': 3, '0.5': 2, '0.75': 0, '1.0': 4, '1.25': 1, '1.5': 8} |
| A | gaussian_blur | 3 | {'0.0': 10, '0.25': 2, '0.5': 3, '0.75': 2, '1.0': 4, '1.25': 0, '1.5': 4} |
| A | gaussian_blur | 4 | {'0.0': 2, '0.25': 3, '0.5': 4, '0.75': 4, '1.0': 5, '1.25': 3, '1.5': 4} |
| B | brightness_dark | all | {'0.0': 14, '0.25': 1, '0.5': 5, '0.75': 7, '1.0': 12, '1.25': 13, '1.5': 48} |
| B | brightness_dark | 1 | {'0.0': 3, '0.25': 1, '0.5': 2, '0.75': 1, '1.0': 4, '1.25': 4, '1.5': 10} |
| B | brightness_dark | 2 | {'0.0': 4, '0.25': 0, '0.5': 0, '0.75': 2, '1.0': 0, '1.25': 3, '1.5': 16} |
| B | brightness_dark | 3 | {'0.0': 6, '0.25': 0, '0.5': 1, '0.75': 1, '1.0': 5, '1.25': 2, '1.5': 10} |
| B | brightness_dark | 4 | {'0.0': 1, '0.25': 0, '0.5': 2, '0.75': 3, '1.0': 3, '1.25': 4, '1.5': 12} |
| B | contrast_low | all | {'0.0': 19, '0.25': 4, '0.5': 9, '0.75': 19, '1.0': 13, '1.25': 16, '1.5': 20} |
| B | contrast_low | 1 | {'0.0': 5, '0.25': 1, '0.5': 5, '0.75': 4, '1.0': 3, '1.25': 2, '1.5': 5} |
| B | contrast_low | 2 | {'0.0': 3, '0.25': 1, '0.5': 2, '0.75': 7, '1.0': 3, '1.25': 5, '1.5': 4} |
| B | contrast_low | 3 | {'0.0': 8, '0.25': 1, '0.5': 1, '0.75': 3, '1.0': 2, '1.25': 4, '1.5': 6} |
| B | contrast_low | 4 | {'0.0': 3, '0.25': 1, '0.5': 1, '0.75': 5, '1.0': 5, '1.25': 5, '1.5': 5} |
| B | gaussian_noise | all | {'0.0': 22, '0.25': 8, '0.5': 6, '0.75': 5, '1.0': 6, '1.25': 14, '1.5': 39} |
| B | gaussian_noise | 1 | {'0.0': 5, '0.25': 3, '0.5': 2, '0.75': 0, '1.0': 3, '1.25': 5, '1.5': 7} |
| B | gaussian_noise | 2 | {'0.0': 4, '0.25': 2, '0.5': 0, '0.75': 1, '1.0': 2, '1.25': 1, '1.5': 15} |
| B | gaussian_noise | 3 | {'0.0': 7, '0.25': 2, '0.5': 2, '0.75': 0, '1.0': 1, '1.25': 3, '1.5': 10} |
| B | gaussian_noise | 4 | {'0.0': 6, '0.25': 1, '0.5': 2, '0.75': 4, '1.0': 0, '1.25': 5, '1.5': 7} |
| B | gaussian_blur | all | {'0.0': 30, '0.25': 9, '0.5': 13, '0.75': 10, '1.0': 9, '1.25': 11, '1.5': 18} |
| B | gaussian_blur | 1 | {'0.0': 10, '0.25': 1, '0.5': 3, '0.75': 3, '1.0': 1, '1.25': 2, '1.5': 5} |
| B | gaussian_blur | 2 | {'0.0': 7, '0.25': 2, '0.5': 3, '0.75': 1, '1.0': 3, '1.25': 3, '1.5': 6} |
| B | gaussian_blur | 3 | {'0.0': 10, '0.25': 2, '0.5': 4, '0.75': 2, '1.0': 1, '1.25': 2, '1.5': 4} |
| B | gaussian_blur | 4 | {'0.0': 3, '0.25': 4, '0.5': 3, '0.75': 4, '1.0': 4, '1.25': 4, '1.5': 3} |

| target | best_alpha_argmax_intersection_fraction |
| --- | --- |
| brightness_dark | 0.960000 |
| contrast_low | 0.820000 |
| gaussian_noise | 0.940000 |
| gaussian_blur | 0.960000 |

## J-ray / PC-ray-safe source selection

| bank | target | zero | alpha1 | best_ray | J_ray | PC_ray_safe | J_ray_retained | PC_ray_safe_retained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 32.296536 | 37.762385 | 32.021471 | 17.546026 | 0.716026 | 0.000000 |
| A | contrast_low | 19.955264 | 28.561473 | 31.488675 | 27.919278 | 19.955264 | 0.690517 | 0.000000 |
| A | gaussian_noise | 30.676644 | 32.499558 | 35.005460 | 32.060532 | 32.764034 | 0.319692 | 0.482208 |
| A | gaussian_blur | 30.901579 | 33.048500 | 35.421874 | 31.856070 | 32.206387 | 0.211157 | 0.288656 |
| B | brightness_dark | 17.546026 | 32.424808 | 37.968292 | 32.167582 | 17.546026 | 0.715961 | 0.000000 |
| B | contrast_low | 19.955264 | 28.111626 | 31.222782 | 27.023720 | 19.955264 | 0.627330 | 0.000000 |
| B | gaussian_noise | 30.676644 | 32.406771 | 34.747803 | 32.107351 | 32.685800 | 0.351425 | 0.493510 |
| B | gaussian_blur | 30.901579 | 33.009919 | 35.560811 | 31.989415 | 32.286396 | 0.233480 | 0.297220 |

| bank | target | metric | pooled_spearman | per_client_spearman_median | valid_clients | positive_rows | harmful_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | DeltaJ | 0.264204 | 0.673633 | 99 | 600 | 0.131667 |
| A | brightness_dark | DeltaC | -0.322799 | -0.954314 | 99 | 0 | None |
| A | contrast_low | DeltaJ | 0.089703 | 0.617355 | 96 | 599 | 0.180301 |
| A | contrast_low | DeltaC | -0.235108 | -0.790569 | 96 | 0 | None |
| A | gaussian_noise | DeltaJ | -0.069131 | 0.036037 | 93 | 576 | 0.173611 |
| A | gaussian_noise | DeltaC | 0.280536 | 0.767193 | 93 | 393 | 0.104326 |
| A | gaussian_blur | DeltaJ | -0.132135 | 0.000000 | 95 | 585 | 0.297436 |
| A | gaussian_blur | DeltaC | 0.018302 | -0.288675 | 95 | 19 | 0.000000 |
| B | brightness_dark | DeltaJ | 0.248697 | 0.668153 | 99 | 600 | 0.131667 |
| B | brightness_dark | DeltaC | -0.327401 | -0.954314 | 99 | 0 | None |
| B | contrast_low | DeltaJ | 0.081651 | 0.535714 | 96 | 599 | 0.203673 |
| B | contrast_low | DeltaC | -0.164262 | -0.632393 | 96 | 0 | None |
| B | gaussian_noise | DeltaJ | -0.033391 | 0.144908 | 92 | 581 | 0.170396 |
| B | gaussian_noise | DeltaC | 0.324072 | 0.796177 | 92 | 427 | 0.096019 |
| B | gaussian_blur | DeltaJ | -0.142981 | -0.074125 | 95 | 572 | 0.309441 |
| B | gaussian_blur | DeltaC | 0.018661 | -0.115728 | 95 | 23 | 0.000000 |

Each within-context pooled correlation has700 client/alpha rows; per-client Spearman uses seven amplitudes with average ranks. Constant vectors are excluded from correlation and valid counts retained. Harm fractions are conditional on strictly positive source improvement; null means no positive rows, not zero harm.

PC-ray-safe always chooses zero for Dark/Contrast because no positive amplitude lowers JS. The corresponding median per-client consistency/task correlations are strongly negative. J-ray captures about72% of best-ray gain on Dark,63–69% on Contrast,32–35% on Noise and21–23% on Blur. PC-ray-safe captures about48–49% on Noise and29–30% on Blur. Source scoring remains inadequate even with the true direction supplied.

## RAY-SOURCE result

[
  {
    "selector": "J_ray",
    "joint_shift_pass": {
      "brightness_dark": false,
      "contrast_low": false,
      "gaussian_noise": false,
      "gaussian_blur": false
    },
    "passed": false
  },
  {
    "selector": "PC_ray_safe",
    "joint_shift_pass": {
      "brightness_dark": false,
      "contrast_low": false,
      "gaussian_noise": false,
      "gaussian_blur": false
    },
    "passed": false
  }
]

**RAY-SOURCE-A fails for each predeclared selector:0/4 shifts jointly retain90% of best-ray gain.** No hybrid selector was assembled from whichever method looked better on a target.

## AMP-A/B/C decision

**AMP-A false; AMP-B applies.** Only2/4 shifts satisfy joint RAY-CAPTURE. In each bank, both Noise and Blur improve by>=1pp over alpha1 while remaining below80% capture, exactly meeting the fixed AMP-B descriptor. AMP-C/INCONCLUSIVE is not selected because amplitude gains are material under that descriptor.

**Primary case T012-B:** CLIENT-LOCK-STRONG plus incomplete true-ray capture supports a persistent client component mixed with transient context correction. T012-A does not hold; this is not evidence that scalar amplitude alone suffices. The small-amplitude-gain premise of T012-C is not met. Source-estimation failure coexists with direction/client mismatch; neither explains everything alone.

## Mechanism vs implementation conclusion

All tests, endpoint regressions, immutability checks and prediction/count audits pass. The states carry both context and persistent client preference, while amplitude is useful but insufficient on Noise/Blur. Current source criteria also fail to select near-oracle amplitude. The evidence motivates explicit client-persistent plus transient-context factorization before a writer; it does not justify enlarging the operator or asserting a particular learned factorization already works.

## Recommended next action

Return to Research Lead with the full decomposition and ray matrices. Preserve zero neutrality and the192-scalar operator. A next bounded package should separate persistent client calibration from transient context state rather than promote J/flip consistency directly into a learned writer. No next stage, new grid, SSL, meta-learning, federation or richer operator was launched.
