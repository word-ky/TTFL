# T003 Moment-Written Context Diagnostic

2/4 PASS; case M-A.

| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark | 4.332 | 27.437 | 5.054 | 1.805 | 1.083 | 52.051 | 50.409 |
| contrast_low | 3.610 | 14.440 | 3.971 | 7.942 | 3.971 | 55.290 | 57.125 |
| gaussian_noise | 20.578 | 19.134 | 19.495 | 32.491 | 31.769 | 55.629 | 56.606 |
| gaussian_blur | 20.217 | 24.188 | 18.412 | 25.632 | 27.076 | 57.038 | 60.297 |

Weighted accuracies (%); macro-client values in summary.csv. T002 CE rows are copied100-client references, not recomputed. Smoke primary rows, if present, use2clients only.

Clean sanity: 25.632% -> 22.744%.

Clamp warning: 4 layer episodes have >10% channels at a clamp boundary. See moment_diagnostics.csv for every layer/context.

Frozen gate: all four gain/gap values>=2pp per corruption;>=2/4 forM-A. No query tuning. Reference uses all train data only; labels are used only to evaluate queries.

```json
[
  {
    "target": "brightness_dark",
    "correct_gain_pp": 23.104694082203327,
    "correct_minus_wrong_clean_pp": 22.382672507961413,
    "correct_minus_wrong_alt_pp": 25.631769804707613,
    "correct_minus_noise_pp": 26.353791498642966,
    "passed": true,
    "wrong_clean_paired": {
      "mean": 21.45046046935022,
      "median": 21.45046046935022,
      "fraction_positive": 1.0,
      "p10": 18.835546700283885,
      "p25": 19.81613936368376,
      "p75": 23.084781575016677,
      "p90": 24.065374238416553
    },
    "wrong_alt_paired": {
      "mean": 23.978550219908357,
      "median": 23.978550219908357,
      "fraction_positive": 1.0,
      "p10": 19.341164650395513,
      "p25": 21.08018423896283,
      "p75": 26.876916200853884,
      "p90": 28.6159357894212
    }
  },
  {
    "target": "contrast_low",
    "correct_gain_pp": 10.830325082226889,
    "correct_minus_wrong_clean_pp": 10.469314252238195,
    "correct_minus_wrong_alt_pp": 6.498194749497335,
    "correct_minus_noise_pp": 10.469314082112124,
    "passed": true,
    "wrong_clean_paired": {
      "mean": 9.490977320820093,
      "median": 9.490977320820093,
      "fraction_positive": 1.0,
      "p10": 6.746680308133364,
      "p25": 7.775791687890887,
      "p75": 11.206162953749299,
      "p90": 12.235274333506823
    },
    "wrong_alt_paired": {
      "mean": 6.176937744021416,
      "median": 6.176937744021416,
      "fraction_positive": 1.0,
      "p10": 5.275791510939598,
      "p25": 5.61372134834528,
      "p75": 6.740154139697552,
      "p90": 7.078083977103233
    }
  },
  {
    "target": "gaussian_noise",
    "correct_gain_pp": -1.4440427914208023,
    "correct_minus_wrong_clean_pp": -0.36101055866113896,
    "correct_minus_wrong_alt_pp": -13.357401381008028,
    "correct_minus_noise_pp": -12.635378940333528,
    "passed": false,
    "wrong_clean_paired": {
      "mean": -0.05674697458744049,
      "median": -0.05674697458744049,
      "fraction_positive": 0.5,
      "p10": -0.910225585103035,
      "p25": -0.590171106159687,
      "p75": 0.47667715698480606,
      "p90": 0.796731635928154
    },
    "wrong_alt_paired": {
      "mean": -14.652140066027641,
      "median": -14.652140066027641,
      "fraction_positive": 0.0,
      "p10": -18.28396402299404,
      "p25": -16.92203003913164,
      "p75": -12.382250092923641,
      "p90": -11.020316109061241
    }
  },
  {
    "target": "gaussian_blur",
    "correct_gain_pp": 3.9711185821772474,
    "correct_minus_wrong_clean_pp": 5.7761732367832295,
    "correct_minus_wrong_alt_pp": -1.4440431868126247,
    "correct_minus_noise_pp": -2.88808853079696,
    "passed": false,
    "wrong_clean_paired": {
      "mean": 4.94268536567688,
      "median": 4.94268536567688,
      "fraction_positive": 1.0,
      "p10": 2.60469913482666,
      "p25": 3.4814439713954926,
      "p75": 6.403926759958267,
      "p90": 7.2806715965271
    },
    "wrong_alt_paired": {
      "mean": -2.0202018320560455,
      "median": -2.0202018320560455,
      "fraction_positive": 0.0,
      "p10": -3.636363297700882,
      "p25": -3.0303027480840683,
      "p75": -1.0101009160280228,
      "p90": -0.404040366411209
    }
  }
]
```