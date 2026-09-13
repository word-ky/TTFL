# T003 Moment-Written Context Diagnostic

1/4 PASS; case REVIEW_M_B_OR_M_C.

| Target | None | Moment correct | Wrong clean | Wrong alt | Noise image | T002 CE correct | T002 CE noise |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_dark | 17.641 | 23.791 | 11.749 | 11.104 | 13.152 | 52.051 | 50.409 |
| contrast_low | 20.088 | 19.376 | 13.598 | 15.200 | 15.533 | 55.290 | 57.125 |
| gaussian_noise | 30.747 | 22.934 | 19.543 | 23.918 | 36.804 | 55.629 | 56.606 |
| gaussian_blur | 30.993 | 23.100 | 21.338 | 21.065 | 31.751 | 57.038 | 60.297 |

Weighted accuracies (%); macro-client values in summary.csv. T002 CE rows are copied100-client references, not recomputed. Smoke primary rows, if present, use2clients only.

Clean sanity: 34.218% -> 24.131%.

Clamp warning: 204 layer episodes have >10% channels at a clamp boundary. See moment_diagnostics.csv for every layer/context.

Frozen gate: all four gain/gap values>=2pp per corruption;>=2/4 forM-A. No query tuning. Reference uses all train data only; labels are used only to evaluate queries.

```json
[
  {
    "target": "brightness_dark",
    "correct_gain_pp": 6.150675015413178,
    "correct_minus_wrong_clean_pp": 12.04202405825558,
    "correct_minus_wrong_alt_pp": 12.687013739262593,
    "correct_minus_noise_pp": 10.63900555436189,
    "passed": true,
    "wrong_clean_paired": {
      "mean": 10.50055639166385,
      "median": 9.789272025227547,
      "fraction_positive": 0.85,
      "p10": -9.547872245311735,
      "p25": 3.06052757659927,
      "p75": 22.60113898664713,
      "p90": 33.144805990159526
    },
    "wrong_alt_paired": {
      "mean": 10.828879859764129,
      "median": 11.592892883345485,
      "fraction_positive": 0.84,
      "p10": -10.543603897094727,
      "p25": 2.777777938172221,
      "p75": 26.240718632470816,
      "p90": 36.151403710246086
    }
  },
  {
    "target": "contrast_low",
    "correct_gain_pp": -0.7114833881137841,
    "correct_minus_wrong_clean_pp": 5.7783097442994045,
    "correct_minus_wrong_alt_pp": 4.175809599758033,
    "correct_minus_noise_pp": 3.843340857046851,
    "passed": false,
    "wrong_clean_paired": {
      "mean": 5.3579199821688235,
      "median": 3.827838972210884,
      "fraction_positive": 0.68,
      "p10": -9.89534929394722,
      "p25": 0.0,
      "p75": 12.441770359873772,
      "p90": 21.765079945325855
    },
    "wrong_alt_paired": {
      "mean": 3.9600254530087113,
      "median": 3.180308360606432,
      "fraction_positive": 0.65,
      "p10": -9.428442120552063,
      "p25": 0.0,
      "p75": 8.533463813364506,
      "p90": 18.33214402198792
    }
  },
  {
    "target": "gaussian_noise",
    "correct_gain_pp": -7.813019453838891,
    "correct_minus_wrong_clean_pp": 3.391182848133269,
    "correct_minus_wrong_alt_pp": -0.9841080176100263,
    "correct_minus_noise_pp": -13.870602847155045,
    "passed": false,
    "wrong_clean_paired": {
      "mean": 3.380391679238528,
      "median": 2.690945193171501,
      "fraction_positive": 0.7,
      "p10": -0.855533881112933,
      "p25": 0.0,
      "p75": 5.470587685704231,
      "p90": 9.836363866925241
    },
    "wrong_alt_paired": {
      "mean": -1.2552839256823063,
      "median": -3.3385099843144417,
      "fraction_positive": 0.37,
      "p10": -17.334667444229126,
      "p25": -9.8451629281044,
      "p75": 5.366580933332443,
      "p90": 19.499999135732654
    }
  },
  {
    "target": "gaussian_blur",
    "correct_gain_pp": -7.892812170502953,
    "correct_minus_wrong_clean_pp": 1.7620852349045002,
    "correct_minus_wrong_alt_pp": 2.0347097094249236,
    "correct_minus_noise_pp": -8.650841114070186,
    "passed": false,
    "wrong_clean_paired": {
      "mean": 1.2350390362553298,
      "median": 0.5342354997992516,
      "fraction_positive": 0.51,
      "p10": -6.291605830192566,
      "p25": -1.1105353012681007,
      "p75": 3.2366071827709675,
      "p90": 8.812643662095077
    },
    "wrong_alt_paired": {
      "mean": 2.318684223573655,
      "median": 1.246408373117447,
      "fraction_positive": 0.53,
      "p10": -10.6855808198452,
      "p25": -1.9967536441981792,
      "p75": 9.026521816849709,
      "p90": 18.20574074983597
    }
  }
]
```