# T002 Prior-Controlled Covariate Context Specificity

Case B; 1/4 corruptions pass. 100 clients, fixed existing checkpoint/support/query.

| Corruption | None | Correct | Wrong clean | Wrong alt | Shuffled | Noise image | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| brightness_dark | 17.641 | 52.051 | 44.325 | 44.431 | 52.796 | 50.409 | True |
| contrast_low | 20.088 | 55.290 | 52.856 | 54.066 | 56.061 | 57.125 | False |
| gaussian_noise | 30.747 | 55.629 | 52.324 | 46.991 | 56.134 | 56.606 | False |
| gaussian_blur | 30.993 | 57.038 | 58.155 | 54.299 | 57.570 | 60.297 | False |

Accuracies are sample-weighted percentages. Macro-client rows and mean support losses/norms are in summary.csv. Paired client differences and percentile summaries are in per_client.csv and summary.json.

PASS requires correct gain >=2pp, both matched-label wrong gaps >=2pp, and correct > noise-image. At least2/4 required overall. No severity/optimizer selection on these query results.

Existing PFLlib static label-skew partitions, merged original labeled CIFAR10 splits with per-client75/25 split; not an official CIFAR10 test benchmark. All conditions within each target share identical corrupted queries. Wrong supports share exact image IDs and labels; only pixel transform differs.

{
  "code_commit": "e09a4a8",
  "checkpoint_sha256": "260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb",
  "baseline": "/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10",
  "support_file_sha256": "855f12223d1164d14c3dd13835ac8a9a73522834c6c49b4228015a3afba2ae85",
  "split_file_sha256": "79b8b6d245ad6785208588c2687b4016be1bd4d3f9da475fca61610c3bc64296",
  "clients": 100,
  "seed": 7,
  "affine_optimizer": "SGD",
  "affine_lr": 0.1,
  "affine_steps": 10,
  "input_observed_range": [
    -1.0,
    1.0
  ],
  "formulas": "z=(x+1)/2; dark=.45*z; contrast=.35*(z-spatial_channel_mean)+mean; noise=clamp(z+.15*N(0,1),0,1); blur=Gaussian5 sigma1.5 reflect; output=2*z-1; noise_image=uniform[0,1]",
  "random_seed": "SHA256(7:client:original_sample_id:transform), first8 bytes little-endian modulo2^63-1; CPU torch.Generator",
  "gpu": "NVIDIA RTX A6000",
  "seconds": 115.9472451210022,
  "peak_gpu_gib": 0.0810694694519043,
  "smoke": false
}

[
  {
    "target": "brightness_dark",
    "correct_gain_pp": 34.410532811936875,
    "correct_minus_wrong_clean_pp": 7.7265776596843025,
    "correct_minus_wrong_alt_pp": 7.620187886544791,
    "correct_minus_noise_pp": 1.6423966314620557,
    "wrong_clean_paired": {
      "mean": 7.491107112262398,
      "median": 3.8711264729499817,
      "fraction_positive": 0.68,
      "p10": -1.1715123057365417,
      "p25": 0.0,
      "p75": 11.11111119389534,
      "p90": 23.676471337676062
    },
    "wrong_alt_paired": {
      "mean": 7.695869781542569,
      "median": 3.265152871608734,
      "fraction_positive": 0.65,
      "p10": -2.6010615378618236,
      "p25": 0.0,
      "p75": 13.775129243731499,
      "p90": 25.181760340929035
    },
    "checks": {
      "gain_ge_2": true,
      "clean_gap_ge_2": true,
      "alt_gap_ge_2": true,
      "beats_noise": true
    },
    "passed": true
  },
  {
    "target": "contrast_low",
    "correct_gain_pp": 35.20180880280591,
    "correct_minus_wrong_clean_pp": 2.43367297099649,
    "correct_minus_wrong_alt_pp": 1.2234857055822488,
    "correct_minus_noise_pp": -1.83522826556937,
    "wrong_clean_paired": {
      "mean": 2.76685138233006,
      "median": 1.9368290901184082,
      "fraction_positive": 0.59,
      "p10": -11.491101197898388,
      "p25": -2.156234532594681,
      "p75": 9.94914397597313,
      "p90": 17.65064612030983
    },
    "wrong_alt_paired": {
      "mean": 1.2187339905649424,
      "median": 0.0,
      "fraction_positive": 0.48,
      "p10": -7.45421439409256,
      "p25": -2.9924973845481873,
      "p75": 5.063290148973465,
      "p90": 11.240655183792116
    },
    "checks": {
      "gain_ge_2": true,
      "clean_gap_ge_2": true,
      "alt_gap_ge_2": false,
      "beats_noise": false
    },
    "passed": false
  },
  {
    "target": "gaussian_noise",
    "correct_gain_pp": 24.881973479739987,
    "correct_minus_wrong_clean_pp": 3.3047411035204064,
    "correct_minus_wrong_alt_pp": 8.637542322645125,
    "correct_minus_noise_pp": -0.9774586664757621,
    "wrong_clean_paired": {
      "mean": 3.251931020990014,
      "median": 2.547873556613922,
      "fraction_positive": 0.78,
      "p10": 0.0,
      "p25": 0.6047263741493225,
      "p75": 4.710514843463898,
      "p90": 6.973684132099154
    },
    "wrong_alt_paired": {
      "mean": 8.448877864517272,
      "median": 7.824458181858063,
      "fraction_positive": 0.81,
      "p10": -0.24260640144348128,
      "p25": 2.726859226822853,
      "p75": 12.99242451786995,
      "p90": 17.87499904632569
    },
    "checks": {
      "gain_ge_2": true,
      "clean_gap_ge_2": true,
      "alt_gap_ge_2": true,
      "beats_noise": false
    },
    "passed": false
  },
  {
    "target": "gaussian_blur",
    "correct_gain_pp": 26.045614729987598,
    "correct_minus_wrong_clean_pp": -1.1170956047242058,
    "correct_minus_wrong_alt_pp": 2.7395440639328683,
    "correct_minus_noise_pp": -3.2581952779313994,
    "wrong_clean_paired": {
      "mean": -1.2756877839565277,
      "median": 0.0,
      "fraction_positive": 0.38,
      "p10": -8.07909570634365,
      "p25": -2.9493097215890884,
      "p75": 1.2600794434547424,
      "p90": 3.202918767929078
    },
    "wrong_alt_paired": {
      "mean": 2.3454648349434137,
      "median": 1.2308746576309204,
      "fraction_positive": 0.6,
      "p10": -2.3287051916122437,
      "p25": -0.1096487045288086,
      "p75": 5.911768972873688,
      "p90": 7.883335798978806
    },
    "checks": {
      "gain_ge_2": true,
      "clean_gap_ge_2": false,
      "alt_gap_ge_2": true,
      "beats_noise": false
    },
    "passed": false
  }
]