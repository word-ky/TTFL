# T002 Prior-Controlled Covariate Context Specificity

Case A; 2/4 corruptions pass. SMOKE ONLY

| Corruption | None | Correct | Wrong clean | Wrong alt | Shuffled | Noise image | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| brightness_dark | 4.332 | 51.264 | 21.661 | 19.856 | 55.957 | 35.379 | True |
| contrast_low | 3.610 | 49.097 | 26.715 | 34.657 | 49.097 | 35.379 | True |
| gaussian_noise | 20.578 | 56.318 | 51.625 | 41.155 | 55.596 | 59.928 | False |
| gaussian_blur | 20.217 | 61.011 | 58.845 | 56.318 | 61.372 | 64.260 | False |

Accuracies are sample-weighted percentages. Macro-client rows and mean support losses/norms are in summary.csv. Paired client differences and percentile summaries are in per_client.csv and summary.json.

PASS requires correct gain >=2pp, both matched-label wrong gaps >=2pp, and correct > noise-image. At least2/4 required overall. No severity/optimizer selection on these query results.

Existing PFLlib static label-skew partitions, merged original labeled CIFAR10 splits with per-client75/25 split; not an official CIFAR10 test benchmark. All conditions within each target share identical corrupted queries. Wrong supports share exact image IDs and labels; only pixel transform differs.

{
  "code_commit": "e09a4a8",
  "checkpoint_sha256": "260657670e4b5beefebc7403487ead624a4ec7e19dd0958c7b281c28713cb9fb",
  "baseline": "/media/wenchang/F/wjq/TTFL/runs/20260913-123718-ttfl-pfl-gpu0/Cifar10",
  "support_file_sha256": "855f12223d1164d14c3dd13835ac8a9a73522834c6c49b4228015a3afba2ae85",
  "split_file_sha256": "79b8b6d245ad6785208588c2687b4016be1bd4d3f9da475fca61610c3bc64296",
  "clients": 2,
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
  "seconds": 2.3959615230560303,
  "peak_gpu_gib": 0.06163597106933594,
  "smoke": true
}

[
  {
    "target": "brightness_dark",
    "correct_gain_pp": 46.931407361254365,
    "correct_minus_wrong_clean_pp": 29.602887392689603,
    "correct_minus_wrong_alt_pp": 31.40794170031909,
    "correct_minus_noise_pp": 15.884474780585357,
    "wrong_clean_paired": {
      "mean": 29.30995300412178,
      "median": 29.30995300412178,
      "fraction_positive": 1.0,
      "p10": 28.488253504037857,
      "p25": 28.79639081656933,
      "p75": 29.823515191674232,
      "p90": 30.131652504205704
    },
    "wrong_alt_paired": {
      "mean": 30.26614412665367,
      "median": 30.26614412665367,
      "fraction_positive": 1.0,
      "p10": 27.06332966685295,
      "p25": 28.26438508927822,
      "p75": 32.26790316402912,
      "p90": 33.46895858645439
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
    "correct_gain_pp": 45.48736408601653,
    "correct_minus_wrong_clean_pp": 22.38267155545713,
    "correct_minus_wrong_alt_pp": 14.44043223393092,
    "correct_minus_noise_pp": 13.71840970180525,
    "wrong_clean_paired": {
      "mean": 21.89876288175583,
      "median": 21.89876288175583,
      "fraction_positive": 1.0,
      "p10": 20.541368424892426,
      "p25": 21.050391346216202,
      "p75": 22.747134417295456,
      "p90": 23.256157338619232
    },
    "wrong_alt_paired": {
      "mean": 14.149925112724304,
      "median": 14.149925112724304,
      "fraction_positive": 1.0,
      "p10": 13.335034251213074,
      "p25": 13.640618324279785,
      "p75": 14.659231901168823,
      "p90": 14.964815974235535
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
    "target": "gaussian_noise",
    "correct_gain_pp": 35.74007429478401,
    "correct_minus_wrong_clean_pp": 4.693144086465942,
    "correct_minus_wrong_alt_pp": 15.162456541285188,
    "correct_minus_noise_pp": -3.6101075716397375,
    "wrong_clean_paired": {
      "mean": 4.996597766876221,
      "median": 4.996597766876221,
      "fraction_positive": 1.0,
      "p10": 4.14539098739624,
      "p25": 4.464593529701233,
      "p75": 5.5286020040512085,
      "p90": 5.847804546356201
    },
    "wrong_alt_paired": {
      "mean": 15.160028636455536,
      "median": 15.160028636455536,
      "fraction_positive": 1.0,
      "p10": 15.1532182097435,
      "p25": 15.155772119760513,
      "p75": 15.164285153150558,
      "p90": 15.166839063167572
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
    "correct_gain_pp": 40.79422465784455,
    "correct_minus_wrong_clean_pp": 2.166063690874118,
    "correct_minus_wrong_alt_pp": 4.693143957358409,
    "correct_minus_noise_pp": -3.2490971501553574,
    "wrong_clean_paired": {
      "mean": 2.35784649848938,
      "median": 2.35784649848938,
      "fraction_positive": 1.0,
      "p10": 1.8198835849761963,
      "p25": 2.02161967754364,
      "p75": 2.6940733194351196,
      "p90": 2.8958094120025635
    },
    "wrong_alt_paired": {
      "mean": 3.8758397102355957,
      "median": 3.8758397102355957,
      "fraction_positive": 1.0,
      "p10": 1.583249568939209,
      "p25": 2.442970871925354,
      "p75": 5.308708548545837,
      "p90": 6.168429851531982
    },
    "checks": {
      "gain_ge_2": true,
      "clean_gap_ge_2": true,
      "alt_gap_ge_2": true,
      "beats_noise": false
    },
    "passed": false
  }
]