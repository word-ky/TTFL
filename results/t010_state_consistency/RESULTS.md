# CODEX -> CHATGPT

## Timestamp

2026-09-13T23:42:52+08:00

## Commit / run ID

Lead 50f324a; runtime 1feaec9bd809434ca15cf20d862f46d77de4f9a2; run 20260913-233408-ttfl-t010-gpu1; A6000 GPU1; 32.85s evaluation. Enclosing delivery commit contains this report.

## Frozen evidence reused

Same CIFAR-10 PFLlib checkpoint, 192-scalar operator, neutral writer, T007R states/pools, T008 references/prototypes, corruption code, historical source_signature, and exact T009 K20 support manifest. All frozen hashes match. 100 clients; no new FL training. This is the historical PFLlib merged-data per-client query split, not the official CIFAR-10 test benchmark.

## New state-consistency score implementation

Score all five frozen candidates including zero by J=||psi||² after sequential affine-corrected blocks, relative to the bank clean moments. Block 2 receives corrected block 1. Fixed argmin order; no labels, gradients, fitting, thresholds or query tuning. The same K20 IDs serve every candidate/context/bank.

## Sanity/regression verification

43 tests PASS. 5,240 candidate scores; 1,048 exact zero-signature/J comparisons, max error 0. Model and states unchanged after every episode. Choices frozen 2026-09-13T15:34:50.884256+00:00, query predictions frozen 2026-09-13T15:34:51.282613+00:00. 985 query policies reused from T009, 14 from T007R, 1 new evaluation. All historical metrics reproduce. Independent saved-prediction integer counts, Fraction gates, decisions and alignment checks pass; macro-class error 7.11e-15 pp.

## Balanced held-out sanity

20/20 correct in each bank; both confusion matrices are diagonal with four examples per context.

## Natural-client state selection

| bank | correct | accuracy | median_true_rank | fraction_true_rank_le2 | passed |
| --- | --- | --- | --- | --- | --- |
| A | 489 | 97.800000 | 1.000000 | 1.000000 | True |
| B | 490 | 98.000000 | 1.000000 | 1.000000 | True |

## Confusion matrices / hard-pair audit

Bank A:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 95 | 0 | 0 | 1 | 4 |
| brightness_dark | 0 | 100 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 98 | 0 | 2 |
| gaussian_noise | 0 | 0 | 0 | 100 | 0 |
| gaussian_blur | 1 | 0 | 3 | 0 | 96 |

Bank B:

| true / selected | clean | brightness_dark | contrast_low | gaussian_noise | gaussian_blur |
| --- | --- | --- | --- | --- | --- |
| clean | 95 | 0 | 0 | 1 | 4 |
| brightness_dark | 0 | 100 | 0 | 0 | 0 |
| contrast_low | 0 | 0 | 99 | 0 | 1 |
| gaussian_noise | 0 | 0 | 0 | 100 | 0 |
| gaussian_blur | 1 | 0 | 3 | 0 | 96 |

| bank | clean_nonzero | blur_zero | blur_contrast |
| --- | --- | --- | --- |
| A | 5 | 1 | 3 |
| B | 5 | 1 | 3 |

| bank | target | corrected | preserved_errors | new_errors | changed_wrong_to_wrong |
| --- | --- | --- | --- | --- | --- |
| A | clean | 3 | 5 | 0 | 0 |
| A | brightness_dark | 0 | 0 | 0 | 0 |
| A | contrast_low | 0 | 0 | 2 | 0 |
| A | gaussian_noise | 0 | 0 | 0 | 0 |
| A | gaussian_blur | 1 | 3 | 1 | 0 |
| A | all | 4 | 8 | 3 | 0 |
| B | clean | 3 | 4 | 1 | 0 |
| B | brightness_dark | 0 | 0 | 0 | 0 |
| B | contrast_low | 0 | 0 | 1 | 0 |
| B | gaussian_noise | 0 | 0 | 0 | 0 |
| B | gaussian_blur | 1 | 2 | 2 | 0 |
| B | all | 4 | 6 | 4 | 0 |

## Per-client mixed retrieval

| bank | target | none | oracle | selected_mixed | wrong_alt | tau_pp | retained_recovery | selected_wrong_margin | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | brightness_dark | 17.546026 | 32.296536 | 32.296536 | 14.903865 | 4.148166 | 1.000000 | 17.392670 | True |
| A | contrast_low | 19.955264 | 28.561473 | 28.291025 | 22.733171 | 3.545856 | 0.968575 | 5.557853 | True |
| A | gaussian_noise | 30.676644 | 32.499558 | 32.499558 | 30.382434 | 0.865512 | 1.000000 | 2.117123 | True |
| A | gaussian_blur | 30.901579 | 33.048500 | 31.768359 | 29.798137 | 0.809278 | 0.403732 | 1.970222 | False |
| B | brightness_dark | 17.546026 | 32.424808 | 32.424808 | 15.489587 | 4.148166 | 1.000000 | 16.935221 | True |
| B | contrast_low | 19.955264 | 28.111626 | 28.025874 | 22.693556 | 3.545856 | 0.989486 | 5.332317 | True |
| B | gaussian_noise | 30.676644 | 32.406771 | 32.406771 | 30.537680 | 0.865512 | 1.000000 | 1.869091 | True |
| B | gaussian_blur | 30.901579 | 33.009919 | 31.736208 | 28.354235 | 0.809278 | 0.395870 | 3.381973 | False |

Dark, Contrast and Noise pass jointly. Blur fails in both banks: only 40.37% / 39.59% of oracle gain retained. Bank-B Blur is worse than T009 (53.89%); this selector has not solved Blur.

## Clean safety

| bank | none | selected | delta_pp | passed |
| --- | --- | --- | --- | --- |
| A | 34.138690 | 33.698442 | -0.440248 | True |
| B | 34.138690 | 33.725322 | -0.413368 | True |

Both losses fall within the predeclared 0.5 pp allowance, improving from T009 losses 0.610321 / 0.616185 pp. Five false clean adaptations remain in each bank; passing this gate does not mean zero harm.

## Restoration-vs-task audit

| bank | role | n | pearson | spearman | positive_J | harmful | harmful_fraction_all | harmful_fraction_positive_J | median_delta_accuracy_positive_J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | selected | 500 | 0.325681 | 0.386188 | 404 | 89 | 0.178000 | 0.220297 | 2.777778 |
| A | true | 500 | 0.315410 | 0.384153 | 399 | 83 | 0.166000 | 0.208020 | 3.061224 |
| B | selected | 500 | 0.317014 | 0.380842 | 404 | 92 | 0.184000 | 0.227723 | 2.795139 |
| B | true | 500 | 0.306415 | 0.376001 | 399 | 87 | 0.174000 | 0.218045 | 2.941176 |

Among selected states with positive DeltaJ, 22.03% / 22.77% still reduce query accuracy. Pearson correlations are only 0.326 / 0.317. Thus lower moment residual is not a reliable per-client guarantee of task benefit. All quantities are per-client percentage-point accuracy changes versus zero. Spearman uses average ranks; constant groups have null correlation. Full per-bank/context statistics and the ten largest positive-J/negative-accuracy disagreements for each role are in restoration_task_audit.json. Ranking was fixed before evaluation: descending DeltaJ, then DeltaAcc/client. These descriptive correlations do not establish that source moment restoration causes task improvement.

## Label-skew audit

| bank | quartile | target | n | accuracy | median_score_margin |
| --- | --- | --- | --- | --- | --- |
| A | 1 | clean | 25 | 88.000000 | 11.479177 |
| A | 1 | gaussian_blur | 25 | 88.000000 | 12.244904 |
| A | 1 | all | 125 | 93.600000 | 19.224369 |
| A | 2 | clean | 25 | 100.000000 | 11.306936 |
| A | 2 | gaussian_blur | 25 | 100.000000 | 15.380751 |
| A | 2 | all | 125 | 100.000000 | 21.361560 |
| A | 3 | clean | 25 | 92.000000 | 11.927444 |
| A | 3 | gaussian_blur | 25 | 96.000000 | 16.046762 |
| A | 3 | all | 125 | 97.600000 | 20.949193 |
| A | 4 | clean | 25 | 100.000000 | 13.379987 |
| A | 4 | gaussian_blur | 25 | 100.000000 | 13.927621 |
| A | 4 | all | 125 | 100.000000 | 20.271257 |
| B | 1 | clean | 25 | 88.000000 | 11.809623 |
| B | 1 | gaussian_blur | 25 | 88.000000 | 11.410332 |
| B | 1 | all | 125 | 94.400000 | 19.917486 |
| B | 2 | clean | 25 | 100.000000 | 11.645819 |
| B | 2 | gaussian_blur | 25 | 100.000000 | 15.792180 |
| B | 2 | all | 125 | 100.000000 | 22.546059 |
| B | 3 | clean | 25 | 92.000000 | 10.749763 |
| B | 3 | gaussian_blur | 25 | 96.000000 | 16.617743 |
| B | 3 | all | 125 | 97.600000 | 19.421815 |
| B | 4 | clean | 25 | 100.000000 | 14.437471 |
| B | 4 | gaussian_blur | 25 | 100.000000 | 13.837714 |
| B | 4 | all | 125 | 100.000000 | 21.447130 |

Frozen T009 quartiles and label statistics reused only after selection/query freezing. Worst clean/blur errors are listed in worst_clean_blur_errors.json by J_true-J_best, with entropy and max-class fraction. The top eight rows all involve single-class support clients 15 and 29 (entropy 0, max-class fraction 1). Both banks still map client 15 Blur to Contrast and client 29 Blur to zero. The low-entropy tail failure remains; changed selection names do not prove semantic-content confounding was removed.

| bank | client | true_context | selected | entropy | max_class_fraction | true_excess |
| --- | --- | --- | --- | --- | --- | --- |
| A | 15 | gaussian_blur | contrast_low | -0.000000 | 1.000000 | 16.482286 |
| B | 15 | gaussian_blur | contrast_low | -0.000000 | 1.000000 | 12.697292 |
| B | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | 8.583462 |
| B | 29 | clean | gaussian_noise | -0.000000 | 1.000000 | 8.401272 |
| B | 15 | clean | gaussian_blur | -0.000000 | 1.000000 | 7.927474 |
| A | 15 | clean | gaussian_blur | -0.000000 | 1.000000 | 7.650946 |
| A | 29 | clean | gaussian_noise | -0.000000 | 1.000000 | 5.592321 |
| A | 29 | gaussian_blur | clean | -0.000000 | 1.000000 | 4.783706 |
| A | 30 | clean | gaussian_blur | 0.371621 | 0.500000 | 4.296095 |
| B | 30 | clean | gaussian_blur | 0.371621 | 0.500000 | 3.665073 |

## OOD descriptive audit

[
  {
    "bank": "A",
    "histogram": {
      "clean": 0,
      "brightness_dark": 0,
      "contrast_low": 0,
      "gaussian_noise": 4,
      "gaussian_blur": 0
    },
    "fraction_zero": 0.0,
    "J_clean": {
      "min": 150.97126154410273,
      "median": 152.77289503351116,
      "mean": 152.75427153322767,
      "max": 154.50003452178552
    },
    "best_score": {
      "min": 82.50088773660539,
      "median": 82.93415970674423,
      "mean": 82.94365423852697,
      "max": 83.40540980401406
    },
    "zero_improvement": {
      "min": 68.47037380749734,
      "median": 69.83873532676694,
      "mean": 69.81061729470068,
      "max": 71.09462471777147
    }
  },
  {
    "bank": "B",
    "histogram": {
      "clean": 0,
      "brightness_dark": 0,
      "contrast_low": 0,
      "gaussian_noise": 4,
      "gaussian_blur": 0
    },
    "fraction_zero": 0.0,
    "J_clean": {
      "min": 154.4515232526435,
      "median": 158.3173976891089,
      "mean": 157.5887047931349,
      "max": 159.26850054167824
    },
    "best_score": {
      "min": 82.99239944326047,
      "median": 87.46360303082042,
      "mean": 86.57328633227363,
      "max": 88.37353982419319
    },
    "zero_improvement": {
      "min": 70.18590961905309,
      "median": 71.1038007926893,
      "mean": 71.01541846086127,
      "max": 71.66816263901336
    }
  }
]

All eight random-noise supports select the Gaussian-noise state; none selects zero. Thus including zero does not provide OOD rejection. No threshold was introduced.

## Failures / uncertainties

Clean tail errors and Blur retention failure persist despite the joint gate passing. Moment/task disagreements remain in the saved audit. Results cover fixed known corruption severities and two existing state banks; no unseen-severity, continuous-state, or open-set success is established.

## SC-ID / SC-RET / Case decision

**SC-ID-A PASS; SC-RET-A PASS; primary case SC-A** under the explicit conjunction rule: 3/4 joint shifted targets plus clean safety in both banks. The instruction’s SC-B description overlaps: clean passes while Blur fails. Preserve that residual rather than claiming complete safe retrieval or changing the original gate.

## Recommended next action

Return to Research Lead for adjudication, explicitly carrying forward clean tail harm and failed Blur retention. A bounded next package could examine the proposed clean↔shift interpolation/unseen severities and OOD rejection separately. No next-stage implementation, SSL, meta-learning, new federation or operator expansion was started.
