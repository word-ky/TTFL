# T001 complete grid

One backbone seed, five support draws. Exploratory same-query grid; independent confirmation not performed.

| Method | Context | LR | Steps | Acc mean ¡À SD (%) | Gain pp |
|---|---|---:|---:|---:|---:|
| none | none | 0.0 | 0 | 18.000 ¡À 0.000 | +0.000 |
| random_no_write | random | 0.0 | 0 | 19.000 ¡À 0.000 | +1.000 |
| bn | correct | 0.0 | 0 | 20.000 ¡À 0.000 | +2.000 |
| prior | correct | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | correct | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| bn | wrong_MNIST | 0.0 | 0 | 13.000 ¡À 0.000 | -5.000 |
| prior | wrong_MNIST | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | wrong_MNIST | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| bn | wrong_USPS | 0.0 | 0 | 18.000 ¡À 0.000 | +0.000 |
| prior | wrong_USPS | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | wrong_USPS | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| bn | wrong_MNISTM | 0.0 | 0 | 18.000 ¡À 0.000 | +0.000 |
| prior | wrong_MNISTM | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | wrong_MNISTM | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| bn | shuffled | 0.0 | 0 | 20.000 ¡À 0.000 | +2.000 |
| prior | shuffled | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | shuffled | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| bn | noise | 0.0 | 0 | 17.000 ¡À 0.000 | -1.000 |
| prior | noise | 0.0 | 0 | 8.000 ¡À 0.000 | -10.000 |
| affine | noise | 0.01 | 1 | 18.000 ¡À 0.000 | +0.000 |
| full | correct | 0.001 | 1 | 18.000 ¡À 0.000 | +0.000 |

Exploratory gate: False
