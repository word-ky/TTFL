# PFLlib 100 clients / 10 participants

One seed; Dirichlet alpha=0.1; 100 rounds; one local epoch; upstream CNN/SGD. Static-client supervised context diagnostic, not dynamic-context evidence.

| Dataset | FedAvg | Affine correct | Gain pp | Wrong client | Correct-wrong pp | Shuffled | Prior correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 92.50 | 94.76 | +2.26 | 92.02 | +2.74 | 94.69 | 98.24 |
| Cifar10 | 34.22 | 57.28 | +23.06 | 25.83 | +31.44 | 58.05 | 79.19 |
| Cifar100 | 11.72 | 13.01 | +1.29 | 11.32 | +1.69 | 13.33 | 36.59 |
| TinyImagenet | 9.64 | 10.43 | +0.78 | 8.82 | +1.60 | 8.76 | 24.96 |

Accuracy is sample-weighted across all100clients; all models run on CUDA. BN N/A because upstream CNN has no BN. Prior uses Laplace1 support counts against uniform classes. Standard PFLlib merges original labeled splits and repartitions75/25, so these are not official held-out image benchmark accuracies.
