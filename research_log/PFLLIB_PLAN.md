# Direct-user benchmark migration, 2026-09-13

The user now requests PFLlib, MNIST/CIFAR-10/CIFAR-100/TinyImageNet,100clients,
10% participation and GPU execution. This direct instruction authorizes FL setup
despite the older T001 stop directive. T001 remains FAIL. Lead commit5647af7
assigned T001B on the old SVHN checkpoint; retain that assignment but do not launch
it alongside this user-directed migration. No SSL/meta-learning authorized.

Default settings stated to user: Dirichlet alpha.1,100rounds,1local epoch. Other
settings reuse PFLlib defaults: CNN,SGD.005,batch10,75/25 client split. Use upstream
dataset utilities to partition merged labeled data, as PFLlib standard generators
do; this is not the official held-out image benchmark split. TinyImageNet train
and labeled val are merged, unlabelled test excluded. Save IDs and class counts.
One seed7 initial benchmark, no tuning. Evaluate all100clients every20rounds and
after final aggregation. Only10selected clients train per round, no random ratio.

Reuse map: unmodified official FedAvg server/client/model/reader; unmodified
separate_data/split_data applied to sample IDs to avoid full-dataset float copies;
small entrypoint for args, provenance, final state_dict, JSON and sampling receipt.
Train baseline first. Then reuse T001 supervised affine update with two spatial
PFLlib CNN blocks, zero initialization, frozen backbone, fixed.1/10steps. Test
correct client support, cyclic wrong client, shuffled labels, noise, random/no-write
and prior on identical final checkpoint. No BN in this exact PFLlib CNN, report N/A.
This is a static label-skew client context diagnostic, not a demonstrated dynamic
context or domain-shift improvement; no claim that client identity is disentangled.

Baseline health: exercise upstream FedAvg on synthetic100clients/10selected/2rounds
on A6000 before method integration. Then test neutral PFLlib operator and frozen
updates. Real-data2roundsmokes on each dataset before fixed100roundruns. Report
smokes separately. Keep every executed result; no artificial acceptance gate.

Remote code root remains /home/wenchang/asdasdsad/wjq/TTFL. Large data/outputs:
/media/wenchang/F/wjq/TTFL (18TB free versus home19GB). Use both A6000 GPUs if free,
separate dataset jobs; data prep is CPU/I/O, actual optimization/inference is CUDA.
