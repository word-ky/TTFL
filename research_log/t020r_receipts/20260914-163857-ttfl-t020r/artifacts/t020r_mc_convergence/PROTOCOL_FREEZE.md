# T020R authorized MC extension

Lead `f0807dd`. Final R is exactly4096; K20 and all scientific inputs/estimator/tolerances unchanged. Retain T020's exact seed namespace `T020|client|bank|context|replica`, full big-endian SHA256 → PCG64. Reuse the immutable first256 q/pi/position/argmax arrays and reproduce all first128/all256 aggregate receipts. Append replicas256..4095; no seed retry or new namespace.

Blocks: Q0=[0,1024), Q1=[1024,2048), Q2=[2048,3072), Q3=[3072,4096), H1=[0,2048), H2=[2048,4096), ALL=[0,4096). Exact accumulated rational utility means determine every block's argmax in the historical candidate order. Vote mode is diagnostic only.

Pass only if: global H1/H2>=99%; global H1/ALL and H2/ALL>=99.5%; each bank×context H1/H2>=98.5%; each context pooled over banks H1/H2>=99%. No threshold adjustment. Failure is T020R-MC2, not a scientific BER diagnosis; no labels/true utilities/query metrics are unsealed on failure.

Use four CPU processes on independent clients, each with the existing two BLAS threads. This affects throughput only; seeds and per-channel operations do not depend on scheduling. Stream one client at a time per process, preserving exact block numerator sums/denominators, mean-max numerator sums, vote counts, final uncertainty, per-cell stream hashes, and samples at replicas256/1024/2048/4095. Existing first256 arrays remain the full prefix archive. Large all-replica text is not generated.

The preparation script only reads allowed posterior and label-free point/BER members, soft channels and frozen utility templates. Cryptographic hashing of upstream bytes is provenance only. No support-label/class-stratified draw, new posterior estimator, feature experiment, writer, state capacity, context detector or FL is introduced.
