# T022 raw neutral-state response

Lead06647a0. Reuse all1807T021 cached paths byte-for-byte and extract only missing paths of the100client×2bank×5true-context×5state×20support grid. Use GPU1, identical model/extractor/preprocessing and states. No query forward. Any historical mismatch stops T022-I.

Phi_L40/Phi_H2048 are float32 concatenations of [Dark−clean,Contrast−clean,Noise−clean,Blur−clean], within the same sample/bank/true corruption. No block scaling, normalization, whitening or learned transformation. Target signature is identical across oracle/source/clean-prototype policies. Only prototype context and utility-template context differ.

Reuse T021 target-excluded class-mean/Gram-CLS/exact-choice machinery with explicit response mode. Oracle prototypes are context matched; source prototypes follow frozen T009 IDs; control prototypes are from clean signatures. Other-client labels calibrate each target, never its own labels. Freeze all6000 actual solutions/48000choices before target composition scoring.

Matched R128 preserves exactK20counts; full SHA256(`T022|matched|L-or-H|client|bank|context|replica`) big-endian PCG64. Seed representation names are L and H (base representation), not Phi_L/Phi_H. Reuse T018/T021 median80%capture gate, real shifted safety versusBBSE-S-01−0.5pp and conditional source clean safety versuszero−0.5pp. Regret/context gates unchanged15%mean/5%p90 acrossbothbanks/allsalts>=3/4shifts. Compare also against T021 absoluteL/H without a new gate.

No BER/SSL/gradient/head/metric/newstate/context detector/FL. If T022-M/N, stop frozen first-moment observer variants and return to Lead. No autonomous T023. Large arrays stay in remote project runs with committed compact manifests.
