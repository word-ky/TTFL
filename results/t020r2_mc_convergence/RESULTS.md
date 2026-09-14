# T020R2 repair complete; sealed T020R-MC2 stop

2026-09-14T17:39:11+08:00. Lead `43d2f85`; runtime `1d7b6b8f7ca3e58fea1ae486cf7505192fdc05a1`; run `20260914-173004-ttfl-t020r2`. **124 remote tests PASS**. Two-client resume and full sealed aggregation completed in 44.66s. Exit1 is the prescribed failed-convergence stop.

## Frozen convergence decision

| Comparison | Agreement | Required | Result |
|---|---:|---:|---|
| Global H1/H2 | 98.8250% | 99% | FAIL |
| Global H1/ALL | 99.2500% | 99.5% | FAIL |
| Global H2/ALL | 99.5750% | 99.5% | PASS |

Every bank/context requires98.5%; every pooled context requires99%:

| Context | A H1/H2 % | B H1/H2 % | Pooled % |
|---|---:|---:|---:|
| clean | 99.250 | 98.250 | 98.750 |
| brightness_dark | 98.375 | 99.500 | 98.938 |
| contrast_low | 98.250 | 99.250 | 98.750 |
| gaussian_noise | 99.250 | 98.875 | 99.062 |
| gaussian_blur | 98.500 | 98.750 | 98.625 |


Three bank/context groups fail (A/Dark, A/Contrast, B/Clean); four pooled contexts fail (all except Noise). H1/H2 differ in **94/8000 slots across38clients/59cells**. Eight template slots share each posterior/bootstrap stream and are correlated. No missing client, imputed decision, rounded-equivalent state or altered threshold enters this result.

**T020R-MC2: persistent MC instability. BER-REGRET-A / BER-CAP-A / BER-SRC-A remain NOT_EXECUTED; no T020-R/C/F/X scientific diagnosis.** True composition, privileged utilities and query outcomes remain sealed. The complete R4096 policy fails the predefined numerical state-identity stability criterion. Its effect on true regret/query accuracy remains unknown; this is not a scientific failure conclusion about the neutral operators. No R increase, seed retry, alternate bootstrap, epsilon tie rule, confidence policy, feature work, SSL/TTT or federation was started.

## Label-free diagnostics

ALL4096 top-two utility-margin p05/median/p95: stable slots [0.002843656678191204, 0.038648361975486545, 0.2571704646331055]; unstable slots [7.043677896600487e-05, 0.0004677409619988223, 0.0021353811878742587] (fractional utility units). BER256→BER4096 changes 161/8000 slots. Quarter-pair agreements, max/p95 exact-mean utility differences, vote differences, disagreement IDs and the full transition matrix are preserved in the CSV files. These diagnostics do not change the policy.

## Cycle-only repair and reuse

Normal active-set arithmetic remains unchanged. Only a repeated working set invokes the existing deterministic1023-face direct-RHS enumeration, followed by the unchanged finite/simplex/KKT/objective checks. No added clipping, renormalization, ridge, relaxed tolerance or external optimizer. Receipts explicitly distinguish the paths.

Exactly **2 cycle fallbacks** occurred: client64/A/Contrast/replica3071 and client87/A/Contrast/replica2688. No additional cycle appeared. Every fallback trace, selected face, objective and KKT is listed in `solver_fallbacks.csv`/JSON. Independent re-enumeration reproduces both frozen-solver cycles and verifies each returned optimum; no other failure mode is converted to a fallback.

Reused98 client pairs byte-for-byte; computed only64/87. This run added **76,800 CLS solves**, retaining **3,763,200 prior extension solves** and the immutable **256,000 first-prefix replicas**. The assembled result has exactly100clients/1000cells/8000unique slots and4,096,000logical replicas; it does not claim4.096M new solves in this run. Max KKT 4.44089e-16, max sum error 4.44089e-16, max updates 17.

## Verification

Before resume:1000 real observations and4020 normal bootstrap samples yield bitwise-identical pi and unchanged normal receipt fields against the frozen implementation (apart from the explicit path flag), zero fallbacks, and8000 unchanged template decisions/argmax sets. Maximum stored cross-platform pi difference6.71456e-13 is below1e-12. Two observed-cycle fixtures, normal no-enumeration behavior and failed-certification stop behavior pass; full remote suite124testsPASS.

After full assembly: independent local replay checks **4000 CLS/position samples**, **8000 legacy aggregates**, **56000 exact block choices**, every final ALL4096 choice, quarter/half/ALL composition, all agreement rows and the unchanged convergence gate. Both cycle cases are independently re-enumerated and the recorded selected faces reconstructed from C/q. All **553 upstream manifest entries** remain unchanged. Local entropy alone uses2e-15 absolute tolerance for the observed Windows/Linux NumPy log last-bit difference; utility means, states and ties remain exact. Zero new model forwards and no true-regret/query-count calculation.

`phaseB_choices_freeze.json` binds all complete unlabeled outputs with `convergence_pass=false`, not permission to unseal. `preserved_98_verification.json` binds reused work; `resume_receipt.json` binds the two new clients and completeness. Exact accumulators and sampled NPZ files are retained locally/remotely under the new run. Original T020/T020R reports and raw receipts remain unchanged.

Local D: space exhaustion interrupted the first artifact extraction, not the experiment. The downloaded archive matched remote SHA256 `aae4ebed93f1ed54559ad0ae84a4820806f34aef7369168867228d2decb038db`. The redundant prior delivery tar was removed, the same archive extracted successfully, and its local transport copy removed after successful extraction. All resulting scientific files passed the checks above; no experiment was restarted.

Return this sealed MC2 result to Lead. The authorized solver repair is complete; the frozen scientific evaluation cannot proceed past the convergence stop.
