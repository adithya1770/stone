# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/pi_run2` — 3 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | INT8 % / escalated % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs baseline per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 131.5 ± 1.7 | 140.4 | 75.0 ± 0.0 | [63, 84] | 0.0 int8 | 149.9 | 78.1 | 75.0 |  |
| always_int8 | 91.7 ± 1.8 | 96.8 | 68.3 ± 0.0 | [56, 79] | 100.0 int8 | 107.5 | 55.3 | 68.3 | 0/4, p=0.12 |
| linucb_cold | 109.3 ± 3.7 | 138.0 | 70.0 ± 0.0 | [57, 80] | 55.0 int8 | 130.0 | 66.9 | 71.3 | 0/3, p=0.25 |
| eightsignal_cold | 101.8 ± 0.2 | 135.0 | 71.7 ± 0.0 | [59, 81] | 74.4 int8 | 121.1 | 62.2 | 70.0 | 0/2, p=0.50 |
| v2_cascade | 90.8 ± 1.1 | 95.3 | 68.3 ± 0.0 | [56, 79] | 0.0 esc | 106.8 | 53.9 | faster than INT8 | 0/4, p=0.12 |
| v2_resource_aware | 91.2 ± 1.6 | 95.5 | 68.3 ± 0.0 | [56, 79] | 0.0 esc | 107.1 | 54.9 | faster than INT8 | 0/4, p=0.12 |
| baseline_fixed | 132.4 ± 2.2 | 141.9 | 80.0 ± 0.0 | [68, 88] | 0.0 int8 | 153.3 | 79.3 | 75.0 | 4/1, p=0.38 |
| v2_resource_aware_fixed | 86.0 ± 0.7 | 176.0 | 75.0 ± 0.0 | [63, 84] | 17.2 esc | 104.7 | 56.6 | faster than INT8 | 3/3, p=1.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.