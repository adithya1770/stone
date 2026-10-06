# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/pi_20261002_1807/suite` — 3 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | INT8 % / escalated % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs Baseline (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 129.1 ± 4.0 | 139.3 | 75.0 ± 0.0 | [63, 84] | 0.0 int8 | 148.4 | 77.9 | 75.0 |  |
| always_int8 | 91.4 ± 0.8 | 96.4 | 68.3 ± 0.0 | [56, 79] | 100.0 int8 | 107.4 | 54.5 | 68.3 | 0/12, p=0.00 |
| linucb_cold | 109.7 ± 4.8 | 137.1 | 71.1 ± 1.9 | [59, 81] | 52.2 int8 | 130.4 | 67.7 | 71.6 | 0/7, p=0.02 |
| eightsignal_cold | 100.1 ± 1.6 | 134.1 | 69.4 ± 1.0 | [57, 80] | 78.9 int8 | 119.0 | 61.4 | 69.9 | 0/10, p=0.00 |
| v2_cascade | 91.5 ± 0.3 | 96.0 | 68.3 ± 0.0 | [56, 79] | 0.0 esc | 107.5 | 54.8 | 68.3 | 0/12, p=0.00 |
| v2_resource_aware | 41.8 ± 0.8 | 92.2 | 30.0 ± 0.0 | [20, 43] | 0.0 esc | 53.9 | 27.2 | faster than INT8 | 0/81, p=0.00 |
| baseline_fixed | 130.5 ± 0.8 | 140.9 | 80.0 ± 0.0 | [68, 88] | 0.0 int8 | 151.2 | 79.0 | 75.0 | 12/3, p=0.04 |
| v2_resource_aware_fixed | 38.9 ± 2.0 | 129.7 | 31.1 ± 1.0 | [21, 44] | 6.1 esc | 52.1 | 28.5 | faster than INT8 | 0/79, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.