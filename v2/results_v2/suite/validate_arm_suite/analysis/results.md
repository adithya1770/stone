# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/validate_arm_suite` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | INT8 % / escalated % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs always_int8 per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 12.3 ± 0.0 | 14.3 | 11.7 ± 0.0 | [6, 22] | 0.0 int8 | 16.4 | 7.8 | 11.7 | 3/4, p=1.00 |
| always_int8 | 6.5 ± 0.0 | 8.5 | 13.3 ± 0.0 | [7, 24] | 100.0 int8 | 9.8 | 4.2 | 13.3 |  |
| linucb_cold | 8.9 ± 0.0 | 13.9 | 11.7 ± 0.0 | [6, 22] | 61.7 int8 | 24.0 | 6.5 | 12.6 | 1/2, p=1.00 |
| eightsignal_cold | 9.1 ± 0.0 | 14.3 | 11.7 ± 0.0 | [6, 22] | 58.3 int8 | 22.2 | 6.7 | 12.6 | 0/1, p=1.00 |
| v2_cascade | 5.6 ± 0.0 | 9.0 | 11.7 ± 0.0 | [6, 22] | 71.7 esc | 9.2 | 4.3 | faster than INT8 | 4/5, p=1.00 |
| v2_resource_aware | 5.3 ± 0.0 | 9.6 | 15.0 ± 0.0 | [8, 26] | 50.0 esc | 10.2 | 3.7 | faster than INT8 | 5/4, p=1.00 |

## Original results (`results/stone_pi_multitrial2`, recomputed with the same code)

| config | stressed lat (ms) | stressed acc (%) | INT8 % |
|---|---|---|---|
| baseline | 130.4 ± 4.5 | 75.0 ± 0.0 | 0.0 ± 0.0 |
| always_int8 | 92.3 ± 2.8 | 68.3 ± 0.0 | 100.0 ± 0.0 |
| linucb_cold | 112.7 ± 3.1 | 70.6 ± 1.0 | 47.8 ± 1.0 |
| linucb_warm | 110.9 ± 5.4 | 71.7 ± 0.0 | 52.8 ± 5.4 |
| egreedy | 129.6 ± 4.9 | 74.4 ± 1.0 | 6.1 ± 1.9 |
| eightsignal_cold | 102.9 ± 0.7 | 70.0 ± 0.0 | 72.2 ± 4.2 |
| eightsignal_warm | 103.7 ± 3.0 | 70.0 ± 0.0 | 73.3 ± 1.7 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.