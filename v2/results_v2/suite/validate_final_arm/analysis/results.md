# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/validate_final_arm` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_rebuilt per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| eightsignal_cold | 8.4 ± 0.0 | 18.6 | 0.0 ± 0.0 | [-0, 20] | 66.7 int8 | 21.3 | 6.0 | faster than INT8 | 0/1, p=1.00 |
| always_int8_rebuilt | 6.5 ± 0.0 | 12.0 | 6.7 ± 0.0 | [1, 30] | 100.0 int8 | 10.8 | 2.0 | faster than INT8 | 0/0, p=1.00 |
| eightsignal_rebuilt | 14.5 ± 0.0 | 47.6 | 6.7 ± 0.0 | [1, 30] | 60.0 int8 | 34.5 | 10.0 | faster than INT8 |  |
| v2_final | 22.7 ± 0.0 | 31.9 | 26.7 ± 0.0 | [11, 52] | 93.3 esc, 6.7 int8 | 35.1 | 12.7 | faster than INT8 | 3/0, p=0.25 |

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