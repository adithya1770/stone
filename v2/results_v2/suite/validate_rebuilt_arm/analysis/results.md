# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/validate_rebuilt_arm` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs always_int8_rebuilt per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| always_int8 | 5.9 ± 0.0 | 8.2 | 0.0 ± 0.0 | [0, 24] | 100.0 int8 | 10.2 | 5.0 | faster than INT8 | 0/0, p=1.00 |
| always_int8_rebuilt | 5.8 ± 0.0 | 8.3 | 0.0 ± 0.0 | [0, 24] | 100.0 int8 | 9.8 | 5.8 | faster than INT8 |  |
| baseline_rebuilt | 18.8 ± 0.0 | 22.8 | 16.7 ± 0.0 | [5, 45] | 0.0 int8 | 22.6 | 10.0 | faster than INT8 | 2/0, p=0.50 |
| v2_rebuilt | 7.4 ± 0.0 | 16.9 | 33.3 ± 0.0 | [14, 61] | 50.0 esc, 50.0 int8 | 11.9 | 6.7 | faster than INT8 | 4/0, p=0.12 |

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