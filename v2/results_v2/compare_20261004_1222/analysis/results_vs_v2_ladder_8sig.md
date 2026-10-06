# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/compare_20261004_1222` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs v2_ladder_8sig per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_ladder | 114.3 ± 0.0 | 124.2 | 74.4 ± 0.0 | [70, 78] | 100.0 int8 | 146.0 | 74.7 | faster than INT8 | 21/49, p=0.00 |
| big_ladder | 203.0 ± 0.0 | 253.3 | 78.8 ± 0.0 | [75, 82] | 100.0 int8 | 242.4 | 143.7 | faster than INT8 | 24/31, p=0.42 |
| eightsignal_ladder | 131.0 ± 0.0 | 232.6 | 74.2 ± 0.0 | [70, 78] | 100.0 int8 | 163.2 | 86.6 | faster than INT8 | 21/50, p=0.00 |
| lite1_ladder | 162.9 ± 0.0 | 191.2 | 82.1 ± 0.0 | [78, 85] | 100.0 int8 | 195.3 | 108.4 | faster than INT8 | 36/27, p=0.31 |
| egreedy_ladder | 112.9 ± 0.0 | 173.5 | 75.2 ± 0.0 | [71, 79] | 100.0 int8 | 132.3 | 71.5 | faster than INT8 | 11/35, p=0.00 |
| linucb_ladder | 144.1 ± 0.0 | 241.8 | 74.8 ± 0.0 | [71, 78] | 100.0 int8 | 179.1 | 97.9 | faster than INT8 | 20/46, p=0.00 |
| v2_ladder_8sig | 152.6 ± 0.0 | 352.6 | 80.2 ± 0.0 | [76, 84] | 25.8 esc, 100.0 int8 | 180.7 | 111.9 | faster than INT8 |  |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.