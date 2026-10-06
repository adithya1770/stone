# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/final_20261003_1029_protocol` — 3 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_cold per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 237.6 ± 4.6 | 316.9 | 75.0 ± 0.0 | [63, 84] | 0.0 int8 | 285.4 | 193.1 | 75.0 | 3/0, p=0.25 |
| always_int8 | 196.5 ± 4.5 | 265.8 | 68.3 ± 0.0 | [56, 79] | 100.0 int8 | 234.4 | 144.9 | 68.3 | 0/1, p=1.00 |
| linucb_cold | 219.6 ± 1.6 | 277.9 | 72.8 ± 1.0 | [61, 83] | 47.2 int8 | 267.3 | 175.2 | 72.1 | 3/1, p=0.62 |
| eightsignal_cold | 210.5 ± 5.7 | 268.6 | 70.0 ± 0.0 | [57, 80] | 63.3 int8 | 256.1 | 166.0 | 70.6 |  |
| always_int8_rebuilt | 194.6 ± 3.7 | 270.0 | 83.3 ± 0.0 | [72, 91] | 100.0 int8 | 231.8 | 144.5 | faster than INT8 | 8/0, p=0.01 |
| baseline_rebuilt | 264.2 ± 1.4 | 363.5 | 78.3 ± 0.0 | [66, 87] | 0.0 int8 | 310.6 | 210.9 | 75.0 | 6/1, p=0.12 |
| eightsignal_rebuilt | 216.8 ± 5.9 | 283.1 | 80.6 ± 1.0 | [68, 88] | 67.8 int8 | 262.4 | 170.2 | 71.6 | 7/1, p=0.07 |
| v2_final | 198.0 ± 6.1 | 269.9 | 83.3 ± 0.0 | [72, 91] | 1.7 esc, 98.3 int8 | 240.8 | 149.8 | 68.6 | 8/0, p=0.01 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.