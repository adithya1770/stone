# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/final_20261003_1218_protocol` — 3 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs baseline per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 127.6 ± 2.4 | 135.1 | 75.0 ± 0.0 | [63, 84] | 0.0 int8 | 146.7 | 75.7 | 75.0 |  |
| always_int8 | 89.7 ± 0.6 | 93.7 | 68.3 ± 0.0 | [56, 79] | 100.0 int8 | 105.1 | 53.9 | 68.3 | 0/4, p=0.12 |
| linucb_cold | 108.1 ± 4.2 | 134.1 | 71.1 ± 1.0 | [59, 81] | 52.8 int8 | 128.5 | 66.4 | 71.6 | 0/3, p=0.25 |
| eightsignal_cold | 100.4 ± 0.7 | 132.5 | 69.4 ± 1.0 | [57, 80] | 72.8 int8 | 119.4 | 61.4 | 70.2 | 0/3, p=0.25 |
| always_int8_rebuilt | 90.2 ± 0.3 | 94.1 | 83.3 ± 0.0 | [72, 91] | 100.0 int8 | 105.8 | 54.3 | 68.4 | 6/1, p=0.12 |
| baseline_rebuilt | 147.1 ± 1.1 | 154.6 | 78.3 ± 0.0 | [66, 87] | 0.0 int8 | 164.7 | 84.7 | 75.0 | 4/2, p=0.69 |
| eightsignal_rebuilt | 106.9 ± 1.4 | 151.2 | 82.2 ± 1.0 | [70, 89] | 67.8 int8 | 126.2 | 66.1 | 71.4 | 6/1, p=0.12 |
| v2_final | 103.4 ± 1.9 | 190.2 | 79.7 ± 0.6 | [68, 88] | 29.1 esc, 70.9 int8 | 124.6 | 68.4 | 70.8 | 4/1, p=0.38 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.