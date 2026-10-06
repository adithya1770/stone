# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/final_20261003_1218_large` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_cold per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 126.9 ± 0.0 | 138.6 | 73.3 ± 0.0 | [68, 78] | 0.0 int8 | 149.7 | 77.8 | 73.3 | 11/6, p=0.33 |
| always_int8 | 92.5 ± 0.0 | 98.2 | 71.3 ± 0.0 | [66, 76] | 100.0 int8 | 108.5 | 55.6 | 71.3 | 3/4, p=1.00 |
| linucb_cold | 111.3 ± 0.0 | 138.7 | 72.0 ± 0.0 | [67, 77] | 53.7 int8 | 132.7 | 68.1 | 72.4 | 6/5, p=1.00 |
| eightsignal_cold | 102.2 ± 0.0 | 136.7 | 71.7 ± 0.0 | [66, 76] | 74.7 int8 | 121.7 | 62.2 | 71.9 |  |
| always_int8_rebuilt | 93.5 ± 0.0 | 101.0 | 86.0 ± 0.0 | [82, 89] | 100.0 int8 | 109.9 | 55.9 | 71.4 | 48/5, p=0.00 |
| baseline_rebuilt | 149.3 ± 0.0 | 160.2 | 85.7 ± 0.0 | [81, 89] | 0.0 int8 | 167.9 | 87.8 | 73.3 | 49/7, p=0.00 |
| eightsignal_rebuilt | 106.0 ± 0.0 | 158.1 | 86.0 ± 0.0 | [82, 89] | 77.3 int8 | 125.6 | 64.1 | 72.1 | 49/6, p=0.00 |
| v2_final | 103.5 ± 0.0 | 198.2 | 85.7 ± 0.0 | [81, 89] | 17.7 esc, 82.3 int8 | 123.8 | 64.7 | 72.0 | 47/5, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.