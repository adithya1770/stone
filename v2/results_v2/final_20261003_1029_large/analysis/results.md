# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/final_20261003_1029_large` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_rebuilt per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 231.9 ± 0.0 | 314.8 | 73.3 ± 0.0 | [68, 78] | 0.0 int8 | 282.3 | 193.0 | 73.3 | 5/45, p=0.00 |
| always_int8 | 192.5 ± 0.0 | 269.2 | 71.3 ± 0.0 | [66, 76] | 100.0 int8 | 229.1 | 141.3 | 71.3 | 4/50, p=0.00 |
| linucb_cold | 216.5 ± 0.0 | 295.3 | 72.7 ± 0.0 | [67, 77] | 38.7 int8 | 264.6 | 175.6 | 72.6 | 5/47, p=0.00 |
| eightsignal_cold | 214.5 ± 0.0 | 298.5 | 72.7 ± 0.0 | [67, 77] | 52.7 int8 | 261.7 | 169.3 | 72.5 | 6/48, p=0.00 |
| always_int8_rebuilt | 193.7 ± 0.0 | 269.9 | 86.0 ± 0.0 | [82, 89] | 100.0 int8 | 229.8 | 142.0 | 71.4 | 3/5, p=0.73 |
| baseline_rebuilt | 259.6 ± 0.0 | 369.8 | 85.7 ± 0.0 | [81, 89] | 0.0 int8 | 307.0 | 208.3 | 73.3 | 3/6, p=0.51 |
| eightsignal_rebuilt | 227.1 ± 0.0 | 346.5 | 86.7 ± 0.0 | [82, 90] | 51.3 int8 | 274.4 | 181.0 | 73.1 |  |
| v2_final | 190.6 ± 0.0 | 270.0 | 85.7 ± 0.0 | [81, 89] | 1.0 esc, 99.0 int8 | 232.9 | 145.6 | faster than INT8 | 3/6, p=0.51 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.