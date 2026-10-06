# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/ladder_20261003_1906` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_cold per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 127.4 ± 0.0 | 137.3 | 57.5 ± 0.0 | [53, 62] | 0.0 int8 | 149.7 | 77.9 | 57.5 | 24/14, p=0.14 |
| always_int8 | 94.2 ± 0.0 | 101.8 | 55.0 ± 0.0 | [51, 59] | 100.0 int8 | 110.3 | 56.2 | 55.0 | 2/4, p=0.69 |
| eightsignal_cold | 106.5 ± 0.0 | 141.7 | 55.4 ± 0.0 | [51, 60] | 72.3 int8 | 126.1 | 64.5 | 55.9 |  |
| fixed_ladder | 116.8 ± 0.0 | 126.5 | 74.4 ± 0.0 | [70, 78] | 100.0 int8 | 149.5 | 76.8 | 56.7 | 115/24, p=0.00 |
| big_ladder | 213.1 ± 0.0 | 265.9 | 78.8 ± 0.0 | [75, 82] | 100.0 int8 | 253.3 | 151.7 | 57.5 | 128/16, p=0.00 |
| eightsignal_ladder | 132.7 ± 0.0 | 217.1 | 74.0 ± 0.0 | [70, 78] | 100.0 int8 | 165.7 | 88.6 | 57.5 | 112/23, p=0.00 |
| v2_ladder | 173.8 ± 0.0 | 367.5 | 82.1 ± 0.0 | [78, 85] | 32.3 esc, 100.0 int8 | 204.8 | 132.7 | 57.5 | 136/8, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.