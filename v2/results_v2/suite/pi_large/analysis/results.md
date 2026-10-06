# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/pi_large` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | INT8 % / escalated % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs eightsignal_cold per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 130.1 ± 0.0 | 147.2 | 73.3 ± 0.0 | [68, 78] | 0.0 int8 | 153.2 | 80.7 | 73.3 | 11/6, p=0.33 |
| always_int8 | 101.5 ± 0.0 | 134.7 | 71.3 ± 0.0 | [66, 76] | 100.0 int8 | 119.7 | 61.4 | 71.3 | 3/4, p=1.00 |
| linucb_cold | 126.8 ± 0.0 | 175.4 | 73.0 ± 0.0 | [68, 78] | 41.0 int8 | 151.2 | 80.7 | 73.1 | 7/3, p=0.34 |
| linucb_warm | 132.9 ± 0.0 | 176.6 | 72.7 ± 0.0 | [67, 77] | 23.0 int8 | 157.1 | 83.9 | 73.3 | 9/6, p=0.61 |
| egreedy | 105.1 ± 0.0 | 145.2 | 71.3 ± 0.0 | [66, 76] | 93.3 int8 | 127.1 | 64.9 | 71.6 | 3/4, p=1.00 |
| eightsignal_cold | 112.2 ± 0.0 | 156.7 | 71.7 ± 0.0 | [66, 76] | 77.7 int8 | 134.1 | 68.9 | 72.1 |  |
| eightsignal_warm | 105.3 ± 0.0 | 146.4 | 71.3 ± 0.0 | [66, 76] | 91.3 int8 | 126.3 | 65.0 | 71.6 | 4/5, p=1.00 |
| baseline_fixed | 136.5 ± 0.0 | 147.5 | 85.0 ± 0.0 | [81, 89] | 0.0 int8 | 159.0 | 82.8 | 73.3 | 48/8, p=0.00 |
| v2_resource_aware_fixed | 92.7 ± 0.0 | 188.7 | 78.0 ± 0.0 | [73, 82] | 22.7 esc | 112.6 | 61.2 | faster than INT8 | 40/21, p=0.02 |
| v2_fixed_8signal | 91.0 ± 0.0 | 180.1 | 78.0 ± 0.0 | [73, 82] | 22.7 esc | 111.7 | 61.4 | faster than INT8 | 40/21, p=0.02 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.