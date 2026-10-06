# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/rebuilt_20261002_2242_large` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs always_int8 per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 130.9 ± 0.0 | 140.0 | 73.3 ± 0.0 | [68, 78] | 0.0 int8 | 149.1 | 77.7 | 73.3 | 15/9, p=0.31 |
| always_int8 | 91.6 ± 0.0 | 96.6 | 71.3 ± 0.0 | [66, 76] | 100.0 int8 | 107.7 | 55.0 | 71.3 |  |
| eightsignal_cold | 102.3 ± 0.0 | 136.1 | 71.3 ± 0.0 | [66, 76] | 75.0 int8 | 121.7 | 61.8 | 71.9 | 3/3, p=1.00 |
| baseline_fixed | 131.4 ± 0.0 | 142.6 | 85.0 ± 0.0 | [81, 89] | 0.0 int8 | 153.0 | 79.8 | 73.3 | 48/7, p=0.00 |
| v2_resource_aware_fixed | 91.5 ± 0.0 | 186.7 | 78.0 ± 0.0 | [73, 82] | 21.7 esc, 0.0 int8 | 111.4 | 60.8 | faster than INT8 | 40/20, p=0.01 |
| always_int8_rebuilt | 93.4 ± 0.0 | 100.4 | 86.0 ± 0.0 | [82, 89] | 100.0 int8 | 109.7 | 55.6 | 71.4 | 47/3, p=0.00 |
| baseline_rebuilt | 149.7 ± 0.0 | 161.7 | 85.7 ± 0.0 | [81, 89] | 0.0 int8 | 168.4 | 87.8 | 73.3 | 50/7, p=0.00 |
| v2_rebuilt | 93.9 ± 0.0 | 102.0 | 86.0 ± 0.0 | [82, 89] | 0.0 esc, 100.0 int8 | 110.2 | 56.5 | 71.4 | 47/3, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.