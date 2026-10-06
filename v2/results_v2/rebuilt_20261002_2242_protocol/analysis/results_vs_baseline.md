# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/rebuilt_20261002_2242_protocol` — 3 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs baseline per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 129.3 ± 0.4 | 135.6 | 75.0 ± 0.0 | [63, 84] | 0.0 int8 | 147.1 | 76.0 | 75.0 |  |
| always_int8 | 89.6 ± 0.5 | 93.5 | 68.3 ± 0.0 | [56, 79] | 100.0 int8 | 104.9 | 53.8 | 68.3 | 0/4, p=0.12 |
| eightsignal_cold | 102.6 ± 1.8 | 133.9 | 70.6 ± 1.0 | [57, 80] | 67.8 int8 | 122.0 | 62.2 | 70.5 | 0/3, p=0.25 |
| baseline_fixed | 129.6 ± 1.3 | 136.0 | 80.0 ± 0.0 | [68, 88] | 0.0 int8 | 150.1 | 77.9 | 75.0 | 4/1, p=0.38 |
| v2_resource_aware_fixed | 85.0 ± 1.0 | 178.5 | 75.0 ± 0.0 | [63, 84] | 17.2 esc, 0.0 int8 | 103.1 | 55.1 | faster than INT8 | 3/3, p=1.00 |
| always_int8_rebuilt | 90.2 ± 0.5 | 93.9 | 83.3 ± 0.0 | [72, 91] | 100.0 int8 | 105.8 | 54.3 | 68.4 | 6/1, p=0.12 |
| baseline_rebuilt | 145.1 ± 0.3 | 153.1 | 78.3 ± 0.0 | [66, 87] | 0.0 int8 | 162.8 | 84.8 | 75.0 | 4/2, p=0.69 |
| v2_rebuilt | 90.5 ± 0.3 | 93.7 | 83.3 ± 0.0 | [72, 91] | 0.0 esc, 100.0 int8 | 105.8 | 54.0 | 68.5 | 6/1, p=0.12 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.