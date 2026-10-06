# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/confirm_20261004_1059` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs fixed_ladder per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_ladder | 109.6 ± 0.0 | 118.1 | 74.4 ± 0.0 | [70, 78] | 100.0 int8 | 144.1 | 73.6 | faster than INT8 |  |
| big_ladder | 200.1 ± 0.0 | 254.6 | 78.8 ± 0.0 | [75, 82] | 100.0 int8 | 239.3 | 143.3 | faster than INT8 | 39/18, p=0.01 |
| eightsignal_ladder | 126.8 ± 0.0 | 212.9 | 74.8 ± 0.0 | [71, 78] | 100.0 int8 | 161.0 | 85.8 | faster than INT8 | 6/4, p=0.75 |
| lite1_ladder | 161.8 ± 0.0 | 186.7 | 82.1 ± 0.0 | [78, 85] | 100.0 int8 | 192.8 | 107.3 | faster than INT8 | 54/17, p=0.00 |
| v2_ladder_8sig | 153.6 ± 0.0 | 338.8 | 80.2 ± 0.0 | [76, 84] | 25.6 esc, 100.0 int8 | 181.3 | 112.6 | faster than INT8 | 49/21, p=0.00 |
| v2_ladder_cpu | 148.6 ± 0.0 | 329.8 | 80.0 ± 0.0 | [76, 83] | 24.0 esc, 100.0 int8 | 173.9 | 108.2 | faster than INT8 | 49/22, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.