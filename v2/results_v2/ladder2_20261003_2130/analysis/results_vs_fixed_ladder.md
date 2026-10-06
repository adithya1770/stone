# STONE v2 — Pi protocol results

Suite: `v2/results_v2/suite/ladder2_20261003_2130` — 1 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs fixed_ladder per image (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_ladder | 114.1 ± 0.0 | 121.0 | 74.4 ± 0.0 | [70, 78] | 100.0 int8 | 144.8 | 74.2 | faster than INT8 |  |
| big_ladder | 204.7 ± 0.0 | 255.6 | 78.8 ± 0.0 | [75, 82] | 100.0 int8 | 243.7 | 144.4 | faster than INT8 | 39/18, p=0.01 |
| eightsignal_ladder | 130.9 ± 0.0 | 220.4 | 76.5 ± 0.0 | [72, 80] | 100.0 int8 | 163.7 | 87.7 | faster than INT8 | 13/3, p=0.02 |
| cpu_ladder | 116.8 ± 0.0 | 126.5 | 74.4 ± 0.0 | [70, 78] | 100.0 int8 | 146.5 | 75.8 | faster than INT8 | 0/0, p=1.00 |
| v2_ladder_8sig | 157.6 ± 0.0 | 342.5 | 80.8 ± 0.0 | [77, 84] | 27.5 esc, 100.0 int8 | 186.1 | 117.3 | faster than INT8 | 51/20, p=0.00 |
| v2_ladder_cpu | 153.9 ± 0.0 | 360.9 | 80.4 ± 0.0 | [77, 84] | 25.8 esc, 100.0 int8 | 179.4 | 111.4 | faster than INT8 | 50/21, p=0.00 |

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, majority over trials); p = exact McNemar test.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.