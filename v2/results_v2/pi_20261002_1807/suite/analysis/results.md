# STONE v2 — Pi protocol results

Suite: `v2/results_v2/pi_20261002_1807/suite` — 0 trial(s); stressed rows = cpu ≥ 70% (original definition).

| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | INT8 % / escalated % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs Baseline (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|

Notes:
- Accuracy CIs use the number of stressed images (same images every trial), so they show real uncertainty; the ± across trials only shows run-to-run agreement.
- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; a router with no per-request intelligence lands on this line.
- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request.