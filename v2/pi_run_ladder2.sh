#!/usr/bin/env bash
# STONE v2 LADDER run 2 (about 60-70 minutes): fixed 150 ms SLA, two device detectors.
#
#   cd ~/stone && PY=$HOME/v2env/bin/python nohup bash v2/pi_run_ladder2.sh > ~/stone/v2/results_v2/ladder2.log 2>&1 &
#
# Same models, photos (Imagewoof test half), stress schedule and operating points as
# pi_run_ladder.sh. Changes after run 1:
#   - the time budget is an absolute SLA (150 ms average in every state) instead of
#     "k x the cheap model's live time", which let the budget grow under stress;
#   - the device state comes from EightSignal (v2_ladder_8sig) or from v2's simple
#     CPU/thermal monitor (v2_ladder_cpu); cpu_ladder = device switching only, CPU monitor.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-ladder2_$(date +%Y%m%d_%H%M)}
N=${N:-800}
[ -f v2/configs/operating_points_ladder.json ] || { echo "ERROR: run pi_run_ladder.sh first (operating_points_ladder.json)"; exit 1; }
echo "STONE v2 LADDER run 2 -> v2/results_v2/suite/$TAG  (throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a))"
$PY -m pytest -q -p no:cacheprovider v2/tests
$PY -m v2.run_suite_v2 --trials 1 --dataset imagewoof_test --iterations "$N" --with-ladder \
    --only fixed_ladder big_ladder eightsignal_ladder cpu_ladder v2_ladder_8sig v2_ladder_cpu --tag "$TAG"
S=v2/results_v2/suite/$TAG
for ref in fixed_ladder eightsignal_ladder cpu_ladder; do
    $PY -m v2.analyse_v2 --suite "$S" --dataset imagewoof_test --ref $ref
    cp "$S/analysis/results.md" "$S/analysis/results_vs_$ref.md"
done
echo "throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo "DONE -> $S/analysis/"
