#!/usr/bin/env bash
# STONE v2 final comparison (about 90 minutes): every way of choosing a model,
# same 800 unseen Imagewoof photos, same session, same stress schedule.
#
#   cd ~/stone && PY=$HOME/v2env/bin/python nohup bash v2/pi_run_compare.sh > ~/stone/v2/results_v2/compare.log 2>&1 &
#
#   fixed_ladder        one model that keeps the budget even under stress (Lite0)
#   lite1_ladder        best single model at about v2's average time (Lite1)
#   big_ladder          best idle model, kept under stress (Lite2)
#   egreedy_ladder      EdgeMLBalancer-style epsilon-greedy over all 6 models
#   linucb_ladder       the original LinUCB (unchanged), Lite2 <-> Lite0
#   eightsignal_ladder  the original EightSignal (unchanged), Lite2 <-> Lite0
#   v2_ladder_8sig      v2: EightSignal + per-photo check + time budget
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-compare_$(date +%Y%m%d_%H%M)}
N=${N:-800}
TARGET=${TARGET:-145}
[ -f v2/configs/operating_points_ladder.json ] || { echo "ERROR: run pi_run_ladder.sh first (operating_points_ladder.json)"; exit 1; }
echo "STONE v2 comparison -> v2/results_v2/suite/$TAG  (target ${TARGET} ms; throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a); temp: $(vcgencmd measure_temp 2>/dev/null || echo n/a))"
$PY -m pytest -q -p no:cacheprovider v2/tests
$PY -m v2.run_suite_v2 --trials 1 --dataset imagewoof_test --iterations "$N" --with-ladder --ladder-target-ms "$TARGET" \
    --only fixed_ladder lite1_ladder big_ladder egreedy_ladder linucb_ladder eightsignal_ladder v2_ladder_8sig --tag "$TAG"
S=v2/results_v2/suite/$TAG
for ref in v2_ladder_8sig fixed_ladder lite1_ladder; do
    $PY -m v2.analyse_v2 --suite "$S" --dataset imagewoof_test --ref $ref
    cp "$S/analysis/results.md" "$S/analysis/results_vs_$ref.md"
done
echo "throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo "DONE -> $S/analysis/"
