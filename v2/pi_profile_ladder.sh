#!/usr/bin/env bash
# Measure Pi latency of the new model ladder (about 10-12 minutes).
#   cd ~/stone && PY=$HOME/v2env/bin/python bash v2/pi_profile_ladder.sh
# Accuracy was measured on the Mac (hardware-independent); only speed needs the Pi.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
R=v2/results_v2/ladder_pi
mkdir -p "$R"
V="int8_224 fp32_224 mnv2q_160 mnv2q_224 efl0q_224 efl1q_240 efl2q_260 efl3q_280 efl4q_300"
[ -d datasets/imagewoof2-320/val ] || { echo "ERROR: datasets/imagewoof2-320/val missing"; exit 1; }
$PY -c "from v2.common.engine import MultiVariantEngine as E; E('$V'.split()); print('all models load')"
echo "throttle flags before: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo "[1/2] idle"
$PY -m v2.bench.profile_variants --dataset imagewoof --condition idle --runs 30 --gap 0.5 --variants $V --out "$R/profile_idle.csv"
sleep 45
echo "[2/2] stressed (stress-ng on all cores)"
$PY -m v2.bench.profile_variants --dataset imagewoof --condition stressed --runs 25 --gap 0.5 --variants $V --out "$R/profile_stressed.csv"
echo "throttle flags after: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo "DONE -> $R"
