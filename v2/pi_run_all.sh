#!/usr/bin/env bash
# STONE v2 — full pipeline on the Raspberry Pi 4 (about 60-75 minutes).
#
#   cd stone && bash v2/pi_run_all.sh
#
# Needs: datasets/imagenette2-320/val (same as the original experiments),
#        stress-ng (sudo apt install stress-ng), pip install -r v2/requirements.txt
# Writes ONLY into v2/results_v2/<tag>/ and v2/configs/. Original files untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1          # no .pyc files in the original runtime/ folder
TAG=${TAG:-pi_$(date +%Y%m%d_%H%M)}
R=v2/results_v2/$TAG
TRIALS=${TRIALS:-3}
LIMIT=${LIMIT:-800}
mkdir -p "$R"

[ -d datasets/imagenette2-320/val ] || { echo "ERROR: datasets/imagenette2-320/val not found"; exit 1; }
$PY -c "import ai_edge_litert, psutil, PIL, numpy" 2>/dev/null || { echo "ERROR: run: $PY -m pip install -r v2/requirements.txt"; exit 1; }
command -v stress-ng >/dev/null || echo "WARNING: stress-ng not found; Python CPU burners will be used instead"
echo "STONE v2 on $(cat /proc/device-tree/model 2>/dev/null || uname -m) -> $R"

echo; echo "[1/8] unit tests"
$PY -m pytest -q -p no:cacheprovider v2/tests

echo; echo "[2/8] preprocessing check (original [0,1] vs Keras [-1,1]) on imagenette"
$PY -m v2.bench.check_preprocessing --dataset imagenette --limit 500 | tee "$R/preprocessing.json"

echo; echo "[3/8] latency profile: idle (0.5 s gaps, like the original protocol)"
$PY -m v2.bench.profile_variants --condition idle --runs 40 --gap 0.5 --out "$R/profile_idle.csv"
sleep 60
echo; echo "[4/8] latency profile: stressed (stress-ng --cpu 0)"
$PY -m v2.bench.profile_variants --condition stressed --runs 30 --gap 0.5 --out "$R/profile_stressed.csv"
sleep 60

echo; echo "[5/8] response table: every variant x $LIMIT held-out imagenette images"
$PY -m v2.bench.build_response_table --dataset imagenette --exclude-his-pool --limit "$LIMIT" --out "$R/table_imagenette.csv"

echo; echo "[6/8] offline replay -> operating points"
$PY -m v2.bench.replay_eval --table "$R/table_imagenette.csv" \
    --profile-idle "$R/profile_idle.csv" --profile-stressed "$R/profile_stressed.csv" \
    --original-only --label "(original preprocessing)" --out-dir "$R/replay_original" \
    --write-config --config-path v2/configs/operating_points.json
$PY -m v2.bench.replay_eval --table "$R/table_imagenette.csv" \
    --profile-idle "$R/profile_idle.csv" --profile-stressed "$R/profile_stressed.csv" \
    --label "(incl. fixed preprocessing)" --out-dir "$R/replay_fixed" \
    --write-config --config-path v2/configs/operating_points_fixed.json
sleep 60

echo; echo "[7/8] original Pi protocol, $TRIALS trials, original configs + v2"
$PY -m v2.run_suite_v2 --trials "$TRIALS" --with-fixed --tag "$TAG"

echo; echo "[8/8] analysis"
$PY -m v2.analyse_v2 --suite "v2/results_v2/suite/$TAG"

echo; echo "DONE. Key files:"
echo "  v2/results_v2/suite/$TAG/analysis/results.md   <- Table-I comparison"
echo "  $R/replay_original/report.md          <- offline evaluation (fair, original preprocessing)"
echo "  $R/replay_fixed/report.md             <- with preprocessing fix"
echo "  $R/preprocessing.json"
