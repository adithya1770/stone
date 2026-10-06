#!/usr/bin/env bash
# Pipeline validation WITHOUT a Pi or imagenette (uses tiny-imagenet, 64 px).
# Latency numbers are from this machine, NOT the Pi; accuracy is on 64 px
# images. Use it to check the code and the routing mechanism, not for claims.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-validate_$(uname -m)}
R=v2/results_v2/$TAG
LIMIT=${LIMIT:-3000}
mkdir -p "$R"
$PY -m pytest -q -p no:cacheprovider v2/tests
$PY -m v2.bench.check_preprocessing --dataset tiny --limit 1000 | tee "$R/preprocessing.json"
$PY -m v2.bench.profile_variants --dataset tiny --condition idle --runs 60 --out "$R/profile_idle.csv"
$PY -m v2.bench.profile_variants --dataset tiny --condition stressed --runs 40 --stressor python --out "$R/profile_stressed.csv"
$PY -m v2.bench.build_response_table --dataset tiny --limit "$LIMIT" --out "$R/table_tiny.csv"
$PY -m v2.bench.replay_eval --table "$R/table_tiny.csv" --profile-idle "$R/profile_idle.csv" \
    --profile-stressed "$R/profile_stressed.csv" --original-only --label "(tiny, original preprocessing)" \
    --out-dir "$R/replay_original" --write-config --config-path "v2/configs/operating_points_$TAG.json"
$PY -m v2.bench.replay_eval --table "$R/table_tiny.csv" --profile-idle "$R/profile_idle.csv" \
    --profile-stressed "$R/profile_stressed.csv" --label "(tiny, incl. fixed preprocessing)" \
    --out-dir "$R/replay_fixed" --write-config --config-path "v2/configs/operating_points_fixed_$TAG.json"
echo "done -> $R"
