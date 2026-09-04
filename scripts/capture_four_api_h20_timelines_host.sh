#!/usr/bin/env bash
# Host-side entry point: lock H20 clocks, then run the 24-case collector in Docker.
set -euo pipefail

CONTAINER=${CONTAINER:-four_api_build}
ROOT=${ROOT:-$(pwd)}
OUT=${OUT:-$ROOT/artifacts/four_api_h20_nsys}
LOCK_SM_CLOCK_MHZ=${LOCK_SM_CLOCK_MHZ:-1830}

for command in docker nvidia-smi; do
  command -v "$command" >/dev/null || { echo "missing host command: $command" >&2; exit 2; }
done

test -d "$ROOT" || { echo "repository path not found: $ROOT" >&2; exit 2; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || {
  echo "container not found: $CONTAINER" >&2
  exit 2
}

# This must run on the host because the validated container is not privileged.
nvidia-smi -lgc "$LOCK_SM_CLOCK_MHZ,$LOCK_SM_CLOCK_MHZ"

mapfile -t clocks < <(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits)
test "${#clocks[@]}" -eq 8
for clock in "${clocks[@]}"; do
  test "$clock" -eq "$LOCK_SM_CLOCK_MHZ" || {
    echo "clock lock verification failed: expected $LOCK_SM_CLOCK_MHZ, got $clock" >&2
    exit 3
  }
done

docker exec \
  -e ROOT="$ROOT" \
  -e OUT="$OUT" \
  -e LOCK_SM_CLOCK_MHZ="$LOCK_SM_CLOCK_MHZ" \
  -e CLOCK_LOCK_MODE=verify \
  -e FORCE="${FORCE:-0}" \
  -e GPU_IDLE_LIMIT_MIB="${GPU_IDLE_LIMIT_MIB:-64}" \
  -e GPU_IDLE_RETRIES="${GPU_IDLE_RETRIES:-45}" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -w "$ROOT" \
  "$CONTAINER" \
  bash scripts/capture_four_api_h20_timelines.sh
