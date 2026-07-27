#!/usr/bin/env bash
# Sample only the GPUs visible in this allocation. Safe to background from an
# sbatch script; stop it with the shell trap on EXIT.
#
# Usage:
#   ./scripts/monitor_gpus.sh <output.csv> [interval_seconds]
set -euo pipefail

OUT="${1:?usage: monitor_gpus.sh <output.csv> [interval_seconds]}"
INTERVAL="${2:-10}"

mkdir -p "$(dirname "${OUT}")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not on PATH" >&2
  exit 1
fi

if [[ ! -f "${OUT}" ]]; then
  echo "timestamp_utc,index,uuid,util_gpu,mem_used_mib,mem_total_mib,temp_c,power_w,sm_clock_mhz,mem_clock_mhz" \
    >"${OUT}"
fi

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi \
    --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    | while IFS=',' read -r index uuid util mem_used mem_total temp power sm_clock mem_clock; do
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "${ts}" \
          "$(echo "${index}" | xargs)" \
          "$(echo "${uuid}" | xargs)" \
          "$(echo "${util}" | xargs)" \
          "$(echo "${mem_used}" | xargs)" \
          "$(echo "${mem_total}" | xargs)" \
          "$(echo "${temp}" | xargs)" \
          "$(echo "${power}" | xargs)" \
          "$(echo "${sm_clock}" | xargs)" \
          "$(echo "${mem_clock}" | xargs)"
      done >>"${OUT}"
  sleep "${INTERVAL}"
done
