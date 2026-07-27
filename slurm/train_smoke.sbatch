#!/bin/bash
# One-node, one-GPU training smoke against Qwen2.5-0.5B-Instruct.
# Proves Pyxis + torch + LoRA wiring. Accuracy numbers from this job are meaningless.
#
# Usage:
#   source configs/cluster.env
#   sbatch --partition="$SLURM_PARTITION" ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} \
#     ${SLURM_QOS:+--qos=$SLURM_QOS} slurm/train_smoke.sbatch
#
#SBATCH --job-name=nebius-train-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/train-smoke-%j.out
#SBATCH --error=logs/train-smoke-%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p "${ROOT}/logs"

if [[ -f "${ROOT}/configs/cluster.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/cluster.env"
fi

: "${SHARED_ROOT:?set SHARED_ROOT in configs/cluster.env}"
: "${TRAIN_IMAGE:?set TRAIN_IMAGE in configs/cluster.env}"
: "${HF_HOME:?set HF_HOME in configs/cluster.env}"

SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_train-smoke_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

MONITOR_PID=""
cleanup() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "run_id=${RUN_ID}"
echo "out=${OUT}"
echo "image=${TRAIN_IMAGE}"
echo "model=${SMOKE_MODEL}"

if [[ -x "${ROOT}/scripts/monitor_gpus.sh" ]]; then
  "${ROOT}/scripts/monitor_gpus.sh" "${OUT}/gpu_monitor.csv" 10 &
  MONITOR_PID=$!
fi

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${HF_HOME}:/hf"
)
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

# Prefer the training container when available. Until TRAIN_IMAGE is a real digest,
# this job will fail at pull time rather than silently running on the host Python.
srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  --container-image="${TRAIN_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,HF_HOME=/hf,HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}",TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
  python -m nebius_poc.train \
    --config configs/train_ranking.yaml \
    --model "${SMOKE_MODEL}" \
    --limit 8 \
    --batch-size 2 \
    --max-steps 4 \
    --results-root /results

ADAPTER_HOST="$(find "${OUT}" -type d -name adapter | head -n 1 || true)"
if [[ -z "${ADAPTER_HOST}" ]]; then
  echo "ERROR: no adapter directory written under ${OUT}" >&2
  exit 1
fi
ADAPTER_CONTAINER="/results/${ADAPTER_HOST#${OUT}/}"
echo "reloading adapter host=${ADAPTER_HOST} container=${ADAPTER_CONTAINER}"

srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  --container-image="${TRAIN_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,HF_HOME=/hf,HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}",TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
  python -m nebius_poc.evaluate \
    --config configs/train_ranking.yaml \
    --model "${SMOKE_MODEL}" \
    --adapter "${ADAPTER_CONTAINER}" \
    --label smoke-tuned \
    --split validation \
    --limit 8 \
    --batch-size 2 \
    --mode forced_choice \
    --results-root /results

echo "train smoke complete: ${OUT}"
