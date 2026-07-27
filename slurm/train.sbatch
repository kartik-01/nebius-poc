#!/bin/bash
# Multi-node LoRA training launcher (2 nodes x 2 GPUs).
# Do not submit until M5 one-GPU smoke has passed and configs/cluster.env is complete.
#
# Usage:
#   source configs/cluster.env
#   sbatch --partition="$SLURM_PARTITION" ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} \
#     ${SLURM_QOS:+--qos=$SLURM_QOS} slurm/train.sbatch
#
# After recipe lock:
#   TRAIN_FINAL=1 RECIPE_LOCK=results/summary/recipe_lock.json \
#     sbatch --partition="$SLURM_PARTITION" slurm/train.sbatch
#
#SBATCH --job-name=nebius-train
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

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

CONFIG="${TRAIN_CONFIG:-configs/train_sft.yaml}"
# After recipe lock: TRAIN_FINAL=1 RECIPE_LOCK=results/summary/recipe_lock.json
TRAIN_FINAL="${TRAIN_FINAL:-0}"
RECIPE_LOCK="${RECIPE_LOCK:-}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_train_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

if [[ "${TRAIN_FINAL}" == "1" && -z "${RECIPE_LOCK}" ]]; then
  echo "TRAIN_FINAL=1 requires RECIPE_LOCK=path/to/recipe_lock.json" >&2
  exit 2
fi

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export MASTER_ADDR MASTER_PORT
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${HF_HOME}:/hf"
)
if [[ -n "${LOCAL_SCRATCH:-}" ]]; then
  MOUNTS+=("${LOCAL_SCRATCH}:/scratch")
fi
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
if [[ -n "${RECIPE_LOCK}" && "${RECIPE_LOCK}" != /* ]]; then
  RECIPE_LOCK="/workspace/${RECIPE_LOCK}"
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

MONITOR_PID=""
cleanup() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -x "${ROOT}/scripts/monitor_gpus.sh" ]]; then
  "${ROOT}/scripts/monitor_gpus.sh" "${OUT}/gpu_monitor.csv" 15 &
  MONITOR_PID=$!
fi

FINAL_ARG=""
LOCK_ARG=""
MODEL_ARG=""
if [[ "${TRAIN_FINAL}" == "1" ]]; then
  FINAL_ARG="--final"
fi
if [[ -n "${RECIPE_LOCK}" ]]; then
  LOCK_ARG="--recipe-lock ${RECIPE_LOCK}"
fi
# TRAIN_MODEL exists so the multi-node path can be rehearsed on the 0.5B model
# before spending an allocation on the 7B run. Leave unset for real training.
if [[ -n "${TRAIN_MODEL:-}" ]]; then
  MODEL_ARG="--model ${TRAIN_MODEL}"
fi
EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "run_id=${RUN_ID} config=${CONFIG} final=${TRAIN_FINAL} master=${MASTER_ADDR}:${MASTER_PORT}"

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-image="${TRAIN_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,HF_HOME=/hf,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,MASTER_ADDR,MASTER_PORT,NCCL_SOCKET_IFNAME,NCCL_IB_HCA,UCX_NET_DEVICES \
  bash -lc "
    set -euo pipefail
    # shellcheck disable=SC1091
    source /workspace/scripts/container_python_setup.sh
    nebius_poc_container_python_setup
    torchrun \
      --nnodes=\"\${SLURM_NNODES}\" \
      --nproc-per-node=${GPUS_PER_NODE} \
      --node-rank=\"\${SLURM_NODEID}\" \
      --rdzv-id=\"\${SLURM_JOB_ID}\" \
      --rdzv-backend=c10d \
      --rdzv-endpoint=\"${MASTER_ADDR}:${MASTER_PORT}\" \
      -m nebius_poc.train \
      --config ${CONFIG} \
      --results-root /results \
      ${FINAL_ARG} ${LOCK_ARG} ${MODEL_ARG} ${EXTRA_ARGS}
  "

echo "training artifacts in ${OUT}"
