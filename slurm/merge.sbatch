#!/bin/bash
# Merge a LoRA adapter into base weights for vLLM serving.
#
# Usage:
#   source configs/cluster.env
#   ADAPTER=results/raw/<train_run>/adapter \
#     MERGE_OUT=$SHARED_ROOT/models/tuned-merged \
#     sbatch --partition="$SLURM_PARTITION" slurm/merge.sbatch
#
#SBATCH --job-name=nebius-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err

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
: "${ADAPTER:?set ADAPTER to the LoRA adapter directory}"
: "${MERGE_OUT:?set MERGE_OUT to the merged model output directory}"

CONFIG="${MERGE_CONFIG:-configs/train_sft.yaml}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_merge_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}" "${MERGE_OUT}"

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${HF_HOME}:/hf"
  "${MERGE_OUT}:${MERGE_OUT}"
)
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi

# Remap adapter under SHARED_ROOT/results into the container.
ADAPTER_ARG="${ADAPTER}"
if [[ "${ADAPTER}" == "${SHARED_ROOT}/results/"* ]]; then
  MOUNTS+=("${SHARED_ROOT}/results:/all_results")
  ADAPTER_ARG="/all_results/${ADAPTER#${SHARED_ROOT}/results/}"
elif [[ "${ADAPTER}" != /* ]]; then
  ADAPTER_ARG="/workspace/${ADAPTER}"
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

echo "merging adapter=${ADAPTER} -> ${MERGE_OUT}"

srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  --container-image="${TRAIN_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,HF_HOME=/hf,HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}",TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  python -m nebius_poc.merge_adapter \
    --config "${CONFIG}" \
    --adapter "${ADAPTER_ARG}" \
    --output "${MERGE_OUT}" \
    --results-root /results

echo "merged model at ${MERGE_OUT}"
echo "merge run artifacts in ${OUT}"
