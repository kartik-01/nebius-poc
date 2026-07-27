#!/bin/bash
# One-node, one-GPU validator smoke. Stops well short of the full two-node qualification.
#
# Usage (after configs/cluster.env is filled and VALIDATOR_IMAGE is set):
#   source configs/cluster.env
#   sbatch --partition="$SLURM_PARTITION" ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} \
#     ${SLURM_QOS:+--qos=$SLURM_QOS} slurm/validate_smoke.sbatch
#
#SBATCH --job-name=nebius-validate-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/validate-smoke-%j.out
#SBATCH --error=logs/validate-smoke-%j.err

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p "${ROOT}/logs"

if [[ -f "${ROOT}/configs/cluster.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/cluster.env"
fi

: "${SHARED_ROOT:?set SHARED_ROOT in configs/cluster.env}"
: "${VALIDATOR_IMAGE:?set VALIDATOR_IMAGE in configs/cluster.env}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_validate-smoke_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

GPU_SMOKE_HOST="${ROOT}/containers/images/gpu_smoke"
SMOKE_ARG=()
MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
)
if [[ -x "${GPU_SMOKE_HOST}" ]]; then
  MOUNTS+=("${GPU_SMOKE_HOST}:/usr/local/bin/gpu_smoke")
  SMOKE_ARG=(--gpu-smoke /usr/local/bin/gpu_smoke)
fi
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

echo "run_id=${RUN_ID}"
echo "out=${OUT}"
echo "image=${VALIDATOR_IMAGE}"
echo "node=$(hostname)"

srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  --container-image="${VALIDATOR_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,LOCAL_SCRATCH,VALIDATOR_IMAGE,PYTHONPATH=/workspace \
  bash -lc '
    set -euo pipefail
    # shellcheck disable=SC1091
    source /workspace/scripts/container_python_setup.sh
    nebius_poc_container_python_setup
    python3 -m validator.cluster_validate \
      --config /workspace/configs/validator.yaml \
      --out /results \
      '"${SMOKE_ARG[*]}"'
  '

echo "smoke artifacts in ${OUT}"
cat "${OUT}/summary.json"
