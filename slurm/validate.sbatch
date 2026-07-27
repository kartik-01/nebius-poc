#!/bin/bash
# Portable cluster qualification + provider-native NCCL checks.
# Every cluster-specific value comes from configs/cluster.env — nothing is hardcoded.
#
# Usage:
#   source configs/cluster.env
#   sbatch slurm/validate.sbatch
#
#SBATCH --job-name=nebius-validate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate-%j.out
#SBATCH --error=logs/validate-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

if [[ -f "${ROOT}/configs/cluster.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/cluster.env"
fi

: "${SLURM_PARTITION:?set SLURM_PARTITION in configs/cluster.env}"
: "${SHARED_ROOT:?set SHARED_ROOT in configs/cluster.env}"
: "${VALIDATOR_IMAGE:?set VALIDATOR_IMAGE in configs/cluster.env}"

# These are declared in the header via #SBATCH when the site supports overrides;
# export them so a site that injects --partition via sbatch CLI still works.
export SLURM_PARTITION SHARED_ROOT LOCAL_SCRATCH VALIDATOR_IMAGE CONTAINER_MOUNTS
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_validate_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}" "${ROOT}/logs"

echo "run_id=${RUN_ID}"
echo "out=${OUT}"
echo "nodes=${SLURM_NODELIST}"
echo "image=${VALIDATOR_IMAGE}"

GPU_SMOKE_HOST="${ROOT}/containers/images/gpu_smoke"
SMOKE_ARGS=()
MOUNTS=("${ROOT}:/workspace" "${OUT}:/results")
if [[ -x "${GPU_SMOKE_HOST}" ]]; then
  MOUNTS+=("${GPU_SMOKE_HOST}:/usr/local/bin/gpu_smoke")
  SMOKE_ARGS=(--gpu-smoke /usr/local/bin/gpu_smoke)
fi
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"
MOUNT_ARGS=(--container-mounts="${MOUNT_CSV}")

# --- Portable checks: one container invocation per allocated node ---
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  --container-image="${VALIDATOR_IMAGE}" \
  "${MOUNT_ARGS[@]}" \
  --container-workdir=/workspace \
  --export=ALL,LOCAL_SCRATCH,VALIDATOR_IMAGE,PYTHONPATH=/workspace \
  python3 -m validator.cluster_validate \
    --config /workspace/configs/validator.yaml \
    --out /results \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}

# --- Provider-native NCCL: discover binaries at runtime, never bake paths in ---
ALL_REDUCE="$(command -v all_reduce_perf || true)"
ALL_REDUCE_MPI="$(command -v all_reduce_perf_mpi || true)"

if [[ -z "${ALL_REDUCE}" && -z "${ALL_REDUCE_MPI}" ]]; then
  echo "WARNING: neither all_reduce_perf nor all_reduce_perf_mpi found on PATH" >&2
  echo "NCCL checks will be recorded as UNKNOWN by the aggregator" >&2
else
  NCCL_BIN="${ALL_REDUCE_MPI:-${ALL_REDUCE}}"
  echo "using NCCL binary: ${NCCL_BIN}"

  # Two cluster-specific details, both discovered rather than assumed:
  #  1. Compute nodes carry /32 pod addresses on different subnets, so UCX cannot
  #     build endpoints between them. MPI only bootstraps the test, so route it
  #     over plain TCP; the collective itself still runs on NCCL over InfiniBand.
  #  2. nccl-tests picks its device by local rank, so every task must see both
  #     node GPUs. --gpus-per-task=1 would hide device 1 from the second rank.
  export OMPI_MCA_pml=ob1
  export OMPI_MCA_btl=tcp,self
  export OMPI_MCA_btl_tcp_if_include="${MPI_TCP_IF:-eth0}"
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,NET

  # Intra-node: 2 GPUs on each node separately.
  mapfile -t NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
  for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 --gpus=2 --nodelist="${node}" \
      "${NCCL_BIN}" -b 8 -e 128M -f 2 -g 2 \
      > "${OUT}/nccl-intra-${node}.log" 2>&1
    tail -n 5 "${OUT}/nccl-intra-${node}.log"
  done

  REPS="${NCCL_REPETITIONS:-3}"
  for rep in $(seq 1 "${REPS}"); do
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
    srun --mpi=pmix --nodes=2 --ntasks=4 --ntasks-per-node=2 --gpus-per-node=2 \
      "${NCCL_BIN}" -b 8 -e 128M -f 2 -g 1 \
      > "${OUT}/nccl-inter-run-${rep}.log" 2>&1
    tail -n 5 "${OUT}/nccl-inter-run-${rep}.log"
  done
fi

# Fold NCCL logs into summary.json without re-running fio or the CUDA smoke.
srun --nodes=1 --ntasks=1 --gpus=0 \
  --container-image="${VALIDATOR_IMAGE}" \
  "${MOUNT_ARGS[@]}" \
  --container-workdir=/workspace \
  --export=ALL,LOCAL_SCRATCH,VALIDATOR_IMAGE,PYTHONPATH=/workspace \
  python3 -m validator.cluster_validate \
    --config /workspace/configs/validator.yaml \
    --out /results \
    --nccl-dir /results \
    --aggregate-only

echo "validation artifacts in ${OUT}"
echo "summary:"
cat "${OUT}/summary.json"
