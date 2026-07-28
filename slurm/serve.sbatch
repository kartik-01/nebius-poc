#!/bin/bash
# Launch vLLM for one topology (P0–P3). Leaves endpoints.json for benchmark.sbatch.
#
# Usage:
#   source configs/cluster.env
#   TOPOLOGY=P0 MODEL_PATH=/path/to/merged sbatch --nodes=1 --gpus-per-node=1 \
#     --partition="$SLURM_PARTITION" slurm/serve.sbatch
#   TOPOLOGY=P1|P2 ... --nodes=1 --gpus-per-node=2
#   TOPOLOGY=P3 ... --nodes=2 --gpus-per-node=2
#
# Cancel when benchmarking is done: scancel $JOBID
#
#SBATCH --job-name=nebius-serve
#SBATCH --cpus-per-task=16
# Weights live on the GPU; the host side only stages them. Without an explicit
# request Slurm reserves the node's entire memory, which starves the benchmark
# client and anyone else sharing the cluster.
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/serve-%j.out
#SBATCH --error=logs/serve-%j.err
# Pass --nodes / --gpus-per-node on the command line to match TOPOLOGY.

set -Eeuo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p "${ROOT}/logs"

if [[ -f "${ROOT}/configs/cluster.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/cluster.env"
fi

: "${SHARED_ROOT:?set SHARED_ROOT in configs/cluster.env}"
: "${VLLM_IMAGE:?set VLLM_IMAGE in configs/cluster.env}"
: "${MODEL_PATH:?set MODEL_PATH to the merged model directory}"
: "${TOPOLOGY:?set TOPOLOGY to P0, P1, P2, or P3}"

TOPO_FILE="${ROOT}/configs/serve_topologies.yaml"

# Compute nodes ship a bare system python without pyyaml, so prefer the repo venv
# on shared storage. This parsing happens outside the container because it decides
# the srun geometry.
PYBIN="python3"
if ! "${PYBIN}" -c "import yaml" >/dev/null 2>&1; then
  if [[ -x "${ROOT}/.venv/bin/python" ]] && "${ROOT}/.venv/bin/python" -c "import yaml" >/dev/null 2>&1; then
    PYBIN="${ROOT}/.venv/bin/python"
  else
    echo "no python with pyyaml available; run 'make install' to create .venv" >&2
    exit 1
  fi
fi

mapfile -t TOPO_FIELDS < <("${PYBIN}" - "${TOPO_FILE}" "${TOPOLOGY}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
topo = cfg["topologies"][sys.argv[2]]
server = cfg["server"]
for key in (
    "nodes", "gpus_per_node", "replicas_per_node", "tensor_parallel_size",
):
    print(topo[key])
for key in (
    "base_port", "dtype", "max_model_len", "gpu_memory_utilization",
    "max_num_seqs", "max_num_batched_tokens",
):
    print(server[key])
print(topo["description"])
PY
)

EXPECT_NODES="${TOPO_FIELDS[0]}"
REPLICAS_PER_NODE="${TOPO_FIELDS[2]}"
TP_SIZE="${TOPO_FIELDS[3]}"
BASE_PORT="${TOPO_FIELDS[4]}"
DTYPE="${TOPO_FIELDS[5]}"
MAX_MODEL_LEN="${TOPO_FIELDS[6]}"
GPU_MEM_UTIL="${TOPO_FIELDS[7]}"
MAX_NUM_SEQS="${TOPO_FIELDS[8]}"
MAX_BATCHED_TOKENS="${TOPO_FIELDS[9]}"
DESCRIPTION="${TOPO_FIELDS[10]}"

export REPLICAS_PER_NODE TP_SIZE BASE_PORT DTYPE MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_BATCHED_TOKENS

if [[ "${SLURM_NNODES}" != "${EXPECT_NODES}" ]]; then
  echo "TOPOLOGY=${TOPOLOGY} needs ${EXPECT_NODES} nodes, got SLURM_NNODES=${SLURM_NNODES}" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_serve-${TOPOLOGY}_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${MODEL_PATH}:${MODEL_PATH}"
)
if [[ -n "${HF_HOME:-}" ]]; then
  MOUNTS+=("${HF_HOME}:/hf")
fi
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

cat > "${OUT}/serve_manifest.json" <<EOF
{
  "topology": "${TOPOLOGY}",
  "description": "${DESCRIPTION}",
  "model_path": "${MODEL_PATH}",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "nodes": "${SLURM_NODELIST}"
}
EOF

echo "serve topology=${TOPOLOGY} (${DESCRIPTION})"
echo "model=${MODEL_PATH} out=${OUT}"

# Inside the container the run directory is mounted at /results.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --container-image="${VLLM_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,OUT=/results,TOPOLOGY,MODEL_PATH,REPLICAS_PER_NODE,TP_SIZE,BASE_PORT,DTYPE,MAX_MODEL_LEN,GPU_MEM_UTIL,MAX_NUM_SEQS,MAX_BATCHED_TOKENS,HF_HOME=/hf \
  bash /workspace/scripts/launch_vllm_replicas.sh

echo "serve job ending; artifacts in ${OUT}"
if [[ -f "${OUT}/endpoints.json" ]]; then
  cat "${OUT}/endpoints.json"
fi
