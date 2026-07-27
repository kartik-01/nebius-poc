#!/bin/bash
# Benchmark a running serve job with vllm bench serve.
#
# Usage:
#   source configs/cluster.env
#   SERVE_RUN=$SHARED_ROOT/results/raw/<serve_run_id> \
#     STAGE=screen \
#     sbatch --partition="$SLURM_PARTITION" --nodes=1 --gpus-per-node=0 \
#       slurm/benchmark.sbatch
#
# STAGE=screen|final|soak
# CONCURRENCY can override the config list for a single point (e.g. soak).
#
#SBATCH --job-name=nebius-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err

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
: "${SERVE_RUN:?set SERVE_RUN to the serve job's results/raw/<run_id> directory}"

STAGE="${STAGE:-screen}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/benchmark.yaml}"
MODEL_NAME="${MODEL_NAME:-merged}"

if [[ ! -f "${SERVE_RUN}/endpoints.json" ]]; then
  echo "missing ${SERVE_RUN}/endpoints.json — is the serve job ready?" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_bench-${STAGE}_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

python3 - "${BENCH_CONFIG}" "${SERVE_RUN}/endpoints.json" "${OUT}" "${STAGE}" <<'PY'
import json, os, sys, yaml
from pathlib import Path

cfg = yaml.safe_load(open(sys.argv[1]))
endpoints = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
stage = sys.argv[4]

conc = os.environ.get("CONCURRENCY")
if conc:
    concurrencies = [int(conc)]
elif stage == "soak":
    # Soak uses one operating point; caller should set CONCURRENCY near 80% goodput.
    concurrencies = [int(os.environ.get("SOAK_CONCURRENCY", cfg["concurrency"][-1]))]
else:
    concurrencies = list(cfg["concurrency"])

reps = {
    "screen": int(cfg["screening_repetitions"]),
    "final": int(cfg["final_repetitions"]),
    "soak": 1,
}.get(stage, 1)

manifest = {
    "stage": stage,
    "serve_run": sys.argv[2],
    "topology": endpoints.get("topology"),
    "base_urls": endpoints.get("base_urls", []),
    "concurrency": concurrencies,
    "repetitions": reps,
    "input_tokens": cfg["input_tokens"],
    "output_tokens": cfg["output_tokens"],
    "warmup_requests": cfg["warmup_requests"],
    "soak_minutes": cfg.get("soak_minutes"),
    "guardrails": cfg["guardrails"],
}
(out / "bench_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${SERVE_RUN}:/serve"
)
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

# Client-only job: no GPUs required. Still use the vLLM image so `vllm bench` exists.
srun --ntasks=1 --kill-on-bad-exit=1 \
  --container-image="${VLLM_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL \
  bash /workspace/scripts/run_vllm_bench.sh /serve/endpoints.json /results "${STAGE}" "${MODEL_NAME}"

echo "benchmark artifacts in ${OUT}"
ls -la "${OUT}"
