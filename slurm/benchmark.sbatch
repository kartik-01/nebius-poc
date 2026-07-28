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
#SBATCH --cpus-per-task=8
# Client-only load generator. Without an explicit request Slurm reserves the whole
# node's memory, which cannot be satisfied while the serve job holds both nodes.
#SBATCH --mem=32G
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
: "${SERVE_RUN:?set SERVE_RUN to the results/raw/<run_id> directory of the serve job}"

STAGE="${STAGE:-screen}"
BENCH_CONFIG="${BENCH_CONFIG:-configs/benchmark.yaml}"

# vLLM serves the model under the exact path it was launched with, and the bench
# client reuses that same string to load the tokenizer. So the served id must be
# the absolute model path, and that path must be mounted at the same location in
# the client container.
SERVED_MODEL="$(sed -n 's/.*"model_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  "${SERVE_RUN}/serve_manifest.json" 2>/dev/null || true)"
MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL}}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "could not determine served model path from ${SERVE_RUN}/serve_manifest.json" >&2
  exit 2
fi

if [[ ! -f "${SERVE_RUN}/endpoints.json" ]]; then
  echo "missing ${SERVE_RUN}/endpoints.json — is the serve job ready?" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_bench-${STAGE}_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

# Compute nodes ship a bare system python without pyyaml; prefer the repo venv.
PYBIN="python3"
if ! "${PYBIN}" -c "import yaml" >/dev/null 2>&1; then
  if [[ -x "${ROOT}/.venv/bin/python" ]] && "${ROOT}/.venv/bin/python" -c "import yaml" >/dev/null 2>&1; then
    PYBIN="${ROOT}/.venv/bin/python"
  else
    echo "no python with pyyaml available; run 'make install' to create .venv" >&2
    exit 1
  fi
fi

"${PYBIN}" - "${BENCH_CONFIG}" "${SERVE_RUN}/endpoints.json" "${OUT}" "${STAGE}" <<'PY'
import json, os, sys, yaml
from pathlib import Path

cfg = yaml.safe_load(open(sys.argv[1]))
endpoints = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
stage = sys.argv[4]

conc = os.environ.get("CONCURRENCY")
if conc:
    # Accepts a single point or a comma-separated list, so a saturation sweep can
    # be driven without editing the committed config.
    concurrencies = [int(part) for part in conc.replace(" ", "").split(",") if part]
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
# Same path inside the client so --model resolves as a local tokenizer directory.
if [[ -d "${MODEL_NAME}" ]]; then
  MOUNTS+=("${MODEL_NAME}:${MODEL_NAME}")
fi
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
