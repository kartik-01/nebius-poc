#!/bin/bash
# Evaluate a base or tuned model. Defaults to the reserved evaluation holdout, the
# share of the category's records that training never sees.
#
# Usage:
#   source configs/cluster.env
#   EVAL_LABEL=base sbatch slurm/evaluate.sbatch
#   EVAL_LABEL=tuned EVAL_ADAPTER=results/raw/<run>/adapter sbatch slurm/evaluate.sbatch
#   # pilot selection set: 100 questions held back inside the adaptation pool.
#   # The pool spans both source splits, so EVAL_SPLIT must be pool, not validation.
#   EVAL_SPLIT=pool IDS_FILE=results/raw/<train_run>/split_manifest.json \
#     EVAL_MODE=forced_choice EVAL_LABEL=pilot-sft sbatch slurm/evaluate.sbatch
#
#SBATCH --job-name=nebius-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate-%j.out
#SBATCH --error=logs/evaluate-%j.err

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

CONFIG="${EVAL_CONFIG:-configs/train_sft.yaml}"
LABEL="${EVAL_LABEL:-base}"
# Matches the default in nebius_poc.evaluate. Leaving this as the raw test split
# would point the launcher at questions the model trained on.
SPLIT="${EVAL_SPLIT:-holdout}"
MODE="${EVAL_MODE:-both}"
ADAPTER="${EVAL_ADAPTER:-}"
IDS_FILE="${IDS_FILE:-}"
LIMIT="${EVAL_LIMIT:-0}"
# Scoring runs without gradients, so evaluation tolerates a much larger batch than
# training does. Raise this to shorten the 460-question holdout evaluations.
BATCH_SIZE="${EVAL_BATCH_SIZE:-0}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_evaluate-${LABEL}_job${SLURM_JOB_ID}"
OUT="${SHARED_ROOT}/results/raw/${RUN_ID}"
mkdir -p "${OUT}"

MOUNTS=(
  "${ROOT}:/workspace"
  "${OUT}:/results"
  "${HF_HOME}:/hf"
)
if [[ -n "${CONTAINER_MOUNTS:-}" ]]; then
  MOUNTS+=("${CONTAINER_MOUNTS}")
fi
MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"

EXTRA=()
if [[ -n "${ADAPTER}" ]]; then
  # Adapter paths are host paths under SHARED_ROOT/results; remap into /results when needed.
  if [[ "${ADAPTER}" == "${OUT}"* ]]; then
    EXTRA+=(--adapter "/results/${ADAPTER#${OUT}/}")
  elif [[ "${ADAPTER}" == "${SHARED_ROOT}/results/"* ]]; then
    # Mount parent results for cross-run adapters.
    MOUNTS+=("${SHARED_ROOT}/results:/all_results")
    MOUNT_CSV="$(IFS=,; echo "${MOUNTS[*]}")"
    EXTRA+=(--adapter "/all_results/${ADAPTER#${SHARED_ROOT}/results/}")
  else
    EXTRA+=(--adapter "${ADAPTER}")
  fi
fi
if [[ -n "${IDS_FILE}" ]]; then
  EXTRA+=(--ids-file "${IDS_FILE}")
fi
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA+=(--limit "${LIMIT}")
fi
if [[ "${BATCH_SIZE}" != "0" ]]; then
  EXTRA+=(--batch-size "${BATCH_SIZE}")
fi

echo "evaluate label=${LABEL} split=${SPLIT} mode=${MODE} config=${CONFIG}"

# Rebuild EXTRA as a string safe to embed in bash -lc.
EXTRA_STR="${EXTRA[*]}"

srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
  --container-image="${TRAIN_IMAGE}" \
  --container-mounts="${MOUNT_CSV}" \
  --container-workdir=/workspace \
  --export=ALL,HF_HOME=/hf,HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}",TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  bash -lc "
    set -euo pipefail
    # shellcheck disable=SC1091
    source /workspace/scripts/container_python_setup.sh
    nebius_poc_container_python_setup
    python -m nebius_poc.evaluate \
      --config ${CONFIG} \
      --label ${LABEL} \
      --split ${SPLIT} \
      --mode ${MODE} \
      --results-root /results \
      ${EXTRA_STR}
  "

echo "evaluation artifacts in ${OUT}"
cat "${OUT}"/*/summary.json 2>/dev/null || cat "${OUT}/summary.json" 2>/dev/null || true
