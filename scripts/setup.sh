#!/usr/bin/env bash
# One-command setup for a fresh clone. Idempotent: every stage checks before it
# acts, so re-running costs seconds and never repeats a download.
#
#   ./scripts/setup.sh --check          report what is present and what is missing
#   ./scripts/setup.sh                  local setup only: venv, config skeleton, tests
#   ./scripts/setup.sh --from <path>    reuse images, wheels and HF cache from an
#                                       existing checkout on shared storage
#   ./scripts/setup.sh --build          build images and prefetch weights (~25 min)
#
# The default does no downloads beyond pip, so it is safe to run before you know
# whether you have cluster access. Cluster-specific values are never guessed:
# configs/cluster.env is created from the template and you fill it in, with
# ./scripts/discover_cluster.sh to read the values off the cluster.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -t 1 ]]; then
  B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
  B=""; D=""; G=""; Y=""; R=""; N=""
fi

CHECK=0
BUILD=0
DONOR=""
MISSING=0

ok()   { printf '  %s%-22s%s %s\n' "${G}" "$1" "${N}" "${2:-}"; }
todo() { printf '  %s%-22s%s %s\n' "${Y}" "$1" "${N}" "${2:-}"; MISSING=1; }
bad()  { printf '  %s%-22s%s %s\n' "${R}" "$1" "${N}" "${2:-}"; MISSING=1; }
head_() { printf '\n%s%s%s\n' "${B}" "$1" "${N}"; }
note() { printf '  %s%s%s\n' "${D}" "$1" "${N}"; }

env_path() { [[ -f configs/cluster.env ]] && grep -E "^$1=" configs/cluster.env | cut -d= -f2- || true; }

# Record a value only when the key is still blank, so a config carried over by
# --from or edited by hand is never clobbered.
fill_env_if_empty() {
  [[ -f configs/cluster.env ]] || return 0
  local key="$1" val="$2" cur
  cur="$(grep -E "^${key}=" configs/cluster.env | head -1 | cut -d= -f2-)"
  [[ -z "${cur}" ]] || return 0
  sed -i "s|^${key}=.*|${key}=${val}|" configs/cluster.env
}

# Read a Slurm value off the cluster, but only accept an unambiguous answer.
# Several partitions or accounts means a choice, and a setup script has no
# business making that choice for someone.
discover_slurm() {
  local value=""
  case "$1" in
    partition) value="$(sinfo -h -o '%P' 2>/dev/null | grep '\*$' | tr -d '*' | sort -u)" ;;
    account)   value="$(sacctmgr -nP show assoc user="${USER}" format=Account 2>/dev/null | sed '/^$/d' | sort -u)" ;;
    qos)       value="$(sacctmgr -nP show assoc user="${USER}" format=QOS 2>/dev/null | tr ',' '\n' | sed '/^$/d' | sort -u)" ;;
  esac
  [[ "$(printf '%s\n' "${value}" | grep -c .)" == "1" ]] || return 1
  printf '%s' "${value}"
}

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (( $# )); do
  case "$1" in
    --check) CHECK=1 ;;
    --build) BUILD=1 ;;
    --from) DONOR="${2:?--from needs a path to an existing checkout}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -n "${DONOR}" ]]; then
  DONOR="$(cd "${DONOR}" && pwd)"
  [[ "${DONOR}" != "${ROOT}" ]] || { echo "--from must be a different checkout" >&2; exit 2; }
fi

# ---------------------------------------------------------------- prerequisites
head_ "Prerequisites"
for tool in python3 git; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool}" "$(command -v "${tool}")"
  else bad "${tool}" "required, not on PATH"; fi
done
# Cluster tools are optional locally: the tests and the display steps do not need them.
for tool in sbatch enroot; do
  if command -v "${tool}" >/dev/null 2>&1; then ok "${tool}" "$(command -v "${tool}")"
  else note "${tool} not found, fine for local work, needed to submit jobs"; fi
done

# ------------------------------------------------------------------------ venv
head_ "Python environment"
if [[ -x .venv/bin/pytest ]]; then
  ok ".venv" "already installed"
elif (( CHECK )); then
  todo ".venv" "run: make install"
else
  note "creating .venv and installing the package (a few minutes)"
  make install >/dev/null
  ok ".venv" "installed"
fi

# ------------------------------------------------------------------- cluster.env
head_ "Cluster configuration"
if [[ -f configs/cluster.env ]]; then
  ok "configs/cluster.env" "present, left untouched"
elif (( CHECK )); then
  todo "configs/cluster.env" "missing"
elif [[ -n "${DONOR}" && -f "${DONOR}/configs/cluster.env" ]]; then
  # Reuse the donor's values, but this checkout must own its own artifact tree.
  sed "s|^SHARED_ROOT=.*|SHARED_ROOT=${ROOT}|" "${DONOR}/configs/cluster.env" >configs/cluster.env
  ok "configs/cluster.env" "copied from donor, SHARED_ROOT repointed here"
else
  cp configs/cluster.env.example configs/cluster.env
  todo "configs/cluster.env" "created from template, fill it in"
  note "run ./scripts/discover_cluster.sh to read the values off the cluster"
fi

# ------------------------------------------------------------------- association
head_ "Slurm association"
if [[ ! -f configs/cluster.env ]]; then
  todo "slurm values" "no configs/cluster.env to write to"
else
  if (( ! CHECK )); then
    # /tmp is the conventional node-local scratch. It is a default rather than a
    # discovery, so it is only ever used to fill a blank.
    for spec in "SLURM_PARTITION:partition" "SLURM_ACCOUNT:account" "SLURM_QOS:qos"; do
      key="${spec%%:*}"
      [[ -n "$(env_path "${key}")" ]] && continue
      if found="$(discover_slurm "${spec#*:}")"; then
        fill_env_if_empty "${key}" "${found}"
      fi
    done
    fill_env_if_empty LOCAL_SCRATCH "/tmp"
  fi
  for key in SLURM_PARTITION SLURM_ACCOUNT SLURM_QOS LOCAL_SCRATCH; do
    value="$(env_path "${key}")"
    if [[ -n "${value}" ]]; then
      ok "${key}" "${value}"
    else
      todo "${key}" "blank, and could not be read unambiguously from the cluster"
    fi
  done
fi

# ------------------------------------------------------------------------ images
head_ "Container images"
if (( BUILD )) && (( ! CHECK )); then
  note "importing base images and building the validator (about 15 minutes)"
  # 'all' imports the images and downloads the wheelhouse in one pass, so the
  # wheelhouse stage below has nothing left to do.
  ./scripts/build_images_enroot.sh all
  ./scripts/build_validator_sqsh.sh
  # The Slurm values still need discovery, but these five are facts about what
  # was just written, not guesses, so leaving them blank would be pointless work.
  fill_env_if_empty SHARED_ROOT "${ROOT}"
  fill_env_if_empty HF_HOME "${ROOT}/hf_cache"
  fill_env_if_empty TRAIN_IMAGE "${ROOT}/containers/images/pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh"
  fill_env_if_empty VALIDATOR_IMAGE "${ROOT}/containers/images/validator.sqsh"
  fill_env_if_empty VLLM_IMAGE "${ROOT}/containers/images/vllm-openai-v0.8.5.sqsh"
  ok "images" "built, paths recorded in configs/cluster.env"
fi

for var in TRAIN_IMAGE VALIDATOR_IMAGE VLLM_IMAGE; do
  path="$(env_path "${var}")"
  if [[ -n "${path}" && -f "${path}" ]]; then
    ok "${var}" "$(du -h "${path}" | cut -f1)"
  else
    todo "${var}" "${path:-unset in configs/cluster.env}"
  fi
done

# -------------------------------------------------------------------- wheelhouse
head_ "Wheelhouse"
# Must live inside this checkout: container_python_setup.sh resolves it at
# /workspace/containers/images/wheels, the mounted repo root inside the container.
if [[ -d containers/images/wheels ]] && compgen -G 'containers/images/wheels/*' >/dev/null; then
  ok "wheels" "$(find containers/images/wheels -type f | wc -l) files"
elif (( CHECK )); then
  todo "wheels" "missing, needed by the training and evaluation jobs"
elif [[ -n "${DONOR}" && -d "${DONOR}/containers/images/wheels" ]]; then
  note "copying the wheelhouse from the donor checkout (about 3.5 GB)"
  mkdir -p containers/images
  cp -r "${DONOR}/containers/images/wheels" containers/images/wheels
  ok "wheels" "copied"
else
  todo "wheels" "pass --from <checkout> to copy, or --build to download"
fi

# --------------------------------------------------------------------- HF assets
head_ "Model and dataset cache"
HF="$(env_path HF_HOME)"
if [[ -n "${HF}" && -d "${HF}" ]]; then
  ok "HF_HOME" "${HF} ($(du -sh "${HF}" 2>/dev/null | cut -f1))"
elif (( CHECK )) || (( ! BUILD )); then
  todo "HF_HOME" "${HF:-unset}, run with --build to prefetch"
else
  .venv/bin/python scripts/prefetch_assets.py --hf-home "${HF:-${ROOT}/hf_cache}" --out results/raw
  ok "HF_HOME" "prefetched"
fi

# ----------------------------------------------------------------------- verify
head_ "Verification"
if (( CHECK )); then
  todo "tests" "not run in --check mode"
elif [[ -x .venv/bin/pytest ]]; then
  # Run once and read the summary out of the captured output, rather than running
  # the suite a second time just to print a line. Not -q: that suppresses the
  # summary line entirely, leaving only progress dots.
  if out="$(.venv/bin/pytest 2>&1)"; then
    ok "tests" "$(printf '%s\n' "${out}" | grep -oE '[0-9]+ passed.*' | tail -1)"
  else
    bad "tests" "failing, run 'make test' to see why"
  fi
else
  bad "tests" "no virtualenv"
fi

# ----------------------------------------------------------------------- summary
head_ "Next"
if (( MISSING )); then
  note "Some items above are still outstanding."
  note "Local work (tests, ./scripts/demo.sh accuracy, ./scripts/demo.sh throughput)"
  note "needs nothing further. Submitting jobs needs configs/cluster.env, the"
  note "wheelhouse, the images and the model cache."
else
  note "Ready. ./scripts/demo.sh walks the whole story, or see docs/RUNBOOK.md."
fi
echo
