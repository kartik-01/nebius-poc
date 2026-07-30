#!/usr/bin/env bash
# Import PoC container images with Enroot (no Docker daemon required).
#
# This login jail cannot `enroot start` (user namespaces blocked), so we cannot
# customize rootfs here. Strategy:
#   - import stock CUDA / PyTorch / vLLM squashfs images
#   - mount the repo + a host-built gpu_smoke into jobs
#   - install LoRA Python deps from containers/images/wheels at job start
#
# Usage (repo root):
#   ./scripts/build_images_enroot.sh          # vllm + cuda runtime + pytorch
#   ./scripts/build_images_enroot.sh vllm
#   ./scripts/build_images_enroot.sh wheels   # pip download only
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG_DIR="${ROOT}/containers/images"
mkdir -p "${IMG_DIR}"
cd "${IMG_DIR}"

TARGET="${1:-all}"

CUDA_RUNTIME_URI="docker://nvidia/cuda:12.4.1-runtime-ubuntu22.04"
TORCH_URI="docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
VLLM_URI="docker://vllm/vllm-openai:v0.8.5"

log() { echo "[build-images] $*"; }

need_import() {
  local out="$1" uri="$2"
  if [[ -f "${out}" ]]; then
    log "reuse existing ${out}"
    return 0
  fi
  log "import ${uri} -> ${out}"
  enroot import -o "${out}" "${uri}"
}

compile_gpu_smoke() {
  if [[ -x "${IMG_DIR}/gpu_smoke" ]]; then
    log "reuse existing gpu_smoke"
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    log "nvcc not on PATH; skip gpu_smoke (validator CUDA check will be UNKNOWN)"
    return 0
  fi
  log "compile gpu_smoke with host nvcc"
  nvcc -O2 -o "${IMG_DIR}/gpu_smoke" "${ROOT}/validator/gpu_smoke.cu"
}

# These wheels are installed inside TRAIN_IMAGE, never on the login node, so they
# must match the container's interpreter rather than this one. The pinned pytorch
# image ships 3.11; this login node runs 3.12. Downloading with the host default
# produces cp312 wheels and the job then dies at startup with "No matching
# distribution found", which is a confusing way to learn about a tag mismatch.
CONTAINER_PYTHON="${CONTAINER_PYTHON:-3.11}"

download_wheels() {
  mkdir -p "${IMG_DIR}/wheels"
  log "download LoRA stack wheels for container python ${CONTAINER_PYTHON} into ${IMG_DIR}/wheels"
  # --python-version requires --only-binary, and the platform list has to cover
  # every manylinux level the dependency tree publishes, since pip treats these
  # as the complete set of acceptable tags rather than a minimum.
  python3 -m pip download -d "${IMG_DIR}/wheels" \
    --python-version "${CONTAINER_PYTHON}" \
    --only-binary=:all: \
    --platform manylinux1_x86_64 \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_5_x86_64 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_27_x86_64 \
    --platform manylinux_2_28_x86_64 \
    "accelerate>=0.34" "datasets>=2.19" "numpy>=1.26" \
    "peft>=0.13" "pyyaml>=6.0" "transformers>=4.56"
}

print_env_hints() {
  echo
  echo "Suggested configs/cluster.env values:"
  [[ -f "${IMG_DIR}/cuda-12.4.1-runtime-ubuntu22.04.sqsh" ]] && \
    echo "# optional thin CUDA base (validator prefers pytorch image for Python):"
  [[ -f "${IMG_DIR}/pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh" ]] && \
    echo "TRAIN_IMAGE=${IMG_DIR}/pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh" && \
    echo "VALIDATOR_IMAGE=${IMG_DIR}/pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh"
  [[ -f "${IMG_DIR}/vllm-openai-v0.8.5.sqsh" ]] && \
    echo "VLLM_IMAGE=${IMG_DIR}/vllm-openai-v0.8.5.sqsh"
  # These hints are informational. Without this the last test above decides the
  # function's exit status, so running a target that builds no images (wheels,
  # gpu-smoke) would return 1 and abort any caller using set -e.
  return 0
}

case "${TARGET}" in
  vllm) need_import "vllm-openai-v0.8.5.sqsh" "${VLLM_URI}" ;;
  cuda) need_import "cuda-12.4.1-runtime-ubuntu22.04.sqsh" "${CUDA_RUNTIME_URI}" ;;
  train|pytorch) need_import "pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh" "${TORCH_URI}" ;;
  wheels) download_wheels ;;
  gpu-smoke) compile_gpu_smoke ;;
  all)
    compile_gpu_smoke
    need_import "vllm-openai-v0.8.5.sqsh" "${VLLM_URI}"
    need_import "cuda-12.4.1-runtime-ubuntu22.04.sqsh" "${CUDA_RUNTIME_URI}"
    need_import "pytorch-2.5.1-cuda12.4-cudnn9-runtime.sqsh" "${TORCH_URI}"
    download_wheels
    ;;
  *)
    echo "usage: $0 [all|vllm|cuda|train|wheels|gpu-smoke]" >&2
    exit 2
    ;;
esac

print_env_hints
