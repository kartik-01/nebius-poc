#!/usr/bin/env bash
# Build the validator image without a Docker daemon.
#
# An Enroot image is a squashfs archive, so it can be assembled from outside
# rather than built by executing commands inside a container. That matters on
# Soperator login nodes, where there is no Docker daemon and user namespaces are
# blocked, so neither `docker build` nor `enroot start --rw` is available.
#
# Nothing here executes code inside the image. The rootfs is unpacked, files are
# copied and extracted into it, and it is resealed. Namespaces are only needed to
# run inside a rootfs, not to construct one.
#
# The CUDA runtime is linked statically into gpu_smoke, so the image needs no CUDA
# libraries at all: libcuda.so arrives from the host driver through the NVIDIA
# container hook at run time. That is what keeps the image small.
#
# Usage:
#   ./scripts/build_validator_sqsh.sh [output.sqsh]
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

IMG_DIR="${ROOT}/containers/images"
OUT="${1:-${IMG_DIR}/validator.sqsh}"
BASE_URI="docker://ubuntu:24.04"
BASE_SQSH="${IMG_DIR}/ubuntu-24.04.sqsh"

# Ubuntu 24.04 to match the host glibc, so the host-compiled gpu_smoke runs inside
# the image. Building against a newer glibc than the base provides is the usual way
# this kind of assembly fails.
# Top-level wants. The dependency closure is resolved below, because
# python3-minimal alone ships a partial stdlib: `import json` fails without
# libpython3.12-stdlib, and that pulls a dozen small libraries with it.
PACKAGES=(
  python3 fio jq pciutils numactl ibverbs-utils libaio1t64
)

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/validator-build.XXXXXX")"
cleanup() { rm -rf "${STAGE}"; }
trap cleanup EXIT

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

require() {
  local missing=()
  for tool in "$@"; do
    command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
  done
  if ((${#missing[@]})); then
    echo "missing required tools: ${missing[*]}" >&2
    exit 1
  fi
}

require enroot unsquashfs mksquashfs dpkg-deb apt-get nvcc

mkdir -p "${IMG_DIR}"

# --- base rootfs -------------------------------------------------------------
if [[ ! -f "${BASE_SQSH}" ]]; then
  log "importing ${BASE_URI}"
  enroot import -o "${BASE_SQSH}" "${BASE_URI}"
fi

log "unpacking base rootfs"
unsquashfs -q -f -d "${STAGE}/rootfs" "${BASE_SQSH}"
ROOTFS="${STAGE}/rootfs"

# --- packages ----------------------------------------------------------------
# dpkg -x extracts files without running maintainer scripts, which needs no root.
# That is fine for self-contained binaries; it would not be for packages that rely
# on post-install configuration.
log "resolving dependency closure for ${#PACKAGES[@]} packages"
mapfile -t CLOSURE < <(
  apt-cache depends --recurse --no-recommends --no-suggests \
    --no-conflicts --no-breaks --no-replaces --no-enhances "${PACKAGES[@]}" 2>/dev/null |
  grep -E '^[[:alnum:]]' | sort -u |
  # debconf and friends arrive through interactive-configuration dependencies and
  # are useless in an image nothing is configured inside of.
  grep -vE '^(cdebconf|debconf|libdebian-installer4|libnewt0|libslang2|libtextwrap1)'
)
log "downloading ${#CLOSURE[@]} packages"
mkdir -p "${STAGE}/debs"
( cd "${STAGE}/debs" && apt-get download "${CLOSURE[@]}" >/dev/null 2>&1 ) || {
  echo "apt-get download failed; is the host apt configured?" >&2
  exit 1
}

log "extracting packages into the rootfs"
for deb in "${STAGE}/debs"/*.deb; do
  dpkg -x "${deb}" "${ROOTFS}"
done

# python3-minimal ships /usr/bin/python3.12; make the generic name resolve too.
if [[ -x "${ROOTFS}/usr/bin/python3.12" && ! -e "${ROOTFS}/usr/bin/python3" ]]; then
  ln -s python3.12 "${ROOTFS}/usr/bin/python3"
fi

# --- pyyaml ------------------------------------------------------------------
# The validator parses YAML config. Take the pure-Python wheel so there is no
# compiled extension to match against the image's interpreter build.
log "vendoring pyyaml"
mkdir -p "${STAGE}/wheels"
python3 -m pip download --no-deps --only-binary=:all: \
  --python-version 3.12 --implementation cp --abi none --platform any \
  -d "${STAGE}/wheels" pyyaml >/dev/null 2>&1 || \
  python3 -m pip download --no-deps -d "${STAGE}/wheels" pyyaml >/dev/null 2>&1

SITE="${ROOTFS}/usr/lib/python3/dist-packages"
mkdir -p "${SITE}"
for wheel in "${STAGE}/wheels"/*.whl; do
  [[ -f "${wheel}" ]] || continue
  ( cd "${SITE}" && unzip -qo "${wheel}" -x '*.dist-info/RECORD' )
done

# --- gpu_smoke ---------------------------------------------------------------
# -cudart static removes the libcudart dependency, so no CUDA runtime has to ship
# in the image. The driver library still comes from the host at run time.
log "compiling gpu_smoke with a static CUDA runtime"
nvcc -O2 -cudart static -o "${ROOTFS}/usr/local/bin/gpu_smoke" validator/gpu_smoke.cu
chmod +x "${ROOTFS}/usr/local/bin/gpu_smoke"

# --- NVIDIA container hook ---------------------------------------------------
# Enroot keeps the image environment in /etc/environment, and the NVIDIA hook reads
# NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES from it to decide what to
# inject. CUDA base images set these; a bare Ubuntu base does not, so without this
# block the container starts with no driver libraries and no nvidia-smi, and
# gpu_smoke reports no devices. The driver itself is still supplied by the host.
log "declaring NVIDIA hook variables"
cat >>"${ROOTFS}/etc/environment" <<'ENVEOF'
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
PATH=/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENVEOF

# --- validator code ----------------------------------------------------------
log "copying validator sources"
mkdir -p "${ROOTFS}/opt/validator/configs"
cp -r validator "${ROOTFS}/opt/validator/"
cp configs/validator.yaml "${ROOTFS}/opt/validator/configs/validator.yaml"
find "${ROOTFS}/opt/validator" -name '__pycache__' -type d -prune -exec rm -rf {} +

# --- seal --------------------------------------------------------------------
log "sealing squashfs"
rm -f "${OUT}"
mksquashfs "${ROOTFS}" "${OUT}" -quiet -no-progress -comp zstd -noappend

log "built ${OUT} ($(du -h "${OUT}" | cut -f1))"
cat <<EOF

Point configs/cluster.env at it:

  VALIDATOR_IMAGE=${OUT}

Then run: sbatch --partition="\$SLURM_PARTITION" slurm/validate.sbatch
EOF
