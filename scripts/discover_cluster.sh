#!/usr/bin/env bash
# Read-only cluster discovery. Writes under .local/ (gitignored).
# Does not modify the system and does not guess values into configs/cluster.env.
#
# Usage:
#   ./scripts/discover_cluster.sh
#   ./scripts/discover_cluster.sh --with-allocation   # needs a working Slurm association
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/.local/discovery/${STAMP}"
mkdir -p "${OUT}"
LATEST="${ROOT}/.local/discovery/latest"
ln -sfn "${STAMP}" "${LATEST}"

WITH_ALLOC=0
if [[ "${1:-}" == "--with-allocation" ]]; then
  WITH_ALLOC=1
fi

log() { printf '%s\n' "$*" | tee -a "${OUT}/discovery.log"; }

run_capture() {
  local name="$1"
  shift
  log "+ $*"
  if "$@" >"${OUT}/${name}.out" 2>"${OUT}/${name}.err"; then
    log "  ok (${name})"
    return 0
  fi
  local rc=$?
  log "  failed rc=${rc} (${name})"
  return 0
}

log "discovery started utc=${STAMP} host=$(hostname) user=${USER}"
log "out=${OUT}"

run_capture sinfo sinfo -Nel
run_capture partitions scontrol show partition
run_capture config scontrol show config
run_capture squeue squeue -a
run_capture assoc sacctmgr -n show assoc format=User%-16,Account%-16,QOS%-24
run_capture assoc_user sacctmgr -n show assoc user="${USER}" format=User%-16,Account%-16,QOS%-24

run_capture which_tools bash -lc '
  for cmd in srun sbatch enroot all_reduce_perf all_reduce_perf_mpi nvidia-smi ibv_devices ibv_devinfo docker; do
    printf "%-22s %s\n" "${cmd}" "$(command -v "${cmd}" || echo MISSING)"
  done
  enroot version 2>/dev/null || true
'

run_capture pyxis_help bash -lc "srun --help | grep -E 'container-image|container-mounts|container-workdir|container-env' || true"
run_capture df df -hT
run_capture mounts mount
run_capture ulimit bash -lc 'ulimit -a'

# Allocation probe — must not be treated as success if accounting rejects us.
log "+ srun --test-only allocation probe"
if srun --test-only --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:05:00 hostname \
  >"${OUT}/alloc_probe.out" 2>"${OUT}/alloc_probe.err"; then
  log "  allocation probe ok"
  ALLOC_OK=1
else
  log "  allocation probe FAILED (see alloc_probe.err)"
  ALLOC_OK=0
fi

if [[ "${WITH_ALLOC}" -eq 1 ]]; then
  if [[ "${ALLOC_OK}" -ne 1 ]]; then
    log "refusing --with-allocation: accounting probe failed"
    exit 2
  fi
  run_capture node_hostname srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 hostname
  run_capture nvidia_smi srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 nvidia-smi
  run_capture nvidia_L srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 nvidia-smi -L
  run_capture nvidia_topo srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 nvidia-smi topo -m
  run_capture ibv_devices srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 bash -lc 'ibv_devices || true'
  run_capture ibv_devinfo srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 bash -lc 'ibv_devinfo -l || true'
  run_capture node_env srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 bash -lc 'env | grep -E "SLURM|NCCL|UCX|CUDA" | sort'
  run_capture node_df srun --partition=earlytalent --nodes=1 --gres=gpu:1 --time=00:10:00 df -hT
fi

# Machine-readable summary of facts we are willing to assert without guessing.
python3 - "${OUT}" "${ALLOC_OK}" <<'PY'
import json, re, sys
from pathlib import Path

out = Path(sys.argv[1])
alloc_ok = sys.argv[2] == "1"

def read(name: str) -> str:
    path = out / name
    return path.read_text() if path.exists() else ""

sinfo = read("sinfo.out")
partition = read("partitions.out")
assoc_user = read("assoc_user.out").strip()
tools = read("which_tools.out")
df = read("df.out")
pyxis = read("pyxis_help.out")
probe_err = read("alloc_probe.err").strip()

partition_name = None
m = re.search(r"PartitionName=(\S+)", partition)
if m:
    partition_name = m.group(1)

qos = None
m = re.search(r"AllowQos=(\S+)", partition)
if m:
    qos = m.group(1).split(",")[0]

tmp_disk = None
m = re.search(r"TMP_DISK\s+(\d+)", sinfo) or re.search(r"\s+(\d+)\s+\d+\s+gpu_", sinfo)
# Prefer the TMP_DISK column value from sinfo -Nel header layout.
for line in sinfo.splitlines():
    if line.startswith("worker-"):
        parts = line.split()
        # NODELIST NODES PARTITION STATE CPUS S:C:T MEMORY TMP_DISK ...
        if len(parts) >= 8 and parts[7].isdigit():
            tmp_disk = int(parts[7])
            break

shared = []
for line in df.splitlines()[1:]:
    cols = line.split()
    if len(cols) >= 7 and cols[-1] in ("/home", "/mnt/data"):
        shared.append({"mount": cols[-1], "fstype": cols[1], "size": cols[2], "avail": cols[4]})

def tool_path(name: str):
    for line in tools.splitlines():
        if line.startswith(name):
            path = line.split(None, 1)[-1].strip()
            return None if path == "MISSING" else path
    return None

summary = {
    "observed": {
        "partition": partition_name,
        "qos_from_partition": qos,
        "user_association": assoc_user or None,
        "allocation_probe_ok": alloc_ok,
        "allocation_probe_error": probe_err or None,
        "tmp_disk_mib_advertised": tmp_disk,
        "shared_mounts": shared,
        "tools": {
            "all_reduce_perf": tool_path("all_reduce_perf"),
            "all_reduce_perf_mpi": tool_path("all_reduce_perf_mpi"),
            "enroot": tool_path("enroot"),
            "srun": tool_path("srun"),
            "sbatch": tool_path("sbatch"),
        },
        "pyxis_flags_documented": bool(pyxis.strip()),
        "pyxis_help": [line.strip() for line in pyxis.splitlines() if line.strip()],
    },
    "still_unknown_without_allocation": [
        "LOCAL_SCRATCH",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "UCX_NET_DEVICES",
        "driver/CUDA versions on compute nodes",
        "Pyxis launch of a real container image",
        "whether /mnt/data is shared across nodes or node-local",
    ],
    "blocked": [] if alloc_ok else [
        "Slurm accounting has no association for this user; job submission fails",
        "one-GPU validator smoke",
        "one-GPU training smoke",
        "in-allocation IB/NCCL interface discovery",
    ],
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

log "wrote ${OUT}/summary.json"
log "discovery finished"
