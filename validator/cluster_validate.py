"""Portable cluster qualification: inventory, CUDA smoke, fio, NCCL aggregation.

This module is what runs inside the validator container. It has no training or
MPI dependencies. NCCL itself is exercised by Nebius-provided binaries outside
this process; we only parse their logs and fold everything into one report.
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from validator.parse_nccl import (
    node_asymmetry,
    parse_nccl_log,
    summarize_repetitions,
)

log = logging.getLogger(__name__)

STATUSES = ("PASS", "WARN", "FAIL", "UNKNOWN", "NOT_OBSERVABLE")
_STATUS_RANK = {name: index for index, name in enumerate(STATUSES)}


def worst_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return "UNKNOWN"
    return max(statuses, key=lambda status: _STATUS_RANK.get(status, 0))


def expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def load_config(path: Path) -> dict:
    return expand_env(yaml.safe_load(Path(path).read_text()) or {})


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _run(command: Sequence[str], timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _first_line(command: Sequence[str]) -> str | None:
    try:
        result = _run(command, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[0].strip()


def collect_gpus() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {
            "status": "NOT_OBSERVABLE",
            "reason": "nvidia-smi not on PATH",
            "count": 0,
            "devices": [],
        }

    query = (
        "uuid,name,memory.total,driver_version,pci.bus_id"
    )
    try:
        result = _run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "NOT_OBSERVABLE",
            "reason": str(exc),
            "count": 0,
            "devices": [],
        }

    if result.returncode != 0:
        return {
            "status": "FAIL",
            "reason": result.stderr.strip() or "nvidia-smi failed",
            "count": 0,
            "devices": [],
        }

    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        devices.append(
            {
                "uuid": parts[0],
                "name": parts[1],
                "memory_mib": _maybe_int(parts[2]),
                "driver_version": parts[3],
                "pci_bus_id": parts[4],
            }
        )

    return {"status": "PASS", "reason": None, "count": len(devices), "devices": devices}


def _maybe_int(token: str) -> int | None:
    try:
        return int(float(token))
    except ValueError:
        return None


def collect_inventory(local_scratch: str | None) -> dict:
    gpus = collect_gpus()
    cpu_model = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        cpu_model = match.group(1).strip() if match else None
    except OSError:
        cpu_model = None

    numa = _first_line(["numactl", "--hardware"])
    topology = None
    if shutil.which("nvidia-smi"):
        try:
            topo = _run(["nvidia-smi", "topo", "-m"], timeout=30)
            topology = topo.stdout if topo.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            topology = None

    ib_devices = []
    if shutil.which("ibv_devices"):
        try:
            result = _run(["ibv_devices"], timeout=15)
            if result.returncode == 0:
                ib_devices = [
                    line.strip()
                    for line in result.stdout.splitlines()[1:]
                    if line.strip()
                ]
        except (OSError, subprocess.SubprocessError):
            ib_devices = []

    mounts = []
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mounts.append({"device": parts[0], "path": parts[1], "fstype": parts[2]})
    except OSError:
        mounts = []

    scratch_status = "PASS"
    scratch_reason = None
    if not local_scratch:
        scratch_status = "UNKNOWN"
        scratch_reason = "LOCAL_SCRATCH not configured"
    elif not Path(local_scratch).exists():
        scratch_status = "FAIL"
        scratch_reason = f"{local_scratch} does not exist"

    return {
        "hostname": socket.gethostname(),
        "slurm_nodename": os.environ.get("SLURMD_NODENAME")
        or os.environ.get("SLURM_NODEID"),
        "slurm": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("SLURM_")
        },
        "utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "numa": numa,
        "gpu_topology": topology,
        "infiniband_devices": ib_devices,
        "mounts": mounts,
        "local_scratch": {
            "path": local_scratch or None,
            "status": scratch_status,
            "reason": scratch_reason,
        },
        "image": {
            "validator_image": os.environ.get("VALIDATOR_IMAGE"),
            "container_hostname": os.environ.get("HOSTNAME"),
        },
        "gpus": gpus,
        "cuda_runtime": _first_line(
            ["bash", "-lc", "echo $CUDA_VERSION"]
        )
        or os.environ.get("CUDA_VERSION"),
    }


def run_gpu_smoke(binary: Path, out_path: Path) -> dict:
    if not binary.exists():
        payload = {
            "status": "UNKNOWN",
            "reason": f"gpu_smoke binary missing: {binary}",
            "elapsed_seconds": None,
        }
        write_json(out_path, payload)
        return payload

    started = time.perf_counter()
    try:
        result = _run([str(binary)], timeout=120)
    except subprocess.TimeoutExpired:
        payload = {
            "status": "FAIL",
            "reason": "gpu_smoke timed out",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(out_path, payload)
        return payload
    except OSError as exc:
        payload = {
            "status": "FAIL",
            "reason": str(exc),
            "elapsed_seconds": None,
        }
        write_json(out_path, payload)
        return payload

    elapsed = time.perf_counter() - started
    stdout = result.stdout.strip()
    detail = {}
    if stdout:
        try:
            detail = json.loads(stdout)
        except json.JSONDecodeError:
            detail = {"raw_stdout": stdout}

    status = "PASS" if result.returncode == 0 else "FAIL"
    payload = {
        "status": status,
        "reason": None if status == "PASS" else (result.stderr.strip() or "nonzero exit"),
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
        "detail": detail,
    }
    write_json(out_path, payload)
    return payload


_STORAGE_CHUNK = 1 << 20


def _direct_write(target: Path, total: int) -> None:
    # O_DIRECT needs the buffer, its address, and the length to be block-aligned;
    # mmap gives us a page-aligned buffer without ctypes.
    import mmap

    buffer = mmap.mmap(-1, _STORAGE_CHUNK)
    buffer.write(os.urandom(_STORAGE_CHUNK))
    view = memoryview(buffer)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
        try:
            written = 0
            while written < total:
                written += os.write(fd, view)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        # mmap.close() raises while any memoryview still exports its pointer.
        view.release()
        buffer.close()


def _buffered_write(target: Path, total: int) -> None:
    chunk = os.urandom(_STORAGE_CHUNK)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        written = 0
        while written < total:
            written += os.write(fd, chunk)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_back(target: Path, direct: bool) -> int:
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(target, flags)
    try:
        if not direct:
            # Without O_DIRECT the file we just wrote is still in page cache, which
            # would measure memory bandwidth instead of the device.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        read = 0
        if direct:
            import mmap

            buffer = mmap.mmap(-1, _STORAGE_CHUNK)
            try:
                while True:
                    got = os.readv(fd, [buffer])
                    if not got:
                        break
                    read += got
            finally:
                buffer.close()
        else:
            while True:
                block = os.read(fd, _STORAGE_CHUNK)
                if not block:
                    break
                read += len(block)
        return read
    finally:
        os.close(fd)


def builtin_storage_benchmark(scratch: str, size_gib: int) -> dict:
    """Bounded sequential write/read used when fio is unavailable.

    Prefers O_DIRECT so the numbers reflect the device rather than page cache, and
    degrades to buffered I/O plus POSIX_FADV_DONTNEED on filesystems that reject it.
    """
    scratch_path = Path(scratch)
    scratch_path.mkdir(parents=True, exist_ok=True)
    target = scratch_path / f"storage-validate-{os.getpid()}.bin"
    total = size_gib * (1 << 30)

    direct = True
    try:
        start = time.perf_counter()
        _direct_write(target, total)
        write_seconds = time.perf_counter() - start
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP, errno.EPERM):
            target.unlink(missing_ok=True)
            return {"status": "FAIL", "reason": f"builtin write failed: {exc}"}
        direct = False
        try:
            start = time.perf_counter()
            _buffered_write(target, total)
            write_seconds = time.perf_counter() - start
        except OSError as buffered_exc:
            target.unlink(missing_ok=True)
            return {"status": "FAIL", "reason": f"builtin write failed: {buffered_exc}"}

    try:
        start = time.perf_counter()
        read_bytes = _read_back(target, direct)
        read_seconds = time.perf_counter() - start
    except OSError as exc:
        return {"status": "FAIL", "reason": f"builtin read failed: {exc}"}
    finally:
        target.unlink(missing_ok=True)

    if read_bytes != total:
        return {
            "status": "FAIL",
            "reason": f"read back {read_bytes} bytes, expected {total}",
        }

    return {
        "status": "PASS",
        "method": "builtin-o_direct" if direct else "builtin-buffered-fadvise",
        "note": (
            "fio unavailable; bounded sequential benchmark from the validator. "
            "Not directly comparable to fio numbers from another platform."
        ),
        "size_bytes": total,
        "write": {
            "status": "PASS",
            "bw_bytes": int(total / write_seconds) if write_seconds > 0 else None,
            "runtime_ms": int(write_seconds * 1000),
        },
        "read": {
            "status": "PASS",
            "bw_bytes": int(total / read_seconds) if read_seconds > 0 else None,
            "runtime_ms": int(read_seconds * 1000),
        },
        "path": str(target),
    }


def run_fio(scratch: str | None, size_gib: int, enabled_shared: bool, out_path: Path) -> dict:
    if enabled_shared:
        # Shared-filesystem fio is opt-in because it can disturb other users.
        pass
    if not scratch:
        payload = {
            "status": "UNKNOWN",
            "reason": "LOCAL_SCRATCH not configured; fio skipped",
        }
        write_json(out_path, payload)
        return payload
    if shutil.which("fio") is None:
        payload = builtin_storage_benchmark(scratch, size_gib)
        write_json(out_path, payload)
        return payload

    scratch_path = Path(scratch)
    scratch_path.mkdir(parents=True, exist_ok=True)
    target = scratch_path / f"fio-validate-{os.getpid()}.bin"
    size = f"{size_gib}G"

    def _fio(mode: str) -> dict:
        command = [
            "fio",
            "--name",
            f"validate-{mode}",
            "--filename",
            str(target),
            "--rw",
            mode,
            "--bs",
            "1M",
            "--size",
            size,
            "--iodepth",
            "16",
            "--direct",
            "1",
            "--ioengine",
            "libaio",
            "--output-format",
            "json",
            "--runtime",
            "30",
            "--time_based",
            "0",
        ]
        try:
            result = _run(command, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "FAIL", "reason": str(exc)}
        if result.returncode != 0:
            return {
                "status": "FAIL",
                "reason": result.stderr.strip() or "fio failed",
                "stdout_tail": result.stdout[-500:],
            }
        try:
            parsed = json.loads(result.stdout)
            job = parsed["jobs"][0]
            key = "read" if "read" in mode else "write"
            stats = job[key]
            return {
                "status": "PASS",
                "bw_bytes": stats.get("bw_bytes"),
                "iops": stats.get("iops"),
                "runtime_ms": stats.get("runtime"),
            }
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            return {"status": "UNKNOWN", "reason": f"fio JSON parse failed: {exc}"}

    try:
        write = _fio("write")
        read = _fio("read") if write.get("status") == "PASS" else {
            "status": "UNKNOWN",
            "reason": "skipped because write failed",
        }
        status = worst_status([write.get("status", "UNKNOWN"), read.get("status", "UNKNOWN")])
        payload = {"status": status, "write": write, "read": read, "path": str(target)}
    finally:
        target.unlink(missing_ok=True)

    write_json(out_path, payload)
    return payload


def check_gpu_expectations(inventory: dict, config: dict) -> dict:
    gpus = inventory["gpus"]
    hard_failures: list[str] = []
    warnings: list[str] = []
    unknown_checks: list[str] = []

    if gpus["status"] == "NOT_OBSERVABLE":
        return {
            "status": "NOT_OBSERVABLE",
            "hard_failures": [],
            "warnings": [],
            "unknown_checks": ["gpu_inventory"],
            "reason": gpus.get("reason"),
        }
    if gpus["status"] == "FAIL":
        return {
            "status": "FAIL",
            "hard_failures": [gpus.get("reason") or "gpu inventory failed"],
            "warnings": [],
            "unknown_checks": [],
        }

    expected = config["expected_gpus_per_node"]
    if gpus["count"] != expected:
        hard_failures.append(
            f"visible GPU count {gpus['count']} != expected_gpus_per_node {expected}"
        )

    pattern = config["expected_gpu_name_pattern"]
    for device in gpus["devices"]:
        if pattern not in device["name"]:
            hard_failures.append(
                f"GPU {device['uuid']} name {device['name']!r} does not contain {pattern!r}"
            )

    names = {device["name"] for device in gpus["devices"]}
    if len(names) > 1:
        hard_failures.append(f"mixed GPU models in allocation: {sorted(names)}")

    status = "FAIL" if hard_failures else "PASS"
    return {
        "status": status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "unknown_checks": unknown_checks,
        "count": gpus["count"],
        "names": sorted(names),
    }


def absolute_threshold_check(config: dict) -> dict:
    """Optional absolute thresholds. Empty config means UNKNOWN, never a fabricated PASS."""
    thresholds = config.get("absolute_thresholds") or {}
    if not thresholds:
        return {
            "status": "UNKNOWN",
            "reason": "absolute_thresholds not configured",
            "unknown_checks": ["absolute_performance"],
        }
    return {
        "status": "UNKNOWN",
        "reason": "absolute threshold evaluation is reserved for an approved reference run",
        "unknown_checks": ["absolute_performance"],
        "configured": thresholds,
    }


def aggregate(
    config: dict,
    inventory: dict,
    compute: dict,
    storage: dict,
    nccl_dir: Path | None,
) -> dict:
    hard_failures: list[str] = []
    warnings: list[str] = []
    unknown_checks: list[str] = []

    gpu_check = check_gpu_expectations(inventory, config)
    hard_failures.extend(gpu_check["hard_failures"])
    warnings.extend(gpu_check["warnings"])
    unknown_checks.extend(gpu_check["unknown_checks"])

    if inventory["local_scratch"]["status"] == "FAIL":
        hard_failures.append(inventory["local_scratch"]["reason"])
    elif inventory["local_scratch"]["status"] == "UNKNOWN":
        unknown_checks.append("local_scratch")

    if compute["status"] == "FAIL":
        hard_failures.append(compute.get("reason") or "gpu_smoke failed")
    elif compute["status"] in ("UNKNOWN", "NOT_OBSERVABLE"):
        unknown_checks.append("gpu_smoke")

    if storage["status"] == "FAIL":
        hard_failures.append(storage.get("reason") or "fio failed")
    elif storage["status"] in ("UNKNOWN", "NOT_OBSERVABLE"):
        unknown_checks.append("storage")

    if nccl_dir and Path(nccl_dir).is_dir():
        network = aggregate_nccl(Path(nccl_dir), config)
    else:
        network = {
            "status": "UNKNOWN",
            "hard_failures": [],
            "warnings": [],
            "unknown_checks": ["nccl_logs_not_provided"],
            "intra": {},
            "inter": {},
            "asymmetry": {},
        }
    hard_failures.extend(network["hard_failures"])
    warnings.extend(network["warnings"])
    unknown_checks.extend(network["unknown_checks"])

    absolute = absolute_threshold_check(config)
    if absolute["status"] == "UNKNOWN":
        unknown_checks.extend(absolute.get("unknown_checks", []))
        # Empty absolute thresholds are expected on a first PoC run. Surface them as
        # a warning so the report never pretends an absolute bar was met.
        warnings.append(absolute["reason"])

    # Absolute-threshold UNKNOWN is informational. Anything else in unknown_checks
    # means a required observation did not happen, so overall cannot be PASS.
    required_unknown = [
        item for item in unknown_checks if item != "absolute_performance"
    ]

    if hard_failures:
        status = "FAIL"
    elif required_unknown:
        status = worst_status(
            [
                gpu_check["status"],
                compute["status"],
                storage["status"],
                network["status"],
                "WARN" if warnings else "UNKNOWN",
            ]
        )
        if status == "PASS":
            status = "UNKNOWN"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "unknown_checks": sorted(set(unknown_checks)),
        "inventory": {
            "hostname": inventory["hostname"],
            "gpu_count": inventory["gpus"]["count"],
            "gpu_names": gpu_check.get("names", []),
            "local_scratch": inventory["local_scratch"],
        },
        "compute": compute,
        "network": network,
        "storage": storage,
        "stability": {
            "nccl_cv": network.get("inter", {}).get("cv"),
            "intra_asymmetry": network.get("asymmetry", {}).get("ratio"),
        },
        "absolute_thresholds": absolute,
        "artifacts": {},
    }


def aggregate_nccl(nccl_dir: Path, config: dict) -> dict:
    hard_failures: list[str] = []
    warnings: list[str] = []
    unknown_checks: list[str] = []

    intra = {}
    for path in sorted(nccl_dir.glob("nccl-intra-*.log")):
        host = path.stem.removeprefix("nccl-intra-")
        parsed = parse_nccl_log(path.read_text())
        write_json(nccl_dir / f"nccl-intra-{host}.json", parsed.to_dict())
        intra[host] = parsed.to_dict()
        if parsed.status == "FAIL":
            hard_failures.append(f"intra-node NCCL on {host}: wrong-value count nonzero")
        elif parsed.status == "UNKNOWN":
            unknown_checks.append(f"nccl_intra_{host}")

    hosts = sorted(intra)
    asymmetry = {"status": "UNKNOWN", "ratio": None, "warnings": [], "unknown_checks": []}
    if len(hosts) >= 2:
        a = intra[hosts[0]].get("avg_busbw_gbs")
        b = intra[hosts[1]].get("avg_busbw_gbs")
        asymmetry = node_asymmetry(a, b, config["node_asymmetry_warn_ratio"])
        warnings.extend(asymmetry["warnings"])
        unknown_checks.extend(asymmetry["unknown_checks"])
    elif not hosts:
        unknown_checks.append("nccl_intra")

    inter_logs = sorted(nccl_dir.glob("nccl-inter-run-*.log"))
    inter_results = [parse_nccl_log(path.read_text()) for path in inter_logs]
    for path, result in zip(inter_logs, inter_results, strict=True):
        write_json(path.with_suffix(".json"), result.to_dict())

    expected_reps = config["nccl_repetitions"]
    if len(inter_results) < expected_reps:
        unknown_checks.append(
            f"nccl_inter_repetitions: found {len(inter_results)}, expected {expected_reps}"
        )

    inter = summarize_repetitions(inter_results, config["variability_warn_cv"])
    hard_failures.extend(inter["hard_failures"])
    warnings.extend(inter["warnings"])
    unknown_checks.extend(inter["unknown_checks"])

    status = "FAIL" if hard_failures else worst_status(
        [inter["status"], asymmetry.get("status", "UNKNOWN")]
        + (["WARN"] if warnings else [])
        + (["UNKNOWN"] if unknown_checks and not inter_results else [])
    )
    if hard_failures:
        status = "FAIL"
    elif warnings and status == "PASS":
        status = "WARN"

    return {
        "status": status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "unknown_checks": unknown_checks,
        "intra": intra,
        "inter": inter,
        "asymmetry": asymmetry,
    }


def render_report(summary: dict) -> str:
    lines = [
        "# Cluster validation report",
        "",
        f"**Overall status:** {summary['status']}",
        "",
        "## Hard failures",
    ]
    if summary["hard_failures"]:
        lines.extend(f"- {item}" for item in summary["hard_failures"])
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    if summary["warnings"]:
        lines.extend(f"- {item}" for item in summary["warnings"])
    else:
        lines.append("- none")

    lines.extend(["", "## Unknown / not observable"])
    if summary["unknown_checks"]:
        lines.extend(f"- {item}" for item in summary["unknown_checks"])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Inventory",
            f"- hostname: {summary['inventory'].get('hostname')}",
            f"- GPU count: {summary['inventory'].get('gpu_count')}",
            f"- GPU names: {', '.join(summary['inventory'].get('gpu_names') or [])}",
            f"- local scratch: {summary['inventory'].get('local_scratch')}",
            "",
            "## Compute",
            f"- status: {summary['compute'].get('status')}",
            f"- reason: {summary['compute'].get('reason')}",
            "",
            "## Storage",
            f"- status: {summary['storage'].get('status')}",
            "",
            "## Network",
            f"- status: {summary['network'].get('status')}",
            f"- NCCL CV: {summary['stability'].get('nccl_cv')}",
            f"- intra-node asymmetry: {summary['stability'].get('intra_asymmetry')}",
            "",
            "## Absolute thresholds",
            f"- {summary['absolute_thresholds'].get('reason')}",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scratch = config.get("local_scratch") or None
    if scratch in ("", "${LOCAL_SCRATCH}"):
        scratch = os.environ.get("LOCAL_SCRATCH") or None

    if args.aggregate_only:
        # Second pass after NCCL logs land. Reuse the portable artifacts already
        # written by the per-node invocations instead of re-running fio/smoke.
        node_files = sorted(out.glob("node-*.json"))
        if not node_files:
            raise SystemExit(f"no node-*.json under {out}; run without --aggregate-only first")
        inventory = _read_json(node_files[0])
        compute_files = sorted(out.glob("gpu-smoke-*.json"))
        storage_files = sorted(out.glob("storage-*.json"))
        compute = _read_json(compute_files[0]) if compute_files else {
            "status": "UNKNOWN",
            "reason": "gpu-smoke artifact missing",
        }
        storage = _read_json(storage_files[0]) if storage_files else {
            "status": "UNKNOWN",
            "reason": "storage artifact missing",
        }
    else:
        inventory = collect_inventory(scratch)
        write_json(out / f"node-{inventory['hostname']}.json", inventory)
        write_json(
            out / "environment.json",
            {
                "utc": datetime.now(UTC).isoformat(),
                "config": config,
                "slurm": inventory["slurm"],
                "image": inventory["image"],
            },
        )
        compute = run_gpu_smoke(
            Path(args.gpu_smoke), out / f"gpu-smoke-{inventory['hostname']}.json"
        )
        storage = run_fio(
            scratch,
            int(config["fio_size_gib"]),
            bool(config.get("fio_shared_enabled", False)),
            out / f"storage-{inventory['hostname']}.json",
        )

    summary = aggregate(config, inventory, compute, storage, args.nccl_dir)
    summary["artifacts"] = {
        "directory": str(out),
        "node": f"node-{inventory['hostname']}.json",
        "compute": f"gpu-smoke-{inventory['hostname']}.json",
        "storage": f"storage-{inventory['hostname']}.json",
    }
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(render_report(summary))

    log.info("validation status=%s out=%s", summary["status"], out)
    for failure in summary["hard_failures"]:
        log.error("FAIL: %s", failure)
    for warning in summary["warnings"]:
        log.warning("WARN: %s", warning)

    return 1 if summary["status"] == "FAIL" else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Portable cluster qualification")
    parser.add_argument("--config", type=Path, default=Path("configs/validator.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--gpu-smoke",
        type=Path,
        default=Path("/usr/local/bin/gpu_smoke"),
        help="path to the compiled CUDA smoke binary",
    )
    parser.add_argument(
        "--nccl-dir",
        type=Path,
        help="directory containing nccl-intra-*.log and nccl-inter-run-*.log",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="fold existing node/NCCL artifacts into summary.json without re-running checks",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
