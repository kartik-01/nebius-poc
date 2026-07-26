"""Run identity, manifests, and the paired base-versus-tuned report.

Nothing in this repository may claim a result without a manifest sitting next to it,
so every CLI opens a run directory through here and closes it the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from nebius_poc.prompts import LABELS
from nebius_poc.stats import align_by_question_id, compare

log = logging.getLogger(__name__)

UNKNOWN = "unknown"

_TRACKED_PACKAGES = ("torch", "transformers", "peft", "datasets", "numpy")


def new_run_id(stage: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    job = os.environ.get("SLURM_JOB_ID", "local")
    return f"{stamp}_{stage}_job{job}"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = UNKNOWN
    return versions


def gpu_inventory() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    inventory = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        inventory.append(
            {
                "index": index,
                "name": properties.name,
                # uuid only exists on newer torch builds; record the gap rather than
                # pretending the device is unidentified.
                "uuid": str(getattr(properties, "uuid", UNKNOWN)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return inventory


def driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if result.returncode != 0 or not result.stdout.strip():
        return UNKNOWN
    return result.stdout.strip().splitlines()[0].strip()


def environment_snapshot() -> dict:
    try:
        import torch

        torch_cuda = torch.version.cuda or UNKNOWN
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ImportError:
        torch_cuda = UNKNOWN
        world_size = 1

    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_runtime": torch_cuda,
        "driver": driver_version(),
        "gpus": gpu_inventory(),
        "world_size": world_size,
        "slurm": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("SLURM_")
        },
    }


def open_run(stage: str, results_root: Path, config: dict | None = None) -> tuple[Path, dict]:
    run_id = new_run_id(stage)
    directory = Path(results_root) / run_id
    directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "stage": stage,
        "started_utc": datetime.now(UTC).isoformat(),
        "ended_utc": None,
        "exit_status": None,
        "config": config or {},
        "environment": environment_snapshot(),
        "artifacts": {},
        "warnings": [],
    }
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def close_run(directory: Path, manifest: dict, exit_status: str, artifacts: dict | None = None):
    manifest["ended_utc"] = datetime.now(UTC).isoformat()
    manifest["exit_status"] = exit_status
    manifest["artifacts"].update(artifacts or {})
    write_json(Path(directory) / "manifest.json", manifest)
    return manifest


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False))


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_checksum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_checksums(directory: Path, patterns: Sequence[str] = ("*.safetensors", "*.json")):
    files = sorted(
        path for pattern in patterns for path in Path(directory).glob(pattern) if path.is_file()
    )
    return {path.name: file_checksum(path) for path in files}


def format_adherence(rows: Sequence[dict]) -> dict:
    parsed = sum(1 for row in rows if row.get("parsed") in LABELS)
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "n": len(rows),
        "parsed": parsed,
        "unparsable": len(rows) - parsed,
        "format_adherence": parsed / len(rows) if rows else 0.0,
        "accuracy": correct / len(rows) if rows else 0.0,
    }


def paired_report(
    base_rows: Sequence[dict],
    tuned_rows: Sequence[dict],
    resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Compare forced-choice outcomes, pairing strictly by stable question ID."""
    base = {row["question_id"]: bool(row["correct"]) for row in base_rows}
    tuned = {row["question_id"]: bool(row["correct"]) for row in tuned_rows}
    if len(base) != len(base_rows) or len(tuned) != len(tuned_rows):
        raise ValueError("duplicate question IDs in the evaluation rows")

    left, right = align_by_question_id(base, tuned)
    return compare(left, right, resamples=resamples, seed=seed).to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pair two evaluation runs into one summary")
    parser.add_argument("--base", type=Path, required=True, help="base forced_choice.jsonl")
    parser.add_argument("--tuned", type=Path, required=True, help="tuned forced_choice.jsonl")
    parser.add_argument("--base-generation", type=Path)
    parser.add_argument("--tuned-generation", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/summary/accuracy.json"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "sources": {"base": str(args.base), "tuned": str(args.tuned)},
        "forced_choice": paired_report(
            read_jsonl(args.base), read_jsonl(args.tuned), args.resamples, args.seed
        ),
    }
    if args.base_generation and args.tuned_generation:
        summary["generation"] = {
            "base": format_adherence(read_jsonl(args.base_generation)),
            "tuned": format_adherence(read_jsonl(args.tuned_generation)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, summary)

    forced = summary["forced_choice"]
    log.info(
        "base %.4f, tuned %.4f, delta %+.2f pp (95%% CI %+.2f to %+.2f), McNemar p=%.4g",
        forced["base_accuracy"],
        forced["tuned_accuracy"],
        forced["delta_pp"],
        forced["ci_low_pp"],
        forced["ci_high_pp"],
        forced["mcnemar_p"],
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
