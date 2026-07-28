"""Distil a finished training run into results/summary/training.json.

The run directory already holds everything (manifest plus rank-zero metrics); this
turns it into the small tracked record that docs/RESULTS.md is generated from, so
no throughput or topology number is ever typed by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path

from nebius_poc.report import read_jsonl, write_json

log = logging.getLogger(__name__)


def topology_from_manifest(manifest: dict) -> dict:
    environment = manifest.get("environment") or {}
    slurm = environment.get("slurm") or {}
    return {
        "world_size": environment.get("world_size"),
        "nodes": slurm.get("SLURM_NNODES"),
        "nodelist": slurm.get("SLURM_JOB_NODELIST"),
        "slurm_job_id": slurm.get("SLURM_JOB_ID"),
        "gpus_visible_to_rank0": [
            {"index": gpu.get("index"), "name": gpu.get("name"), "uuid": gpu.get("uuid")}
            for gpu in environment.get("gpus") or []
        ],
        "driver": environment.get("driver"),
        "cuda_runtime": environment.get("cuda_runtime"),
        "packages": environment.get("packages"),
    }


def throughput(rows: Sequence[dict], world_size: int | None) -> dict:
    # Step 1 carries model-load and warm-up cost, so it is excluded from the medians.
    steady = [row for row in rows if row.get("global_step", 0) > 1] or list(rows)
    step_seconds = [row["step_seconds"] for row in steady if row.get("step_seconds")]
    samples = [row["samples_per_second"] for row in steady if row.get("samples_per_second")]
    peak_bytes = [row["peak_gpu_bytes"] for row in rows if row.get("peak_gpu_bytes")]
    return {
        "steps": len(rows),
        "steps_excluding_first": len(steady),
        "median_step_seconds": statistics.median(step_seconds) if step_seconds else None,
        "median_samples_per_second": statistics.median(samples) if samples else None,
        "peak_gpu_bytes_rank0": max(peak_bytes) if peak_bytes else None,
        "peak_gpu_gib_rank0": round(max(peak_bytes) / 2**30, 2) if peak_bytes else None,
        "world_size": world_size,
    }


def loss_trace(rows: Sequence[dict]) -> dict:
    losses = [row["loss"] for row in rows if row.get("loss") is not None]
    finite = [value for value in losses if value == value and abs(value) != float("inf")]
    return {
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "median_loss": statistics.median(losses) if losses else None,
        "nan_or_inf_count": len(losses) - len(finite),
    }


def build_summary(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = read_jsonl(run_dir / "metrics.jsonl")
    environment = manifest.get("environment") or {}
    config = manifest.get("config") or {}

    adapter = run_dir / "adapter"
    return {
        "run_id": manifest.get("run_id"),
        "stage": manifest.get("stage"),
        "started_utc": manifest.get("started_utc"),
        "ended_utc": manifest.get("ended_utc"),
        "exit_status": manifest.get("exit_status"),
        "objective": config.get("objective"),
        "model": config.get("model"),
        "dataset": config.get("dataset"),
        "lora": config.get("lora"),
        "training": config.get("training"),
        "topology": topology_from_manifest(manifest),
        "throughput": throughput(rows, environment.get("world_size")),
        "loss": loss_trace(rows),
        "artifacts": {
            "run_dir": str(run_dir),
            "adapter": str(adapter) if adapter.is_dir() else None,
            "metrics": str(run_dir / "metrics.jsonl"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a training run for results/summary")
    parser.add_argument("--run-dir", type=Path, required=True, help="a train run directory")
    parser.add_argument("--out", type=Path, default=Path("results/summary/training.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = build_summary(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, summary)

    throughput_block = summary["throughput"]
    log.info(
        "world_size=%s steps=%s median_step=%.3fs peak=%.1f GiB exit=%s",
        throughput_block["world_size"],
        throughput_block["steps"],
        throughput_block["median_step_seconds"] or 0.0,
        throughput_block["peak_gpu_gib_rank0"] or 0.0,
        summary["exit_status"],
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
