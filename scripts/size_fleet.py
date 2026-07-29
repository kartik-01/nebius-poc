#!/usr/bin/env python3
"""Turn measured goodput into a fleet size and a cost per million output tokens.

The customer's question is not "how fast is one GPU", it is "how many do we
reserve, and what does a million tokens cost". This converts the benchmark into
that answer using the per-GPU figure that was actually measured, and it refuses to
invent a GPU price: pass one, or the cost columns stay empty.

Per-GPU goodput is the right basis because replication scaled linearly across 1, 2
and 4 GPUs in this PoC. That linearity is the assumption the whole calculation
rests on, and it is only demonstrated to 4 GPUs, so the output says so.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path


def per_gpu_goodput(report: dict) -> tuple[str, float, int]:
    """Highest measured per-GPU goodput among topologies that passed guardrails."""
    best = None
    for topology in report["topologies"]:
        passing = topology.get("best_passing")
        if not passing:
            continue
        rate = passing.get("output_tokens_per_s_per_gpu")
        if rate and (best is None or rate > best[1]):
            best = (topology["topology"], float(rate), int(topology["gpu_count"]))
    if best is None:
        raise SystemExit("no topology passed guardrails; nothing to size from")
    return best


def size_fleet(
    per_gpu: float,
    target_tokens_per_s: float,
    gpus_per_replica: int = 1,
    headroom: float = 0.2,
) -> dict:
    # Size against a derated figure. The measured peak sits right at the guardrail
    # boundary, and running production at 100 % of a benchmark leaves nothing for
    # traffic spikes or a degraded node.
    usable = per_gpu * (1.0 - headroom)
    gpus = math.ceil(target_tokens_per_s / usable)
    # Round up to whole replicas so the fleet is actually deployable.
    gpus = math.ceil(gpus / gpus_per_replica) * gpus_per_replica
    return {
        "target_output_tokens_per_s": target_tokens_per_s,
        "headroom_fraction": headroom,
        "usable_tokens_per_s_per_gpu": usable,
        "gpus_required": gpus,
        "delivered_tokens_per_s": gpus * usable,
    }


def cost_block(gpus: int, per_gpu: float, headroom: float, gpu_hour_usd: float | None) -> dict:
    usable = per_gpu * (1.0 - headroom)
    tokens_per_gpu_hour = usable * 3600.0
    block: dict = {
        "output_tokens_per_gpu_hour": tokens_per_gpu_hour,
        "gpu_hours_per_million_output_tokens": 1e6 / tokens_per_gpu_hour,
    }
    if gpu_hour_usd is None:
        block["note"] = "pass --gpu-hour-usd to add cost; no price is assumed here"
        return block
    block["gpu_hour_usd"] = gpu_hour_usd
    block["usd_per_million_output_tokens"] = (1e6 / tokens_per_gpu_hour) * gpu_hour_usd
    block["fleet_usd_per_hour"] = gpus * gpu_hour_usd
    block["fleet_usd_per_month_730h"] = gpus * gpu_hour_usd * 730.0
    return block


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Size a serving fleet from measured goodput")
    parser.add_argument("--inference", type=Path, default=Path("results/summary/inference.json"))
    parser.add_argument(
        "--target-tokens-per-s",
        type=float,
        action="append",
        help="repeatable target output-token rate to size for",
    )
    parser.add_argument(
        "--gpu-hour-usd", type=float, help="GPU price; omitted means no cost output"
    )
    parser.add_argument("--headroom", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("results/summary/sizing.json"))
    args = parser.parse_args(argv)

    report = json.loads(args.inference.read_text())
    topology, per_gpu, gpus_measured = per_gpu_goodput(report)
    targets = args.target_tokens_per_s or [10_000, 50_000, 100_000, 500_000]

    scenarios = [size_fleet(per_gpu, target, headroom=args.headroom) for target in targets]
    for scenario in scenarios:
        scenario["cost"] = cost_block(
            scenario["gpus_required"], per_gpu, args.headroom, args.gpu_hour_usd
        )

    payload = {
        "basis": {
            "source": str(args.inference),
            "topology": topology,
            "gpus_measured": gpus_measured,
            "measured_tokens_per_s_per_gpu": per_gpu,
            "workload": "512 input tokens, 128 output tokens per request",
            "guardrails": report.get("guardrails"),
        },
        "assumptions": [
            "Replication scales linearly. Measured flat per-GPU goodput across 1, 2 "
            "and 4 GPUs; not demonstrated beyond 4.",
            f"Sized at {int((1 - args.headroom) * 100)} % of measured peak, because the "
            "peak sits at the guardrail boundary.",
            "H200 measurements. H100 capacity must be re-measured on the target SKU.",
            "Request shape is the benchmark's, not the customer's traffic.",
        ],
        "scenarios": scenarios,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"basis: {topology}, {per_gpu:,.0f} output tok/s per GPU "
        f"measured on {gpus_measured} GPUs"
    )
    print(f"sized at {int((1 - args.headroom) * 100)} % of peak\n")
    header = f"{'target tok/s':>14}{'GPUs':>8}{'GPU-h / 1M tok':>17}"
    if args.gpu_hour_usd is not None:
        header += f"{'USD / 1M tok':>15}{'fleet USD/mo':>15}"
    print(header)
    for scenario in scenarios:
        cost = scenario["cost"]
        row = (
            f"{scenario['target_output_tokens_per_s']:>14,.0f}"
            f"{scenario['gpus_required']:>8}"
            f"{cost['gpu_hours_per_million_output_tokens']:>17.3f}"
        )
        if args.gpu_hour_usd is not None:
            row += (
                f"{cost['usd_per_million_output_tokens']:>15.2f}"
                f"{cost['fleet_usd_per_month_730h']:>15,.0f}"
            )
        print(row)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
