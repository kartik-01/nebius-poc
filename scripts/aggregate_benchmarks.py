#!/usr/bin/env python3
"""Aggregate vLLM bench shards into results/summary/inference.json.

Fleet percentiles are recomputed from per-request records when available. Endpoint
p95 values are never averaged together — that would understate the true tail.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def load_json(path: Path):
    return json.loads(path.read_text())


def discover_shards(bench_root: Path) -> list[Path]:
    return sorted(path for path in bench_root.glob("**/bench.json") if path.is_file())


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_requests(payload: dict) -> list[dict]:
    """Normalize detailed per-request records from a few vLLM bench shapes."""
    candidates = []
    for key in ("detailed", "requests", "per_request", "results"):
        value = payload.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            candidates = value
            break
    rows = []
    for item in candidates:
        ttft = _as_float(
            item.get("ttft_ms")
            or item.get("ttft")
            or item.get("latency_ttft_ms")
            or item.get("time_to_first_token_ms")
        )
        tpot = _as_float(
            item.get("tpot_ms")
            or item.get("tpot")
            or item.get("latency_tpot_ms")
            or item.get("time_per_output_token_ms")
        )
        e2e = _as_float(
            item.get("e2e_ms")
            or item.get("latency_ms")
            or item.get("request_latency_ms")
            or item.get("end_to_end_latency_ms")
        )
        out_tokens = _as_float(
            item.get("output_tokens")
            or item.get("generated_tokens")
            or item.get("n_tokens")
        )
        error = bool(item.get("error") or item.get("failed") or item.get("exception"))
        # Some exports store successes only; treat missing error as success.
        if item.get("success") is False:
            error = True
        rows.append(
            {
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "e2e_ms": e2e,
                "output_tokens": out_tokens,
                "error": error,
            }
        )
    return rows


def summarize_requests(rows: Sequence[dict], *, gpu_count: int, wall_seconds: float | None) -> dict:
    total = len(rows)
    errors = sum(1 for row in rows if row["error"])
    ok = [row for row in rows if not row["error"]]
    ttfts = [row["ttft_ms"] for row in ok if row["ttft_ms"] is not None]
    tpots = [row["tpot_ms"] for row in ok if row["tpot_ms"] is not None]
    e2es = [row["e2e_ms"] for row in ok if row["e2e_ms"] is not None]
    out_tokens = sum(row["output_tokens"] or 0.0 for row in ok)

    if wall_seconds is None or wall_seconds <= 0:
        # Fall back to max e2e as a weak duration proxy when the client omitted it.
        wall_seconds = (max(e2es) / 1000.0) if e2es else None

    output_tok_s = (out_tokens / wall_seconds) if wall_seconds else None
    return {
        "requests": total,
        "errors": errors,
        "error_rate": (errors / total) if total else 1.0,
        "output_tokens": out_tokens,
        "wall_seconds": wall_seconds,
        "output_tokens_per_s": output_tok_s,
        "output_tokens_per_s_per_gpu": (
            (output_tok_s / gpu_count) if output_tok_s is not None and gpu_count else None
        ),
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "tpot_ms": {
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
        },
        "e2e_ms": {
            "p50": percentile(e2es, 0.50),
            "p95": percentile(e2es, 0.95),
        },
    }


def summarize_from_summary_only(payload: dict, *, gpu_count: int) -> dict:
    """Best-effort when detailed records are absent — marks the gap explicitly."""
    output_tok_s = _as_float(
        payload.get("output_throughput")
        or payload.get("output_tokens_per_second")
        or payload.get("tokens_per_second")
    )
    return {
        "requests": payload.get("num_prompts") or payload.get("completed") or 0,
        "errors": payload.get("failed") or 0,
        "error_rate": _as_float(payload.get("error_rate")) or 0.0,
        "output_tokens": payload.get("total_output_tokens"),
        "wall_seconds": payload.get("duration"),
        "output_tokens_per_s": output_tok_s,
        "output_tokens_per_s_per_gpu": (
            (output_tok_s / gpu_count) if output_tok_s is not None and gpu_count else None
        ),
        "ttft_ms": {
            "p50": _as_float(payload.get("mean_ttft_ms")),
            "p95": _as_float(payload.get("p95_ttft_ms")),
            "p99": _as_float(payload.get("p99_ttft_ms")),
        },
        "tpot_ms": {
            "p50": _as_float(payload.get("mean_tpot_ms")),
            "p95": _as_float(payload.get("p95_tpot_ms")),
            "p99": _as_float(payload.get("p99_tpot_ms")),
        },
        "e2e_ms": {
            "p50": _as_float(payload.get("mean_latency_ms") or payload.get("median_e2el_ms")),
            "p95": _as_float(payload.get("p95_latency_ms") or payload.get("p95_e2el_ms")),
        },
        "detailed_records": False,
    }


def meets_guardrails(summary: dict, guardrails: dict) -> tuple[bool, list[str]]:
    failures = []
    p95_ttft = (summary.get("ttft_ms") or {}).get("p95")
    p95_tpot = (summary.get("tpot_ms") or {}).get("p95")
    error_rate = summary.get("error_rate")

    if p95_ttft is None or p95_ttft > guardrails["p95_ttft_ms"]:
        failures.append(
            f"p95_ttft_ms {p95_ttft} exceeds {guardrails['p95_ttft_ms']}"
            if p95_ttft is not None
            else "p95_ttft_ms missing"
        )
    if p95_tpot is None or p95_tpot > guardrails["p95_tpot_ms"]:
        failures.append(
            f"p95_tpot_ms {p95_tpot} exceeds {guardrails['p95_tpot_ms']}"
            if p95_tpot is not None
            else "p95_tpot_ms missing"
        )
    if error_rate is None or error_rate > guardrails["max_error_rate"]:
        failures.append(
            f"error_rate {error_rate} exceeds {guardrails['max_error_rate']}"
        )
    return not failures, failures


def aggregate_point(
    shard_paths: Sequence[Path],
    *,
    gpu_count: int,
    guardrails: dict,
) -> dict:
    all_rows: list[dict] = []
    summary_only = []
    for path in shard_paths:
        payload = load_json(path)
        rows = extract_requests(payload)
        if rows:
            all_rows.extend(rows)
        else:
            summary_only.append(summarize_from_summary_only(payload, gpu_count=gpu_count))

    if all_rows:
        # Prefer the sum of per-shard durations when present in sibling meta.
        wall = None
        durations = []
        for path in shard_paths:
            payload = load_json(path)
            duration = _as_float(payload.get("duration") or payload.get("elapsed_time"))
            if duration:
                durations.append(duration)
        if durations:
            # Concurrent replica benches share wall time ≈ max shard duration.
            wall = max(durations)
        summary = summarize_requests(all_rows, gpu_count=gpu_count, wall_seconds=wall)
        summary["detailed_records"] = True
    elif summary_only:
        def _sum(field: str) -> float:
            return sum(item.get(field) or 0.0 for item in summary_only)

        def _worst(group: str, stat: str) -> float | None:
            values = [
                (item.get(group) or {}).get(stat)
                for item in summary_only
                if (item.get(group) or {}).get(stat) is not None
            ]
            return max(values) if values else None

        output_tok_s = _sum("output_tokens_per_s")
        requests = _sum("requests")
        errors = _sum("errors")
        summary = {
            "requests": int(requests),
            "errors": int(errors),
            "error_rate": (errors / requests) if requests else 0.0,
            "output_tokens": _sum("output_tokens") or None,
            "wall_seconds": max(
                (item.get("wall_seconds") or 0.0 for item in summary_only), default=None
            )
            or None,
            "output_tokens_per_s": output_tok_s,
            "output_tokens_per_s_per_gpu": (
                output_tok_s / gpu_count if gpu_count else None
            ),
            "ttft_ms": {stat: _worst("ttft_ms", stat) for stat in ("p50", "p95", "p99")},
            "tpot_ms": {stat: _worst("tpot_ms", stat) for stat in ("p50", "p95", "p99")},
            "e2e_ms": {stat: _worst("e2e_ms", stat) for stat in ("p50", "p95")},
            "detailed_records": False,
            "replicas": len(summary_only),
        }
        if len(summary_only) > 1:
            summary["warnings"] = [
                "no per-request records from this vLLM build; counts and throughput are "
                "summed exactly, percentiles are the worst replica rather than a true "
                "fleet percentile"
            ]
        else:
            summary["warnings"] = [
                "aggregated from summary metrics only; single replica so percentiles are exact"
            ]
    else:
        summary = {
            "requests": 0,
            "errors": 0,
            "error_rate": 1.0,
            "output_tokens_per_s": None,
            "detailed_records": False,
        }

    ok, failures = meets_guardrails(summary, guardrails)
    goodput = summary["output_tokens_per_s"] if ok else 0.0
    summary["passes_guardrails"] = ok
    summary["guardrail_failures"] = failures
    summary["output_token_goodput"] = goodput
    return summary


def gpu_count_for_topology(topology: str | None) -> int:
    return {"P0": 1, "P1": 2, "P2": 2, "P3": 4}.get(topology or "", 1)


def select_winner(points: Sequence[dict]) -> dict | None:
    eligible = [point for point in points if point.get("passes_guardrails")]
    if not eligible:
        return None
    return max(eligible, key=lambda point: point.get("output_token_goodput") or 0.0)


def aggregate_bench_root(bench_root: Path, config: dict) -> dict:
    plan_path = bench_root / "bench_plan.json"
    plan = load_json(plan_path) if plan_path.exists() else {}
    topology = plan.get("topology")
    gpu_count = gpu_count_for_topology(topology)
    guardrails = config["guardrails"]

    # Group shards by concurrency (and stage/rep when present in path name).
    groups: dict[tuple, list[Path]] = {}
    for path in discover_shards(bench_root):
        meta_path = path.parent / "meta.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            key = (meta.get("stage"), meta.get("concurrency"), meta.get("repetition"))
        else:
            key = ("unknown", path.parent.name, 1)
        groups.setdefault(key, []).append(path)

    points = []
    for key, shards in sorted(groups.items(), key=lambda item: item[0]):
        stage, concurrency, repetition = key
        summary = aggregate_point(shards, gpu_count=gpu_count, guardrails=guardrails)
        points.append(
            {
                "stage": stage,
                "concurrency": concurrency,
                "repetition": repetition,
                "topology": topology,
                "gpu_count": gpu_count,
                "shards": [str(path) for path in shards],
                **summary,
            }
        )

    # Final-stage median across repetitions per concurrency.
    finals = [point for point in points if point["stage"] == "final"]
    by_conc: dict[int, list[dict]] = {}
    for point in finals:
        by_conc.setdefault(int(point["concurrency"]), []).append(point)

    final_medians = []
    for concurrency, reps in sorted(by_conc.items()):
        goodputs = [point["output_token_goodput"] for point in reps]
        final_medians.append(
            {
                "concurrency": concurrency,
                "repetitions": len(reps),
                "median_goodput": statistics.median(goodputs) if goodputs else None,
                "min_goodput": min(goodputs) if goodputs else None,
                "max_goodput": max(goodputs) if goodputs else None,
                "all_pass": all(point["passes_guardrails"] for point in reps),
            }
        )

    winner = select_winner(points)
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "bench_root": str(bench_root),
        "topology": topology,
        "gpu_count": gpu_count,
        "objective": config.get("objective", "output_token_goodput"),
        "guardrails": guardrails,
        "points": points,
        "final_medians": final_medians,
        "selected": {
            "tag": None
            if winner is None
            else {
                "stage": winner["stage"],
                "concurrency": winner["concurrency"],
                "repetition": winner["repetition"],
                "output_token_goodput": winner["output_token_goodput"],
                "output_tokens_per_s_per_gpu": winner.get("output_tokens_per_s_per_gpu"),
                "ttft_p95_ms": (winner.get("ttft_ms") or {}).get("p95"),
                "tpot_p95_ms": (winner.get("tpot_ms") or {}).get("p95"),
            },
            "reason": (
                "max output_token_goodput among points that pass guardrails"
                if winner
                else "no point passed guardrails"
            ),
        },
        "notes": [
            "Warm-up exclusion depends on the bench client; "
            "prefer runs configured with warmup_requests.",
            "Fleet p95 is recomputed from raw requests when detailed records exist.",
            "These guardrails are PoC demonstration limits, not customer SLOs.",
            "Four H200s do not validate 512-H100 reservation behavior.",
        ],
    }


def consolidate(reports: Sequence[dict], config: dict) -> dict:
    """Combine per-topology reports into the single tracked inference summary.

    Each bench root covers one topology, so the comparison across P0–P3 has to be
    assembled here. Per-GPU goodput is what decides the recommendation: absolute
    goodput always favours whichever topology was given more GPUs.
    """
    topologies = []
    for report in reports:
        selected = report["selected"]["tag"]
        topologies.append(
            {
                "topology": report["topology"],
                "gpu_count": report["gpu_count"],
                "bench_root": report["bench_root"],
                "best_passing": selected,
                "final_medians": report["final_medians"],
                "points": report["points"],
            }
        )

    ranked = [item for item in topologies if item["best_passing"]]
    best_absolute = max(
        ranked, key=lambda item: item["best_passing"]["output_token_goodput"], default=None
    )
    best_per_gpu = max(
        ranked,
        key=lambda item: item["best_passing"].get("output_tokens_per_s_per_gpu") or 0.0,
        default=None,
    )
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "objective": config.get("objective", "output_token_goodput"),
        "guardrails": config["guardrails"],
        "topologies": topologies,
        "recommendation": {
            "highest_absolute_goodput": None
            if best_absolute is None
            else {
                "topology": best_absolute["topology"],
                "gpu_count": best_absolute["gpu_count"],
                **best_absolute["best_passing"],
            },
            "highest_per_gpu_goodput": None
            if best_per_gpu is None
            else {
                "topology": best_per_gpu["topology"],
                "gpu_count": best_per_gpu["gpu_count"],
                **best_per_gpu["best_passing"],
            },
        },
        "notes": [
            "Each topology was swept to its own saturation point; a fixed concurrency "
            "understates larger topologies because per-replica load falls as replicas are added.",
            "Guardrails are PoC demonstration limits, not customer SLOs.",
            "Four H200s do not validate 512-H100 reservation behavior.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate vLLM benchmark shards")
    parser.add_argument("--bench-root", type=Path, required=True, action="append")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/summary/inference.json"))
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    reports = [aggregate_bench_root(root, config) for root in args.bench_root]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if len(reports) == 1:
        report = reports[0]
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        selected = report["selected"]["tag"]
        if selected:
            print(
                f"selected concurrency={selected['concurrency']} "
                f"goodput={selected['output_token_goodput']:.2f} tok/s "
                f"per_gpu={selected['output_tokens_per_s_per_gpu']}"
            )
        else:
            print("no configuration passed guardrails")
    else:
        combined = consolidate(reports, config)
        args.out.write_text(json.dumps(combined, indent=2) + "\n")
        for item in combined["topologies"]:
            best = item["best_passing"]
            if best:
                print(
                    f"{item['topology']:>3} ({item['gpu_count']} gpu): "
                    f"{best['output_token_goodput']:>9.1f} tok/s at c{best['concurrency']} "
                    f"({best['output_tokens_per_s_per_gpu']:.1f}/gpu)"
                )
            else:
                print(f"{item['topology']:>3}: no point passed guardrails")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
