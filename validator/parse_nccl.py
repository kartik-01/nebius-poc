"""Parse all_reduce_perf / all_reduce_perf_mpi stdout into structured results.

The validator never rebuilds MPI or NCCL tests. It only reads the Nebius-provided
binaries' logs and folds them into the same PASS/WARN/FAIL report as the portable
checks. Keep the regexes conservative: a field we cannot read becomes UNKNOWN, not
a guessed zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# One data row from the size/algbw table. Columns vary slightly across nccl-tests
# versions; we only require size and the out-of-place #wrong / busbw fields.
_ROW = re.compile(
    r"^\s*(?P<size>\d+)\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<dtype>\S+)\s+"
    r"(?P<redop>\S+)\s+"
    r"(?P<time_oop>\S+)\s+"
    r"(?P<algbw_oop>\S+)\s+"
    r"(?P<busbw_oop>\S+)\s+"
    r"(?P<wrong_oop>\d+)"
    r"(?:\s+(?P<time_ip>\S+)\s+(?P<algbw_ip>\S+)\s+(?P<busbw_ip>\S+)\s+(?P<wrong_ip>\d+))?"
)

_AVG_BUSBW = re.compile(
    r"Avg bus bandwidth\s*:\s*(?P<busbw>[0-9.]+)", re.IGNORECASE
)
_OUT_OF_BOUNDS = re.compile(
    r"Out of bounds values\s*:\s*(?P<count>\d+)\s*(?P<status>\S+)?", re.IGNORECASE
)
_RANK = re.compile(
    r"#\s*Rank\s+(?P<rank>\d+)\s+Group\s+(?P<group>\d+)\s+Pid\s+(?P<pid>\d+)\s+"
    r"on\s+(?P<node>\S+)\s+device\s+(?P<device>\d+)",
    re.IGNORECASE,
)
_SOCKET_FALLBACK = re.compile(
    r"NCCL WARN.*(?:NET/Socket|using socket)|"
    r"NCCL INFO.*(?:Using network Socket|NET/Socket)",
    re.IGNORECASE,
)
_IB_TRANSPORT = re.compile(
    r"NCCL INFO.*(?:Using network (?:IB|OFI)|NET/IB|NET/OFI)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NcclRow:
    size_bytes: int
    count: int
    dtype: str
    redop: str
    time_us: float
    algbw_gbs: float
    busbw_gbs: float
    wrong: int


@dataclass
class NcclParseResult:
    status: str  # PASS | FAIL | UNKNOWN
    rows: list[NcclRow] = field(default_factory=list)
    ranks: list[dict] = field(default_factory=list)
    avg_busbw_gbs: float | None = None
    out_of_bounds: int | None = None
    total_wrong: int = 0
    socket_fallback: bool = False
    ib_transport: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload


def _as_float(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_nccl_log(text: str) -> NcclParseResult:
    """Parse one all_reduce_perf log. Incomplete output is UNKNOWN, not PASS."""
    if not text or not text.strip():
        return NcclParseResult(status="UNKNOWN", notes=["empty log"])

    rows: list[NcclRow] = []
    ranks: list[dict] = []
    notes: list[str] = []
    avg_busbw: float | None = None
    out_of_bounds: int | None = None

    for line in text.splitlines():
        rank = _RANK.search(line)
        if rank:
            ranks.append(
                {
                    "rank": int(rank.group("rank")),
                    "group": int(rank.group("group")),
                    "pid": int(rank.group("pid")),
                    "node": rank.group("node"),
                    "device": int(rank.group("device")),
                }
            )
            continue

        avg = _AVG_BUSBW.search(line)
        if avg:
            avg_busbw = float(avg.group("busbw"))
            continue

        bounds = _OUT_OF_BOUNDS.search(line)
        if bounds:
            out_of_bounds = int(bounds.group("count"))
            continue

        match = _ROW.match(line)
        if not match:
            continue

        time_us = _as_float(match.group("time_oop"))
        algbw = _as_float(match.group("algbw_oop"))
        busbw = _as_float(match.group("busbw_oop"))
        if time_us is None or algbw is None or busbw is None:
            notes.append(f"skipped non-numeric row at size {match.group('size')}")
            continue

        rows.append(
            NcclRow(
                size_bytes=int(match.group("size")),
                count=int(match.group("count")),
                dtype=match.group("dtype"),
                redop=match.group("redop"),
                time_us=time_us,
                algbw_gbs=algbw,
                busbw_gbs=busbw,
                wrong=int(match.group("wrong_oop")),
            )
        )

    total_wrong = sum(row.wrong for row in rows)
    if out_of_bounds is not None:
        total_wrong += out_of_bounds

    socket_fallback = bool(_SOCKET_FALLBACK.search(text))
    ib_transport = bool(_IB_TRANSPORT.search(text))

    if not rows and avg_busbw is None:
        # Header-only or truncated mid-run. Do not treat as a clean pass.
        return NcclParseResult(
            status="UNKNOWN",
            ranks=ranks,
            socket_fallback=socket_fallback,
            ib_transport=ib_transport,
            notes=notes + ["no bandwidth rows found"],
        )

    status = "FAIL" if total_wrong > 0 else "PASS"
    if total_wrong > 0:
        notes.append(f"nonzero wrong-value count: {total_wrong}")

    return NcclParseResult(
        status=status,
        rows=rows,
        ranks=ranks,
        avg_busbw_gbs=avg_busbw,
        out_of_bounds=out_of_bounds,
        total_wrong=total_wrong,
        socket_fallback=socket_fallback,
        ib_transport=ib_transport,
        notes=notes,
    )


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def summarize_repetitions(
    results: Sequence[NcclParseResult],
    warn_cv: float,
) -> dict:
    """Fold several multi-node runs into one stability block.

    Absolute bandwidth thresholds are never invented here. When the caller has not
    configured one, the absolute check stays UNKNOWN.
    """
    if not results:
        return {
            "status": "UNKNOWN",
            "hard_failures": ["no NCCL repetition logs"],
            "warnings": [],
            "unknown_checks": ["nccl_inter_repetitions"],
            "busbw_gbs": {},
            "cv": None,
            "total_wrong": 0,
            "socket_fallback": False,
        }

    hard_failures: list[str] = []
    warnings: list[str] = []
    unknown_checks: list[str] = []

    for index, result in enumerate(results, start=1):
        if result.status == "FAIL":
            hard_failures.append(f"nccl_inter_run_{index}: wrong-value count nonzero")
        elif result.status == "UNKNOWN":
            unknown_checks.append(f"nccl_inter_run_{index}")

    busbw = [result.avg_busbw_gbs for result in results if result.avg_busbw_gbs is not None]
    # Fall back to the largest message size's out-of-place busbw when the summary
    # line is missing, which some truncated logs still leave behind.
    if not busbw:
        for result in results:
            if result.rows:
                busbw.append(max(row.busbw_gbs for row in result.rows))

    cv = coefficient_of_variation(busbw) if len(busbw) >= 2 else None
    if cv is not None and cv > warn_cv:
        warnings.append(
            f"multi-node busbw CV {cv:.3f} exceeds warn threshold {warn_cv:.3f}"
        )

    if any(result.socket_fallback for result in results):
        warnings.append("NCCL socket fallback observed in at least one repetition")

    total_wrong = sum(result.total_wrong for result in results)

    if hard_failures:
        status = "FAIL"
    elif unknown_checks and not busbw:
        status = "UNKNOWN"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "status": status,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "unknown_checks": unknown_checks,
        "busbw_gbs": {
            "values": busbw,
            "min": min(busbw) if busbw else None,
            "median": sorted(busbw)[len(busbw) // 2] if busbw else None,
            "max": max(busbw) if busbw else None,
        },
        "cv": cv,
        "total_wrong": total_wrong,
        "socket_fallback": any(result.socket_fallback for result in results),
        "ib_transport": any(result.ib_transport for result in results),
    }


def node_asymmetry(intra_a: float | None, intra_b: float | None, warn_ratio: float) -> dict:
    """Relative gap between the two nodes' intra-node all-reduce busbw."""
    if intra_a is None or intra_b is None:
        return {
            "status": "UNKNOWN",
            "ratio": None,
            "warnings": [],
            "unknown_checks": ["intra_node_asymmetry"],
        }

    baseline = max(intra_a, intra_b)
    if baseline == 0:
        return {
            "status": "UNKNOWN",
            "ratio": None,
            "warnings": [],
            "unknown_checks": ["intra_node_asymmetry"],
        }

    ratio = abs(intra_a - intra_b) / baseline
    warnings = []
    if ratio > warn_ratio:
        warnings.append(
            f"intra-node busbw asymmetry {ratio:.3f} exceeds warn threshold {warn_ratio:.3f}"
        )
    return {
        "status": "WARN" if warnings else "PASS",
        "ratio": ratio,
        "warnings": warnings,
        "unknown_checks": [],
        "busbw_gbs": {"a": intra_a, "b": intra_b},
    }
