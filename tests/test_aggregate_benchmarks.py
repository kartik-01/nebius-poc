import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_benchmarks", ROOT / "scripts" / "aggregate_benchmarks.py"
)
agg = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(agg)


def _guardrails():
    return {"p95_ttft_ms": 2000, "p95_tpot_ms": 100, "max_error_rate": 0.0}


def test_percentile_interpolates():
    assert agg.percentile([0.0, 10.0, 20.0, 30.0], 0.5) == 15.0
    assert agg.percentile([1.0], 0.95) == 1.0
    assert agg.percentile([], 0.95) is None


def test_guardrails_fail_on_high_ttft():
    ok, failures = agg.meets_guardrails(
        {
            "ttft_ms": {"p95": 3000},
            "tpot_ms": {"p95": 50},
            "error_rate": 0.0,
        },
        _guardrails(),
    )
    assert not ok
    assert any("ttft" in item for item in failures)


def test_aggregate_point_recomputes_fleet_p95_from_raw_requests(tmp_path):
    def row(ttft, tpot, e2e):
        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_ms": e2e,
            "output_tokens": 128,
            "error": False,
        }

    shard = tmp_path / "bench.json"
    shard.write_text(
        json.dumps(
            {
                "duration": 10.0,
                "detailed": [
                    row(100, 20, 500),
                    row(200, 25, 600),
                    row(300, 30, 700),
                    row(400, 35, 750),
                    row(1500, 40, 800),
                    row(1600, 45, 850),
                    row(1700, 50, 900),
                    row(1800, 55, 950),
                    row(1900, 60, 1000),
                    row(2000, 65, 1050),
                ],
            }
        )
    )
    summary = agg.aggregate_point([shard], gpu_count=1, guardrails=_guardrails())
    assert summary["detailed_records"] is True
    assert summary["passes_guardrails"] is True
    # Interpolated p95 over this series sits in the upper tail, not the mean.
    assert summary["ttft_ms"]["p95"] >= 1500
    assert summary["output_token_goodput"] > 0


def test_select_winner_ignores_failing_points():
    points = [
        {"passes_guardrails": False, "output_token_goodput": 999},
        {"passes_guardrails": True, "output_token_goodput": 100},
        {"passes_guardrails": True, "output_token_goodput": 200},
    ]
    assert agg.select_winner(points)["output_token_goodput"] == 200
    assert agg.select_winner([points[0]]) is None


def test_aggregate_bench_root_writes_selection(tmp_path):
    root = tmp_path / "bench"
    (root / "screen_c8_r1_e0").mkdir(parents=True)
    (root / "bench_plan.json").write_text(
        json.dumps({"topology": "P0", "base_urls": ["http://h:8000/v1"]})
    )
    (root / "screen_c8_r1_e0" / "meta.json").write_text(
        json.dumps({"stage": "screen", "concurrency": 8, "repetition": 1})
    )
    (root / "screen_c8_r1_e0" / "bench.json").write_text(
        json.dumps(
            {
                "duration": 5.0,
                "detailed": [
                    {
                        "ttft_ms": 200,
                        "tpot_ms": 30,
                        "e2e_ms": 400,
                        "output_tokens": 128,
                        "error": False,
                    }
                ]
                * 10,
            }
        )
    )

    config = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text())
    report = agg.aggregate_bench_root(root, config)
    assert report["topology"] == "P0"
    assert report["selected"]["tag"]["concurrency"] == 8
    assert report["points"][0]["passes_guardrails"] is True
