from pathlib import Path

import pytest
from validator.parse_nccl import (
    coefficient_of_variation,
    node_asymmetry,
    parse_nccl_log,
    summarize_repetitions,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_valid_log_is_pass_with_zero_wrong():
    result = parse_nccl_log(_load("nccl_valid.log"))

    assert result.status == "PASS"
    assert result.total_wrong == 0
    assert result.out_of_bounds == 0
    assert result.avg_busbw_gbs == pytest.approx(18.1775)
    assert len(result.rows) == 4
    assert result.rows[-1].size_bytes == 134217728
    assert {rank["node"] for rank in result.ranks} == {"worker-0"}
    assert len(result.ranks) == 2


def test_nonzero_wrong_values_are_fail():
    result = parse_nccl_log(_load("nccl_wrong_values.log"))

    assert result.status == "FAIL"
    assert result.total_wrong == 3
    assert any("wrong-value" in note for note in result.notes)


def test_incomplete_log_is_unknown_not_pass():
    result = parse_nccl_log(_load("nccl_incomplete.log"))

    assert result.status == "UNKNOWN"
    assert result.rows == []
    assert result.avg_busbw_gbs is None
    assert len(result.ranks) == 4
    assert result.ib_transport is True
    assert any("no bandwidth rows" in note for note in result.notes)


def test_empty_log_is_unknown():
    result = parse_nccl_log("")
    assert result.status == "UNKNOWN"
    assert result.notes == ["empty log"]


def test_socket_fallback_is_flagged_without_failing_correctness():
    result = parse_nccl_log(_load("nccl_socket_fallback.log"))

    assert result.status == "PASS"
    assert result.socket_fallback is True
    assert result.total_wrong == 0


def test_summarize_repetitions_warns_on_high_cv():
    # Three clean runs with deliberately different busbw so CV > 0.10.
    texts = [
        "# Avg bus bandwidth    : 100.0 (GB/s)\n"
        "     1048576        262144     float     sum   10.00  100.00  100.00      0\n",
        "# Avg bus bandwidth    : 40.0 (GB/s)\n"
        "     1048576        262144     float     sum   25.00   40.00   40.00      0\n",
        "# Avg bus bandwidth    : 70.0 (GB/s)\n"
        "     1048576        262144     float     sum   15.00   70.00   70.00      0\n",
    ]
    results = [parse_nccl_log(text) for text in texts]
    summary = summarize_repetitions(results, warn_cv=0.10)

    assert summary["status"] == "WARN"
    assert summary["cv"] > 0.10
    assert summary["total_wrong"] == 0
    assert summary["busbw_gbs"]["min"] == pytest.approx(40.0)
    assert summary["busbw_gbs"]["max"] == pytest.approx(100.0)
    assert any("CV" in warning for warning in summary["warnings"])


def test_summarize_repetitions_fails_on_wrong_values():
    results = [
        parse_nccl_log(_load("nccl_valid.log")),
        parse_nccl_log(_load("nccl_wrong_values.log")),
        parse_nccl_log(_load("nccl_valid.log")),
    ]
    summary = summarize_repetitions(results, warn_cv=0.10)

    assert summary["status"] == "FAIL"
    assert summary["total_wrong"] == 3
    assert any("wrong-value" in failure for failure in summary["hard_failures"])


def test_summarize_repetitions_marks_incomplete_unknown():
    results = [
        parse_nccl_log(_load("nccl_incomplete.log")),
        parse_nccl_log(_load("nccl_incomplete.log")),
    ]
    summary = summarize_repetitions(results, warn_cv=0.10)

    assert summary["status"] == "UNKNOWN"
    assert summary["unknown_checks"] == ["nccl_inter_run_1", "nccl_inter_run_2"]


def test_summarize_empty_input_is_unknown():
    summary = summarize_repetitions([], warn_cv=0.10)
    assert summary["status"] == "UNKNOWN"
    assert "no NCCL repetition logs" in summary["hard_failures"]


def test_node_asymmetry_warns_above_threshold():
    result = node_asymmetry(100.0, 80.0, warn_ratio=0.10)
    assert result["status"] == "WARN"
    assert result["ratio"] == pytest.approx(0.20)


def test_node_asymmetry_unknown_when_a_side_is_missing():
    result = node_asymmetry(100.0, None, warn_ratio=0.10)
    assert result["status"] == "UNKNOWN"
    assert "intra_node_asymmetry" in result["unknown_checks"]


def test_coefficient_of_variation_none_for_short_series():
    assert coefficient_of_variation([]) is None
    assert coefficient_of_variation([1.0]) is None
    assert coefficient_of_variation([10.0, 10.0]) == pytest.approx(0.0)


def test_rooted_table_does_not_shift_wrong_column():
    """all_reduce_perf emits a `root` column; ignoring it moved the in-place time
    into #wrong and turned clean runs into FAIL."""
    result = parse_nccl_log(_load("nccl_valid_rooted.log"))

    assert result.status == "PASS"
    assert result.total_wrong == 0
    assert result.avg_busbw_gbs == pytest.approx(22.2768)
    assert result.rows[-1].size_bytes == 134217728
    assert result.rows[-1].busbw_gbs == pytest.approx(122.28)
    assert result.rows[-1].time_us == pytest.approx(1646.40)
    assert {rank["node"] for rank in result.ranks} == {"worker-0", "worker-1"}
