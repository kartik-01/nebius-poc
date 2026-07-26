from pathlib import Path

from validator.cluster_validate import (
    absolute_threshold_check,
    aggregate,
    check_gpu_expectations,
    expand_env,
    render_report,
    worst_status,
)


def _config(**overrides):
    base = {
        "expected_nodes": 2,
        "expected_gpus_total": 4,
        "expected_gpus_per_node": 2,
        "expected_gpu_name_pattern": "H200",
        "local_scratch": "/scratch",
        "fio_size_gib": 1,
        "fio_shared_enabled": False,
        "nccl_repetitions": 3,
        "variability_warn_cv": 0.10,
        "node_asymmetry_warn_ratio": 0.10,
        "absolute_thresholds": {},
    }
    base.update(overrides)
    return base


def _inventory(count=2, names=None, scratch_status="PASS"):
    names = names or ["NVIDIA H200"] * count
    return {
        "hostname": "worker-0",
        "local_scratch": {"path": "/scratch", "status": scratch_status, "reason": None},
        "gpus": {
            "status": "PASS",
            "count": count,
            "devices": [
                {
                    "uuid": f"GPU-{index}",
                    "name": names[index],
                    "memory_mib": 140000,
                    "driver_version": "550.54",
                    "pci_bus_id": f"0000:{index:02d}:00.0",
                }
                for index in range(count)
            ],
        },
    }


def test_worst_status_orders_fail_above_warn():
    assert worst_status(["PASS", "WARN"]) == "WARN"
    assert worst_status(["WARN", "FAIL"]) == "FAIL"
    assert worst_status(["UNKNOWN", "PASS"]) == "UNKNOWN"
    assert worst_status([]) == "UNKNOWN"


def test_expand_env_resolves_scratch(monkeypatch):
    monkeypatch.setenv("LOCAL_SCRATCH", "/mnt/local")
    assert expand_env({"local_scratch": "${LOCAL_SCRATCH}"})["local_scratch"] == "/mnt/local"


def test_gpu_count_mismatch_is_hard_failure():
    result = check_gpu_expectations(_inventory(count=1), _config())
    assert result["status"] == "FAIL"
    assert any("GPU count" in item for item in result["hard_failures"])


def test_gpu_name_pattern_mismatch_is_hard_failure():
    result = check_gpu_expectations(
        _inventory(names=["NVIDIA A100", "NVIDIA A100"]), _config()
    )
    assert result["status"] == "FAIL"
    assert any("does not contain" in item for item in result["hard_failures"])


def test_matching_gpus_pass():
    assert check_gpu_expectations(_inventory(), _config())["status"] == "PASS"


def test_absolute_thresholds_empty_are_unknown_not_pass():
    result = absolute_threshold_check(_config(absolute_thresholds={}))
    assert result["status"] == "UNKNOWN"
    assert "absolute_performance" in result["unknown_checks"]


def test_aggregate_fails_on_cuda_smoke(tmp_path):
    summary = aggregate(
        _config(),
        _inventory(),
        {"status": "FAIL", "reason": "CUDA error"},
        {"status": "PASS"},
        nccl_dir=None,
    )
    assert summary["status"] == "FAIL"
    assert any("gpu_smoke" in item or "CUDA" in item for item in summary["hard_failures"])


def test_aggregate_does_not_fabricate_network_pass_without_logs():
    summary = aggregate(
        _config(),
        _inventory(),
        {"status": "PASS", "reason": None},
        {"status": "UNKNOWN", "reason": "LOCAL_SCRATCH not configured"},
        nccl_dir=None,
    )
    assert "nccl_logs_not_provided" in summary["unknown_checks"]
    assert summary["network"]["status"] == "UNKNOWN"


def test_aggregate_folds_nccl_logs(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    nccl_dir = tmp_path / "nccl"
    nccl_dir.mkdir()
    (nccl_dir / "nccl-intra-worker-0.log").write_text(
        fixtures.joinpath("nccl_valid.log").read_text()
    )
    (nccl_dir / "nccl-intra-worker-1.log").write_text(
        fixtures.joinpath("nccl_valid.log").read_text()
    )
    for rep in (1, 2, 3):
        (nccl_dir / f"nccl-inter-run-{rep}.log").write_text(
            fixtures.joinpath("nccl_valid.log").read_text()
        )

    summary = aggregate(
        _config(),
        _inventory(),
        {"status": "PASS", "reason": None},
        {"status": "PASS"},
        nccl_dir=nccl_dir,
    )

    assert summary["network"]["status"] in ("PASS", "WARN")
    assert summary["network"]["inter"]["total_wrong"] == 0
    assert summary["hard_failures"] == []
    assert (nccl_dir / "nccl-intra-worker-0.json").exists()


def test_aggregate_fails_when_nccl_has_wrong_values(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    nccl_dir = tmp_path / "nccl"
    nccl_dir.mkdir()
    (nccl_dir / "nccl-inter-run-1.log").write_text(
        fixtures.joinpath("nccl_wrong_values.log").read_text()
    )

    summary = aggregate(
        _config(),
        _inventory(),
        {"status": "PASS", "reason": None},
        {"status": "PASS"},
        nccl_dir=nccl_dir,
    )
    assert summary["status"] == "FAIL"
    assert summary["network"]["inter"]["total_wrong"] == 3


def test_report_mentions_overall_status():
    summary = aggregate(
        _config(),
        _inventory(),
        {"status": "PASS", "reason": None},
        {"status": "PASS"},
        nccl_dir=None,
    )
    text = render_report(summary)
    assert f"**Overall status:** {summary['status']}" in text
    assert "Hard failures" in text
