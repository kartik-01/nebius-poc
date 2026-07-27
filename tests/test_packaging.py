from pathlib import Path

import pytest
import yaml

import nebius_poc

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

REQUIRED_KEYS = {
    "validator.yaml": {
        "expected_nodes",
        "expected_gpus_total",
        "expected_gpus_per_node",
        "expected_gpu_name_pattern",
        "local_scratch",
        "fio_size_gib",
        "fio_shared_enabled",
        "nccl_repetitions",
        "variability_warn_cv",
        "node_asymmetry_warn_ratio",
        "absolute_thresholds",
    },
    "train_sft.yaml": {"objective", "model", "dataset", "lora", "training"},
    "train_ranking.yaml": {"objective", "model", "dataset", "lora", "training"},
    "evaluation.yaml": {"mode", "candidate_labels", "bootstrap_resamples", "seed", "generation"},
    "benchmark.yaml": {
        "objective",
        "concurrency",
        "input_tokens",
        "output_tokens",
        "warmup_requests",
        "guardrails",
    },
    "serve_topologies.yaml": {"topologies", "server"},
}


def load_config(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def test_package_imports():
    assert nebius_poc.__version__


@pytest.mark.parametrize("name", sorted(REQUIRED_KEYS))
def test_config_parses_and_has_required_keys(name):
    config = load_config(name)
    missing = REQUIRED_KEYS[name] - config.keys()
    assert not missing, f"{name} is missing {sorted(missing)}"


def test_pilot_configs_differ_only_by_objective():
    # The pilot only means anything if the objective is the one thing that changes.
    sft = load_config("train_sft.yaml")
    ranking = load_config("train_ranking.yaml")

    assert sft["objective"] == "completion_sft"
    assert ranking["objective"] == "candidate_ranking"

    sft.pop("objective")
    ranking.pop("objective")
    assert sft == ranking


def test_adaptation_split_is_not_the_official_test_split():
    # Test split is held back for the final base-vs-tuned comparison.
    for name in ("train_sft.yaml", "train_ranking.yaml"):
        dataset = load_config(name)["dataset"]
        assert dataset["adaptation_split"] != dataset["final_test_split"]
        assert dataset["adaptation_split"] == "validation"


def test_pilot_split_sizes_match_the_adaptation_pool():
    dataset = load_config("train_sft.yaml")["dataset"]
    assert dataset["pilot_train_size"] + dataset["pilot_validation_size"] == 170


def test_guardrails_are_declared_before_benchmarking():
    guardrails = load_config("benchmark.yaml")["guardrails"]
    assert guardrails["p95_ttft_ms"] > 0
    assert guardrails["p95_tpot_ms"] > 0
    assert guardrails["max_error_rate"] == 0.0


def test_validator_dockerfile_keeps_runtime_thin():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "containers" / "validator.Dockerfile"
    ).read_text()
    # Last FROM stage is the runtime image. The build stage is allowed to use nvcc.
    runtime = dockerfile[dockerfile.rfind("\nFROM ") :]
    for forbidden in ("openmpi", "mpich", "transformers", "torch", "vllm", "peft", "nvcc"):
        assert forbidden not in runtime.lower(), forbidden
    assert "AS build" in dockerfile
    assert "gpu_smoke" in dockerfile
