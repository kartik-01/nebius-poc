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
    # Ranking forwards four candidates per question, so it has to use a smaller
    # micro-batch to fit in memory; accumulation restores the same effective batch.
    # That split is a memory detail, not an experimental variable, so compare the
    # effective batch and require everything else to match exactly.
    sft = load_config("train_sft.yaml")
    ranking = load_config("train_ranking.yaml")

    assert sft["objective"] == "completion_sft"
    assert ranking["objective"] == "candidate_ranking"

    def effective_batch(config: dict) -> int:
        training = config["training"]
        return training["per_device_batch_size"] * training["gradient_accumulation_steps"]

    assert effective_batch(sft) == effective_batch(ranking)

    for config in (sft, ranking):
        config.pop("objective")
        for key in ("per_device_batch_size", "gradient_accumulation_steps"):
            config["training"].pop(key)
    assert sft == ranking


def test_a_real_share_of_the_category_is_reserved_for_evaluation():
    # The holdout is what makes the base-vs-tuned comparison meaningful, so it has to
    # be a substantial slice rather than a token one.
    for name in ("train_sft.yaml", "train_ranking.yaml"):
        dataset = load_config(name)["dataset"]
        assert dataset["adaptation_split"] == "validation"
        assert dataset["adaptation_split"] != dataset["trainable_split"]
        assert 0.2 <= dataset["holdout_fraction"] <= 0.5


def test_internal_selection_set_is_large_enough_to_separate_recipes():
    # Twenty questions moved in five-point steps and could not rank close candidates.
    dataset = load_config("train_sft.yaml")["dataset"]
    assert dataset["pilot_validation_size"] >= 50


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


def test_slurm_launchers_cover_the_offline_scaffolded_stages():
    root = Path(__file__).resolve().parents[1] / "slurm"
    expected = {
        "validate.sbatch",
        "validate_smoke.sbatch",
        "train.sbatch",
        "train_smoke.sbatch",
        "evaluate.sbatch",
        "merge.sbatch",
        "serve.sbatch",
        "benchmark.sbatch",
    }
    assert expected.issubset({path.name for path in root.glob("*.sbatch")})
    train = (root / "train.sbatch").read_text()
    assert "TRAIN_FINAL" in train
    assert "RECIPE_LOCK" in train
    assert "--final" in train
    merge = (root / "merge.sbatch").read_text()
    assert "nebius_poc.merge_adapter" in merge
