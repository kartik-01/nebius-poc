"""Pilot comparison and the machine-readable recipe lock.

The official test split never enters this module. Selection uses only the internal
held-out adaptation IDs, scored with the shared forced-choice evaluator.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from nebius_poc.data import load_config
from nebius_poc.evaluate import accuracy, mean_gold_choice_nll
from nebius_poc.report import read_jsonl, write_json

log = logging.getLogger(__name__)


def score_pilot(rows: Sequence[dict], label: str) -> dict:
    return {
        "label": label,
        "n": len(rows),
        "mean_gold_choice_nll": mean_gold_choice_nll(rows),
        "forced_choice_accuracy": accuracy(rows),
    }


def select_objective(candidates: Sequence[dict]) -> dict:
    """Lower mean gold-choice NLL wins; accuracy is the tie-break only."""
    if not candidates:
        raise ValueError("no pilot candidates to compare")
    if len({item["n"] for item in candidates}) != 1:
        raise ValueError("pilot candidates must cover the same number of questions")

    ranked = sorted(
        candidates,
        key=lambda item: (item["mean_gold_choice_nll"], -item["forced_choice_accuracy"]),
    )
    winner = ranked[0]
    return {
        "primary_metric": "mean_gold_choice_nll",
        "secondary_metric": "forced_choice_accuracy",
        "candidates": list(candidates),
        "winner_label": winner["label"],
        "winner_mean_gold_choice_nll": winner["mean_gold_choice_nll"],
        "winner_forced_choice_accuracy": winner["forced_choice_accuracy"],
    }


def build_recipe_lock(
    config: dict,
    selection: dict,
    *,
    learning_rate: float | None = None,
    lora_rank: int | None = None,
    max_length: int | None = None,
    epochs: int | None = None,
    sources: dict | None = None,
    notes: Sequence[str] | None = None,
) -> dict:
    training = config["training"]
    lora = config["lora"]
    return {
        "locked_utc": datetime.now(UTC).isoformat(),
        "objective": config["objective"],
        "learning_rate": float(
            learning_rate if learning_rate is not None else training["learning_rate"]
        ),
        "lora_rank": int(lora_rank if lora_rank is not None else lora["rank"]),
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": list(lora["target_modules"]),
        "epochs": int(epochs if epochs is not None else training["epochs"]),
        "max_length": int(max_length if max_length is not None else training["max_length"]),
        "per_device_batch_size": training["per_device_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "seed": training["seed"],
        "model": dict(config["model"]),
        "dataset": dict(config["dataset"]),
        "augmentation": {
            "max_variants_per_question": config["dataset"]["max_variants_per_question"],
            "policy": "safe_choice_permutation",
        },
        "checkpoint_rule": {
            "retention": training.get("checkpoint_retention", 2),
            "select": "final_adapter_after_locked_epochs",
        },
        "selection": selection,
        "sources": sources or {},
        "notes": list(notes or []),
    }


def apply_recipe_lock(config: dict, lock: dict) -> dict:
    """Overlay a locked recipe onto a training config.

    The lock wins for objective and hyperparameters so final training cannot silently
    drift from the pilot decision. Dataset identity is checked, not overwritten, so a
    mismatched lock fails loudly.
    """
    dataset = config["dataset"]
    locked_dataset = lock.get("dataset") or {}
    for key in ("id", "config", "adaptation_split", "final_test_split", "seed"):
        if key in locked_dataset and dataset.get(key) != locked_dataset[key]:
            raise ValueError(
                f"recipe lock dataset.{key}={locked_dataset[key]!r} does not match "
                f"config {dataset.get(key)!r}"
            )

    updated = {
        **config,
        "objective": lock["objective"],
        "lora": {
            **config["lora"],
            "rank": int(lock["lora_rank"]),
            "alpha": lock.get("lora_alpha", config["lora"]["alpha"]),
            "dropout": lock.get("lora_dropout", config["lora"]["dropout"]),
            "target_modules": list(
                lock.get("lora_target_modules") or config["lora"]["target_modules"]
            ),
        },
        "training": {
            **config["training"],
            "learning_rate": float(lock["learning_rate"]),
            "epochs": int(lock["epochs"]),
            "max_length": int(lock["max_length"]),
            "per_device_batch_size": int(
                lock.get("per_device_batch_size", config["training"]["per_device_batch_size"])
            ),
            "gradient_accumulation_steps": int(
                lock.get(
                    "gradient_accumulation_steps",
                    config["training"]["gradient_accumulation_steps"],
                )
            ),
            "seed": int(lock.get("seed", config["training"]["seed"])),
        },
        "model": {
            **config["model"],
            **{
                key: lock["model"][key]
                for key in ("id", "revision", "dtype")
                if lock.get("model") and key in lock["model"] and lock["model"][key] is not None
            },
        },
    }
    return updated


def load_recipe_lock(path: Path) -> dict:
    import json

    lock = json.loads(Path(path).read_text())
    required = {
        "objective",
        "learning_rate",
        "lora_rank",
        "epochs",
        "max_length",
        "dataset",
    }
    missing = required - lock.keys()
    if missing:
        raise ValueError(f"{path} is missing lock fields: {sorted(missing)}")
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare pilot forced-choice runs and optionally write recipe_lock.json"
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "CONFIG", "FORCED_CHOICE_JSONL"),
        required=True,
        help="repeatable: label, training config path, forced_choice.jsonl from the internal set",
    )
    parser.add_argument(
        "--lock",
        action="store_true",
        help="write results/summary/recipe_lock.json for the winning candidate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary/recipe_lock.json"),
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="free-form note recorded in the lock (repeatable)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scored = []
    configs = {}
    sources = {}
    for label, config_path, jsonl_path in args.candidate:
        config = load_config(Path(config_path))
        rows = read_jsonl(Path(jsonl_path))
        scored.append(score_pilot(rows, label))
        configs[label] = (config, Path(config_path))
        sources[label] = {
            "config": str(config_path),
            "forced_choice": str(jsonl_path),
            "objective": config["objective"],
        }

    selection = select_objective(scored)
    log.info(
        "winner=%s mean_gold_nll=%.4f accuracy=%.4f",
        selection["winner_label"],
        selection["winner_mean_gold_choice_nll"],
        selection["winner_forced_choice_accuracy"],
    )
    for item in selection["candidates"]:
        log.info(
            "  %-20s nll=%.4f acc=%.4f",
            item["label"],
            item["mean_gold_choice_nll"],
            item["forced_choice_accuracy"],
        )

    comparison_path = args.out.with_name("pilot_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        comparison_path,
        {
            "generated_utc": datetime.now(UTC).isoformat(),
            "selection": selection,
            "sources": sources,
        },
    )
    log.info("wrote %s", comparison_path)

    if not args.lock:
        return 0

    winner = selection["winner_label"]
    config, config_path = configs[winner]
    # If the user passed hyperparameter overrides, they apply to the locked recipe.
    # The config file on disk stays unchanged so the pilot remains reproducible.
    lock = build_recipe_lock(
        config,
        selection,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_length=args.max_length,
        epochs=args.epochs,
        sources={"winner_config": str(config_path), **sources},
        notes=args.note
        or [
            "Locked from internal pilot comparison only; official test was not inspected.",
        ],
    )
    write_json(args.out, lock)
    log.info("locked recipe -> %s (objective=%s)", args.out, lock["objective"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
