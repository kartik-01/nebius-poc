"""LoRA fine-tuning for both predeclared objectives.

A plain torch loop rather than a framework trainer. The candidate-ranking objective
needs a custom loss over a reshaped (questions x candidates) batch, and writing the
loop directly keeps every line explainable and the two objectives symmetric.

This module can only ever read the adaptation pool. `load_adaptation_pool` refuses
any split but `validation`, so the official test set is unreachable from here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from nebius_poc.data import (
    Variant,
    expand,
    load_adaptation_pool,
    load_config,
    split_adaptation_pool,
    split_manifest,
)
from nebius_poc.objectives import (
    ScoringBatch,
    build_scoring_batch,
    candidate_scores,
    completion_loss,
    encode_candidates,
    gold_completion_batch,
    ranking_loss,
    validate_training_config,
)
from nebius_poc.report import close_run, open_run, write_json

log = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass(frozen=True)
class EncodedExample:
    qid: str
    variant_id: str
    prompt_ids: list[int]
    candidate_ids: list[list[int]]
    gold: int


def cosine_schedule_with_warmup(optimizer, total_steps: int, warmup_ratio: float):
    """Linear warmup, then cosine decay to zero.

    Written out rather than taken from OneCycleLR, whose div_factor defaults start and
    end the run at learning rates nobody asked for, and which are easy to miss when
    reading the metrics back.
    """
    warmup = max(1, round(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        # bf16 matmul on CPU is slow and unevenly supported. The CPU path only exists
        # to prove the wiring, so drop to fp32 instead of failing.
        return torch.float32
    return DTYPES[name]


def encode_variants(variants: Sequence[Variant], tokenizer) -> list[EncodedExample]:
    encoded = []
    for variant in variants:
        prompt_ids, candidate_ids = encode_candidates(
            tokenizer, variant.question, variant.choices
        )
        encoded.append(
            EncodedExample(
                qid=variant.source_qid,
                variant_id=variant.variant_id,
                prompt_ids=prompt_ids,
                candidate_ids=candidate_ids,
                gold=variant.answer,
            )
        )
    return encoded


def collate(
    examples: Sequence[EncodedExample], objective: str, pad_token_id: int, max_length: int
) -> tuple[ScoringBatch, torch.Tensor]:
    pairs = [(example.prompt_ids, example.candidate_ids) for example in examples]
    gold = [example.gold for example in examples]

    if objective == "candidate_ranking":
        batch = build_scoring_batch(pairs, pad_token_id, max_length)
    elif objective == "completion_sft":
        batch = gold_completion_batch(pairs, gold, pad_token_id, max_length)
    else:
        raise ValueError(f"unknown objective {objective!r}")

    return batch, torch.tensor(gold, dtype=torch.long)


def objective_loss(
    logits: torch.Tensor, batch: ScoringBatch, gold: torch.Tensor, objective: str
) -> torch.Tensor:
    if objective == "candidate_ranking":
        scores = candidate_scores(logits, batch.labels, batch.num_candidates)
        return ranking_loss(scores, gold)
    return completion_loss(logits, batch.labels)


def setup_distributed(preferred: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available() if preferred == "auto" else preferred == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch reports no CUDA devices")

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl" if use_cuda else "gloo")

    return rank, world_size, local_rank, device


def load_tokenizer(model_id: str, revision: str | None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_id: str, revision: str | None, dtype: torch.dtype, lora: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )
    return get_peft_model(base, peft_config)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, step: int, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(directory / "adapter")
    torch.save(
        {
            "global_step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        directory / "training_state.pt",
    )
    return directory


def load_checkpoint(model, optimizer, scheduler, directory: Path, device) -> int:
    """Restore adapter weights and optimizer state, returning the completed step count."""
    directory = Path(directory)
    unwrap(model).load_adapter(
        str(directory / "adapter"), adapter_name="default", is_trainable=True
    )

    state = torch.load(directory / "training_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["global_step"])


def prune_checkpoints(root: Path, keep: int) -> None:
    saved = sorted(Path(root).glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    for stale in saved[:-keep] if keep > 0 else saved:
        shutil.rmtree(stale, ignore_errors=True)


def build_dataset(config: dict, tokenizer, limit: int | None) -> tuple[list[EncodedExample], dict]:
    dataset = config["dataset"]
    pool = load_adaptation_pool(dataset["id"], dataset["config"], dataset["adaptation_split"])
    train, held_out = split_adaptation_pool(
        pool, dataset["pilot_train_size"], dataset["pilot_validation_size"], dataset["seed"]
    )
    if limit:
        train = train[:limit]

    variants = expand(train, dataset["max_variants_per_question"])
    manifest = split_manifest(train, held_out, dataset["seed"], dataset)
    manifest["training_rows_after_augmentation"] = len(variants)
    return encode_variants(variants, tokenizer), manifest


def run(config: dict, args: argparse.Namespace) -> Path:
    validate_training_config(config)

    rank, world_size, local_rank, device = setup_distributed(args.device)
    is_main = rank == 0

    # Fold the CLI overrides in before the manifest is written, so the manifest
    # describes the run that actually happened rather than the file on disk.
    training = dict(config["training"])
    if args.batch_size:
        training["per_device_batch_size"] = args.batch_size
    config = {**config, "training": training}
    if args.model:
        config["model"] = {**config["model"], "id": args.model}

    objective = config["objective"]
    model_id = config["model"]["id"]
    revision = config["model"].get("revision")

    # Offset by rank so dropout masks differ across replicas while the data order,
    # which DistributedSampler seeds separately, stays reproducible.
    set_seed(training["seed"] + rank)

    if training.get("tf32") and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_dir, manifest = (
        open_run(f"train-{objective}", args.results_root, config) if is_main else (None, None)
    )

    tokenizer = load_tokenizer(model_id, revision)
    examples, split_info = build_dataset(config, tokenizer, args.limit)
    if is_main:
        write_json(run_dir / "split_manifest.json", split_info)
    log.info("rank %d: %d training rows", rank, len(examples))

    dtype = resolve_dtype(config["model"]["dtype"], device)
    model = build_model(model_id, revision, dtype, config["lora"]).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    sampler = (
        DistributedSampler(
            examples, num_replicas=world_size, rank=rank, shuffle=True, seed=training["seed"]
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        examples,
        batch_size=training["per_device_batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=partial(
            collate,
            objective=objective,
            pad_token_id=tokenizer.pad_token_id,
            max_length=training["max_length"],
        ),
        drop_last=False,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.01),
        fused=device.type == "cuda",
    )

    accumulation = training["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(loader) // accumulation)
    total_steps = args.max_steps or steps_per_epoch * training["epochs"]
    scheduler = cosine_schedule_with_warmup(
        optimizer, total_steps, training.get("warmup_ratio", 0.05)
    )

    metrics_path = run_dir / "metrics.jsonl" if is_main else None
    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(model, optimizer, scheduler, args.resume_from, device)
        log.info("resumed from %s at step %d", args.resume_from, global_step)

    model.train()
    started = time.perf_counter()

    # Checkpoints are written at epoch boundaries, so resuming replays whole epochs
    # rather than trying to reposition inside the sampler.
    for epoch in range(global_step // steps_per_epoch, training["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        step_started = time.perf_counter()
        for index, (batch, gold) in enumerate(loader):
            batch, gold = batch.to(device), gold.to(device)
            output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            loss = objective_loss(output.logits, batch, gold, objective) / accumulation
            loss.backward()

            if (index + 1) % accumulation:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, training.get("max_grad_norm", 1.0)
            )
            # Read before stepping, so the logged rate is the one that produced this
            # update rather than the one queued for the next.
            applied_lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main:
                elapsed = time.perf_counter() - step_started
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": loss.item() * accumulation,
                    "learning_rate": applied_lr,
                    "grad_norm": float(grad_norm),
                    "step_seconds": elapsed,
                    "samples_per_second": (
                        training["per_device_batch_size"] * accumulation * world_size / elapsed
                    ),
                    "peak_gpu_bytes": (
                        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
                    ),
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if global_step % 10 == 0 or global_step == 1:
                    log.info("step %d loss %.4f", global_step, record["loss"])

            step_started = time.perf_counter()
            if global_step >= total_steps:
                break

        if is_main:
            save_checkpoint(
                model, optimizer, scheduler, global_step, run_dir / f"checkpoint-{global_step}"
            )
            prune_checkpoints(run_dir, training.get("checkpoint_retention", 2))
        if global_step >= total_steps:
            break

    if world_size > 1:
        torch.distributed.barrier()

    final = None
    if is_main:
        final = run_dir / "adapter"
        unwrap(model).save_pretrained(final)
        close_run(
            run_dir,
            manifest,
            "ok",
            {
                "adapter": str(final),
                "metrics": str(metrics_path),
                "split_manifest": str(run_dir / "split_manifest.json"),
                "global_steps": global_step,
                "training_rows": len(examples),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        log.info("adapter written to %s", final)

    if world_size > 1:
        torch.distributed.destroy_process_group()
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on the MMLU adaptation pool")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", help="override the configured model, used by the smoke path")
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="cap the number of source questions")
    parser.add_argument("--resume-from", type=Path, help="a checkpoint-N directory to continue")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
