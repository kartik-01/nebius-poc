"""Evaluate one model on a split, by forced choice and by deterministic generation.

Forced choice is the primary metric: it measures answer selection without letting
output formatting interfere. Generation is reported separately because a model that
learns to emit a bare letter has improved its formatting, not its law.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from nebius_poc.data import Question, evaluation_holdout, load_config, load_split
from nebius_poc.objectives import build_scoring_batch, candidate_scores, encode_candidates
from nebius_poc.prompts import LABELS, render_prompt
from nebius_poc.report import close_run, format_adherence, open_run, write_json, write_jsonl
from nebius_poc.train import resolve_dtype, setup_distributed

log = logging.getLogger(__name__)

# Strict, uppercase only. Leniency here would inflate the format-adherence number,
# which is the very thing this metric exists to measure.
_ANSWER = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str) -> str | None:
    match = _ANSWER.search(text)
    return match.group(1) if match else None


def load_model(model_id: str, revision: str | None, adapter: Path | None, dtype, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    return model.to(device).eval()


@torch.no_grad()
def forced_choice(
    model, tokenizer, questions: Sequence[Question], max_length: int, batch_size: int, device
) -> list[dict]:
    rows: list[dict] = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        encoded = [
            encode_candidates(tokenizer, item.subject, item.question, item.choices)
            for item in chunk
        ]
        batch = build_scoring_batch(encoded, tokenizer.pad_token_id, max_length).to(device)

        output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        scores = candidate_scores(output.logits, batch.labels, batch.num_candidates)
        scores = scores.float().cpu()
        probabilities = torch.softmax(scores, dim=-1)

        for item, score, probability in zip(chunk, scores, probabilities, strict=True):
            prediction = int(score.argmax())
            row = {
                "question_id": item.qid,
                "gold_answer": LABELS[item.answer],
                "prediction": LABELS[prediction],
                "predicted_probability": float(probability[prediction]),
                "correct": prediction == item.answer,
            }
            for index, label in enumerate(LABELS):
                row[f"score_{label.lower()}"] = float(score[index])
                row[f"prob_{label.lower()}"] = float(probability[index])
            rows.append(row)
    return rows


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    questions: Sequence[Question],
    max_new_tokens: int,
    batch_size: int,
    device,
) -> list[dict]:
    # Decoder-only batched generation needs the padding on the left, otherwise the
    # continuation starts after a run of pad tokens.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        rows: list[dict] = []
        for start in range(0, len(questions), batch_size):
            chunk = questions[start : start + batch_size]
            prompts = [
                render_prompt(item.subject, item.question, item.choices) for item in chunk
            ]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)

            generated = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            completions = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            for item, text in zip(chunk, completions, strict=True):
                parsed = parse_answer(text)
                rows.append(
                    {
                        "question_id": item.qid,
                        "gold_answer": LABELS[item.answer],
                        "generated": text.strip(),
                        "parsed": parsed,
                        "correct": parsed == LABELS[item.answer],
                    }
                )
        return rows
    finally:
        tokenizer.padding_side = original_side


def accuracy(rows: Sequence[dict]) -> float:
    return sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0


def gold_choice_nll(row: dict) -> float:
    """Negative log-likelihood of the gold letter under the forced-choice scores."""
    gold = str(row["gold_answer"]).lower()
    key = f"score_{gold}"
    if key not in row:
        raise KeyError(f"missing {key} in forced-choice row")
    return -float(row[key])


def mean_gold_choice_nll(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return sum(gold_choice_nll(row) for row in rows) / len(rows)


def load_id_list(path: Path) -> list[str]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        for key in ("pilot_validation_ids", "question_ids", "ids"):
            if key in payload:
                return [str(item) for item in payload[key]]
        raise ValueError(f"{path} JSON has no id list")
    return [line.strip() for line in text.splitlines() if line.strip()]


def filter_questions_by_ids(
    questions: Sequence[Question], ids: Sequence[str]
) -> list[Question]:
    wanted = set(ids)
    selected = [question for question in questions if question.qid in wanted]
    missing = wanted - {question.qid for question in selected}
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"{len(missing)} ids not found in split (e.g. {sample})")
    # Keep the caller-provided order so paired comparisons stay stable.
    by_id = {question.qid: question for question in selected}
    return [by_id[qid] for qid in ids]


def run(config: dict, evaluation: dict, args: argparse.Namespace) -> Path:
    from nebius_poc.train import load_tokenizer

    _, _, _, device = setup_distributed("auto" if args.device == "auto" else args.device)

    model_id = args.model or config["model"]["id"]
    # The pinned revision belongs to the configured model; a smoke override must
    # fall back to that model's own default revision.
    revision = None if model_id != config["model"]["id"] else config["model"].get("revision")
    dataset = config["dataset"]

    label = args.label or ("tuned" if args.adapter else "base")
    run_dir, manifest = open_run(f"evaluate-{label}", args.results_root, config)

    if args.split == "holdout":
        # The reserved share of the category's records. Training cannot reach these:
        # load_adaptation_pool carves them out from the same seed and never returns
        # them, and the split manifest records both id lists for audit.
        questions = evaluation_holdout(
            dataset["id"],
            dataset["config"],
            revision=dataset.get("revision"),
            trainable_split=dataset["trainable_split"],
            holdout_fraction=dataset["holdout_fraction"],
            seed=dataset["seed"],
        )
    else:
        questions = load_split(dataset["id"], dataset["config"], args.split)
    if args.ids_file:
        ids = load_id_list(args.ids_file)
        questions = filter_questions_by_ids(questions, ids)
        log.info("filtered to %d ids from %s", len(questions), args.ids_file)
    if args.limit:
        questions = questions[: args.limit]
    log.info("evaluating %s on %d %s questions", label, len(questions), args.split)

    tokenizer = load_tokenizer(model_id, revision)
    dtype = resolve_dtype(config["model"]["dtype"], device)
    model = load_model(model_id, revision, args.adapter, dtype, device)

    artifacts: dict = {"split": args.split, "questions": len(questions), "label": label}
    summary: dict = {"label": label, "split": args.split, "n": len(questions)}

    if args.mode in ("forced_choice", "both"):
        rows = forced_choice(
            model, tokenizer, questions, config["training"]["max_length"], args.batch_size, device
        )
        write_jsonl(run_dir / "forced_choice.jsonl", rows)
        summary["forced_choice_accuracy"] = accuracy(rows)
        summary["mean_gold_choice_nll"] = mean_gold_choice_nll(rows)
        artifacts["forced_choice"] = str(run_dir / "forced_choice.jsonl")
        log.info(
            "forced-choice accuracy %.4f, mean gold NLL %.4f",
            summary["forced_choice_accuracy"],
            summary["mean_gold_choice_nll"],
        )

    if args.mode in ("generation", "both"):
        rows = generate_answers(
            model,
            tokenizer,
            questions,
            evaluation["generation"]["max_new_tokens"],
            args.batch_size,
            device,
        )
        write_jsonl(run_dir / "generation.jsonl", rows)
        summary["generation"] = format_adherence(rows)
        artifacts["generation"] = str(run_dir / "generation.jsonl")
        log.info(
            "generation accuracy %.4f, format adherence %.4f",
            summary["generation"]["accuracy"],
            summary["generation"]["format_adherence"],
        )

    write_json(run_dir / "summary.json", summary)
    close_run(run_dir, manifest, "ok", artifacts)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a model on an MMLU split")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", help="name used in the run directory and summary")
    parser.add_argument("--split", default="holdout")
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="JSON split_manifest or newline id list; restricts evaluation to those IDs",
    )
    parser.add_argument("--mode", choices=("forced_choice", "generation", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluation = yaml.safe_load(args.eval_config.read_text())
    run(load_config(args.config), evaluation, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
