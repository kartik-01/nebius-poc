"""Both predeclared training objectives, and the scoring they share with evaluation.

Completion-only SFT and candidate ranking are built on the same primitive: score a
continuation conditioned on a prompt, with the prompt masked out of the loss. Keeping
one implementation means the forced-choice evaluator and the ranking trainer cannot
disagree about where the prompt ends and the answer begins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nebius_poc.prompts import CANDIDATE_STRINGS, LABELS, render_prompt

IGNORE_INDEX = -100

OBJECTIVES = ("completion_sft", "candidate_ranking")


@dataclass(frozen=True)
class ScoringBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    num_candidates: int

    def to(self, device) -> ScoringBatch:
        return ScoringBatch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.labels.to(device),
            self.num_candidates,
        )


def encode_candidates(
    tokenizer,
    subject: str,
    question: str,
    choices: Sequence[str],
    labels: Sequence[str] = CANDIDATE_STRINGS,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize one question into a prompt prefix and the candidate continuations.

    The prompt stops at "Answer:" and each candidate carries its own leading space,
    so the candidate tokens attach exactly where the model would start generating.
    Evaluation and training both come through here so that boundary is identical in
    both paths.
    """
    prompt_text = render_prompt(subject, question, choices)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    candidate_ids = [
        tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels
    ]

    for label, ids in zip(labels, candidate_ids, strict=True):
        if not ids:
            raise ValueError(f"candidate {label!r} tokenized to nothing")
    return list(prompt_ids), [list(ids) for ids in candidate_ids]


def _fit(
    prompt_ids: Sequence[int], candidate_ids: Sequence[int], max_length: int
) -> tuple[list[int], int]:
    if not candidate_ids:
        raise ValueError("candidate cannot be empty")
    if len(candidate_ids) >= max_length:
        raise ValueError(
            f"candidate needs {len(candidate_ids)} tokens, max_length is {max_length}"
        )

    # Trim the front of the prompt, never the candidate. The scored tokens have to
    # survive intact, and the end of the prompt is the part the answer hangs on.
    room = max_length - len(candidate_ids)
    kept = list(prompt_ids[-room:]) if len(prompt_ids) > room else list(prompt_ids)
    return kept + list(candidate_ids), len(kept)


def build_scoring_batch(
    examples: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]],
    pad_token_id: int,
    max_length: int,
) -> ScoringBatch:
    """Lay out one row per (question, candidate) pair, question-major.

    Row order is what makes the later `view(-1, num_candidates)` line up, so all of
    question 0's candidates come first, then question 1's, and so on.
    """
    if not examples:
        raise ValueError("no examples to score")

    counts = {len(candidates) for _, candidates in examples}
    if len(counts) != 1:
        raise ValueError(f"every question needs the same candidate count, saw {sorted(counts)}")
    num_candidates = counts.pop()
    if num_candidates == 0:
        raise ValueError("no candidates to score")

    rows = [
        _fit(prompt_ids, candidate, max_length)
        for prompt_ids, candidates in examples
        for candidate in candidates
    ]

    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)

    for row, (ids, prompt_length) in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        labels[row, prompt_length : len(ids)] = torch.tensor(
            ids[prompt_length:], dtype=torch.long
        )

    return ScoringBatch(input_ids, attention_mask, labels, num_candidates)


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed log-likelihood of the unmasked tokens in each row."""
    # The logit at position t predicts the token at t+1, so both sides shift by one.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    keep = shift_labels != IGNORE_INDEX
    # gather rejects -100, so park the masked entries on a valid index and drop them
    # afterwards with the mask.
    gather_index = shift_labels.masked_fill(~keep, 0).unsqueeze(-1)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, gather_index).squeeze(-1)
    return (token_log_probs * keep).sum(dim=-1)


def candidate_scores(
    logits: torch.Tensor, labels: torch.Tensor, num_candidates: int
) -> torch.Tensor:
    flat = sequence_log_probs(logits, labels)
    if flat.numel() % num_candidates:
        raise ValueError(f"{flat.numel()} rows do not divide into {num_candidates} candidates")
    return flat.view(-1, num_candidates)


def ranking_loss(scores: torch.Tensor, gold_index: torch.Tensor) -> torch.Tensor:
    """Cross-entropy across the four candidate scores for each question.

    The wrong choices act as explicit negatives, which is the whole point of this
    objective: it optimizes the same quantity the forced-choice evaluator measures.
    Scores stay as summed log-likelihood rather than per-token averages, since the
    four candidates are the same length and normalizing would only add a knob.
    """
    if scores.ndim != 2:
        raise ValueError(f"expected (questions, candidates), got {tuple(scores.shape)}")
    if gold_index.shape != scores.shape[:1]:
        raise ValueError(
            f"{gold_index.numel()} gold labels for {scores.shape[0]} questions"
        )
    return F.cross_entropy(scores, gold_index)


def completion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def gold_completion_batch(
    examples: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]],
    gold_index: Sequence[int],
    pad_token_id: int,
    max_length: int,
) -> ScoringBatch:
    """Completion-only SFT is the candidate batch narrowed to the correct answer."""
    if len(examples) != len(gold_index):
        raise ValueError(f"{len(examples)} examples against {len(gold_index)} gold labels")

    narrowed = [
        (prompt_ids, [candidates[gold]])
        for (prompt_ids, candidates), gold in zip(examples, gold_index, strict=True)
    ]
    return build_scoring_batch(narrowed, pad_token_id, max_length)


_REQUIRED_SECTIONS = ("objective", "model", "dataset", "lora", "training")

_POSITIVE_INTS = {
    "lora": ("rank", "alpha"),
    "training": ("epochs", "per_device_batch_size", "gradient_accumulation_steps", "max_length"),
    "dataset": ("pilot_train_size", "pilot_validation_size", "max_variants_per_question"),
}


def validate_training_config(config: dict) -> None:
    missing = set(_REQUIRED_SECTIONS) - config.keys()
    if missing:
        raise ValueError(f"config is missing required sections: {sorted(missing)}")

    if config["objective"] not in OBJECTIVES:
        raise ValueError(
            f"unknown objective {config['objective']!r}, expected one of {list(OBJECTIVES)}"
        )

    for section, fields in _POSITIVE_INTS.items():
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"{section}.{field} is required")
            value = config[section][field]
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{field} must be a positive integer, got {value!r}")

    dropout = config["lora"]["dropout"]
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"lora.dropout must be in [0, 1), got {dropout}")

    if not config["lora"].get("target_modules"):
        raise ValueError("lora.target_modules cannot be empty")

    learning_rate = config["training"]["learning_rate"]
    if not isinstance(learning_rate, int | float) or learning_rate <= 0:
        raise ValueError(f"training.learning_rate must be positive, got {learning_rate!r}")

    variants = config["dataset"]["max_variants_per_question"]
    if variants > len(LABELS):
        raise ValueError(
            f"max_variants_per_question cannot exceed {len(LABELS)}, got {variants}"
        )

    if config["dataset"]["adaptation_split"] == config["dataset"]["final_test_split"]:
        raise ValueError("adaptation_split and final_test_split must differ")
