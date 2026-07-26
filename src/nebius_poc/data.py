"""MMLU record handling: stable IDs, the adaptation split, and safe choice permutation.

The official test split is never touched here. Everything in this module operates on
the category validation split, which the plan calls the adaptation pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from nebius_poc.prompts import LABELS

log = logging.getLogger(__name__)

ADAPTATION_POOL_SPLIT = "validation"


@dataclass(frozen=True)
class Question:
    qid: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class Variant:
    source_qid: str
    variant_id: str
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int
    original_answer: int
    permutation: tuple[int, ...]
    augmentation_applied: bool
    augmentation_skip_reason: str | None


def _canonical(text: str) -> str:
    return " ".join(text.split())


def stable_id(subject: str, question: str, choices: Sequence[str], answer: int) -> str:
    parts = [_canonical(subject), _canonical(question)]
    parts.extend(_canonical(choice) for choice in choices)
    parts.append(str(answer))
    # Unit separator, so a choice containing the delimiter cannot forge a collision.
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def build_question(record: dict) -> Question:
    choices = tuple(str(choice) for choice in record["choices"])
    if len(choices) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} choices, got {len(choices)}")

    answer = int(record["answer"])
    if not 0 <= answer < len(LABELS):
        raise ValueError(f"answer index {answer} outside 0..{len(LABELS) - 1}")

    subject = str(record.get("subject", ""))
    question = str(record["question"])
    return Question(
        qid=stable_id(subject, question, choices, answer),
        subject=subject,
        question=question,
        choices=choices,
        answer=answer,
    )


def load_split(dataset_id: str, config: str, split: str) -> list[Question]:
    from datasets import load_dataset

    rows = load_dataset(dataset_id, config, split=split)
    questions = [build_question(row) for row in rows]

    unique = {question.qid for question in questions}
    if len(unique) != len(questions):
        raise ValueError(f"{len(questions) - len(unique)} duplicate question IDs in {split}")
    return questions


def load_adaptation_pool(dataset_id: str, config: str, split: str) -> list[Question]:
    # The only entry point training code is allowed to call. Refusing anything but the
    # validation split is what makes test leakage structurally impossible rather than
    # a rule someone has to remember.
    if split != ADAPTATION_POOL_SPLIT:
        raise ValueError(
            f"adaptation pool must come from the '{ADAPTATION_POOL_SPLIT}' split, got '{split}'"
        )
    return load_split(dataset_id, config, split)


def _stratified_take(groups: dict[int, list[Question]], wanted: int) -> dict[int, int]:
    # Largest remainder, so the held-out set keeps the pool's label balance and the
    # two parts still add up to exactly the requested sizes.
    pool = sum(len(group) for group in groups.values())
    exact = {label: len(group) * wanted / pool for label, group in groups.items()}
    take = {label: int(value) for label, value in exact.items()}

    shortfall = wanted - sum(take.values())
    by_remainder = sorted(groups, key=lambda label: (-(exact[label] - take[label]), label))
    for label in by_remainder[:shortfall]:
        take[label] += 1

    for label, count in take.items():
        if count > len(groups[label]):
            raise ValueError(f"label {label} cannot supply {count} of {len(groups[label])}")
    return take


def split_adaptation_pool(
    questions: Sequence[Question],
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[list[Question], list[Question]]:
    total = train_size + validation_size
    if len(questions) != total:
        raise ValueError(f"pool holds {len(questions)} records but the split expects {total}")

    groups: dict[int, list[Question]] = {}
    # Sort by ID first so the result does not depend on the order the rows arrived in.
    for question in sorted(questions, key=lambda item: item.qid):
        groups.setdefault(question.answer, []).append(question)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    take = _stratified_take(groups, validation_size)
    train: list[Question] = []
    validation: list[Question] = []
    for label in sorted(groups):
        group = groups[label]
        validation.extend(group[: take[label]])
        train.extend(group[take[label] :])

    train.sort(key=lambda item: item.qid)
    validation.sort(key=lambda item: item.qid)
    return train, validation


# Applied to answer options. A choice that names another option by letter, or that
# aggregates the other options, stops meaning the same thing once they move.
_CHOICE_LABEL_PATTERNS = (
    re.compile(
        r"\b(all|none|both|any|either|neither)\s+of\s+the\s+(above|below|following)\b", re.I
    ),
    re.compile(r"\b(choices?|options?|answers?|alternatives?)\s+\(?[A-D]\)?\b", re.I),
    re.compile(r"\b[A-D]\s+(and|or)\s+[A-D]\b"),
    re.compile(r"\(\s*[A-D]\s*\)\s*(and|or)\s*\(\s*[A-D]\s*\)"),
)

# Applied to the question stem, which is narrower on purpose. Law questions name
# parties "A" and "B" constantly, and those letters have nothing to do with the
# answer labels.
_STEM_LABEL_PATTERNS = (
    re.compile(r"\b(all|none|both)\s+of\s+the\s+above\b", re.I),
    re.compile(r"\b(choices?|options?|answers?)\s+\(?[A-D]\)?\b", re.I),
)


def permutation_skip_reason(question: Question) -> str | None:
    for pattern in _STEM_LABEL_PATTERNS:
        if pattern.search(question.question):
            return f"stem references answer labels: {pattern.pattern}"

    for index, choice in enumerate(question.choices):
        for pattern in _CHOICE_LABEL_PATTERNS:
            if pattern.search(choice):
                return f"choice {LABELS[index]} references answer labels: {pattern.pattern}"
    return None


def rotate_choices(question: Question, shift: int) -> tuple[tuple[str, ...], int, tuple[int, ...]]:
    """Rotate the options so the correct answer lands `shift` positions later.

    Returns the new choices, the new answer index, and the source mapping where
    entry `i` is the original index of the option now sitting at position `i`.
    """
    size = len(question.choices)
    source = tuple((position - shift) % size for position in range(size))
    choices = tuple(question.choices[origin] for origin in source)
    return choices, (question.answer + shift) % size, source


def variants_for(question: Question, max_variants: int) -> Iterator[Variant]:
    identity = tuple(range(len(question.choices)))
    skip_reason = permutation_skip_reason(question)

    if skip_reason is not None or max_variants <= 1:
        yield Variant(
            source_qid=question.qid,
            variant_id=f"{question.qid}:0",
            subject=question.subject,
            question=question.question,
            choices=question.choices,
            answer=question.answer,
            original_answer=question.answer,
            permutation=identity,
            augmentation_applied=False,
            augmentation_skip_reason=skip_reason,
        )
        return

    for shift in range(min(max_variants, len(question.choices))):
        choices, answer, source = rotate_choices(question, shift)
        yield Variant(
            source_qid=question.qid,
            variant_id=f"{question.qid}:{shift}",
            subject=question.subject,
            question=question.question,
            choices=choices,
            answer=answer,
            original_answer=question.answer,
            permutation=source,
            augmentation_applied=shift != 0,
            augmentation_skip_reason=None,
        )


def expand(questions: Iterable[Question], max_variants: int) -> list[Variant]:
    return [variant for question in questions for variant in variants_for(question, max_variants)]


def augmentation_audit(
    questions: Sequence[Question], max_variants: int, sample_size: int = 12
) -> dict:
    """Build the artifact that gets read by hand before the recipe is locked.

    Pattern matching narrows the risk of a permutation changing a question's meaning.
    It does not prove the remaining permutations are safe, so the applied sample is
    here to be eyeballed.
    """
    skipped = []
    applied = []
    for question in sorted(questions, key=lambda item: item.qid):
        reason = permutation_skip_reason(question)
        if reason is not None:
            skipped.append(
                {
                    "qid": question.qid,
                    "reason": reason,
                    "question": question.question,
                    "choices": list(question.choices),
                }
            )
        elif len(applied) < sample_size:
            applied.append(
                {
                    "qid": question.qid,
                    "question": question.question,
                    "original_choices": list(question.choices),
                    "original_answer": LABELS[question.answer],
                    "variants": [
                        {
                            "variant_id": variant.variant_id,
                            "choices": list(variant.choices),
                            "answer": LABELS[variant.answer],
                        }
                        for variant in variants_for(question, max_variants)
                    ],
                }
            )

    return {
        "max_variants_per_question": max_variants,
        "questions": len(questions),
        "skipped_count": len(skipped),
        "augmented_count": len(questions) - len(skipped),
        "skipped": skipped,
        "applied_sample": applied,
    }


def split_manifest(
    train: Sequence[Question], validation: Sequence[Question], seed: int, dataset: dict
) -> dict:
    def balance(questions: Sequence[Question]) -> dict[str, int]:
        counts = dict.fromkeys(LABELS, 0)
        for question in questions:
            counts[LABELS[question.answer]] += 1
        return counts

    return {
        "dataset": dataset,
        "seed": seed,
        "adaptation_pool_size": len(train) + len(validation),
        "pilot_train_size": len(train),
        "pilot_validation_size": len(validation),
        "pilot_train_label_balance": balance(train),
        "pilot_validation_label_balance": balance(validation),
        "pilot_train_ids": [question.qid for question in train],
        "pilot_validation_ids": [question.qid for question in validation],
    }


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    missing = {"model", "dataset", "lora", "training", "objective"} - config.keys()
    if missing:
        raise ValueError(f"{path} is missing required sections: {sorted(missing)}")
    return config


def _run_directory(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = root / f"{stamp}_prepare-data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the adaptation split and augmentation audit"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_sft.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    dataset = config["dataset"]

    questions = load_adaptation_pool(
        dataset["id"], dataset["config"], dataset["adaptation_split"]
    )
    log.info("loaded %d records from the adaptation pool", len(questions))

    train, validation = split_adaptation_pool(
        questions,
        dataset["pilot_train_size"],
        dataset["pilot_validation_size"],
        dataset["seed"],
    )

    audit = augmentation_audit(train, dataset["max_variants_per_question"])
    manifest = split_manifest(train, validation, dataset["seed"], dataset)

    out = _run_directory(args.results_root)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "augmentation_audit.json").write_text(json.dumps(audit, indent=2))

    log.info("pilot train %d, pilot validation %d", len(train), len(validation))
    log.info(
        "augmentation: %d of %d questions skipped as unsafe",
        audit["skipped_count"],
        audit["questions"],
    )
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
