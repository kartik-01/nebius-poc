"""Prompt rendering and the A/B/C/D candidate strings.

Training and evaluation both import from here so the completion boundary can never
drift between them.

The prompt is the standard MMLU zero-shot completion format: a one-line subject
header, the question, the four labelled options, and a bare `Answer:` line that the
model continues. The base checkpoint has no chat template, and a plain completion
prompt is also what the original MMLU evaluation and lm-evaluation-harness use, so
the numbers here stay comparable to published figures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

LABELS = ("A", "B", "C", "D")


def format_subject(subject: str) -> str:
    return " ".join(subject.replace("_", " ").split())


def render_prompt(subject: str, question: str, choices: Sequence[str]) -> str:
    if len(choices) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} choices, got {len(choices)}")
    options = "\n".join(
        f"{label}. {text}" for label, text in zip(LABELS, choices, strict=True)
    )
    return (
        f"The following is a multiple choice question about {format_subject(subject)}. "
        "Answer with a single letter.\n\n"
        f"{question}\n{options}\nAnswer:"
    )


def candidate_completion(answer_index: int) -> str:
    return CANDIDATE_STRINGS[answer_index]


# The prompt ends at "Answer:" with no trailing space, so the continuation carries
# the space. Tokenizers treat " A" and "A" as different tokens, and training and
# evaluation must agree on which one they score.
CANDIDATE_STRINGS = tuple(f" {label}" for label in LABELS)


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    # Nearest-rank rather than interpolated. Different libraries interpolate
    # differently and this number goes straight into the max_length decision.
    rank = math.ceil(fraction * len(sorted_values))
    return sorted_values[max(rank - 1, 0)]


def profile_sequence_lengths(
    texts: Sequence[str],
    tokenize: Callable[[str], Sequence[int]],
    limits: Sequence[int] = (1024, 2048),
) -> dict:
    if not texts:
        raise ValueError("nothing to profile")

    lengths = sorted(len(tokenize(text)) for text in texts)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": _percentile(lengths, 0.50),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": lengths[-1],
        "truncated_at": {
            str(limit): sum(1 for length in lengths if length > limit) for limit in limits
        },
    }
