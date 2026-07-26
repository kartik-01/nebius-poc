"""Prompt rendering and the A/B/C/D candidate strings.

Training and evaluation both import from here so the completion boundary can never
drift between them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

LABELS = ("A", "B", "C", "D")

SYSTEM_MESSAGE = (
    "You are answering a multiple-choice question. Select the single best answer.\n"
    "Reply with only A, B, C, or D."
)


def render_user_message(question: str, choices: Sequence[str]) -> str:
    if len(choices) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} choices, got {len(choices)}")
    rendered = "\n".join(
        f"{label}. {text}" for label, text in zip(LABELS, choices, strict=True)
    )
    return f"Question:\n{question}\n\nChoices:\n{rendered}\n\nAnswer:"


def build_messages(question: str, choices: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": render_user_message(question, choices)},
    ]


def candidate_completion(answer_index: int) -> str:
    return LABELS[answer_index]


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
