"""Paired comparison between the base and tuned models.

Both models answer the same test questions, so every statistic here is paired.
Comparing two independent accuracy figures would throw away that structure and
widen the interval for no reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

# Bootstrap draws are chunked to keep the index matrix off the heap in one piece.
# The seed and this constant together fix the result, so do not tune it casually.
_BOOTSTRAP_CHUNK = 1000


@dataclass(frozen=True)
class PairedComparison:
    n: int
    base_accuracy: float
    tuned_accuracy: float
    delta_pp: float
    ci_low_pp: float
    ci_high_pp: float
    confidence: float
    resamples: int
    seed: int
    both_correct: int
    both_wrong: int
    base_only_correct: int
    tuned_only_correct: int
    mcnemar_p: float

    def to_dict(self) -> dict:
        return asdict(self)


def align_by_question_id(
    base: Mapping[str, bool], tuned: Mapping[str, bool]
) -> tuple[list[bool], list[bool]]:
    if base.keys() != tuned.keys():
        only_base = sorted(base.keys() - tuned.keys())
        only_tuned = sorted(tuned.keys() - base.keys())
        raise ValueError(
            f"question sets differ: {len(only_base)} only in base, "
            f"{len(only_tuned)} only in tuned"
        )

    order = sorted(base)
    return [bool(base[qid]) for qid in order], [bool(tuned[qid]) for qid in order]


def _as_array(values: Sequence[bool]) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError("expected a flat sequence of per-question outcomes")
    if array.size == 0:
        raise ValueError("nothing to compare")
    return array


def paired_bootstrap_ci(
    base: Sequence[bool],
    tuned: Sequence[bool],
    resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    left = _as_array(base)
    right = _as_array(tuned)
    if left.size != right.size:
        raise ValueError(f"length mismatch: {left.size} against {right.size}")

    # Resample the per-question difference, which is what keeps the pairing intact.
    difference = right.astype(np.int8) - left.astype(np.int8)
    n = difference.size

    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    drawn = 0
    while drawn < resamples:
        size = min(_BOOTSTRAP_CHUNK, resamples - drawn)
        index = rng.integers(0, n, size=(size, n))
        means[drawn : drawn + size] = difference[index].mean(axis=1)
        drawn += size

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low) * 100.0, float(high) * 100.0


def mcnemar_exact(base_only_correct: int, tuned_only_correct: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs alone.

    Under the null the discordant pairs split like a fair coin, so this is a
    binomial tail doubled. The exact form is used because the discordant count can
    be small enough that the chi-square approximation misbehaves.
    """
    discordant = base_only_correct + tuned_only_correct
    if discordant == 0:
        return 1.0

    smaller = min(base_only_correct, tuned_only_correct)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def compare(
    base: Sequence[bool],
    tuned: Sequence[bool],
    resamples: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> PairedComparison:
    left = _as_array(base)
    right = _as_array(tuned)
    if left.size != right.size:
        raise ValueError(f"length mismatch: {left.size} against {right.size}")

    base_only = int(np.count_nonzero(left & ~right))
    tuned_only = int(np.count_nonzero(~left & right))
    low, high = paired_bootstrap_ci(left, right, resamples, seed, confidence)

    base_accuracy = float(left.mean())
    tuned_accuracy = float(right.mean())

    return PairedComparison(
        n=int(left.size),
        base_accuracy=base_accuracy,
        tuned_accuracy=tuned_accuracy,
        delta_pp=(tuned_accuracy - base_accuracy) * 100.0,
        ci_low_pp=low,
        ci_high_pp=high,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        both_correct=int(np.count_nonzero(left & right)),
        both_wrong=int(np.count_nonzero(~left & ~right)),
        base_only_correct=base_only,
        tuned_only_correct=tuned_only,
        mcnemar_p=mcnemar_exact(base_only, tuned_only),
    )
