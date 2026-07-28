import pytest

from nebius_poc.prompts import (
    CANDIDATE_STRINGS,
    LABELS,
    candidate_completion,
    format_subject,
    profile_sequence_lengths,
    render_prompt,
)

CHOICES = ["Damages", "Rescission", "Specific performance", "Nothing"]


def test_prompt_labels_every_choice():
    text = render_prompt("professional_law", "What is the remedy?", CHOICES)
    for label, choice in zip(LABELS, CHOICES, strict=True):
        assert f"{label}. {choice}" in text


def test_prompt_ends_at_the_completion_boundary():
    # Training and evaluation both append the candidate right here, so the trailing
    # text has to stay byte-identical between them. No trailing space: the candidate
    # carries it.
    text = render_prompt("professional_law", "What is the remedy?", CHOICES)
    assert text.endswith("\nAnswer:")


def test_prompt_names_the_subject_readably():
    text = render_prompt("professional_law", "What is the remedy?", CHOICES)
    assert "professional law" in text
    assert "professional_law" not in text


def test_format_subject_normalises_underscores_and_spacing():
    assert format_subject("high_school_psychology") == "high school psychology"
    assert format_subject("  professional__law ") == "professional law"


def test_prompt_rejects_the_wrong_choice_count():
    with pytest.raises(ValueError, match="choices"):
        render_prompt("professional_law", "What is the remedy?", CHOICES[:3])


def test_candidates_carry_a_leading_space():
    # The prompt stops at "Answer:", so the space belongs to the continuation.
    # Tokenizers score " A" and "A" differently and both paths must agree.
    assert CANDIDATE_STRINGS == (" A", " B", " C", " D")
    assert [candidate_completion(index) for index in range(4)] == list(CANDIDATE_STRINGS)


def test_sequence_length_profile_uses_nearest_rank():
    texts = [" ".join(["token"] * count) for count in range(1, 101)]
    profile = profile_sequence_lengths(texts, str.split, limits=(50, 99))

    assert profile["count"] == 100
    assert profile["min"] == 1
    assert profile["median"] == 50
    assert profile["p95"] == 95
    assert profile["p99"] == 99
    assert profile["max"] == 100


def test_sequence_length_profile_counts_truncation_per_limit():
    texts = [" ".join(["token"] * count) for count in range(1, 101)]
    profile = profile_sequence_lengths(texts, str.split, limits=(50, 99))

    assert profile["truncated_at"]["50"] == 50
    assert profile["truncated_at"]["99"] == 1


def test_sequence_length_profile_needs_input():
    with pytest.raises(ValueError):
        profile_sequence_lengths([], str.split)
