import pytest

from nebius_poc.prompts import (
    LABELS,
    SYSTEM_MESSAGE,
    build_messages,
    candidate_completion,
    profile_sequence_lengths,
    render_user_message,
)

CHOICES = ["Damages", "Rescission", "Specific performance", "Nothing"]


def test_user_message_labels_every_choice():
    text = render_user_message("What is the remedy?", CHOICES)
    for label, choice in zip(LABELS, CHOICES, strict=True):
        assert f"{label}. {choice}" in text


def test_user_message_ends_at_the_completion_boundary():
    # Training and evaluation both append the answer letter right here, so the
    # trailing text has to stay byte-identical between them.
    assert render_user_message("What is the remedy?", CHOICES).endswith("\n\nAnswer:")


def test_user_message_rejects_the_wrong_choice_count():
    with pytest.raises(ValueError, match="choices"):
        render_user_message("What is the remedy?", CHOICES[:3])


def test_messages_carry_the_system_instruction():
    messages = build_messages("What is the remedy?", CHOICES)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_MESSAGE


def test_candidate_completion_maps_index_to_letter():
    assert [candidate_completion(index) for index in range(4)] == list(LABELS)


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
