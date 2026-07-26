import pytest

from nebius_poc.evaluate import accuracy, parse_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        (" B ", "B"),
        ("C. Specific performance", "C"),
        ("The answer is D.", "D"),
        ("**B**", "B"),
        ("C or D", "C"),
    ],
)
def test_parse_answer_takes_the_first_standalone_label(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize("text", ["", "Damages", "b", "BAD", "42", "I and II only"])
def test_parse_answer_rejects_everything_else(text, expected=None):
    # Uppercase only and word-bounded. A lenient parser would inflate the format
    # adherence figure, which is the one thing this metric is supposed to expose.
    assert parse_answer(text) is expected


def test_parse_answer_is_fooled_by_a_sentence_opening_with_a_label():
    # Known limitation, tolerable because generation is capped at a few new tokens.
    # It is recorded here so nobody "fixes" the strictness without noticing the trade.
    assert parse_answer("A contractor may recover damages.") == "A"


def test_accuracy_counts_only_correct_rows():
    rows = [{"correct": True}, {"correct": False}, {"correct": True}, {"correct": False}]
    assert accuracy(rows) == pytest.approx(0.5)


def test_accuracy_of_nothing_is_zero():
    assert accuracy([]) == 0.0
