import pytest

from helpers import question_from, record
from nebius_poc.data import (
    augmentation_audit,
    build_question,
    expand,
    permutation_skip_reason,
    split_adaptation_pool,
    split_manifest,
    stable_id,
    variants_for,
)
from nebius_poc.prompts import LABELS


def test_stable_id_is_deterministic():
    fields = record("Is the contract enforceable?", ["Yes", "No", "Only in part", "Unclear"], 1)
    assert build_question(fields).qid == build_question(dict(fields)).qid


def test_stable_id_ignores_whitespace_formatting():
    original = question_from("Is the  contract enforceable?", ["Yes", "No", "Maybe", "Never"], 1)
    respaced = question_from(
        "  Is the contract   enforceable?  ", [" Yes", "No ", "Maybe", "Never"], 1
    )
    assert original.qid == respaced.qid


def test_stable_id_tracks_the_answer_index():
    choices = ["Yes", "No", "Maybe", "Never"]
    assert question_from("Q?", choices, 1).qid != question_from("Q?", choices, 2).qid


def test_stable_id_is_not_confused_by_field_boundaries():
    # Naive concatenation would give these the same digest.
    left = stable_id("law", "ab", ["c", "d", "e", "f"], 0)
    right = stable_id("law", "a", ["bc", "d", "e", "f"], 0)
    assert left != right


def test_build_question_rejects_a_bad_answer_index():
    with pytest.raises(ValueError, match="outside"):
        build_question(record("Q?", ["a", "b", "c", "d"], 4))


def test_build_question_rejects_the_wrong_number_of_choices():
    with pytest.raises(ValueError, match="choices"):
        build_question(record("Q?", ["a", "b", "c"], 0))


def test_split_is_deterministic(adaptation_pool):
    first = split_adaptation_pool(adaptation_pool, 20, 42)
    second = split_adaptation_pool(adaptation_pool, 20, 42)
    assert [q.qid for q in first[0]] == [q.qid for q in second[0]]
    assert [q.qid for q in first[1]] == [q.qid for q in second[1]]


def test_split_does_not_depend_on_input_order(adaptation_pool):
    forward = split_adaptation_pool(adaptation_pool, 20, 42)
    backward = split_adaptation_pool(list(reversed(adaptation_pool)), 20, 42)
    assert [q.qid for q in forward[1]] == [q.qid for q in backward[1]]


def test_split_sizes_and_no_overlap(adaptation_pool):
    train, validation = split_adaptation_pool(adaptation_pool, 20, 42)
    assert len(train) == 150
    assert len(validation) == 20
    assert {q.qid for q in train}.isdisjoint({q.qid for q in validation})
    assert {q.qid for q in train + validation} == {q.qid for q in adaptation_pool}


def test_split_keeps_the_label_balance(adaptation_pool):
    _, validation = split_adaptation_pool(adaptation_pool, 20, 42)
    counts = [sum(1 for q in validation if q.answer == label) for label in range(4)]
    assert counts == [5, 5, 5, 5]


def test_split_refuses_an_internal_set_that_swallows_the_pool(adaptation_pool):
    with pytest.raises(ValueError, match="leaves no training data"):
        split_adaptation_pool(adaptation_pool[:20], 20, 42)


def test_split_refuses_an_already_augmented_pool(adaptation_pool):
    # Augmentation has to happen after the split, or the same question leaks into
    # both halves wearing different permutations.
    with pytest.raises(ValueError):
        split_adaptation_pool(expand(adaptation_pool, 4), 20, 42)


def test_augmentation_never_sees_the_held_out_questions(adaptation_pool):
    train, validation = split_adaptation_pool(adaptation_pool, 20, 42)
    sources = {variant.source_qid for variant in expand(train, 4)}
    assert sources.isdisjoint({q.qid for q in validation})


@pytest.mark.parametrize(
    "choice",
    [
        "All of the above",
        "None of the above",
        "Both A and B",
        "A and C only",
        "Choices A and D",
        "Option B is correct",
        "The answer C is correct",
        "(A) or (C)",
        "a and b",
    ],
)
def test_choices_that_name_other_options_are_not_permuted(choice):
    question = question_from("Which is correct?", ["First", "Second", "Third", choice], 0)
    assert permutation_skip_reason(question) is not None


def test_stem_that_names_options_is_not_permuted():
    question = question_from(
        "Is option B a valid defense?", ["Yes", "No", "Only partly", "Never"], 0
    )
    assert permutation_skip_reason(question) is not None


def test_roman_numeral_choices_are_still_permuted():
    # These refer to statements in the stem, not to answer positions, so rotating
    # the options leaves them meaning the same thing.
    question = question_from(
        "Which of the numbered statements are correct?",
        ["I only", "I and II only", "II and III only", "I, II, and III"],
        answer=1,
    )
    assert permutation_skip_reason(question) is None


def test_party_letters_in_the_stem_do_not_block_permutation():
    question = question_from(
        "A sues B for breach after C repudiates. Who prevails?",
        ["A prevails", "B prevails", "C prevails", "Nobody prevails"],
        answer=0,
    )
    assert permutation_skip_reason(question) is None


def test_four_variants_put_the_answer_in_every_position(safe_question):
    variants = list(variants_for(safe_question, 4))
    assert len(variants) == 4
    assert sorted(variant.answer for variant in variants) == [0, 1, 2, 3]
    assert len({variant.variant_id for variant in variants}) == 4


def test_permutation_preserves_the_choices_and_the_correct_text(safe_question):
    correct = safe_question.choices[safe_question.answer]
    for variant in variants_for(safe_question, 4):
        assert sorted(variant.choices) == sorted(safe_question.choices)
        assert variant.choices[variant.answer] == correct
        assert variant.original_answer == safe_question.answer


def test_permutation_mapping_reconstructs_the_variant(safe_question):
    for variant in variants_for(safe_question, 4):
        rebuilt = tuple(safe_question.choices[origin] for origin in variant.permutation)
        assert rebuilt == variant.choices


def test_original_variant_is_not_marked_as_augmented(safe_question):
    variants = list(variants_for(safe_question, 4))
    assert variants[0].choices == safe_question.choices
    assert variants[0].augmentation_applied is False
    assert all(variant.augmentation_applied for variant in variants[1:])


def test_unsafe_question_yields_only_the_original_with_a_reason():
    question = question_from(
        "Which is correct?", ["First", "Second", "Third", "All of the above"], 0
    )
    variants = list(variants_for(question, 4))
    assert len(variants) == 1
    assert variants[0].augmentation_applied is False
    assert variants[0].augmentation_skip_reason is not None
    assert variants[0].choices == question.choices


def test_max_variants_of_one_disables_augmentation(safe_question):
    variants = list(variants_for(safe_question, 1))
    assert len(variants) == 1
    assert variants[0].augmentation_applied is False


def test_audit_reports_skipped_records_and_a_sample(safe_question):
    unsafe = question_from(
        "Which is correct?", ["First", "Second", "Third", "None of the above"], 0
    )
    audit = augmentation_audit([safe_question, unsafe], max_variants=4)

    assert audit["questions"] == 2
    assert audit["skipped_count"] == 1
    assert audit["augmented_count"] == 1
    assert audit["skipped"][0]["qid"] == unsafe.qid
    assert audit["skipped"][0]["reason"]

    assert len(audit["applied_sample"]) == 1
    sample = audit["applied_sample"][0]
    assert sample["qid"] == safe_question.qid
    assert len(sample["variants"]) == 4
    assert sorted(variant["answer"] for variant in sample["variants"]) == list(LABELS)


def test_audit_sample_is_capped(adaptation_pool):
    audit = augmentation_audit(adaptation_pool, max_variants=4, sample_size=3)
    assert len(audit["applied_sample"]) == 3


def test_split_manifest_records_ids_and_balance(adaptation_pool):
    train, validation = split_adaptation_pool(adaptation_pool, 20, 42)
    manifest = split_manifest(train, validation, 42, {"id": "cais/mmlu", "config": "x"})

    assert manifest["pilot_train_size"] == len(adaptation_pool) - 20
    assert manifest["pilot_validation_size"] == 20
    assert len(manifest["pilot_train_ids"]) == len(adaptation_pool) - 20
    assert len(set(manifest["pilot_train_ids"]) & set(manifest["pilot_validation_ids"])) == 0
    assert sum(manifest["pilot_validation_label_balance"].values()) == 20
