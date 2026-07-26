import pytest

from nebius_poc.stats import align_by_question_id, compare, mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_matches_a_hand_computed_case():
    # b=1, c=4. Discordant n=5, tail = C(5,0) + C(5,1) = 6, so p = 2 * 6 / 32.
    assert mcnemar_exact(1, 4) == pytest.approx(0.375)


def test_mcnemar_is_one_without_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)


def test_mcnemar_never_exceeds_one():
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_survives_a_realistic_discordant_count():
    # Big binomial coefficients, so this checks the exact arithmetic does not
    # overflow at test-split scale.
    p = mcnemar_exact(120, 260)
    assert 0.0 < p < 0.001


def test_compare_counts_the_contingency_table():
    base = [True, True, False, False]
    tuned = [True, False, True, False]
    result = compare(base, tuned, resamples=200)

    assert result.n == 4
    assert result.both_correct == 1
    assert result.base_only_correct == 1
    assert result.tuned_only_correct == 1
    assert result.both_wrong == 1


def test_compare_reports_the_delta_in_percentage_points():
    base = [True, False, False, False]
    tuned = [True, True, True, False]
    result = compare(base, tuned, resamples=200)

    assert result.base_accuracy == pytest.approx(0.25)
    assert result.tuned_accuracy == pytest.approx(0.75)
    assert result.delta_pp == pytest.approx(50.0)


def test_bootstrap_is_reproducible():
    base = [True, False] * 40
    tuned = [True, True, True, False] * 20
    assert paired_bootstrap_ci(base, tuned, resamples=500, seed=42) == paired_bootstrap_ci(
        base, tuned, resamples=500, seed=42
    )


def test_bootstrap_is_reproducible_across_chunk_boundaries():
    base = [True, False] * 40
    tuned = [True, True, True, False] * 20
    assert paired_bootstrap_ci(base, tuned, resamples=2500, seed=42) == paired_bootstrap_ci(
        base, tuned, resamples=2500, seed=42
    )


def test_bootstrap_responds_to_the_seed():
    base = [True, False] * 40
    tuned = [True, True, True, False] * 20
    assert paired_bootstrap_ci(base, tuned, resamples=500, seed=42) != paired_bootstrap_ci(
        base, tuned, resamples=500, seed=7
    )


def test_bootstrap_interval_contains_the_observed_delta():
    base = [True, False] * 50
    tuned = [True, True, True, False] * 25
    low, high = paired_bootstrap_ci(base, tuned, resamples=2000)
    observed = (sum(tuned) - sum(base)) / len(base) * 100
    assert low <= observed <= high


def test_bootstrap_interval_is_degenerate_when_nothing_changes():
    outcomes = [True, False, True, True]
    assert paired_bootstrap_ci(outcomes, outcomes, resamples=200) == (0.0, 0.0)


def test_compare_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compare([True, False], [True], resamples=100)


def test_compare_rejects_empty_input():
    with pytest.raises(ValueError, match="nothing to compare"):
        compare([], [], resamples=100)


def test_align_orders_both_models_the_same_way():
    base, tuned = align_by_question_id({"b": True, "a": False}, {"a": True, "b": False})
    assert base == [False, True]
    assert tuned == [True, False]


def test_align_rejects_mismatched_question_sets():
    with pytest.raises(ValueError, match="question sets differ"):
        align_by_question_id({"a": True}, {"b": True})
