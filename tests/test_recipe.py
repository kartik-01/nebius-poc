import json

import pytest

from helpers import question_from
from nebius_poc.evaluate import (
    filter_questions_by_ids,
    gold_choice_nll,
    load_id_list,
    mean_gold_choice_nll,
)
from nebius_poc.profile import recommend_max_length
from nebius_poc.recipe import (
    apply_recipe_lock,
    build_recipe_lock,
    load_recipe_lock,
    score_pilot,
    select_objective,
)


def _row(qid, gold, scores, correct):
    labels = "ABCD"
    row = {
        "question_id": qid,
        "gold_answer": gold,
        "correct": correct,
    }
    for label, score in zip(labels, scores, strict=True):
        row[f"score_{label.lower()}"] = score
    return row


def test_gold_choice_nll_negates_the_gold_score():
    row = _row("q0", "B", scores=(-1.0, -0.2, -3.0, -4.0), correct=True)
    assert gold_choice_nll(row) == pytest.approx(0.2)


def test_mean_gold_choice_nll_averages_rows():
    rows = [
        _row("q0", "A", (-0.5, -2.0, -2.0, -2.0), True),
        _row("q1", "C", (-2.0, -2.0, -1.5, -2.0), True),
    ]
    assert mean_gold_choice_nll(rows) == pytest.approx(1.0)


CANDIDATES = [
    {"label": "sft", "n": 100, "mean_gold_choice_nll": 1.2, "forced_choice_accuracy": 0.55},
    {"label": "ranking", "n": 100, "mean_gold_choice_nll": 1.24, "forced_choice_accuracy": 0.57},
]


def test_select_objective_defaults_to_the_reported_metric():
    # Accuracy is what the final comparison reports, so it leads once the internal
    # set is large enough to resolve single questions.
    selection = select_objective(CANDIDATES)
    assert selection["winner_label"] == "ranking"
    assert selection["primary_metric"] == "forced_choice_accuracy"


def test_select_objective_can_lead_on_gold_nll():
    # On a 20-question set accuracy moves in five-point steps, so the continuous
    # metric has to lead instead.
    selection = select_objective(CANDIDATES, "mean_gold_choice_nll")
    assert selection["winner_label"] == "sft"
    assert selection["primary_metric"] == "mean_gold_choice_nll"


def test_select_objective_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="primary metric"):
        select_objective(CANDIDATES, "loss")


def test_select_objective_uses_the_other_metric_as_tiebreak():
    selection = select_objective(
        [
            {"label": "sft", "n": 20, "mean_gold_choice_nll": 1.0, "forced_choice_accuracy": 0.4},
            {
                "label": "ranking",
                "n": 20,
                "mean_gold_choice_nll": 1.0,
                "forced_choice_accuracy": 0.7,
            },
        ]
    )
    assert selection["winner_label"] == "ranking"


def test_select_objective_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="same number"):
        select_objective(
            [
                {"label": "a", "n": 20, "mean_gold_choice_nll": 1.0, "forced_choice_accuracy": 0.5},
                {"label": "b", "n": 19, "mean_gold_choice_nll": 0.9, "forced_choice_accuracy": 0.5},
            ]
        )


def _training_config(objective="candidate_ranking"):
    return {
        "objective": objective,
        "model": {"id": "Qwen/Qwen2.5-7B", "revision": "abc", "dtype": "bfloat16"},
        "dataset": {
            "id": "cais/mmlu",
            "config": "professional_law",
            "adaptation_split": "validation",
            "final_test_split": "test",
            "pilot_train_size": 150,
            "pilot_validation_size": 20,
            "seed": 42,
            "max_variants_per_question": 4,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "training": {
            "learning_rate": 1e-4,
            "epochs": 5,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_length": 1024,
            "seed": 42,
            "checkpoint_retention": 2,
        },
    }


def test_build_recipe_lock_records_the_winning_objective():
    config = _training_config()
    selection = select_objective(
        [
            score_pilot(
                [_row("q0", "A", (-0.1, -2, -2, -2), True)] * 4,
                "ranking",
            )
        ]
    )
    lock = build_recipe_lock(config, selection, notes=["pilot only"])
    assert lock["objective"] == "candidate_ranking"
    assert lock["lora_rank"] == 16
    assert lock["notes"] == ["pilot only"]
    assert lock["selection"]["winner_label"] == "ranking"


def test_apply_recipe_lock_overlays_hyperparameters():
    config = _training_config("completion_sft")
    lock = build_recipe_lock(
        _training_config("candidate_ranking"),
        select_objective(
            [score_pilot([_row("q0", "A", (-0.1, -2, -2, -2), True)] * 4, "ranking")]
        ),
        learning_rate=2e-4,
        lora_rank=8,
        epochs=3,
        max_length=2048,
    )
    updated = apply_recipe_lock(config, lock)
    assert updated["objective"] == "candidate_ranking"
    assert updated["training"]["learning_rate"] == pytest.approx(2e-4)
    assert updated["lora"]["rank"] == 8
    assert updated["training"]["epochs"] == 3
    assert updated["training"]["max_length"] == 2048
    # Dataset identity is checked, not rewritten.
    assert updated["dataset"] == config["dataset"]


def test_apply_recipe_lock_rejects_a_mismatched_dataset():
    config = _training_config()
    lock = build_recipe_lock(
        config,
        select_objective(
            [score_pilot([_row("q0", "A", (-0.1, -2, -2, -2), True)] * 4, "ranking")]
        ),
    )
    lock["dataset"] = {**lock["dataset"], "seed": 99}
    with pytest.raises(ValueError, match="dataset.seed"):
        apply_recipe_lock(config, lock)


def test_load_recipe_lock_requires_core_fields(tmp_path):
    path = tmp_path / "recipe_lock.json"
    path.write_text(json.dumps({"objective": "completion_sft"}))
    with pytest.raises(ValueError, match="missing lock fields"):
        load_recipe_lock(path)


def test_recommend_max_length_picks_smallest_covering_99_percent():
    profile = {
        "count": 100,
        "truncated_at": {"1024": 0, "2048": 0},
    }
    assert recommend_max_length(profile) == 1024

    profile = {"count": 100, "truncated_at": {"1024": 3, "2048": 0}}
    assert recommend_max_length(profile) == 2048


def test_filter_questions_by_ids_preserves_caller_order():
    questions = [
        question_from(f"q{i}", ["a", "b", "c", "d"], answer=i % 4) for i in range(5)
    ]
    ids = [questions[3].qid, questions[1].qid]
    filtered = filter_questions_by_ids(questions, ids)
    assert [item.qid for item in filtered] == ids


def test_load_id_list_reads_split_manifest(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps({"pilot_validation_ids": ["a", "b", "c"]}))
    assert load_id_list(path) == ["a", "b", "c"]


def test_load_id_list_reads_newline_file(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("x\ny\n")
    assert load_id_list(path) == ["x", "y"]
