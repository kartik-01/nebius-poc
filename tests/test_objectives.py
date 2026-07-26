from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from fakes import VOCAB_SIZE, FakeTokenizer, TinyCausalLM
from nebius_poc.objectives import (
    IGNORE_INDEX,
    build_scoring_batch,
    candidate_scores,
    completion_loss,
    encode_candidates,
    gold_completion_batch,
    ranking_loss,
    sequence_log_probs,
    validate_training_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CHOICES = ["Damages", "Rescission", "Specific performance", "Nothing"]


def test_batch_rows_are_question_major():
    batch = build_scoring_batch([([1], [[10], [11]]), ([2], [[20], [21]])], 0, 8)
    assert batch.input_ids.tolist() == [[1, 10], [1, 11], [2, 20], [2, 21]]
    assert batch.num_candidates == 2


def test_labels_mask_the_prompt():
    batch = build_scoring_batch([([1, 2, 3], [[9], [8]])], pad_token_id=0, max_length=16)
    assert (batch.labels[:, :3] == IGNORE_INDEX).all()
    assert batch.labels[0, 3].item() == 9
    assert batch.labels[1, 3].item() == 8


def test_multi_token_candidates_keep_every_token():
    batch = build_scoring_batch([([1, 2], [[7, 8, 9], [5, 6, 4]])], 0, 16)
    assert batch.labels[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9]
    assert batch.input_ids[1].tolist() == [1, 2, 5, 6, 4]


def test_ragged_prompts_pad_to_the_widest_row():
    batch = build_scoring_batch([([1], [[9], [8]]), ([1, 2, 3, 4], [[9], [8]])], 0, 16)
    assert batch.input_ids.shape == (4, 5)
    assert batch.attention_mask[0].tolist() == [1, 1, 0, 0, 0]
    assert batch.attention_mask[2].tolist() == [1, 1, 1, 1, 1]
    assert (batch.labels[0, 2:] == IGNORE_INDEX).all()


def test_long_prompt_is_trimmed_from_the_front():
    prompt = list(range(1, 21))
    batch = build_scoring_batch([(prompt, [[99]])], 0, max_length=5)
    assert batch.input_ids[0].tolist() == [17, 18, 19, 20, 99]
    assert batch.labels[0].tolist() == [IGNORE_INDEX] * 4 + [99]


def test_candidate_that_cannot_fit_is_rejected():
    with pytest.raises(ValueError, match="max_length"):
        build_scoring_batch([([1], [[1, 2, 3, 4]])], 0, max_length=4)


def test_uneven_candidate_counts_are_rejected():
    with pytest.raises(ValueError, match="same candidate count"):
        build_scoring_batch([([1], [[2]]), ([1], [[2], [3]])], 0, 8)


def test_sequence_log_probs_ignores_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, VOCAB_SIZE)
    labels = torch.full((2, 6), IGNORE_INDEX)
    labels[0, 3] = 5
    labels[1, 3] = 5
    labels[1, 4] = 6

    scored = sequence_log_probs(logits, labels)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    assert scored[0].item() == pytest.approx(log_probs[0, 2, 5].item(), rel=1e-5)
    expected = log_probs[1, 2, 5] + log_probs[1, 3, 6]
    assert scored[1].item() == pytest.approx(expected.item(), rel=1e-5)


def test_scores_reshape_question_major():
    torch.manual_seed(0)
    logits = torch.randn(8, 5, VOCAB_SIZE)
    labels = torch.full((8, 5), IGNORE_INDEX)
    labels[:, 4] = torch.arange(8) % VOCAB_SIZE

    flat = sequence_log_probs(logits, labels)
    scores = candidate_scores(logits, labels, num_candidates=4)

    assert scores.shape == (2, 4)
    for question in range(2):
        for candidate in range(4):
            assert scores[question, candidate].item() == flat[question * 4 + candidate].item()


def test_candidate_scores_reject_a_ragged_row_count():
    logits = torch.randn(3, 5, VOCAB_SIZE)
    labels = torch.full((3, 5), IGNORE_INDEX)
    labels[:, 4] = 1
    with pytest.raises(ValueError, match="do not divide"):
        candidate_scores(logits, labels, num_candidates=4)


def test_ranking_loss_targets_the_gold_candidate():
    scores = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
    gold = torch.tensor([1])
    expected = -F.log_softmax(scores, dim=-1)[0, 1]
    assert ranking_loss(scores, gold).item() == pytest.approx(expected.item(), rel=1e-6)


def test_ranking_loss_drops_when_the_gold_score_rises():
    gold = torch.tensor([2])
    weak = ranking_loss(torch.tensor([[1.0, 1.0, 1.0, 1.0]]), gold)
    strong = ranking_loss(torch.tensor([[1.0, 1.0, 4.0, 1.0]]), gold)
    assert strong.item() < weak.item()


def test_ranking_loss_rejects_a_gold_label_mismatch():
    with pytest.raises(ValueError, match="gold labels"):
        ranking_loss(torch.zeros(2, 4), torch.tensor([0]))


def test_completion_loss_only_counts_candidate_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, VOCAB_SIZE)
    labels = torch.full((1, 5), IGNORE_INDEX)
    labels[0, 4] = 7

    expected = F.cross_entropy(logits[0, 3].unsqueeze(0).float(), torch.tensor([7]))
    assert completion_loss(logits, labels).item() == pytest.approx(expected.item(), rel=1e-6)


def test_prompt_positions_receive_no_gradient():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, VOCAB_SIZE, requires_grad=True)
    labels = torch.full((1, 6), IGNORE_INDEX)
    labels[0, 4] = 11
    labels[0, 5] = 12

    completion_loss(logits, labels).backward()

    # Only positions 3 and 4 predict the two scored tokens.
    assert torch.count_nonzero(logits.grad[0, :3]) == 0
    assert torch.count_nonzero(logits.grad[0, 5]) == 0
    assert torch.count_nonzero(logits.grad[0, 3]) > 0
    assert torch.count_nonzero(logits.grad[0, 4]) > 0


def test_ranking_gradients_reach_the_model():
    model = TinyCausalLM()
    batch = build_scoring_batch([([1, 2, 3], [[10], [11], [12], [13]])], 0, 16)

    output = model(batch.input_ids, batch.attention_mask)
    scores = candidate_scores(output.logits, batch.labels, batch.num_candidates)
    ranking_loss(scores, torch.tensor([2])).backward()

    assert torch.count_nonzero(model.head.weight.grad) > 0
    assert torch.count_nonzero(model.embed.weight.grad) > 0


def test_ranking_step_reduces_the_loss():
    model = TinyCausalLM()
    batch = build_scoring_batch([([1, 2, 3], [[10], [11], [12], [13]])], 0, 16)
    gold = torch.tensor([2])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    def loss_now():
        output = model(batch.input_ids, batch.attention_mask)
        scores = candidate_scores(output.logits, batch.labels, batch.num_candidates)
        return ranking_loss(scores, gold)

    before = loss_now()
    optimizer.zero_grad()
    before.backward()
    optimizer.step()

    assert loss_now().item() < before.item()


def test_gold_completion_batch_keeps_only_the_correct_answer():
    examples = [([1], [[10], [11], [12], [13]]), ([2], [[20], [21], [22], [23]])]
    batch = gold_completion_batch(examples, [2, 0], pad_token_id=0, max_length=8)

    assert batch.num_candidates == 1
    assert batch.input_ids.tolist() == [[1, 12], [2, 20]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, 12], [IGNORE_INDEX, 20]]


def test_gold_completion_batch_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="gold labels"):
        gold_completion_batch([([1], [[2], [3]])], [0, 1], 0, 8)


def test_encode_candidates_shares_one_prompt_across_choices():
    tokenizer = FakeTokenizer()
    prompt_ids, candidates = encode_candidates(tokenizer, "What is the remedy?", CHOICES)

    assert len(candidates) == 4
    assert all(len(ids) == 1 for ids in candidates)
    assert len({tuple(ids) for ids in candidates}) == 4
    assert len(prompt_ids) > 0


def test_encode_candidates_ends_at_the_generation_boundary():
    tokenizer = FakeTokenizer()
    prompt_ids, _ = encode_candidates(tokenizer, "What is the remedy?", CHOICES)
    opener = tokenizer("<assistant>")["input_ids"]
    assert prompt_ids[-len(opener) :] == opener


def test_encode_candidates_handles_multi_token_labels():
    tokenizer = FakeTokenizer()
    _, candidates = encode_candidates(
        tokenizer, "What is the remedy?", CHOICES, labels=["AA", "BB", "CC", "DD"]
    )
    assert all(len(ids) == 2 for ids in candidates)


@pytest.mark.parametrize("name", ["train_sft.yaml", "train_ranking.yaml"])
def test_shipped_training_configs_validate(name):
    validate_training_config(yaml.safe_load((CONFIG_DIR / name).read_text()))


def config_with(**overrides):
    config = yaml.safe_load((CONFIG_DIR / "train_sft.yaml").read_text())
    for dotted, value in overrides.items():
        section, _, field = dotted.partition(".")
        if field:
            config[section][field] = value
        else:
            config[section] = value
    return config


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        validate_training_config(config_with(objective="reinforce"))


def test_missing_section_is_rejected():
    config = config_with()
    del config["lora"]
    with pytest.raises(ValueError, match="missing required sections"):
        validate_training_config(config)


def test_non_positive_learning_rate_is_rejected():
    with pytest.raises(ValueError, match="learning_rate"):
        validate_training_config(config_with(**{"training.learning_rate": 0}))


def test_zero_rank_is_rejected():
    with pytest.raises(ValueError, match="lora.rank"):
        validate_training_config(config_with(**{"lora.rank": 0}))


def test_out_of_range_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        validate_training_config(config_with(**{"lora.dropout": 1.0}))


def test_empty_target_modules_is_rejected():
    with pytest.raises(ValueError, match="target_modules"):
        validate_training_config(config_with(**{"lora.target_modules": []}))


def test_too_many_variants_is_rejected():
    with pytest.raises(ValueError, match="max_variants_per_question"):
        validate_training_config(config_with(**{"dataset.max_variants_per_question": 5}))


def test_training_on_the_test_split_is_rejected():
    with pytest.raises(ValueError, match="must differ"):
        validate_training_config(config_with(**{"dataset.adaptation_split": "test"}))
