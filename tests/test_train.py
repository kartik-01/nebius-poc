import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fakes import FakeTokenizer, TinyCausalLM
from helpers import question_from
from nebius_poc.data import expand, load_adaptation_pool
from nebius_poc.objectives import IGNORE_INDEX
from nebius_poc.prompts import CANDIDATE_STRINGS, LABELS
from nebius_poc.train import (
    build_dataset,
    collate,
    cosine_schedule_with_warmup,
    encode_variants,
    load_checkpoint,
    main,
    objective_loss,
    prune_checkpoints,
    resolve_dtype,
    save_checkpoint,
)


class SaveableModel(TinyCausalLM):
    """Stands in for the PEFT wrapper, whose adapter I/O is all checkpointing touches."""

    loaded_from = None

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"stub")

    def load_adapter(self, directory, adapter_name="default", is_trainable=False):
        self.loaded_from = (Path(directory), adapter_name, is_trainable)


@pytest.fixture
def encoded(safe_question):
    return encode_variants(expand([safe_question], 4), FakeTokenizer())


def test_encode_variants_keeps_the_variant_identity(safe_question, encoded):
    assert len(encoded) == 4
    assert {example.qid for example in encoded} == {safe_question.qid}
    assert len({example.variant_id for example in encoded}) == 4
    # The safe permutation walks the answer through every label position.
    assert sorted(example.gold for example in encoded) == [0, 1, 2, 3]


def test_encode_variants_produces_one_candidate_per_label(encoded):
    for example in encoded:
        assert len(example.candidate_ids) == len(LABELS)
        assert all(ids for ids in example.candidate_ids)


def test_ranking_collate_lays_out_every_candidate(encoded):
    batch, gold = collate(encoded, "candidate_ranking", pad_token_id=0, max_length=512)

    assert batch.num_candidates == len(LABELS)
    assert batch.input_ids.shape[0] == len(encoded) * len(LABELS)
    assert gold.tolist() == [example.gold for example in encoded]


def test_sft_collate_keeps_only_the_gold_continuation(encoded):
    batch, gold = collate(encoded, "completion_sft", pad_token_id=0, max_length=512)

    assert batch.num_candidates == 1
    assert batch.input_ids.shape[0] == len(encoded)

    tokenizer = FakeTokenizer()
    for row, example in enumerate(batch.labels.tolist()):
        scored = [token for token in example if token != IGNORE_INDEX]
        expected = tokenizer(CANDIDATE_STRINGS[encoded[row].gold])["input_ids"]
        assert scored == expected

    assert gold.tolist() == [example.gold for example in encoded]


def test_collate_rejects_an_unknown_objective(encoded):
    with pytest.raises(ValueError, match="unknown objective"):
        collate(encoded, "dpo", pad_token_id=0, max_length=512)


@pytest.mark.parametrize("objective", ["completion_sft", "candidate_ranking"])
def test_both_objectives_backpropagate_into_the_model(encoded, objective):
    model = TinyCausalLM()
    batch, gold = collate(encoded, objective, pad_token_id=0, max_length=512)

    output = model(batch.input_ids, batch.attention_mask)
    loss = objective_loss(output.logits, batch, gold, objective)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.count_nonzero(model.head.weight.grad) > 0


def test_ranking_loss_falls_when_the_gold_choice_wins(encoded):
    model = TinyCausalLM()
    batch, gold = collate(encoded[:1], "candidate_ranking", pad_token_id=0, max_length=512)
    logits = model(batch.input_ids, batch.attention_mask).logits.detach()

    before = objective_loss(logits, batch, gold, "candidate_ranking")

    # Reward exactly the tokens the gold candidate is scored on.
    boosted = logits.clone()
    gold_row = int(gold[0])
    scored = batch.labels[gold_row, 1:] != IGNORE_INDEX
    positions = torch.nonzero(scored).flatten()
    boosted[gold_row, positions, batch.labels[gold_row, positions + 1]] += 10.0

    assert objective_loss(boosted, batch, gold, "candidate_ranking") < before


def test_prune_checkpoints_orders_by_step_not_by_name(tmp_path):
    for step in (2, 9, 10, 11):
        (tmp_path / f"checkpoint-{step}").mkdir()

    prune_checkpoints(tmp_path, keep=2)

    remaining = sorted(path.name for path in tmp_path.glob("checkpoint-*"))
    assert remaining == ["checkpoint-10", "checkpoint-11"]


def schedule(total_steps=20, warmup_ratio=0.05, lr=1e-4):
    optimizer = torch.optim.AdamW(TinyCausalLM().parameters(), lr=lr)
    return optimizer, cosine_schedule_with_warmup(optimizer, total_steps, warmup_ratio)


def test_schedule_warms_up_then_decays_to_zero():
    total_steps = 20
    optimizer, scheduler = schedule(total_steps=total_steps, warmup_ratio=0.25)

    rates = []
    for _ in range(total_steps + 1):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()  # no grads to apply, just keeps the real call order
        scheduler.step()

    peak = max(rates)
    assert rates.index(peak) == 4  # warmup spans 5 steps and peaks on the last of them
    assert rates[:5] == sorted(rates[:5])
    assert rates[4:] == sorted(rates[4:], reverse=True)
    assert peak == pytest.approx(1e-4)
    assert rates[total_steps] == pytest.approx(0.0)


def test_schedule_never_starts_at_zero():
    # A degenerate warmup would waste the first optimizer steps entirely, which is easy
    # to miss on a short pilot run.
    _, scheduler = schedule(total_steps=2, warmup_ratio=0.05)
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4)


def test_save_checkpoint_writes_the_adapter_and_the_resume_state(tmp_path):
    model = SaveableModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = cosine_schedule_with_warmup(optimizer, total_steps=10, warmup_ratio=0.1)

    save_checkpoint(model, optimizer, scheduler, 7, tmp_path / "checkpoint-7")

    assert (tmp_path / "checkpoint-7" / "adapter" / "adapter_model.safetensors").exists()
    state = torch.load(tmp_path / "checkpoint-7" / "training_state.pt", weights_only=False)
    assert state["global_step"] == 7
    assert "state" in state["optimizer"]
    assert state["scheduler"]["last_epoch"] == 0


def test_resume_restores_the_step_count_and_the_adapter(tmp_path):
    model = SaveableModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = cosine_schedule_with_warmup(optimizer, total_steps=10, warmup_ratio=0.1)

    optimizer.step()
    scheduler.step()
    save_checkpoint(model, optimizer, scheduler, 3, tmp_path / "checkpoint-3")

    fresh = SaveableModel()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
    fresh_scheduler = cosine_schedule_with_warmup(
        fresh_optimizer, total_steps=10, warmup_ratio=0.1
    )

    step = load_checkpoint(
        fresh, fresh_optimizer, fresh_scheduler, tmp_path / "checkpoint-3", torch.device("cpu")
    )

    assert step == 3
    assert fresh_scheduler.last_epoch == scheduler.last_epoch
    assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()
    # The adapter has to come back trainable, otherwise the resumed run silently
    # optimizes nothing.
    assert fresh.loaded_from == (tmp_path / "checkpoint-3" / "adapter", "default", True)


def test_cpu_runs_fall_back_to_fp32():
    assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.float32
    assert resolve_dtype("bfloat16", torch.device("cuda", 0)) is torch.bfloat16


def test_training_cannot_reach_the_official_test_split():
    with pytest.raises(ValueError, match="adaptation pool must come from"):
        load_adaptation_pool("cais/mmlu", "professional_law", "test")


def test_importing_the_clis_pulls_in_no_model_stack():
    # transformers and peft are imported inside the functions that need them. Keeping
    # it that way is what stops `--help` or a unit test from touching the Hub.
    probe = (
        "import sys, nebius_poc.train, nebius_poc.evaluate, nebius_poc.merge_adapter;"
        "leaked = {'transformers', 'peft'} & sys.modules.keys();"
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_question_fixture_stays_permutable():
    # Guards the fixture itself: if it ever trips the unsafe-choice detector the
    # variant assertions above would silently start testing a single row.
    question = question_from("Which remedy applies?", ["One", "Two", "Three", "Four"], answer=2)
    assert len(expand([question], 4)) == 4


def _dataset_config():
    return {
        "dataset": {
            "id": "cais/mmlu",
            "config": "professional_law",
            "adaptation_split": "validation",
            "final_test_split": "test",
            "pilot_train_size": 150,
            "pilot_validation_size": 20,
            "seed": 42,
            "max_variants_per_question": 4,
            "revision": None,
        }
    }


def test_pilot_build_dataset_uses_the_150_split(adaptation_pool, monkeypatch):
    monkeypatch.setattr(
        "nebius_poc.train.load_adaptation_pool",
        lambda *args, **kwargs: adaptation_pool,
    )
    examples, manifest = build_dataset(_dataset_config(), FakeTokenizer(), limit=None, final=False)

    assert manifest["training_mode"] == "pilot"
    assert manifest["trained_question_count"] == 150
    assert set(manifest["trained_question_ids"]) == set(manifest["pilot_train_ids"])
    assert len(examples) == manifest["training_rows_after_augmentation"]


def test_final_build_dataset_trains_the_full_pool(adaptation_pool, monkeypatch):
    monkeypatch.setattr(
        "nebius_poc.train.load_adaptation_pool",
        lambda *args, **kwargs: adaptation_pool,
    )
    examples, manifest = build_dataset(_dataset_config(), FakeTokenizer(), limit=None, final=True)

    assert manifest["training_mode"] == "final"
    assert manifest["trained_question_count"] == 170
    assert set(manifest["trained_question_ids"]) == {question.qid for question in adaptation_pool}
    # Pilot IDs stay in the manifest for audit even though nothing is held out of training.
    assert len(manifest["pilot_train_ids"]) == 150
    assert len(manifest["pilot_validation_ids"]) == 20
    assert sorted(manifest["trained_question_ids"]) == manifest["trained_question_ids"]
    assert len(examples) == manifest["training_rows_after_augmentation"]


def test_final_cli_requires_a_recipe_lock():
    config = Path(__file__).resolve().parents[1] / "configs" / "train_sft.yaml"
    with pytest.raises(SystemExit, match="--final requires --recipe-lock"):
        main(["--config", str(config), "--final"])
