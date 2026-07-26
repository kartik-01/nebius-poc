import json

import pytest

from nebius_poc.report import (
    close_run,
    directory_checksums,
    environment_snapshot,
    file_checksum,
    format_adherence,
    new_run_id,
    open_run,
    paired_report,
    read_jsonl,
    write_jsonl,
)


def rows(outcomes, prefix="q"):
    return [
        {"question_id": f"{prefix}{index}", "correct": outcome}
        for index, outcome in enumerate(outcomes)
    ]


def test_run_id_carries_the_stage_and_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    run_id = new_run_id("train-completion_sft")
    assert run_id.endswith("_train-completion_sft_job12345")
    assert run_id.split("_")[0].endswith("Z")


def test_run_id_falls_back_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert new_run_id("evaluate").endswith("_evaluate_joblocal")


def test_open_run_writes_a_manifest_before_any_work(tmp_path):
    directory, manifest = open_run("evaluate", tmp_path, {"objective": "completion_sft"})

    written = json.loads((directory / "manifest.json").read_text())
    assert written["exit_status"] is None
    assert written["ended_utc"] is None
    assert written["config"]["objective"] == "completion_sft"
    assert manifest["run_id"] == directory.name


def test_close_run_records_status_and_artifacts(tmp_path):
    directory, manifest = open_run("merge", tmp_path)
    close_run(directory, manifest, "ok", {"merged_path": "/tmp/model"})

    written = json.loads((directory / "manifest.json").read_text())
    assert written["exit_status"] == "ok"
    assert written["ended_utc"] is not None
    assert written["artifacts"]["merged_path"] == "/tmp/model"


def test_environment_snapshot_captures_slurm_variables(monkeypatch):
    monkeypatch.setenv("SLURM_NODELIST", "worker-[0-1]")
    snapshot = environment_snapshot()

    assert snapshot["slurm"]["SLURM_NODELIST"] == "worker-[0-1]"
    assert snapshot["packages"]["torch"] != "unknown"


def test_checksums_change_with_content(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text("{}")
    second.write_text("{}")

    assert file_checksum(first) == file_checksum(second)

    second.write_text('{"x": 1}')
    digests = directory_checksums(tmp_path)
    assert digests["a.json"] != digests["b.json"]


def test_jsonl_round_trips(tmp_path):
    path = tmp_path / "forced_choice.jsonl"
    original = rows([True, False, True])
    write_jsonl(path, original)
    assert read_jsonl(path) == original


def test_format_adherence_separates_wrong_from_unparsable():
    summary = format_adherence(
        [
            {"parsed": "A", "correct": True},
            {"parsed": "B", "correct": False},
            {"parsed": None, "correct": False},
            {"parsed": "Z", "correct": False},
        ]
    )

    assert summary["parsed"] == 2
    assert summary["unparsable"] == 2
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["format_adherence"] == pytest.approx(0.5)


def test_paired_report_pairs_by_id_not_by_position():
    base = rows([True, False, False, False])
    tuned = list(reversed(rows([False, False, False, True])))

    report = paired_report(base, tuned, resamples=200, seed=7)

    # Only q0 is right for base and only q3 for tuned, so both pairs are discordant.
    # Pairing by row order instead would have reported one agreement and none.
    assert report["n"] == 4
    assert report["both_correct"] == 0
    assert report["base_only_correct"] == 1
    assert report["tuned_only_correct"] == 1
    assert report["delta_pp"] == pytest.approx(0.0)


def test_paired_report_rejects_mismatched_question_sets():
    with pytest.raises(ValueError, match="question sets differ"):
        paired_report(rows([True, False]), rows([True, False], prefix="r"), resamples=100)


def test_paired_report_rejects_duplicate_ids():
    duplicated = rows([True, False]) + [{"question_id": "q0", "correct": False}]
    with pytest.raises(ValueError, match="duplicate question IDs"):
        paired_report(duplicated, duplicated, resamples=100)
