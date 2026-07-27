#!/usr/bin/env python3
"""Download and pin model/dataset snapshots to shared storage.

Writes an asset manifest so later jobs can run with HF_HUB_OFFLINE=1. Never prints
credentials. Supports --dry-run and a tokenizer-boundary check used by M5/M6.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


def _resolve_revision(repo_id: str, revision: str | None, *, repo_type: str = "model") -> str:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, revision=revision, repo_type=repo_type)
    sha = getattr(info, "sha", None) or getattr(info, "id", None)
    if not sha:
        raise RuntimeError(f"could not resolve revision for {repo_id}")
    return str(sha)


def prefetch_model(model_id: str, revision: str | None, hf_home: Path, dry_run: bool) -> dict:
    resolved = _resolve_revision(model_id, revision, repo_type="model")
    record = {
        "id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "hf_home": str(hf_home),
    }
    if dry_run:
        record["status"] = "dry_run"
        return record

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=model_id,
        revision=resolved,
        cache_dir=str(hf_home),
    )
    record["local_path"] = path
    record["status"] = "downloaded"
    return record


def prefetch_dataset(
    dataset_id: str, config: str, revision: str | None, hf_home: Path, dry_run: bool
) -> dict:
    resolved = _resolve_revision(dataset_id, revision, repo_type="dataset")
    record = {
        "id": dataset_id,
        "config": config,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "hf_home": str(hf_home),
    }
    if dry_run:
        record["status"] = "dry_run"
        return record

    from datasets import load_dataset

    # Materialize the category splits we actually use. The official test split is
    # downloaded here for evaluation jobs, not for training.
    for split in ("validation", "test", "dev"):
        load_dataset(dataset_id, config, split=split, revision=resolved)
    record["splits"] = ["validation", "test", "dev"]
    record["status"] = "downloaded"
    return record


def tokenizer_boundary_check(model_id: str, revision: str | None) -> dict:
    """Record how the chat template and A/B/C/D candidates tokenize.

    Training and evaluation both depend on this boundary. Capture it once against
    the real tokenizer so a later template change cannot silently drift.
    """
    from transformers import AutoTokenizer

    from nebius_poc.prompts import LABELS, SYSTEM_MESSAGE, build_messages

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    messages = build_messages(
        "Which remedy applies?",
        ["Damages", "Rescission", "Specific performance", "Nothing"],
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    candidates = {}
    for label in LABELS:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        candidates[label] = {"token_ids": ids, "tokens": tokens}

    # Leading whitespace: some templates insert a space or newline before the
    # assistant content. Record what the template actually emitted at the end.
    trailing = prompt[-20:]
    return {
        "model_id": model_id,
        "revision": revision,
        "system_message": SYSTEM_MESSAGE,
        "prompt_preview_tail": trailing,
        "prompt_token_count": len(prompt_ids),
        "candidates": candidates,
        "notes": [
            "candidate strings are the bare letters A/B/C/D with no leading space",
            "prompt was built with add_generation_prompt=True",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prefetch and pin HF assets")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--smoke-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--dataset-config", default="professional_law")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path("hf_cache"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/raw"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="prefetch the 0.5B smoke model only; skip the 7B weights",
    )
    parser.add_argument(
        "--boundary-check-only",
        action="store_true",
        help="run the tokenizer boundary check without downloading new weights",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / f"{stamp}_prefetch_joblocal"
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "started_utc": datetime.now(UTC).isoformat(),
        "dry_run": args.dry_run,
        "hf_home": str(args.hf_home),
        "assets": {},
    }

    boundary_model = args.smoke_model if args.smoke_only or args.boundary_check_only else args.model

    if args.boundary_check_only:
        resolved = _resolve_revision(boundary_model, args.model_revision, repo_type="model")
        boundary = tokenizer_boundary_check(boundary_model, resolved)
        (out / "tokenizer_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
        manifest["assets"]["tokenizer_boundary"] = boundary
        manifest["ended_utc"] = datetime.now(UTC).isoformat()
        (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log.info("wrote %s", out / "tokenizer_boundary.json")
        return 0

    if args.smoke_only:
        manifest["assets"]["smoke_model"] = prefetch_model(
            args.smoke_model, args.model_revision, args.hf_home, args.dry_run
        )
        boundary_model = args.smoke_model
    else:
        manifest["assets"]["model"] = prefetch_model(
            args.model, args.model_revision, args.hf_home, args.dry_run
        )
        manifest["assets"]["smoke_model"] = prefetch_model(
            args.smoke_model, args.model_revision, args.hf_home, args.dry_run
        )
        boundary_model = args.model

    manifest["assets"]["dataset"] = prefetch_dataset(
        args.dataset,
        args.dataset_config,
        args.dataset_revision,
        args.hf_home,
        args.dry_run,
    )

    if not args.dry_run:
        asset_key = "model" if "model" in manifest["assets"] else "smoke_model"
        resolved = manifest["assets"][asset_key]["resolved_revision"]
        boundary = tokenizer_boundary_check(boundary_model, resolved)
        (out / "tokenizer_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
        manifest["assets"]["tokenizer_boundary"] = {
            "path": str(out / "tokenizer_boundary.json"),
            "model_id": boundary["model_id"],
            "candidates": boundary["candidates"],
            "prompt_token_count": boundary["prompt_token_count"],
        }

    manifest["ended_utc"] = datetime.now(UTC).isoformat()
    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", out / "asset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
