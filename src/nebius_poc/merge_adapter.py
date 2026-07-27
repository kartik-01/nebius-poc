"""Fold a LoRA adapter into the base weights for serving.

vLLM can serve adapters directly, but merging keeps the benchmark honest: the served
artifact is then a plain dense model with no adapter overhead in the measurement, and
the customer gets one directory they can copy.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch

from nebius_poc.data import load_config
from nebius_poc.prompts import build_messages
from nebius_poc.report import close_run, directory_checksums, open_run, write_json
from nebius_poc.train import DTYPES, load_tokenizer

log = logging.getLogger(__name__)

VERIFICATION_QUESTION = "Which body of law governs the formation of contracts for goods?"
VERIFICATION_CHOICES = (
    "The Uniform Commercial Code",
    "The Restatement of Torts",
    "The Federal Rules of Evidence",
    "The Bankruptcy Code",
)


def merge(model_id: str, revision: str | None, adapter: Path, dtype: torch.dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    return PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()


@torch.no_grad()
def verify_generation(model, tokenizer, max_new_tokens: int = 16) -> str:
    """One greedy generation so a broken merge fails here and not in the vLLM job."""
    prompt = tokenizer.apply_chat_template(
        build_messages(VERIFICATION_QUESTION, VERIFICATION_CHOICES),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def copy_base_tokenizer(model_id: str, revision: str | None, output: Path, tokenizer) -> None:
    """Publish the base model's own tokenizer files alongside the merged weights.

    A LoRA merge never changes the tokenizer, so re-serialising it with
    save_pretrained only risks writing a schema the serving runtime cannot read.
    That is not hypothetical: transformers 5 emits `extra_special_tokens` as a list,
    and the transformers 4.x inside the pinned vLLM image rejects it. Copying the
    pinned originals keeps training, evaluation, and serving byte-identical.
    """
    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(
            snapshot_download(
                model_id,
                revision=revision,
                allow_patterns=list(_TOKENIZER_FILES),
            )
        )
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the merge
        log.warning("could not resolve base tokenizer snapshot (%s); saving via transformers", exc)
        tokenizer.save_pretrained(output)
        return

    copied = []
    for name in _TOKENIZER_FILES:
        source = snapshot / name
        if source.is_file():
            shutil.copy2(source, output / name)
            copied.append(name)
    if not copied:
        log.warning("no tokenizer files found in %s; saving via transformers", snapshot)
        tokenizer.save_pretrained(output)
        return
    log.info("copied base tokenizer files: %s", ", ".join(copied))


def run(config: dict, args: argparse.Namespace) -> Path:
    model_id = args.model or config["model"]["id"]
    revision = config["model"].get("revision")
    dtype = DTYPES[args.dtype or config["model"]["dtype"]]

    run_dir, manifest = open_run("merge", args.results_root, config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log.info("merging %s into %s", args.adapter, model_id)
    tokenizer = load_tokenizer(model_id, revision)
    merged = merge(model_id, revision, args.adapter, dtype)
    merged.save_pretrained(output, safe_serialization=True)
    copy_base_tokenizer(model_id, revision, output, tokenizer)

    sample = verify_generation(merged.eval(), tokenizer)
    log.info("verification generation: %r", sample)

    report = {
        "base_model": model_id,
        "base_revision": revision,
        "adapter": str(args.adapter),
        "merged_path": str(output),
        "dtype": str(dtype),
        "adapter_checksums": directory_checksums(args.adapter),
        "merged_checksums": directory_checksums(output),
        "verification_prompt": VERIFICATION_QUESTION,
        "verification_generation": sample,
    }
    write_json(run_dir / "merge.json", report)
    close_run(run_dir, manifest, "ok", {"merged_path": str(output)})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base weights")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="override the configured base model")
    parser.add_argument("--dtype", choices=tuple(DTYPES))
    parser.add_argument("--results-root", type=Path, default=Path("results/raw"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
