"""Profile rendered prompt lengths against a real tokenizer.

Used before locking max_length. Prefers the adaptation-pool prompts so the decision
is driven by the data that will actually be trained on.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from nebius_poc.data import expand, load_adaptation_pool, load_config, split_adaptation_pool
from nebius_poc.prompts import LABELS, build_messages, profile_sequence_lengths
from nebius_poc.report import write_json

log = logging.getLogger(__name__)


def recommend_max_length(profile: dict, limits: Sequence[int] = (1024, 2048)) -> int:
    """Smallest configured limit covering at least 99% of examples."""
    count = profile["count"]
    for limit in sorted(limits):
        truncated = profile["truncated_at"][str(limit)]
        if truncated / count <= 0.01:
            return limit
    return max(limits)


def rendered_texts(variants, tokenizer) -> list[str]:
    texts = []
    for variant in variants:
        prompt = tokenizer.apply_chat_template(
            build_messages(variant.question, variant.choices),
            tokenize=False,
            add_generation_prompt=True,
        )
        # Include the gold letter so the profile matches the SFT sequence, not just
        # the prompt prefix.
        texts.append(prompt + LABELS[variant.answer])
    return texts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile adaptation-pool sequence lengths")
    parser.add_argument("--config", type=Path, default=Path("configs/train_sft.yaml"))
    parser.add_argument("--model", help="override model id (defaults to config)")
    parser.add_argument("--revision", help="tokenizer revision")
    parser.add_argument("--out", type=Path, default=Path("results/raw"))
    parser.add_argument("--limits", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="profile the 150-example pilot train split instead of the full 170 pool",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from transformers import AutoTokenizer

    config = load_config(args.config)
    dataset = config["dataset"]
    model_id = args.model or config["model"]["id"]
    revision = args.revision or config["model"].get("revision")

    pool = load_adaptation_pool(dataset["id"], dataset["config"], dataset["adaptation_split"])
    if args.pilot_only:
        pool, _ = split_adaptation_pool(
            pool, dataset["pilot_train_size"], dataset["pilot_validation_size"], dataset["seed"]
        )

    variants = expand(pool, dataset["max_variants_per_question"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    texts = rendered_texts(variants, tokenizer)

    def tokenize(text: str):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    profile = profile_sequence_lengths(texts, tokenize, limits=args.limits)
    recommended = recommend_max_length(profile, args.limits)
    payload = {
        "model_id": model_id,
        "revision": revision,
        "questions": len(pool),
        "rows_after_augmentation": len(variants),
        "pilot_only": args.pilot_only,
        "profile": profile,
        "recommended_max_length": recommended,
        "configured_max_length": config["training"]["max_length"],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sequence_length_profile.json"
    write_json(path, payload)
    log.info(
        "n=%d median=%d p99=%d max=%d recommended_max_length=%d (configured=%d)",
        profile["count"],
        profile["median"],
        profile["p99"],
        profile["max"],
        recommended,
        config["training"]["max_length"],
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
