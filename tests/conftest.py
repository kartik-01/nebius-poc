import os

import pytest

from helpers import question_from

# Set before any test module imports the Hugging Face stack. A stray from_pretrained
# or load_dataset then fails loudly instead of quietly downloading 15 GB.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture
def safe_question():
    return question_from(
        "A contractor abandons the job halfway. What is the owner's best remedy?",
        [
            "Expectation damages measured by the cost of completion",
            "Restitution of payments already made",
            "Specific performance of the building contract",
            "Nominal damages only",
        ],
        answer=0,
    )


@pytest.fixture
def adaptation_pool():
    # 170 records, matching the real professional_law validation split size so the
    # 150/20 split exercises the same arithmetic it will on the cluster.
    return [
        question_from(
            f"Scenario {index}: which outcome is most likely on these facts?",
            [f"Holding {index}-{suffix}" for suffix in ("w", "x", "y", "z")],
            answer=index % 4,
        )
        for index in range(170)
    ]
