# Nebius LLM Training and Inference PoC

A reproducible proof-of-concept for a customer evaluating Nebius before reserving
512 H100 GPUs for six months.

The PoC runs on the provided evaluation cluster: 2 nodes, 4 allocated H200 GPUs,
Slurm with Enroot and Pyxis containers.

## Status

Complete. All four stages ran on the cluster and produced the artifacts under
`results/summary/`. Every reported value is generated from a run manifest.

## Headline results

| | Result |
| --- | --- |
| Cluster qualification | No hard failures. 0 NCCL wrong values across 5 runs, 0.36 % run-to-run variance, 0.14 % node asymmetry, InfiniBand confirmed with no socket fallback |
| Multi-node training | Qwen2.5-7B, LoRA, 2 nodes and 4 GPUs, world size 4, 310 steps, 89.2 GiB peak per GPU, no NaN or Inf |
| **Accuracy** | **53.48 % to 58.70 %, +5.22 points on 460 held-out questions. Statistically significant: 95 % CI +1.30 to +9.13, McNemar p = 0.013** |
| Format adherence | 100 % before and after, so the gain is answer selection rather than output formatting |
| Inference | 24,483 output tok/s on 4 GPUs under latency guardrails. Replication beats tensor parallelism by 21.5 % at equal GPU count, and per-GPU throughput is flat across 1, 2, and 4 GPUs |
| Reliability, serving | 90,000 requests over 596 s with zero errors |
| Reliability, training | Job killed after step 163 of 310 and resumed from checkpoint; all 4 ranks continued at step 156 and completed normally |

## Scope: what this proves, and what it does not

> This PoC validates the software stack, allocated GPU health, two-node
> collective communication, distributed execution, accuracy methodology, and
> serving architecture. It does not validate 512-GPU topology behaviour or
> absolute H100 performance. A production reservation should be gated by a larger
> topology-aware validation on the target H100 SKU.

That distinction is deliberate and is repeated in the design document and the
demo narrative. Four H200s can prove a method is sound and a stack is healthy.
They cannot predict rack-to-rack congestion, large-job failure rates, or
checkpoint throughput at 512 GPUs.

## What it does

1. **Qualify the cluster.** A lightweight portable container collects GPU
   inventory, runs a bounded CUDA compute check and a bounded local-disk check,
   and the workflow runs provider-native NCCL all-reduce tests intra-node and
   across both nodes. Everything lands in one machine-readable
   PASS/WARN/FAIL/UNKNOWN report.
2. **Train.** BF16 LoRA fine-tune of `Qwen/Qwen2.5-7B` on the `professional_law`
   category of `cais/mmlu`, using PyTorch DDP across 2 nodes and 4 GPUs.
3. **Measure accuracy.** Base against tuned on a reserved holdout the training
   code structurally cannot reach, scored by zero-shot conditional
   log-likelihood over A/B/C/D, with paired bootstrap and McNemar statistics.
4. **Measure performance.** vLLM serving benchmarks optimising output-token
   goodput under declared latency and error-rate guardrails, comparing tensor
   parallelism against replication at equal GPU count.

## How the data is split

MMLU ships `professional_law` as 5 dev, 170 validation, and 1,534 test questions.
Training on the 170-example official adaptation split is not enough to adapt a 7B
model, so the category's own records are split here instead:

| Partition | Questions |
| --- | --- |
| Adaptation pool | 1,244 |
| Evaluation holdout | 460 |

Nothing evaluated is ever trained on. The partition is deterministic in the stable
question IDs, the holdout is carved out inside the loader and never returned, the
manifest builder raises if a reserved ID reaches the pool, and a test proves it
raises. Verified on the final run: 1,244 trained, 460 reserved, intersection zero.

## Layout

```
configs/      cluster.env and YAML for validator, training, evaluation, benchmarking
containers/   Dockerfiles for the validator and training images
slurm/        sbatch entry points
src/          nebius_poc package: data, prompts, objectives, training, evaluation, stats
validator/    portable cluster checks and NCCL log parsing
scripts/      cluster discovery, asset prefetch, GPU monitoring, benchmark aggregation, demo driver
tests/        offline CPU tests, no network and no model downloads
docs/         DESIGN, RUNBOOK, RESULTS, DEMO
results/      run artifacts and curated summaries
```

`results/raw/` holds full logs and checkpoints and is not tracked.
`results/summary/` holds the small curated JSON files the documents are generated
from, and those are tracked.

## Prerequisites

- Python 3.11 or newer
- Access to a Slurm cluster with Enroot and Pyxis, and an accounting association
  permitting GPU allocation
- Outbound network from the login node for the one-time model and dataset
  prefetch

## Getting started

One command sets up a fresh clone:

```bash
./scripts/setup.sh --check              # report what is present and what is missing
./scripts/setup.sh                      # venv, config skeleton, tests
./scripts/setup.sh --from <checkout>    # reuse images, wheels and cache from another checkout
./scripts/setup.sh --build              # build images and prefetch weights, about 25 minutes
```

It is idempotent, so re-running costs seconds and repeats no downloads. The
default does nothing slower than `pip`, which makes it safe to run before you
know whether you have cluster access.

Cluster-specific values are never guessed. `configs/cluster.env` is created from
the template for you to fill in, and `./scripts/discover_cluster.sh` reads the
values off the cluster without writing them anywhere.

The individual steps remain available if you would rather drive them yourself:

```bash
make install        # create .venv and install the package with dev extras
make test           # offline unit tests, no network, no GPU
make lint           # ruff
make prepare-data   # build the split manifest and the augmentation audit
cp configs/cluster.env.example configs/cluster.env
cp .env.example .env
```

`make smoke` runs the whole train, evaluate, compare, and merge chain against
`Qwen/Qwen2.5-0.5B` on CPU. It is opt-in because it is the only local step that
downloads weights, and it checks the plumbing rather than producing meaningful
accuracy. The 7B model is never fetched implicitly.

## Running it

```bash
./scripts/demo.sh                        # every step, pausing between them
./scripts/demo.sh accuracy               # a single step
./scripts/demo.sh --reuse                # recorded runs only, nothing reaches the cluster
./scripts/demo.sh throughput --measure   # run the real serving sweep, about 40 minutes
./scripts/demo.sh --list                 # the step names
```

Steps are `preflight`, `validate`, `train`, `accuracy`, `throughput`.

`validate` and `train` submit real Slurm jobs and follow them until they finish
on their own, so nothing needs interrupting. `accuracy` and `throughput` read the
tracked summaries under `results/summary/` and need no cluster access at all. A
step that fails does not stop the ones after it.

`--reuse` shows the last completed validation and training runs instead of
submitting new ones, so the whole walkthrough works while the cluster is busy.
`--measure` does the opposite for serving: it merges the adapter, sweeps all four
topologies to their own saturation points, runs a soak, and rebuilds
`results/summary/inference.json` from what it measured. Servers are cancelled on
exit, including on Ctrl-C, so an interrupted sweep cannot leave GPUs held.

For the underlying sbatch commands, monitoring, and troubleshooting, see
`docs/RUNBOOK.md`.

## Documentation

- `docs/RESULTS.md` for measured results, generated from `results/summary/`
- `docs/RUNBOOK.md` for exact reproduction, monitoring, and troubleshooting steps
- `docs/DESIGN.md` for the decision log and the alternatives that were rejected
