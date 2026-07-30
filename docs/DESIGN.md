# Design notes

Why the PoC is built the way it is, what was rejected, and what surprised us once
it ran on real hardware. Measured numbers live in `docs/RESULTS.md`.

## Claim boundary

This PoC validates the software stack, allocated GPU health, two-node collective
communication, distributed execution, accuracy methodology, and serving
architecture. It does not validate 512-GPU topology behaviour or absolute H100
performance.

## Core choices

| Area | Choice | Why |
| --- | --- | --- |
| Scheduler | Slurm with Enroot and Pyxis | Matches the assigned Nebius environment |
| Storage | Shared NFS `/home` for code, caches and artifacts; node-local `/tmp` for transient I/O | Every rank must read the same weights and write to one artifact tree, which needs shared storage. Scratch stays node-local so benchmark and checkpoint I/O does not cross NFS |
| Framework | Plain PyTorch DDP with peft, no trainer abstraction | The candidate-ranking objective scores four sequences per question and reshapes them, which a stock `SFTTrainer` cannot express |
| Base model | `Qwen/Qwen2.5-7B`, pinned revision | Permissive licence, ungated, strong 7B, supported by both Transformers and vLLM |
| Adaptation | BF16 LoRA, DDP across 2 nodes and 2 GPUs each | Base fits per GPU, simplest reliable multi-node path |
| Data split | Category records split 1,244 / 460 | The official 170-example adaptation split cannot adapt a 7B model |
| Prompt | Standard MMLU zero-shot completion format | The base checkpoint has no chat template, and this is the format the benchmark was built around |
| Primary accuracy metric | Forced-choice log-likelihood | Measures answer selection independently of output formatting |
| Pilot selection metric | Forced-choice accuracy, NLL secondary | Accuracy is what the comparison reports, and 100 questions resolve single answers |
| Final training gate | `--final` refuses to run without `recipe_lock.json` | Stops silent drift from the pilot decision |
| Serving | vLLM, goodput under fixed guardrails | Standard stack, and goodput is what fleet sizing depends on |
| Topologies | P0 through P3 | Controlled comparison of tensor parallelism against replication at equal GPU count |
| Benchmark weights | `merge_and_unload` before serving | Keeps adapter overhead out of the topology comparison |

## Category and split design

`professional_law` was chosen for the largest test split in MMLU, 1,534
questions, which buys statistical power for the paired comparison. It is also one
of the hardest and most knowledge-dependent categories, which bounds what
adaptation can achieve. Both facts are worth stating.

The split design is the decision most likely to be questioned, so here is the
reasoning in full.

MMLU's official semantics are 5 dev examples for few-shot prompting, 170
validation, and 1,534 test. Preserving those semantics means training on 170
questions. An earlier iteration of this PoC did exactly that and measured +0.85
points with a confidence interval spanning zero. The constraint was the data, not
the method.

The assignment asks to "fine tune the model against at least one category of your
choosing from the `cais/mmlu` dataset". It does not mandate which split trains and
which evaluates. For a dataset that ships one large split and two tiny ones, the
normal workflow is to split it yourself, which is exactly what you would do with a
customer's own data.

So the category's records are partitioned into a 1,244-question adaptation pool
and a 460-question evaluation holdout, deterministic in the stable SHA-256
question IDs and stratified by answer label. The guarantee is structural rather
than procedural: `load_adaptation_pool` carves out the holdout and never returns
it, `split_manifest` raises if a reserved ID appears in the pool, and a test
proves it raises.

The cost is comparability with published MMLU leaderboard figures. That was
already lost, because this PoC is zero-shot and published numbers are 5-shot.
What is preserved is the thing that matters: base and tuned measured under
identical conditions on questions neither has seen.

## Prompt design

The standard MMLU completion format. A subject header, the question, four
labelled options, then `Answer:` which the model continues.

Two properties matter. Training and evaluation import the same rendering and
encoding code, so the completion boundary cannot drift between them. And the
candidates are `" A"` through `" D"` with a leading space, because the prompt ends
at `Answer:` with none. The pinned tokenizer maps each to a single token, which
was verified against the real tokenizer and recorded rather than assumed, and the
masking was confirmed by decoding actual training batches.

Single-token candidates matter because the forced-choice score is then one clean
log-probability per option, with no length normalisation to argue about.

Chain-of-thought was not used. MMLU provides no verified reasoning traces, so
training on generated rationales would introduce a second model and a second
source of error into a result meant to be clean.

## LoRA versus full fine-tuning

LoRA is the primary path. The frozen 7B base fits comfortably on each H200, the
trainable adapter is small, and DDP stays simple. Full-parameter FSDP2 would
demonstrate a heavier collective pattern and is a reasonable appendix, but it is a
systems demonstration rather than a model-quality argument.

QLoRA was not used. Quantisation solves a memory constraint that does not exist
here: 15 GB of weights on a 139.8 GB card.

## DDP versus FSDP

DDP, deliberately. Every rank holds a full BF16 base copy and only adapter
gradients move, which is enough to prove multi-node launch, NCCL health under real
training traffic, and checkpoint behaviour.

FSDP shards parameters, gradients, and optimizer state to train models that do not
fit on one device. This model fits roughly nine times over, so sharding would add
all-gather traffic on every forward pass to solve a problem that does not exist.
It would be slower for no benefit. FSDP becomes the right answer when the customer
trains something that genuinely does not fit, and that is a different exercise.

## Framework choice

A plain PyTorch loop rather than a framework trainer. TRL's `SFTTrainer` would be
a reasonable choice for completion-only SFT and less code to maintain, but it
cannot express the candidate-ranking objective, which needs all four
prompt-plus-candidate sequences scored, reshaped into (questions × candidates),
and cross-entropy applied across the four.

Writing the loop directly also means every line is explainable, which matters
when the design choices have to be defended.

## Answer-position augmentation

Each safe question expands into four variants with the correct answer placed once
in each position, removing answer-position bias from the training signal and
multiplying 1,244 questions into 4,964 rows.

Permutation is unsafe when a choice refers to other choices by label, for example
"both A and B". Those are detected and left unpermuted, and every skip is recorded
in a reviewable audit artifact.

This is not theoretical. One question in the pool has the gold answer written as
the lowercase string `"a and b"`. The original detector was case-sensitive and
missed it, and permuting that question would have produced three training rows
with the wrong label attached. The audit surfaced it. Pattern matching reduces
semantic risk but does not eliminate it, which is why the artifact is meant to be
read by a human before the recipe is locked.

## Objectives, and how the selector changed

Both objectives were implemented and unit tested before any holdout data was
inspected, so neither is a post-hoc rescue.

Completion SFT trains the model to emit the gold letter. Candidate ranking scores
all four sequences, reshapes them per question, and applies cross-entropy across
the four scores, using the wrong answers as explicit negatives and matching the
evaluation metric directly.

The two configurations are kept comparable rather than identical. Ranking forwards
four candidates per question, so a micro-batch of 8 questions puts 32 sequences on
the GPU and runs out of memory. It uses a micro-batch of 2 with 4 accumulation
steps, giving the same 8 sequences in flight as SFT and the same effective batch
of 8 questions per optimiser step. Only the memory mechanics differ, and a test
asserts the effective batch stays equal.

The selection metric had to change with the data. When the internal set was 20
questions, accuracy moved in 5-point steps and could not separate close
candidates, so the continuous gold-choice NLL had to lead. At 100 questions
accuracy resolves single answers and is the metric actually reported. That
distinction is not cosmetic: on the final comparison the two metrics disagree,
because with this much data the model reorders candidates rather than only
sharpening confidence. The selector is now an explicit flag recorded in the lock
rather than an assumption buried in the code.

Ranking was selected because it optimises the reported metric, not because its
2-point lead over SFT was significant. It is not; that gap is 2 questions out of
100. Worth recording alongside: ranking costs 2.5× the compute, so on this task
the objectives are equivalent within noise and SFT is considerably cheaper.

## Learning rate, and what the pilot caught

2e-5 with 2 epochs, both measured rather than inherited.

An earlier iteration at lr 1e-4 drove training loss to roughly 1e-6 while internal
accuracy fell from 0.60 to 0.45, below the untrained baseline. With a small
adaptation pool and a 7B model, that setting memorises rather than adapts.

2 epochs sits inside the standard 1-3 range for supervised fine-tuning. The
argument for it here is not convention but the measurement above: the only
evidence available on this data points toward less training, not more.

This is the most transferable finding in the PoC. A customer fine-tuning on a
small in-house dataset will hit exactly this, and the fix is a smaller learning
rate plus a held-out set that can detect the problem before the real run.

## Evaluation protocol

Zero-shot, deliberately. Published MMLU numbers are usually 5-shot and will be
higher. The comparison here is base against tuned under identical conditions, so
the protocol only has to be consistent, not conventional. Worth stating explicitly
when anyone compares against public leaderboards.

Predictions are paired by stable question ID, so base and tuned are compared
question by question rather than in aggregate. The reported statistics are a
paired bootstrap confidence interval (10,000 resamples, seed 42) and an exact
McNemar test on the discordant pairs.

Generation is scored separately from forced choice, and format adherence is
reported on its own. Improving output formatting is not the same as improving
knowledge, and conflating them is the easiest way to overstate a result. In this
PoC format adherence was 100 % on both sides, which is what lets the accuracy gain
be attributed to answer selection.

## Inference aggregation

Goodput is counted in output tokens per second that satisfy guardrails fixed
before benchmarking: p95 TTFT under 2000 ms, p95 TPOT under 100 ms, zero errors.
Fixing them first means the operating point is not selected after seeing the data.

Each topology is driven to its own saturation point rather than a shared fixed
concurrency. This mattered more than expected. At concurrency 64 P1 and P2
differed by 0.6 % and looked equivalent; at saturation P2 leads by 21.5 %. A
fixed-concurrency sweep also makes larger topologies look inefficient, because
per-replica load falls as replicas are added: an early sweep suggested 44 %
scaling efficiency, and once each topology reached saturation, scaling was linear.

The pinned vLLM build does not emit per-request records, so fleet percentiles
cannot be rebuilt from raw data for multi-replica topologies. Counts and
throughput are summed exactly. Percentiles are taken from the worst replica, never
averaged and never from an arbitrary one, and every affected point is labelled.

## Rejected alternatives

| Rejected | Why |
| --- | --- |
| Shipping the validator as a stock image with code mounted in | Works, but it is not the lightweight portable container the brief asks for. Assembling the squashfs directly gives a real 64 MB image without a Docker daemon |
| Preserving MMLU's official split semantics | Leaves 170 training examples, which measured +0.85 points with a CI spanning zero. The purist choice produced an unusable result |
| Fabricating absolute NCCL or GPU thresholds | A threshold copied from another topology produces a meaningless pass. The validator reports UNKNOWN until a vendor figure or approved reference run exists |
| Averaging endpoint p95 values | Understates fleet tail latency. The worst replica is used instead |
| Reporting a fixed-concurrency topology comparison | Under-loads larger topologies and hides the tensor-parallelism penalty |
| FSDP for the primary run | Shards a model that fits nine times over, adding communication for no benefit |
| TRL `SFTTrainer` as the only loop | Cannot express the ranking objective's questions-by-candidates layout |
| Serving live LoRA adapters for the primary benchmark | Couples adapter runtime overhead into the topology comparison |
| Switching checkpoints to create formatting headroom | Measured and rejected: the base checkpoint is also 100 % format-adherent, so model choice does not create headroom |
| Training on MMLU `auxiliary_train` | Not category-specific, so it answers a different question than the one asked |
| Re-evaluating the holdout until something clears significance | The holdout was scored once, after the recipe was locked |

## Known limitations

The effect size is imprecise. The confidence interval runs from +1.3 to +9.1
points, so the direction and significance are solid while the magnitude is not
pinned down. Describe it as a real but modest gain.

Failure recovery was exercised by cancelling a job and resuming it, which is a
scheduler cancellation rather than a real node eviction. The distinction matters:
a hardware failure can also corrupt an in-flight checkpoint write, which this does
not test.

The evaluation holdout is 460 questions. That is larger than most MMLU categories
provide in total, and it was sufficient to establish significance, but it is
narrower than the full 1,534-question split.

Warm-up requests are not excluded from the benchmark, because the pinned vLLM
build has no warm-up flag.

The inference measurements used 2 of 8 GPUs per node and 2 of 8 available HCAs.
Nothing here should be extrapolated to a full node or to a 512-GPU fabric.

## Production follow-up

A reservation decision should not rest on this. The staged qualification we would
recommend on the target H100 capacity: repeat the validator on the production
topology; run topology-aware NCCL tests across representative racks; run a
customer-representative sharded training workload; test distributed checkpoint
save and restore; measure storage under concurrent checkpoint and data-loading
pressure; exercise node failure and job resume; sustain training for a
representative duration; reproduce inference capacity against the customer's own
request-length distribution and SLOs; and validate scheduler, quota, and
observability processes. Size the reservation from those measurements, not from
four H200s.
