# Results

Every number here comes from a file under `results/summary/`. Nothing is typed by
hand, and the generating artifact is named next to each section.

Hardware: 4 NVIDIA H200 GPUs across 2 nodes of a Nebius Soperator Slurm cluster,
which is the allocation this PoC was given. Read section 6 before quoting any of
this in a capacity discussion.

## 1. Cluster qualification

Source: `results/summary/cluster_validation.json`

Overall status is WARN with no hard failures. The single warning is that no
absolute bandwidth thresholds were configured, which is deliberate: a threshold
copied from a different cluster topology produces a pass or fail that means
nothing. Thresholds should come from Nebius or from an approved reference run on
the same SKU.

| Check | Result |
| --- | --- |
| GPUs | 2 per node, 4 total, all UUIDs recorded |
| GPU model | NVIDIA H200, consistent across the allocation |
| CUDA compute smoke | PASS on every allocated GPU, 0 numerical mismatches |
| Intra-node NCCL, worker-0 | 60.68 GB/s average bus bandwidth |
| Intra-node NCCL, worker-1 | 60.77 GB/s average bus bandwidth |
| Node asymmetry | 0.14 % |
| Inter-node NCCL, 3 repetitions | 22.46, 22.33, 22.27 GB/s |
| Run-to-run variation | 0.36 % coefficient of variation |
| NCCL correctness | 0 wrong values, 0 out-of-bounds, across all 5 runs |
| Transport | InfiniBand verbs, no socket fallback |
| Node-local storage (fio, O_DIRECT) | 4.46 GB/s write, 6.35 GB/s read |

Two results matter more than the bandwidth figures.

**Correctness.** All five NCCL runs reported zero wrong values. A fabric that is
fast but silently corrupts a reduction is worse than a slow one, and this is the
check that catches it.

**Stability.** Three repetitions of the multi-node all-reduce landed within
0.36 % of each other, and the two nodes agree to within 0.14 % on the intra-node
test. That is the evidence the allocation is configured consistently, which is
the real question behind "is this cluster reliable".

The transport flags deserve a look too. NCCL selected InfiniBand verbs and never
fell back to TCP sockets. Socket fallback is a common misconfiguration on new
clusters, and it costs most of your interconnect bandwidth while everything still
appears to work.

### The validator container

**64 MB**, built by `scripts/build_validator_sqsh.sh`, and it carries only what the
checks need: a Python collector, the compiled `gpu_smoke` binary, `fio`, `jq`,
`pciutils`, `numactl`, and `ibverbs-utils`. No compiler, no MPI, no PMIx, no
training framework, and no CUDA libraries. Building MPI into a portable validator
is where these images usually break.

Getting there required working around two constraints on this cluster, and the
solution is worth describing because a customer on Soperator will hit both.

**No Docker daemon, and user namespaces are blocked**, so neither `docker build`
nor `enroot start --rw` is available. Neither is actually necessary. An Enroot
image is a squashfs archive, so it can be assembled from outside rather than built
by executing commands inside a container: unpack a base rootfs with `unsquashfs`,
populate it with `dpkg -x` (which extracts without root and without running
maintainer scripts), copy in the binaries, and reseal with `mksquashfs`. Namespaces
are only needed to run inside a rootfs, not to construct one.

**The CUDA runtime is linked statically** into `gpu_smoke` with `nvcc -cudart
static`, producing a 970 KB binary with no `libcudart` dependency. `libcuda.so`
still arrives from the host driver through the NVIDIA container hook at run time,
so the image ships no CUDA at all. That is what takes it from the 2.2 GB a CUDA
base image would cost down to 64 MB.

One detail that is easy to miss: the NVIDIA hook decides what to inject by reading
`NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`. CUDA base images set
these; a bare Ubuntu base does not. Without declaring them in `/etc/environment`,
where Enroot keeps the image environment, the container starts with no driver
libraries, no `nvidia-smi`, and `gpu_smoke` reports no devices.

Building the image immediately paid for itself by exposing a latent bug. The
storage check invoked `fio --time_based 0`, but `--time_based` is a boolean flag,
so fio read `0` as a job filename and failed. That code path had never executed
because `fio` had never been present in any image the validator ran in. The
storage numbers above come from real `fio` with `O_DIRECT`.

Communication testing uses the NCCL binaries the cluster already provides
(`all_reduce_perf_mpi`) rather than a rebuilt copy. The validator parses their
output and folds it into a single report. That split is what keeps the image small
while still exercising the vendor's own tooling, and it avoids compiling MPI and
PMIx into a portable image.

## 2. Data and splits

Source: `results/raw/<run>/split_manifest.json`

MMLU ships this category as 5 dev, 170 validation, and 1,534 test questions. The
official adaptation split is 170 examples, which is not enough to adapt a 7B
model: an earlier iteration of this PoC trained on exactly that and measured a
change indistinguishable from noise.

The category's own records are therefore split here instead.

| Partition | Questions |
| --- | --- |
| Adaptation pool (170 validation + 70 % of test) | 1,244 |
| ... internal set for recipe selection | 100 |
| ... pilot training | 1,144 |
| Evaluation holdout (30 % of test) | 460 |
| Training rows after safe augmentation | 4,964 |

The partition is deterministic in the stable SHA-256 question IDs and stratified
by answer label. Nothing evaluated is ever trained on, and that is enforced
rather than promised: `load_adaptation_pool` carves the holdout out and never
returns it, `split_manifest` raises if a holdout ID appears in the pool, and a
test proves it raises. Verified on the final run: **1,244 trained, 460 reserved,
intersection zero.**

This does mean the numbers are not comparable to published MMLU leaderboard
scores. They were not comparable anyway, since this PoC is zero-shot and
published figures are 5-shot. What matters here is that base and tuned are
measured under identical conditions on questions neither has seen.

### Answer-position augmentation

Each safe question expands into four variants with the correct answer placed once
in each position. This removes answer-position bias from the training signal.

Permutation is unsafe when a choice refers to other choices by label, for example
"both A and B". Those are detected and left unpermuted, and every skip is recorded
in a reviewable audit artifact. This is not theoretical: one question in the pool
has the gold answer written as the lowercase string `"a and b"`. The original
detector was case-sensitive and missed it, and permuting that question would have
produced three training rows with the wrong label attached. The audit is what
surfaced it.

### Sequence length

Source: `results/raw/*_profile/sequence_length_profile.json`

Rendered prompts against the pinned tokenizer: minimum 91 tokens, median 248,
p95 436, p99 599, maximum 769. Nothing truncates at 1024, so `max_length: 1024`
covers the data with room to spare. Moving to 2048 would have doubled attention
cost for no benefit.

## 3. Training

Source: `results/summary/training.json`, `results/summary/recipe_lock.json`

| Property | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-7B, revision `d1497293` |
| Dataset | cais/mmlu, `professional_law`, revision `c30699e8` |
| Method | BF16 LoRA, rank 16, alpha 32, dropout 0.05 |
| Objective | Four-choice candidate ranking |
| Learning rate | 2e-5 |
| Epochs | 2 |
| Sequence length | 1024 |
| Micro-batch and accumulation | 2 questions, 4 accumulation steps |
| Topology | 2 nodes, 2 GPUs per node, world size 4, PyTorch DDP |
| Training rows | 4,964 from 1,244 questions |
| Steps | 310 |
| Median step time | 1.617 s |
| Peak GPU memory, rank 0 | 89.2 GiB of 139.8 GiB |
| NaN or Inf in metrics | 0 |
| Exit status | ok |

The run manifest records world size 4 on `worker-[0-1]` with four distinct GPU
UUIDs, so the multi-node claim is checkable rather than asserted.

### Failure recovery

Source: `results/raw/*_train_job201/`, `results/raw/*_train_job202/`

Checkpoints are written at epoch boundaries and carry the adapter plus optimizer
and scheduler state. To check that a resume actually works rather than trusting
that it should, a training run was killed and restarted.

| Stage | Result |
| --- | --- |
| Checkpoint written | `checkpoint-155`, at the epoch-1 boundary |
| Job killed | `scancel` after step 163 of 310 |
| Checkpoint after the kill | Adapter and `training_state.pt` both intact |
| Resume | All 4 ranks logged `resumed from ... at step 155` |
| First step executed | **156**, not 1 |
| Completion | Step 310, final loss 0.0515, 0 NaN or Inf, adapter written |

The step numbers are the evidence. A resume that silently restarts from step 1
would look successful in the logs and waste the entire run, so the check is that
it continued rather than that it finished.

Cost of the interruption: 12m00s of wall clock across both jobs against 10m30s
uninterrupted, so about 1m30s of overhead. That is one extra job startup plus the
8 steps between `checkpoint-155` and the kill. The 155 steps before the checkpoint
were preserved, not repeated.

Because checkpoints land at epoch boundaries, the worst case is losing the whole
interrupted epoch, 155 steps here. This run died early in epoch 2 and lost 8.

For a customer planning months of training, this is the reliability question that
matters more than a throughput soak, and it is worth repeating on the target
cluster with a real node eviction rather than a scheduler cancellation.

### Prompt

The standard MMLU zero-shot completion format: a one-line subject header, the
question, four labelled options, and a bare `Answer:` line the model continues.

Two properties matter. Training and evaluation import the same rendering and
encoding code, so the completion boundary cannot drift between them. And the
candidates are `" A"` through `" D"` with a leading space, because the prompt ends
at `Answer:` with none; the pinned tokenizer maps each to a single token (362,
425, 356, 422). That was verified against the real tokenizer and recorded, not
assumed, and the masking was confirmed by decoding actual training batches:
exactly one scored position per row, holding the gold candidate.

Chain-of-thought was not used. MMLU provides no verified reasoning traces, so
training on generated rationales would inject a second model's errors into a
result meant to be clean.

### Recipe selection

Source: `results/summary/pilot_comparison.json`

Both objectives were implemented and unit tested before any holdout data was
inspected, so neither is a post-hoc rescue. Selection used a fixed 100-question
internal set carved out of the adaptation pool. The holdout was never loaded.

| Configuration | Internal accuracy | Mean gold NLL | Train time |
| --- | --- | --- | --- |
| Base model, no training | 0.52 | 1.186 | n/a |
| Completion SFT, 2 epochs, lr 2e-5 | 0.55 | 1.203 | 3m56s |
| Candidate ranking, 2 epochs, lr 2e-5 | **0.57** | 1.244 | 9m42s |

Both objectives beat the untrained model. The 2-point gap between them is 2
questions out of 100 and sits inside the noise floor, so ranking is not claimed
as a significant winner. It was selected because it optimises the forced-choice
metric this PoC reports, using the three wrong answers as explicit negatives.

Two things worth recording. **Ranking costs 2.5× the compute** because it forwards
four candidates per question, so on this task the objectives are equivalent within
noise and SFT is considerably cheaper. And the selection metric had to change:
gold NLL led the selection when the internal set was 20 questions and accuracy
moved in 5-point steps, but at 100 questions accuracy resolves single questions
and is the metric actually reported. Here the two metrics disagree, because with
this much data the model reorders candidates rather than only sharpening
confidence. The selector is recorded in the lock rather than left implicit.

### Learning rate

2e-5 with 2 epochs, and both are measured rather than inherited. An earlier
iteration at lr 1e-4 drove training loss to roughly 1e-6 while internal accuracy
fell *below* the untrained baseline. That is overfitting, and it is the most
transferable finding in this PoC: a customer fine-tuning on a small in-house
dataset will hit exactly this, and needs a held-out set that can detect it.

## 4. Accuracy, base versus tuned

Source: `results/summary/accuracy.json`

Both models were scored on all 460 questions of the reserved holdout, using
identical evaluation code, identical prompts, and predictions paired by stable
question ID.

### Forced choice

| Metric | Value |
| --- | --- |
| Questions | 460 |
| Base accuracy | 53.48 % |
| Tuned accuracy | 58.70 % |
| Difference | **+5.22 percentage points** |
| 95 % bootstrap CI | **+1.30 to +9.13 points** |
| Exact McNemar p | **0.0127** |
| Both correct | 215 |
| Both wrong | 159 |
| Base right, tuned wrong | 31 |
| Tuned right, base wrong | 55 |

**The improvement is statistically significant.** The confidence interval excludes
zero and McNemar rejects at p = 0.013. The tuned model won 55 questions and lost
31, a net gain of 24 out of 460.

State the magnitude honestly. The interval runs from +1.3 to +9.1 points, so the
direction and significance are solid while the effect size is imprecise. Describe
it as a real but modest gain rather than pinning it to a single number.

### Generation

| Metric | Base | Tuned |
| --- | --- | --- |
| Accuracy | 54.35 % | 58.26 % |
| Format adherence | 100 % | 100 % |
| Unparseable outputs | 0 | 0 |

**Format adherence was already perfect before training.** Given a standard MMLU
prompt, the base model emitted a valid A/B/C/D on all 460 questions, and so did
the tuned model.

This matters more than it looks. The most common way a fine-tuning result gets
inflated is that the base model cannot produce parseable output, fine-tuning
fixes the formatting, and the whole jump is reported as an accuracy improvement.
There was no formatting deficit here, so every point of the gain is better answer
selection. Generation accuracy moved +3.91 points independently, with format held
at 100 % on both sides, which corroborates it from a second direction.

### A negative result worth reporting

An earlier iteration used `Qwen2.5-7B-Instruct` with a chat prompt and 170
training examples, and measured +0.85 points with a confidence interval spanning
zero (p = 0.283). Two changes were tested against that.

Switching to the base checkpoint **did not help by itself**. The hypothesis was
that a non-instruction-tuned model would have formatting headroom for
fine-tuning to recover. Measured on 200 questions, the base model's format
adherence is also 100 %. There is no headroom on either checkpoint, so model
choice does not create it.

What changed the result was **training data volume**: 170 questions to 1,244. The
constraint was never the model or the method. It was the data, and the earlier
design starved the experiment by preserving MMLU's official split semantics.

`results/instruct-baseline/` retains the earlier numbers.

## 5. Inference performance

Source: `results/summary/inference.json`

Optimised for output-token goodput, meaning output tokens per second that satisfy
declared latency and error limits. Guardrails were fixed before benchmarking at
p95 TTFT under 2000 ms, p95 TPOT under 100 ms, and zero request errors. These are
demonstration limits chosen for the PoC. They are not customer SLOs.

Workload is 512 input tokens and 128 output tokens per request from a
deterministic prompt set with a fixed seed. Each topology was driven to its own
saturation point rather than a shared fixed load, and all four were measured
against the merged model that was actually shipped.

| Topology | GPUs | Max goodput under guardrails | Concurrency | Per GPU |
| --- | --- | --- | --- | --- |
| P0, one TP=1 replica | 1 | 6,109 tok/s | 128 | 6,109 |
| P1, one TP=2 replica | 2 | 10,105 tok/s | 256 | 5,052 |
| P2, two TP=1 replicas | 2 | **12,279 tok/s** | 256 | **6,140** |
| P3, four TP=1 replicas | 4 | **24,483 tok/s** | 512 | **6,121** |

Every topology fails at the next concurrency step up, always on p95 TTFT
exceeding 2000 ms and never on errors. That is what makes "maximum under
guardrails" a real claim rather than an arbitrary operating point.

### Replicas beat tensor parallelism at equal GPU count

P1 and P2 both use two GPUs. P2 delivers 12,279 tok/s against P1's 10,105, which
is **21.5 % more throughput**, at lower latency. P1 also degrades badly past its
peak: at concurrency 512 its throughput stops rising and p95 TTFT reaches
4,075 ms, while P2 keeps climbing.

The reason is straightforward. A 7B model in BF16 needs about 15 GB and an H200
has 139.8 GB. Tensor parallelism across two GPUs buys no memory headroom and
charges NVLink communication on every forward pass. Splitting a model that
already fits is a cost with no matching benefit.

This comparison is only visible under load. An earlier sweep at a fixed
concurrency of 64 put the two topologies within 0.6 % of each other and they
looked equivalent. Benchmark at production load, or the result will mislead you.

The finding also reproduced independently: an earlier campaign on a different
merged model measured 22.8 % for the same comparison.

### Replication scales linearly

Per-GPU goodput is 6,109 at one GPU, 6,140 at two, and 6,121 at four. P3 delivers
almost exactly twice P2 for twice the hardware. Within measurement noise, adding
replicas adds capacity in proportion, which makes fleet sizing a division problem.

### Repeatability

Three repetitions at each operating point.

| Topology | Median goodput | Spread across runs |
| --- | --- | --- |
| P2 at concurrency 256 | 12,145 tok/s | 1.94 % |
| P3 at concurrency 512 | 24,206 tok/s | 1.50 % |

### Reliability soak

P3 at roughly 80 % of its measured peak, sustained.

| Metric | Value |
| --- | --- |
| Duration | 596.3 s |
| Requests | 90,000 |
| Errors | 0 |
| Sustained goodput | 19,407 tok/s |
| p95 TTFT | 580 ms |
| p95 TPOT | 13.18 ms |

No errors and no latency drift across the window. Across the whole benchmark
campaign, zero requests failed.

### Translating goodput into fleet size

Source: `results/summary/sizing.json`

The measured per-GPU figure converts directly into capacity planning. At 6,140
output tok/s per GPU, derated to 80 % because the peak sits at the guardrail
boundary, serving costs **0.057 GPU-hours per million output tokens**.

| Target output tok/s | GPUs | GPU-hours / 1M output tokens |
| --- | --- | --- |
| 10,000 | 3 | 0.057 |
| 50,000 | 11 | 0.057 |
| 100,000 | 21 | 0.057 |
| 500,000 | 102 | 0.057 |

No GPU price is assumed anywhere in this repository. `scripts/size_fleet.py`
accepts `--gpu-hour-usd` if you want cost columns; without it the output stays a
measurement. Multiply the GPU-hours figure by your own quoted rate.

Two limits on this table. It is **serving capacity only**, and on a reservation of
several hundred GPUs the training side is likely the larger share, which this PoC
does not size. And it assumes replication keeps scaling linearly, which was
measured across 1, 2 and 4 GPUs and not beyond.

### Measurement caveats

**Warm-up requests are not excluded.** `configs/benchmark.yaml` carries a
`warmup_requests` value, but the pinned vLLM build has no warm-up flag, so it acts
only as a floor on the request count and every request lands in the measurement.
With 1,024 to 4,096 requests per shard the effect of a few cold-start requests is
small, but these runs are not warm-up-excluded and should not be described that
way.

**Multi-replica percentiles are approximate.** vLLM 0.8.5 does not support
`--save-detailed`, so per-request records are unavailable and a true fleet
percentile cannot be reconstructed for P2 and P3. Request counts, error counts,
and throughput are summed exactly. Percentiles are taken from the **worst
replica**, never averaged and never from an arbitrary one, and every affected
point is labelled in `inference.json`. A build that emits detailed records would
remove the approximation.

## 6. Monitoring

Every run leaves the evidence needed to see what happened.

| Artifact | Contents |
| --- | --- |
| `gpu_monitor.csv` | Per-GPU utilisation, memory, temperature, power, and clocks sampled every 15 s, for allocated GPUs only |
| `metrics.jsonl` | Rank-zero per-step loss, learning rate, gradient norm, step time, samples/s, peak GPU bytes |
| `manifest.json` | GPU UUIDs, driver and CUDA versions, package versions, Slurm job and node list, full config, exit status |
| `split_manifest.json` | Every trained and reserved question ID |
| Slurm | `squeue`, `sacct`, and per-job stdout/stderr under `logs/` |

The monitoring script samples only GPUs visible in the allocation, so it is safe
on a shared cluster. `docs/RUNBOOK.md` has the live commands.

## 7. Reproducing this

`docs/RUNBOOK.md` has the exact commands in order, with cluster-specific values
pulled from `configs/cluster.env`.

## 8. What this does and does not establish

Established: the container and driver stack work on this cluster; the allocated
GPUs are healthy; two-node collective communication is correct and stable;
distributed training runs end to end and is traceable to manifests; fine-tuning
produces a statistically significant accuracy improvement on held-out questions;
the serving architecture is characterised with a defensible operating point.

Not established: scaling behaviour at 512 GPUs; H100 performance, since every
number is from H200s; rack-to-rack congestion at production scale; mean time
between failures for long jobs; checkpoint and storage throughput under
production pressure; failure recovery across hundreds of workers; scheduler
behaviour under a large reservation.

Use this wording when the results are presented:

> This PoC validates the software stack, allocated GPU health, two-node
> collective communication, distributed execution, accuracy methodology, and
> serving architecture. It does not validate 512-GPU topology behaviour or
> absolute H100 performance. A production reservation should be gated by a larger
> topology-aware validation on the target H100 SKU.

One further scoping note on the inference figures. The measurement used 2 of 8
GPUs per node and, according to the NCCL logs, 2 of the 8 available 400 Gb/s
HCAs. A full node engages all eight. The inter-node numbers here reflect roughly
a quarter of the node's network capability by construction and should not be
extrapolated.
