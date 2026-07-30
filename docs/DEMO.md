# Demo narrative

The assignment asks for a short presentation and does not set a length. Confirm the slot with the
coordinator. Absent that, plan about 15 minutes of prepared material and expect to spend 25 to 30
delivering it, because a technical audience deciding on a six-month reservation will interrupt, and
every interruption is a good sign.

Rehearse in tiers rather than to a clock, so the talk survives both a short slot and a room full of
questions:

| Tier | Content | Time |
| --- | --- | --- |
| Must say | Scope boundary, the validator's three fields, the accuracy result with both pre-empts, the throughput table with the replication finding, the closing gate recommendation | about 8 min |
| If there is room | How the 64 MB image was built, the learning-rate finding, the split redesign, failure recovery, the monitoring artifacts | about 7 min |
| Answer only | The negative result, the failure log, the rejected alternatives | as asked |

Never lead with the third tier. Those answers are strong because they arrive under questioning.

You are presenting **to the customer**, not to Nebius. They are a small VC-funded
startup, mostly ML engineers without deep infrastructure background, deciding
whether to commit to 512 H100s for six months. Structure the talk around the
questions they actually have, not around the order you built things.

Every number below is in `results/summary/`, so open the file if anyone asks
where it came from.

## Opening, 30 seconds

> Four H200s across two nodes. This validates the software stack, GPU health,
> two-node collectives, distributed training, the accuracy method, and a serving
> operating point. It does not validate 512-GPU topology behaviour or H100
> performance.

Say the boundary first. It buys credibility for everything after it and stops the
room extrapolating a four-GPU number to a 512-GPU decision.

## 1. "Is the cluster we would be renting actually healthy?" (90 s)

Run the validator live; it takes about a minute. Then show three fields:

- `hard_failures: []`
- **0 NCCL wrong values across 5 runs**
- `socket_fallback: false`

Explain why those three. A fabric that is fast but silently corrupts a reduction
is worse than a slow one, and the wrong-value count is what catches it. Socket
fallback is a common misconfiguration on new clusters that costs most of your
interconnect bandwidth while everything still appears to work.

Then the reliability numbers: three repetitions within **0.36 %** of each other,
two nodes agreeing to within **0.14 %**. That is what answers "is this reliable",
and it is a better lead than any peak bandwidth figure.

Worth thirty seconds on how the image was built, because a customer on Soperator
hits the same wall. There is no Docker daemon on the login node and user
namespaces are blocked, so neither `docker build` nor `enroot start --rw` works.
Neither is needed: an Enroot image is a squashfs, so it can be assembled from
outside with `unsquashfs`, `dpkg -x`, and `mksquashfs`. Namespaces are only needed
to run inside a rootfs, not to build one. Linking the CUDA runtime statically into
`gpu_smoke` removed the CUDA dependency entirely, which is what makes the image
**64 MB rather than the 2.2 GB a CUDA base would cost.**

If asked whether that was worth the effort: building it exposed a bug that had
been there since the code was written. The storage check passed `--time_based 0`
to fio, but that is a boolean flag, so fio read `0` as a job filename and failed.
The path had never run, because fio had never been present in any image the
validator used.

If asked whether the bandwidth is good: it is stable and symmetric, and grading it
absolutely requires a reference figure for this SKU and topology, which is why the
validator reports UNKNOWN rather than inventing a threshold.

## 2. "Can we actually train on it?" (2 min, running underneath)

Submit the final training job here and let it run for the rest of the talk. It
takes about 10 minutes, which covers everything that follows.

While it runs, explain the choices they will ask about:

**Prompt.** Standard MMLU completion format. Training and evaluation share one
encoding function so the completion boundary cannot drift. Candidates are `" A"`
through `" D"`, single tokens under the pinned tokenizer, verified against the
real tokenizer rather than assumed.

**LoRA, not full fine-tuning.** The frozen base fits on each GPU and only the
adapter moves.

**DDP, not FSDP.** A 15 GB model on a 139.8 GB card. Sharding would add all-gather
traffic on every forward pass to solve a memory problem that does not exist.

**2 epochs at lr 2e-5, measured not assumed.** At 1e-4 the training loss went to
1e-6 while held-out accuracy fell *below* the untrained model. That is the most
transferable finding here: a customer fine-tuning on a small in-house dataset will
hit exactly this.

**The split.** MMLU gives this category 170 validation questions, which cannot
adapt a 7B model. The category's own records are split instead: 1,244 for
training, 460 reserved. Nothing evaluated is ever trained on, and that is enforced
in code rather than promised.

## 3. "Does fine-tuning actually improve anything?" (2 min)

> **53.48 % to 58.70 %. Plus 5.22 points, 95 % confidence interval +1.30 to
> +9.13, McNemar p = 0.013**, on 460 questions the model never saw.

Then pre-empt the two challenges before they are asked.

**"Is it just formatting?"** No. Format adherence was **100 % before training and
100 % after**, zero unparseable outputs on either side. There was no formatting
deficit to recover, so the gain is answer selection. Generation accuracy moved
+3.91 points independently with format held at 100 %, which corroborates it.

**"Did you leak?"** 1,244 trained, 460 reserved, **intersection zero**, asserted
in code and recorded in the manifest. Base and tuned scored on identical question
IDs in identical order.

Be honest about magnitude. The interval runs +1.3 to +9.1, so call it a real but
modest gain rather than pinning it to one number.

If they ask why the gain is not larger: the category is knowledge-heavy and 1,244
examples cannot teach law. What adaptation reaches is answer-selection behaviour.
If they want larger gains, the lever is data volume, not GPUs.

## 4. "How much serving capacity do we get?" (2 min)

This is the procurement question, so give it room.

| Topology | GPUs | Goodput | Per GPU |
| --- | --- | --- | --- |
| P0, one replica | 1 | 6,109 tok/s | 6,109 |
| P1, one TP=2 replica | 2 | 10,105 tok/s | 5,052 |
| P2, two replicas | 2 | **12,279 tok/s** | **6,140** |
| P3, four replicas | 4 | **24,483 tok/s** | **6,121** |

**Replication beats tensor parallelism by 21.5 % at equal GPU count.** A 7B model
needs 15 GB and the card has 139.8 GB; splitting it buys no memory and charges
NVLink traffic on every forward pass.

**Per-GPU throughput is flat across 1, 2, and 4 GPUs**, so capacity scales
linearly and fleet sizing is division.

Add the methodological point, because it separates a careful engineer from someone
who ran a benchmark: at a fixed concurrency of 64 those two topologies were 0.6 %
apart and looked identical. The 21.5 % gap only appears once each topology is
driven to its own saturation point. Benchmarking everything at one load
under-loads the larger configurations and produces the wrong recommendation.

Reliability: **90,000 requests over 596 seconds at 80 % of peak, zero errors**, no
latency drift. Across the whole campaign no request failed.

For training reliability, which matters more on a six-month reservation: a run was
killed after step 163 of 310 and resumed from its last checkpoint. All four ranks
restarted at step 156, not step 1, and ran to completion with a valid adapter. The
interruption cost about 90 seconds of overhead, one job startup plus 8 replayed
steps. **A node failure costs the current epoch at worst, and nothing beyond it.**

## 5. "How do we run and monitor this ourselves?" (1 min)

Do not skip this. The assignment names documentation and monitoring explicitly,
and it is the section most candidates treat as an afterthought.

Show `docs/RUNBOOK.md`: exact commands in order, plus a troubleshooting table
built from failures that actually happened here.

Then show the monitoring artifacts concretely:

| Artifact | Contents |
| --- | --- |
| `gpu_monitor.csv` | Per-GPU utilisation, memory, temperature, power, clocks every 15 s |
| `metrics.jsonl` | Per-step loss, learning rate, gradient norm, step time, peak memory |
| `manifest.json` | GPU UUIDs, driver, package versions, config, exit status |
| `split_manifest.json` | Every trained and reserved question ID |

The line that lands:

> No number in this repository is typed by hand. Every figure in the results
> document is generated from these files.

## Closing, 30 seconds

> Four H200s validated the stack, the collectives, distributed training, the
> accuracy method, and a serving operating point. They cannot tell you about
> 512-GPU fabric behaviour, large-job failure rates, or checkpoint throughput at
> scale. Before committing, I would run a staged qualification on the reserved
> H100 capacity. Those steps are in the design document.

Recommend a next gate, not a verdict.

## Have ready, do not lead with

**The failures.** If asked whether anything went wrong: a learning rate that
destroyed accuracy, an NCCL log parser that reported a healthy cluster as failed,
a tokenizer schema mismatch between the training and serving images, and a soak
that would have recorded nothing. Each is a problem this customer would otherwise
hit themselves.

**The negative result.** An earlier iteration used the instruct checkpoint with
170 training examples and measured +0.85 points with a CI spanning zero. Two
changes were tested. Switching to the base checkpoint did nothing, because it is
also 100 % format-adherent, so model choice does not create headroom. Increasing
training data from 170 to 1,244 questions is what produced the significant result.
That answers "did you try the base model?" with a measurement.

## Likely questions

**Why didn't you use MMLU's official test split?** The official adaptation split
is 170 questions, which cannot adapt a 7B model; an earlier iteration proved that.
The category's own records are split instead, with a reserved holdout training
cannot reach. Comparability with published leaderboard numbers was already gone,
since this is zero-shot and published figures are 5-shot.

**Is 460 evaluation questions enough?** It is larger than most MMLU categories
provide in total, and it resolved the effect at p = 0.013. A wider interval only
matters if it fails to detect the effect, and it did not.

**Why 2 epochs and not more?** It sits in the standard 1-3 range, and more
importantly the only evidence on this data points toward less training: at a
higher rate and more epochs, held-out accuracy dropped below baseline.

**Why not TRL or FSDP?** TRL's `SFTTrainer` cannot express the ranking objective's
questions-by-candidates layout. FSDP shards a model that fits nine times over,
adding communication for no benefit.

**Are the p95 numbers exact?** For single-replica topologies yes. For P2 and P3
the pinned vLLM build emits no per-request records, so percentiles are the worst
replica rather than a pooled distribution. Conservative, and labelled in the
output.

**What happens if a node dies mid-training?** You lose the interrupted epoch at
worst. I tested it: killed a job after step 163 of 310, resumed from the
checkpoint, and all four ranks picked up at step 156 and finished normally. That
one cost about 90 seconds because it died early in the epoch. Checkpoints carry the
adapter plus optimizer and scheduler state. Worth repeating on your cluster with a
real eviction, since I simulated it with a scheduler cancellation.

**How many GPUs would we need?** On the serving side, one GPU sustains about
4,900 output tok/s inside the latency guardrails, which is 0.057 GPU-hours per
million output tokens. Give me your expected request rate and I can size the
inference fleet. Training capacity is a separate calculation and on a reservation
this size it is the larger part. `scripts/size_fleet.py` runs the arithmetic; it
assumes no GPU price, so multiply by your own rate.

**Was warm-up excluded?** No. The pinned build has no warm-up flag, so every
request is in the measurement. With 1,024 to 4,096 requests per shard the effect
is small, but the runs are not warm-up-excluded and are not described as such.

## Live demo sequence

Running the whole pipeline live takes about 45 minutes, which does not fit the
slot. Run the validator live, start training and let it run underneath the talk,
and show finished artifacts for the rest.

### Before the call

```bash
cd ~/nebius-poc && source configs/cluster.env
sinfo && squeue -u "$USER"
```

Both workers idle, nothing of yours queued. These are slow and must not be
attempted live:

```bash
./scripts/build_images_enroot.sh all              # ~15 min, one-time
.venv/bin/python scripts/prefetch_assets.py --hf-home "$HF_HOME" --out results/raw   # ~10 min, one-time
make prepare-data                                  # seconds
```

### Driving it

`scripts/demo.sh` runs each step with readable output and no Ctrl-C anywhere:

```bash
./scripts/demo.sh              # every step, pausing between them
./scripts/demo.sh validate     # or one step at a time
./scripts/demo.sh --list
```

Steps: `preflight`, `validate`, `train`, `accuracy`, `throughput`.

Live steps submit a Slurm job and show progress until it finishes on its own.
Interesting log lines are echoed as they appear so the screen is never blank.

### Timing

The tier-1 spine, at the pace the training job sets. Stretch the gaps with tier-2 material if the
slot is longer, and let questions consume it if the room is engaged.

| Time | Action |
| --- | --- |
| 0:00 | Opening, scope boundary |
| 0:30 | `./scripts/demo.sh validate`, about 1 minute |
| 1:30 | Walk the validator output |
| 2:30 | `./scripts/demo.sh train`, runs about 10 minutes underneath the rest |
| 3:00 | Phase 1 design justifications while it runs |
| 5:00 | `./scripts/demo.sh accuracy` |
| 7:00 | `./scripts/demo.sh throughput` |
| 9:00 | Back to the training job, now finished, show world size 4 |
| 9:30 | Close |

The validator and training both want all four GPUs, so keep them sequential. The
training job then fills the talk instead of being dead air you narrate around.

**If the cluster is contended**, don't wait on camera. Every display step reads
from `results/` and works identically without a live job.

**If someone opens `configs/train_ranking.yaml`** they will see the pilot starting
values rather than the locked ones. The YAML records where the pilot started, the
lock records what was selected, and the log they just watched printed
`lr=2e-05`. That is the guardrail working.

### Full reproduction, for the customer

1. `make install && make test` (local, offline)
2. `./scripts/discover_cluster.sh`, then fill `configs/cluster.env`
3. `./scripts/build_images_enroot.sh all`
4. `.venv/bin/python scripts/prefetch_assets.py`
5. `make prepare-data`
6. `sbatch slurm/validate.sbatch`
7. Two pilots, then `.venv/bin/python -m nebius_poc.recipe --lock`
8. `TRAIN_FINAL=1 RECIPE_LOCK=... sbatch slurm/train.sbatch`
9. Two evaluation jobs on the holdout, then `.venv/bin/python -m nebius_poc.report`
10. `.venv/bin/python -m nebius_poc.summarize_training`
11. `sbatch slurm/merge.sbatch`
12. Serve and benchmark each topology, then `scripts/aggregate_benchmarks.py`

Steps 1 through 5 need no GPU. Step 6 onward needs the allocation.
