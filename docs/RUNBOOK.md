# Runbook

Exact steps to reproduce, monitor, and troubleshoot the PoC. Every cluster-specific value is an
environment variable or a placeholder, so this runs against a different Nebius allocation without
editing scripts.

## Reproducing from a fresh clone

Git carries code, configs, and the curated summaries. It does not carry ~79 GB of runtime state.
Only the wheelhouse has to be rebuilt or copied; the rest can be pointed at from
`configs/cluster.env`.

```bash
git clone <repo> ~/repro && cd ~/repro
./scripts/setup.sh --check                    # what is present, what is missing
./scripts/setup.sh --from /path/to/checkout   # or --build with nothing to borrow from
```

`setup.sh` handles the venv, the config skeleton, the wheelhouse, and the verification run, and it
is idempotent, so re-running costs seconds and repeats no downloads. `--from` reuses the images,
wheelhouse and Hugging Face cache of an existing checkout on shared storage and rewrites only
`SHARED_ROOT`, which is the one value that has to belong to the new clone. `--build` imports the
images and prefetches the weights instead, which takes about 25 minutes.

It never guesses cluster-specific values. Without `--from` it copies
`configs/cluster.env.example` and leaves you to fill it in;
`./scripts/discover_cluster.sh` reads partition, account, QoS and scratch off the cluster without
writing them anywhere.

### Doing it by hand

```bash
mkdir -p containers/images                 # gitignored, so the clone has no such directory
cp -r <existing>/containers/images/wheels containers/images/wheels
cp configs/cluster.env.example configs/cluster.env
./scripts/discover_cluster.sh
make install && make test
```

Set `SHARED_ROOT` to the new clone. `HF_HOME` and the three image paths may point back at an
existing checkout. With no checkout to borrow from, rebuild instead:
`./scripts/build_images_enroot.sh all`, `./scripts/build_validator_sqsh.sh`, then
`python3 scripts/prefetch_assets.py --hf-home "$PWD/hf_cache" --out results/raw`.

The wheelhouse is the one thing that cannot be an absolute path elsewhere:
`scripts/container_python_setup.sh` resolves it at `/workspace/containers/images/wheels`, the
mounted repo root inside the container. `make test` needs no network and no GPU and passes in a
bare clone, so a failure there is the environment, not the cluster.

`results/summary/recipe_lock.json` is tracked, so the pilots need not be repeated. The final chain
is validation (34 s), training (10 m 30 s), two evaluations (3 m 34 s), merge (1 m 30 s), and the
serving sweeps (about 30 m), using the commands in the sections below.

**Expect the accuracy delta to move slightly.** The split is deterministic, so a rerun scores the
identical 460 holdout questions, but training is not bit-reproducible: seeds are set,
`torch.use_deterministic_algorithms` is not, and TF32 is on. Expect to land near +5.22 rather than
exactly on it.

### Then run it

```bash
./scripts/demo.sh              # every step, pausing between them
./scripts/demo.sh accuracy     # a single step
./scripts/demo.sh --reuse      # show the last training run instead of submitting a new one
```

Steps are `preflight`, `validate`, `train`, `accuracy`, `throughput`. `validate` and `train` submit
Slurm jobs and follow them to completion; `accuracy` and `throughput` read `results/summary/` and
need no cluster access. A failing step does not stop the ones after it. The sections below give the
underlying sbatch commands for running any stage on its own.

## Local installation

```bash
make install
make test
make lint
```

## Offline tests

`make test` requires no network, no GPU, and downloads no models. It is safe to run anywhere.
`tests/conftest.py` sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before anything imports
the Hugging Face stack, so an accidental download fails the suite instead of succeeding quietly.

## Local smoke run

Opt-in, because it is the only local step that downloads weights. It runs the whole chain
(train, evaluate base, evaluate tuned, paired comparison, adapter merge) against
`Qwen/Qwen2.5-0.5B`, which shares the tokenizer and prompt format with the 7B model:

```bash
make smoke
```

Roughly 1 GB of downloads and a few minutes on CPU. It proves the wiring only; the accuracy
numbers it prints are meaningless at that model size and sample count. Override the pieces with
`make smoke SMOKE_MODEL=... SMOKE_ROOT=... SMOKE_QUESTIONS=...`.

The 7B model is never fetched automatically. Every command above passes `--model` explicitly, and
the real base model arrives only through the asset prefetch step below.

## Container build and publish

Preferred on this cluster (no Docker daemon on the login node):

```bash
./scripts/build_images_enroot.sh all    # import stock images + download wheels
# or one of: vllm | cuda | train | wheels | gpu-smoke
```

`enroot import` works on the login node. `enroot start` / rootfs customization does **not**
(user namespaces are blocked in this jail). Jobs therefore:

- use stock PyTorch / vLLM squashfs paths in `configs/cluster.env`;
- mount the repo at `/workspace`;
- install LoRA Python deps from `containers/images/wheels` via
  `scripts/container_python_setup.sh`;
- mount host-built `containers/images/gpu_smoke` into the validator job.

If you later have a machine with a Docker daemon, you can still build the custom
Dockerfiles and `enroot import` the results for thinner images.

## Asset prefetch

Pinned revisions are recorded in `configs/train_sft.yaml` and
`configs/train_ranking.yaml`:

- model `Qwen/Qwen2.5-7B`: `d149729398750b98c0af14eb82c78cfe92750796`
- smoke model `Qwen/Qwen2.5-0.5B`: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- dataset `cais/mmlu`: `c30699e8356da336a370243923dbaf21066bb9fe`

```bash
# Resolve revisions without downloading:
.venv/bin/python scripts/prefetch_assets.py --dry-run --hf-home "$HF_HOME"

# Smoke-model boundary check (0.5B tokenizer, same prompt format as 7B):
.venv/bin/python scripts/prefetch_assets.py --smoke-only --boundary-check-only --hf-home "$HF_HOME"

# Full prefetch of 7B + 0.5B + cais/mmlu professional_law (networked, once):
.venv/bin/python scripts/prefetch_assets.py --hf-home "$HF_HOME"
```

Artifacts: `results/raw/<stamp>_prefetch_*/asset_manifest.json` and `tokenizer_boundary.json`.

**Check the boundary file before trusting anything downstream.** It records the
prompt tail and how each candidate tokenizes. On the pinned model the prompt ends
at `Answer:` and `" A"` through `" D"` map to single tokens 362, 425, 356, 422. If
a candidate tokenizes to more than one token, or the tail is not `Answer:`, stop:
training and evaluation would be scoring different things.

## Slurm discovery

```bash
./scripts/discover_cluster.sh                 # read-only, always safe
./scripts/discover_cluster.sh --with-allocation  # needs a working Slurm association
```

Output lands in `.local/discovery/<stamp>/` (gitignored), with `summary.json` and a `latest` symlink.
`configs/cluster.env` is filled only with observed values. Anything that needs a GPU allocation
(`LOCAL_SCRATCH`, NCCL/UCX interfaces, image digests) stays blank until proven.

Observed on this cluster:

- partition `earlytalent`, QoS `gpulimit`
- `/home` is NFS shared; `/tmp` is node-local ext4 and is the usable `LOCAL_SCRATCH`
- Pyxis flags present on `srun`
- `all_reduce_perf` and `all_reduce_perf_mpi` at `/usr/bin` on both workers
- 8 ConnectX-7 HCAs per node at 400 Gb/s NDR, GPUs paired NV18 over NVLink
- no Docker daemon on the login node, and Enroot cannot customise a rootfs
  because user namespaces are blocked, so only `enroot import` works

## First job: allocated GPUs per node

The smallest useful job on the cluster, and the one to run before anything else. It needs no
container, no repo layout, and no `configs/cluster.env`, so it isolates "can I schedule at all"
from every later failure mode:

```bash
sbatch --partition="$SLURM_PARTITION" ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} \
  ${SLURM_QOS:+--qos=$SLURM_QOS} slurm/gpu_info.sbatch
```

It confirms the account, partition and QoS are usable, that both nodes schedule, and that each
task sees the GPUs it was given. Expected output, one block per node:

```
=== worker-0 ===
CUDA_VISIBLE_DEVICES=0,1
0, GPU-b7e1a88f-..., NVIDIA H200, 580.159.04, 143771 MiB
1, GPU-c0782e0c-..., NVIDIA H200, 580.159.04, 143771 MiB
```

Four distinct UUIDs across two hosts is the pass condition. Repeated UUIDs mean the GPU request
shape is wrong, which is why the `srun` inside asks for `--gpus-per-node=2` rather than
`--gpus-per-task=1`.

## Shared cluster etiquette

Other users may be on this cluster, so every job declares bounded resources rather than taking a
node by default:

- **GPUs.** Nothing exceeds the 4-GPU allocation. `train`, `validate` and `gpu_info` take 2 nodes
  at 2 GPUs each; `evaluate`, `merge` and the smoke jobs take 1.
- **Memory.** `serve.sbatch` and `benchmark.sbatch` set `--mem` explicitly (256G and 32G). Without
  it a vLLM server reserves the node's whole memory and the benchmark that needs to attach sits in
  `PENDING (Resources)` behind it.
- **CPU and time.** `--cpus-per-task` is 2 to 16, never the whole node, and every job has a
  `--time` ceiling from 5 minutes to 4 hours, so nothing can hold an allocation indefinitely.
- **Storage.** Transient I/O goes to node-local `/tmp` via `LOCAL_SCRATCH`, not to shared `/home`.
- **Serve jobs do not self-exit.** They hold GPUs until cancelled. Cancel them as soon as the
  benchmark finishes.

## Cluster validation

Offline parser and aggregation tests (no GPU, no Slurm):

```bash
python -m validator.cluster_validate --help
make test   # includes tests/test_nccl_parser.py and tests/test_cluster_validate.py
```

On the cluster, after `configs/cluster.env` is filled:

```bash
mkdir -p logs
sbatch --partition="$SLURM_PARTITION" \
  ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
  ${SLURM_QOS:+--qos="$SLURM_QOS"} \
  slurm/validate.sbatch
```

Artifacts land under `$SHARED_ROOT/results/raw/<run_id>/` including `summary.json` and `report.md`.
Absolute NCCL/GPU bandwidth thresholds stay `UNKNOWN` unless you set them from a vendor figure or
an approved reference run on the same platform. The validator will never invent one.

## Data splits

`make prepare-data` writes the split manifest and the augmentation audit. Read the
audit before locking a recipe: it lists every question whose choices were left
unpermuted because they reference other options by label, and pattern matching
reduces that risk without eliminating it.

The category's 1,709 records partition as 1,244 into the adaptation pool
(170 validation plus 70 % of test) and 460 reserved as the evaluation holdout. The
partition is deterministic in the stable question IDs and stratified by answer
label. Training cannot reach the holdout: `load_adaptation_pool` carves it out and
never returns it, and `split_manifest` raises if a reserved ID appears in the pool.

To audit disjointness on any finished run:

```bash
python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
t=set(m['trained_question_ids']); h=set(m['evaluation_holdout_ids'])
print('trained',len(t),'reserved',len(h),'intersection',len(t&h))
" results/raw/<train_run>/split_manifest.json
```

## Pilot training

Two runs on the pilot split (adaptation pool minus the 100-question internal set),
identical except for the objective:

```bash
# Local / container CLI
.venv/bin/python -m nebius_poc.train --config configs/train_sft.yaml
.venv/bin/python -m nebius_poc.train --config configs/train_ranking.yaml

# Multi-node Slurm (one objective per job)
TRAIN_CONFIG=configs/train_sft.yaml sbatch --partition="$SLURM_PARTITION" \
  ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} ${SLURM_QOS:+--qos="$SLURM_QOS"} \
  slurm/train.sbatch
TRAIN_CONFIG=configs/train_ranking.yaml sbatch ... slurm/train.sbatch
```

Each writes `results/raw/<run_id>/` containing `manifest.json`, `split_manifest.json`,
`metrics.jsonl`, retained checkpoints, and the final `adapter/`.

Multi-node runs go through `slurm/train.sbatch` and `torchrun`; the CLI reads `RANK`,
`WORLD_SIZE`, and `LOCAL_RANK` from the environment and needs no extra flags.

To continue after a preemption, point at a retained checkpoint:

```bash
.venv/bin/python -m nebius_poc.train --config configs/train_sft.yaml \
  --resume-from results/raw/<run_id>/checkpoint-<step>
```

Checkpoints are written at epoch boundaries, so a resume replays from the start of the interrupted
epoch rather than mid-epoch.

Before locking, profile sequence lengths and score both pilots plus the untrained
base on the internal 100 only. Use `--split pool`, because the internal set spans
both source splits and cannot be filtered out of just one of them:

```bash
.venv/bin/python -m nebius_poc.profile --config configs/train_ranking.yaml

# on the cluster, one job each
EVAL_LABEL=pilot-base EVAL_SPLIT=pool EVAL_MODE=forced_choice \
  IDS_FILE=results/raw/<train_run>/split_manifest.json \
  sbatch --partition="$SLURM_PARTITION" slurm/evaluate.sbatch

EVAL_LABEL=pilot-rank EVAL_SPLIT=pool EVAL_MODE=forced_choice \
  IDS_FILE=results/raw/<train_run>/split_manifest.json \
  EVAL_ADAPTER=$SHARED_ROOT/results/raw/<rank_run>/adapter \
  sbatch --partition="$SLURM_PARTITION" slurm/evaluate.sbatch
```

## Recipe locking

Compare the two internal forced-choice runs. The primary metric is declared with
`--select-on` and recorded in the lock. Use `forced_choice_accuracy` when the
internal set is large enough to resolve single questions, which is the metric the
final comparison reports; use `mean_gold_choice_nll` when it is small enough that
accuracy moves in coarse steps. The two can disagree, so the choice is explicit
rather than implied. This never loads the evaluation holdout.

```bash
.venv/bin/python -m nebius_poc.recipe \
  --candidate sft configs/train_sft.yaml <sft_forced_choice.jsonl> \
  --candidate ranking configs/train_ranking.yaml <rank_forced_choice.jsonl> \
  --select-on forced_choice_accuracy \
  --lock --out results/summary/recipe_lock.json
```

Do not evaluate on the holdout until after this file exists, and evaluate it once.

## Final training

Requires `results/summary/recipe_lock.json` from the step above. Trains on all
1,244 adaptation-pool questions, never the reserved holdout, with the locked
objective and hyperparameters overlaid onto the config:

```bash
# Direct CLI
.venv/bin/python -m nebius_poc.train --config configs/train_sft.yaml \
  --final --recipe-lock results/summary/recipe_lock.json

# Multi-node Slurm (2×2). TRAIN_FINAL=1 refuses to start without RECIPE_LOCK.
TRAIN_FINAL=1 RECIPE_LOCK=results/summary/recipe_lock.json \
  TRAIN_CONFIG=configs/train_sft.yaml \
  sbatch --partition="$SLURM_PARTITION" \
    ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
    ${SLURM_QOS:+--qos="$SLURM_QOS"} \
    slurm/train.sbatch
```

`split_manifest.json` still records the pilot 150/20 IDs for audit, and adds
`training_mode=final` plus `trained_question_ids` covering the full pool. Do not look at
official test evaluation until this run finishes and the adapter is saved.

## Evaluation

Score the base and the tuned model on the reserved holdout, then pair the results.
`--split` defaults to `holdout`, which reproduces the same partition from the same
seed the training loader used to carve it out.

```bash
.venv/bin/python -m nebius_poc.report \
  --base results/raw/<base_run_id>/forced_choice.jsonl \
  --tuned results/raw/<tuned_run_id>/forced_choice.jsonl \
  --base-generation results/raw/<base_run_id>/generation.jsonl \
  --tuned-generation results/raw/<tuned_run_id>/generation.jsonl \
  --out results/summary/accuracy.json
```

Training cannot reach the holdout: `load_adaptation_pool` partitions the records
and returns only the trainable share, and it is the only loader `train.py`
imports.

On the cluster, run base and tuned as two jobs. They are independent, so putting
them on separate nodes halves wall time. `EVAL_BATCH_SIZE` matters: scoring runs
without gradients, so 16 roughly halves the runtime against the default of 8.

```bash
EVAL_LABEL=final-base EVAL_SPLIT=holdout EVAL_MODE=both EVAL_BATCH_SIZE=16 \
  EVAL_CONFIG=configs/train_ranking.yaml \
  sbatch --partition="$SLURM_PARTITION" slurm/evaluate.sbatch

EVAL_LABEL=final-tuned EVAL_SPLIT=holdout EVAL_MODE=both EVAL_BATCH_SIZE=16 \
  EVAL_CONFIG=configs/train_ranking.yaml \
  EVAL_ADAPTER=$SHARED_ROOT/results/raw/<train_run>/adapter \
  sbatch --partition="$SLURM_PARTITION" slurm/evaluate.sbatch
```

Confirm the comparison is genuinely paired before trusting it:

```bash
python3 -c "
import json,sys
def ids(p): return [json.loads(l)['question_id'] for l in open(p)]
b,t=ids(sys.argv[1]),ids(sys.argv[2])
print(len(b),len(t),'identical ids and order:',b==t)
" results/raw/<base_run>/forced_choice.jsonl results/raw/<tuned_run>/forced_choice.jsonl
```

Then write the tracked training summary so the results document can be generated
rather than typed:

```bash
.venv/bin/python -m nebius_poc.summarize_training \
  --run-dir results/raw/<train_run_id> \
  --out results/summary/training.json
```

## Adapter merge

```bash
.venv/bin/python -m nebius_poc.merge_adapter --config configs/train_sft.yaml \
  --adapter results/raw/<train_run_id>/adapter --output <shared_root>/models/tuned-merged
```

On the cluster (one GPU, uses `TRAIN_IMAGE`):

```bash
ADAPTER=$SHARED_ROOT/results/raw/<train_run>/adapter \
  MERGE_OUT=$SHARED_ROOT/models/tuned-merged \
  sbatch --partition="$SLURM_PARTITION" \
    ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
    ${SLURM_QOS:+--qos="$SLURM_QOS"} \
    slurm/merge.sbatch
```

Writes checksums for both the adapter and the merged weights, and runs one greedy generation so a
broken merge surfaces here rather than inside the vLLM job.

## vLLM launch

Merge first, then serve one topology at a time. Match Slurm resources to the topology:

```bash
# P0: 1 GPU  |  P1/P2: 2 GPUs on 1 node  |  P3: 2 nodes × 2 GPUs
TOPOLOGY=P0 MODEL_PATH=<merged_model> \
  sbatch --nodes=1 --gpus-per-node=1 --partition="$SLURM_PARTITION" slurm/serve.sbatch
```

The serve job writes `endpoints.json` under `results/raw/<serve_run>/` and sleeps until
`scancel`. Topologies are defined in `configs/serve_topologies.yaml`.

## Benchmarking

Always derive `SERVE_RUN` from the job id rather than retyping the timestamp. The
run directory is stamped at job start, not at submit time, and guessing it wastes
an allocation:

```bash
SERVE_RUN=$(ls -d "$SHARED_ROOT"/results/raw/*serve-P2_job<jobid> | head -1)
```

Screening sweep. `CONCURRENCY` accepts a comma-separated list and overrides
`configs/benchmark.yaml`:

```bash
SERVE_RUN="$SERVE_RUN" STAGE=screen CONCURRENCY="64,128,256,512" \
  sbatch --partition="$SLURM_PARTITION" --nodes=1 --gpus-per-node=0 \
  slurm/benchmark.sbatch
```

Sweep each topology until it fails a guardrail, not to a shared fixed load. A
fixed concurrency under-loads the larger topologies: with four replicas, a total
of 64 is only 16 per replica, and the comparison measures idle time instead of
capacity. This changes conclusions, not just numbers.

Final stage, three repetitions at the chosen operating point:

```bash
SERVE_RUN="$SERVE_RUN" STAGE=final CONCURRENCY=256 sbatch ... slurm/benchmark.sbatch
```

Soak. `SOAK_PROMPTS` is the fleet-wide request count and must be sized so the run
finishes on its own, because the client only writes results at the end. Estimate
it as `target_tok_s / output_tokens * seconds`:

```bash
SERVE_RUN="$SERVE_RUN" STAGE=soak CONCURRENCY=256 SOAK_PROMPTS=85620 \
  sbatch ... slurm/benchmark.sbatch
```

Aggregate. Pass `--bench-root` once per bench run to build the consolidated
summary:

```bash
.venv/bin/python scripts/aggregate_benchmarks.py \
  --bench-root results/raw/<p0_screen> \
  --bench-root results/raw/<p1_screen> \
  --bench-root results/raw/<p2_final> \
  --bench-root results/raw/<p3_final> \
  --bench-root results/raw/<soak> \
  --out results/summary/inference.json
```

Fleet p95 is recomputed from per-request records when the vLLM build emits them.
vLLM 0.8.5 does not, so counts and throughput are summed exactly and percentiles
fall back to the worst replica. Endpoint p95 values are never averaged, and every
approximated point is labelled in the output.

## Monitoring

The training sbatch scripts background `scripts/monitor_gpus.sh` against the job's result
directory (`gpu_monitor.csv`, 15 s in `train.sbatch` and 10 s in `train_smoke.sbatch`). It samples
only GPUs visible in the allocation via `nvidia-smi`. Stop is handled by the job's EXIT trap.

The serve and benchmark jobs do not sample GPUs. Throughput and latency come from the load
generator, and per-GPU telemetry was not needed to interpret those runs. If you want it for a
serving campaign, background the same script from `slurm/serve.sbatch` the way `train.sbatch`
does, or watch the node directly:

```bash
srun --overlap --jobid <serve_jobid> nvidia-smi
```

Rank-zero training metrics land in `metrics.jsonl` (step, loss, learning rate, grad norm, step
time, samples/s, peak GPU bytes). Use Nebius Grafana / NCCL Inspector when available; they are
not required to interpret a run.

Live Slurm checks while a job is up:

```bash
squeue -u "$USER"
scontrol show job <jobid>
tail -f logs/<jobname>-<jobid>.out
```

## Cancellation and cleanup

```bash
scancel <jobid>                 # one job
scancel -u "$USER"              # everything you own
# Serve jobs sleep until cancelled; that is intentional so a benchmark can attach.
```

Artifacts under `$SHARED_ROOT/results/raw/` and `$HF_HOME` are intentionally retained across jobs.
Delete a specific raw run only after its summary has been copied into `results/summary/` (or you
are sure you do not need it). Do not wipe `HF_HOME` mid-campaign, because offline jobs expect the
prefetch cache.

## Common failures and diagnosis

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `Invalid account or account/partition combination` | No Slurm association for the user | `sacctmgr show assoc user=$USER`; ask admin to add Account/QoS |
| `TRAIN_IMAGE` / `VALIDATOR_IMAGE` unset | Images not imported | Build elsewhere, `enroot import`, set paths in `configs/cluster.env` |
| `TRAIN_FINAL=1 requires RECIPE_LOCK=...` | Final train without lock | Run `nebius_poc.recipe --lock` first |
| `--final requires --recipe-lock` | Same, CLI path | Pass `--recipe-lock results/summary/recipe_lock.json` |
| Hugging Face download mid-job | Prefetch incomplete or offline flags off | Re-run `scripts/prefetch_assets.py`; export `HF_HUB_OFFLINE=1` |
| Pyxis cannot pull image | No registry auth / wrong ref | Prefer a local `.sqsh` via enroot import |
| NCCL hang / asymmetric bandwidth | Fabric or interface mismatch | Re-run validator; confirm `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` from discovery |
| Serve job up but bench cannot connect | Wrong `SERVE_RUN` or endpoints not ready | Wait for `endpoints.json`; curl one endpoint before `sbatch` bench |
| Merge OOM on login CPU | Merge needs a GPU allocation | Use `slurm/merge.sbatch`, not a login-node process |
| Benchmark job stuck in `PENDING (Resources)` while the serve job runs | Serve reserved the node's full memory | Both scripts now set an explicit `--mem`. Check `scontrol show job <serve_id>` for `AllocTRES` |
| `ModuleNotFoundError: No module named 'yaml'` in an sbatch script | Compute-node system python has no pyyaml | The scripts prefer `.venv/bin/python` on shared storage. Run `make install` |
| `python: command not found` inside the vLLM container | The official image ships `python3` only | Resolved at runtime by `scripts/launch_vllm_replicas.sh` |
| `'list' object has no attribute 'keys'` when vLLM loads the merged model | transformers version skew between training and serving images | The merge copies the base model's own tokenizer files instead of re-serialising them |
| `vllm: error: unrecognized arguments: --backend` | Benchmark flags differ across vLLM releases | `run_vllm_bench.sh` probes `--help` and only passes supported flags |
| Benchmark returns `Not Found` on every request | Doubled `/v1` in the URL | `--base-url` takes the server root; the script strips a trailing `/v1` from `endpoints.json` |
| Soak produces load but no `bench.json` | Client only writes results on completion | Size `SOAK_PROMPTS` so the run finishes inside the window |
| CUDA OOM on the ranking objective | Four candidates per question multiply activations | Lower `per_device_batch_size` and raise `gradient_accumulation_steps` to keep the effective batch |
| Training loss near zero, held-out accuracy below base | Learning rate too high for the data volume | Drop the rate before adding data. See the pilot sweep in `results/summary/pilot_comparison.json` |
| Fine-tuning shows no measurable accuracy gain | Too little training data, not a broken recipe | Check the adaptation pool size. 170 examples measured +0.85 pp with a CI spanning zero; 1,244 measured +5.22 pp at p = 0.013 |
| `adaptation pool must start from the 'validation' split` | Passing `test` to the training loader | The loader partitions `test` itself. Pass `validation` and set `trainable_split` in the config |
| `N evaluation questions appear in the adaptation pool` | Holdout seed or fraction changed between training and evaluation | Both sides derive the partition from `dataset.seed` and `holdout_fraction`. Keep them identical |
| `ids not found in split` during pilot evaluation | Internal set spans both source splits | Use `EVAL_SPLIT=pool`, not `validation` |
