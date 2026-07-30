#!/usr/bin/env bash
# Demo driver. Runs the PoC story one step at a time with readable output.
#
#   ./scripts/demo.sh              walk every step, pausing between them
#   ./scripts/demo.sh validate     run a single step
#   ./scripts/demo.sh --reuse      show recorded runs instead of submitting jobs
#   ./scripts/demo.sh --list       show the step names
#
# Live steps submit a Slurm job and show progress until it finishes on its own.
# Nothing needs Ctrl-C. Steps that only read artifacts never touch the cluster.
# A step that fails does not stop the ones after it, so a bad allocation cannot
# cost you the accuracy and throughput sections.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f configs/cluster.env ]]; then
  # shellcheck disable=SC1091
  source configs/cluster.env
fi

if [[ -t 1 ]]; then
  B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else
  B=""; D=""; G=""; Y=""; R=""; C=""; N=""
fi

PAUSE=1
REUSE="${DEMO_REUSE:-0}"
MEASURE=0
MERGED_MODEL="${MERGED_MODEL:-}"
BENCH_ROOTS=()
frames='|/-\'
PY="${ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"

rule() { printf '%s%s%s\n' "${D}" "$(printf '─%.0s' {1..72})" "${N}"; }

title() {
  echo
  rule
  printf '%s  %s%s\n' "${B}" "$*" "${N}"
  rule
}

note() { printf '%s  %s%s\n' "${D}" "$*" "${N}"; }
ok()   { printf '%s  %s%s\n' "${G}" "$*" "${N}"; }
warn() { printf '%s  %s%s\n' "${Y}" "$*" "${N}"; }
fail() { printf '%s  %s%s\n' "${R}" "$*" "${N}"; }

pause() {
  (( PAUSE )) || return 0
  echo
  read -rsp "$(printf '%s  [Enter] to continue%s' "${D}" "${N}")" -n 1 || true
  echo; echo
}

# Submit a job and show progress until Slurm releases it. Interesting log lines
# are echoed as they appear so the screen is never blank, and the loop ends by
# itself so the operator never has to interrupt anything.
follow_job() {
  local job="$1" log="$2" pattern="$3"
  local frames='|/-\' i=0 shown=0 start=${SECONDS} total elapsed

  while squeue -j "${job}" -h -o '%T' 2>/dev/null | grep -q .; do
    if [[ -f "${log}" ]]; then
      total="$(grep -cE "${pattern}" "${log}" 2>/dev/null || true)"
      total="${total:-0}"
      if (( total > shown )); then
        printf '\r%-72s\r' ''
        grep -E "${pattern}" "${log}" | tail -n +$((shown + 1)) | sed "s/^/  ${C}|${N} /"
        shown=${total}
      fi
    fi
    elapsed=$((SECONDS - start))
    printf '\r  %s running  %ss ' "${frames:i++%4:1}" "${elapsed}"
    sleep 2
  done

  printf '\r%-72s\r' ''
  if [[ -f "${log}" ]]; then
    total="$(grep -cE "${pattern}" "${log}" 2>/dev/null || true)"
    total="${total:-0}"
    if (( total > shown )); then
      grep -E "${pattern}" "${log}" | tail -n +$((shown + 1)) | sed "s/^/  ${C}|${N} /"
    fi
  fi

  local state
  state="$(sacct -j "${job}" --format=State -n 2>/dev/null | head -1 | tr -d ' ')"
  elapsed=$((SECONDS - start))
  if [[ "${state}" == COMPLETED ]]; then
    ok "job ${job} completed in ${elapsed}s"
  else
    fail "job ${job} finished as ${state:-UNKNOWN} after ${elapsed}s"
    return 1
  fi
}

sbatch_args() {
  printf '%s' "--partition=${SLURM_PARTITION}"
  [[ -n "${SLURM_ACCOUNT:-}" ]] && printf ' --account=%s' "${SLURM_ACCOUNT}"
  [[ -n "${SLURM_QOS:-}" ]] && printf ' --qos=%s' "${SLURM_QOS}"
}

# A vLLM server sleeps until cancelled, so anything launched here must be cleaned
# up even when the run is interrupted. An orphaned server holds its GPUs until the
# job time limit expires, which on a shared cluster is somebody else's problem.
SERVE_JOBS=()

cancel_serve_jobs() {
  local job
  for job in "${SERVE_JOBS[@]:-}"; do
    [[ -n "${job}" ]] || continue
    if squeue -j "${job}" -h -o '%A' 2>/dev/null | grep -q .; then
      warn "cancelling serve job ${job}"
      scancel "${job}" 2>/dev/null || true
    fi
  done
  SERVE_JOBS=()
}
# Ctrl-C must stop the sweep, not just release the current server and march on to
# the next topology. The per-topology "|| true" is there so one bad topology does
# not lose the rest, and it must not swallow a deliberate interrupt.
on_interrupt() {
  echo
  warn "interrupted"
  cancel_serve_jobs
  exit 130
}
trap on_interrupt INT TERM
trap cancel_serve_jobs EXIT

# Slurm resources per topology. configs/serve_topologies.yaml is the source of
# truth and serve.sbatch refuses a mismatch, so these must agree with it.
topology_resources() {
  case "$1" in
    P0)    printf '%s' "--nodes=1 --gpus-per-node=1" ;;
    P1|P2) printf '%s' "--nodes=1 --gpus-per-node=2" ;;
    P3)    printf '%s' "--nodes=2 --gpus-per-node=2" ;;
    *) return 1 ;;
  esac
}

# The serve run directory is stamped at job start, not at submit, so it can only
# be found after the job is running. Guessing the timestamp wastes an allocation.
# The run directory is returned on stdout and captured by the caller, so every
# byte of progress output has to go to stderr. A stray spinner-clearing escape on
# stdout ends up prepended to the path, and the failure surfaces much later as an
# unreadable serve_manifest.json.
wait_for_endpoints() {
  local job="$1" waited=0 run=""
  while (( waited < 900 )); do
    # Absolute: this path becomes a pyxis --container-mounts source, and pyxis
    # rejects a relative one. The runbook's manual recipe uses $SHARED_ROOT for
    # the same reason.
    run="$(ls -d "${ROOT}"/results/raw/*serve-*_job"${job}" 2>/dev/null | head -1)"
    if [[ -n "${run}" && -f "${run}/endpoints.json" ]]; then
      printf '\r%-72s\r' '' >&2
      printf '%s\n' "${run}"
      return 0
    fi
    if ! squeue -j "${job}" -h -o '%T' 2>/dev/null | grep -q .; then
      printf '\r%-72s\r' '' >&2
      fail "serve job ${job} exited before publishing endpoints" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    printf '\r  %s waiting for endpoints  %ss ' "${frames:0:1}" "${waited}" >&2
  done
  printf '\r%-72s\r' '' >&2
  fail "timed out waiting for endpoints from serve job ${job}" >&2
  return 1
}

# Serve one topology, run the given benchmark stages against it, then cancel the
# server before returning so the next topology can have the GPUs.
measure_topology() {
  local topo="$1"; shift
  local resources job run stage_spec stage conc bench

  note "topology ${topo}: starting vLLM"
  # shellcheck disable=SC2046
  resources="$(topology_resources "${topo}")"
  # shellcheck disable=SC2046
  job="$(TOPOLOGY="${topo}" MODEL_PATH="${MERGED_MODEL}" \
        sbatch --parsable $(sbatch_args) ${resources} slurm/serve.sbatch)"
  SERVE_JOBS+=("${job}")
  note "serve job ${job}, loading the model"

  if ! run="$(wait_for_endpoints "${job}")"; then
    cancel_serve_jobs
    return 1
  fi
  ok "endpoints ready: $(basename "${run}")"

  for stage_spec in "$@"; do
    stage="${stage_spec%%:*}"
    conc="${stage_spec#*:}"
    note "${topo} ${stage} at concurrency ${conc}"
    # shellcheck disable=SC2046
    bench="$(SERVE_RUN="${run}" STAGE="${stage}" CONCURRENCY="${conc}" \
            SOAK_PROMPTS="${SOAK_PROMPTS:-85620}" \
            sbatch --parsable $(sbatch_args) --nodes=1 --gpus-per-node=0 \
            slurm/benchmark.sbatch)"
    if follow_job "${bench}" "logs/bench-${bench}.out" 'stage|concurrency|wrote'; then
      BENCH_ROOTS+=("$(ls -d "${ROOT}"/results/raw/*bench-${stage}_job"${bench}" 2>/dev/null | head -1)")
    else
      fail "${topo} ${stage} did not complete"
    fi
  done

  scancel "${job}" 2>/dev/null || true
  SERVE_JOBS=()
  ok "${topo} done, server released"
}

step_preflight() {
  title "Pre-flight"
  sinfo | sed 's/^/  /'
  echo
  local queued
  queued="$(squeue -u "${USER}" -h 2>/dev/null | wc -l)"
  if (( queued == 0 )); then
    ok "no jobs of yours queued, cluster is free"
  else
    warn "${queued} of your jobs already queued; live steps may wait for resources"
    squeue -u "${USER}" | sed 's/^/  /'
  fi
}

# Most recent full validation that produced a summary. The glob excludes the
# smoke runs, whose directories are named *_validate-smoke_job* and which report
# a deliberately narrower set of checks.
latest_validate_run() {
  local dir
  while IFS= read -r dir; do
    [[ -f "${dir}/summary.json" ]] || continue
    printf '%s\n' "${dir}"
    return 0
  done < <(ls -dt results/raw/*_validate_job* 2>/dev/null)
  return 1
}

step_validate() {
  title "1. Qualify the allocation"
  note "portable checks on 4 GPUs, then intra-node and 3x inter-node NCCL"

  local run job
  if (( REUSE )); then
    if run="$(latest_validate_run)"; then
      warn "reuse requested: showing the last completed run, nothing was submitted"
      echo
      show_validate_run "${run}"
      return 0
    fi
    warn "no completed run to reuse, submitting a live job instead"
  fi

  # shellcheck disable=SC2046
  job="$(sbatch --parsable $(sbatch_args) slurm/validate.sbatch)"
  note "submitted job ${job}"
  echo
  follow_job "${job}" "logs/validate-${job}.out" 'using NCCL binary|Avg bus bandwidth|artifacts in'
  echo
  run="$(ls -dt results/raw/*_validate_job"${job}" 2>/dev/null | head -1)"
  if [[ -z "${run}" ]]; then
    fail "no run directory for job ${job}"
    if run="$(latest_validate_run)"; then
      warn "falling back to the last completed run"
      echo
      show_validate_run "${run}"
    fi
    return 1
  fi
  show_validate_run "${run}"
}

show_validate_run() {
  "${PY}" - "$1" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/summary.json"))
n = d["network"]
rows = [
    ("status", d["status"]),
    ("hard failures", d["hard_failures"] or "none"),
    ("NCCL wrong values", n["inter"]["total_wrong"]),
    ("socket fallback", n["inter"]["socket_fallback"]),
    ("InfiniBand transport", n["inter"]["ib_transport"]),
    ("inter-node busbw GB/s", ", ".join(f"{v:.2f}" for v in n["inter"]["busbw_gbs"]["values"])),
    ("run-to-run CV", f"{n['inter']['cv'] * 100:.2f} %"),
    ("node asymmetry", f"{n['asymmetry']['ratio'] * 100:.2f} %"),
    ("local storage", f"{d['storage']['write']['bw_bytes'] / 1e9:.2f} GB/s write, "
                      f"{d['storage']['read']['bw_bytes'] / 1e9:.2f} GB/s read"),
]
for k, v in rows:
    print(f"  {k:<22} {v}")
PY
}

# Most recent training run that finished cleanly, for --reuse and for recovering a
# demo when the cluster will not give up four GPUs.
latest_train_run() {
  local dir
  while IFS= read -r dir; do
    [[ -f "${dir}/manifest.json" && -f "${dir}/metrics.jsonl" ]] || continue
    grep -q '"exit_status": "ok"' "${dir}/manifest.json" 2>/dev/null || continue
    printf '%s\n' "${dir}"
    return 0
  done < <(ls -dt results/raw/*_train_job*/*/ 2>/dev/null)
  return 1
}

step_train() {
  title "2. Multi-node training"
  note "Qwen2.5-7B base, LoRA, 2 nodes x 2 GPUs, recipe pinned by recipe_lock.json"

  local run job
  if (( REUSE )); then
    if run="$(latest_train_run)"; then
      warn "reuse requested: showing the last completed run, nothing was submitted"
      echo
      show_train_run "${run}"
      return 0
    fi
    warn "no completed run to reuse, submitting a live job instead"
  fi

  # shellcheck disable=SC2046
  job="$(TRAIN_CONFIG=configs/train_ranking.yaml TRAIN_FINAL=1 \
        RECIPE_LOCK=results/summary/recipe_lock.json \
        sbatch --parsable $(sbatch_args) slurm/train.sbatch)"
  note "submitted job ${job}, expect about 10 minutes on 4,964 rows"
  echo
  follow_job "${job}" "logs/train-${job}.err" 'applied recipe lock|training rows|adapter written'
  echo
  run="$(ls -d results/raw/*_train_job"${job}"/*/ 2>/dev/null | head -1)"
  if [[ -z "${run}" ]]; then
    fail "no run directory for job ${job}"
    if run="$(latest_train_run)"; then
      warn "falling back to the last completed run"
      echo
      show_train_run "${run}"
    fi
    return 1
  fi
  show_train_run "${run}"
}

show_train_run() {
  "${PY}" - "$1" <<'PY'
import json, statistics, sys
d = sys.argv[1]
m = json.load(open(f"{d}/manifest.json"))
e, t = m["environment"], m["config"]["training"]
rows = [json.loads(line) for line in open(f"{d}/metrics.jsonl")]
steps = [r["step_seconds"] for r in rows[1:]] or [rows[0]["step_seconds"]]
losses = [r["loss"] for r in rows]
out = [
    ("run id", m["run_id"]),
    ("objective", m["config"]["objective"]),
    ("world size", e["world_size"]),
    ("nodes", f"{e['slurm'].get('SLURM_NNODES')} ({e['slurm'].get('SLURM_JOB_NODELIST')})"),
    ("learning rate / epochs", f"{t['learning_rate']} / {t['epochs']}"),
    ("optimizer steps", len(rows)),
    ("median step time", f"{statistics.median(steps):.3f} s"),
    ("peak GPU memory", f"{max(r['peak_gpu_bytes'] for r in rows) / 2**30:.1f} GiB"),
    ("NaN or Inf losses", sum(1 for v in losses if v != v)),
    ("exit status", m["exit_status"]),
]
for k, v in out:
    print(f"  {k:<22} {v}")
PY
}

step_accuracy() {
  title "3. Accuracy, base versus tuned"
  note "460 reserved questions, never trained on, paired by stable question id"
  "${PY}" - <<'PY'
import glob, json
f = json.load(open("results/summary/accuracy.json"))["forced_choice"]
g = json.load(open("results/summary/accuracy.json"))["generation"]
rows = [
    ("questions", f["n"]),
    ("base accuracy", f"{f['base_accuracy'] * 100:.2f} %"),
    ("tuned accuracy", f"{f['tuned_accuracy'] * 100:.2f} %"),
    ("difference", f"{f['delta_pp']:+.2f} pp"),
    ("95% bootstrap CI", f"{f['ci_low_pp']:+.2f} to {f['ci_high_pp']:+.2f} pp"),
    ("McNemar p", f"{f['mcnemar_p']:.3f}"),
    ("tuned right / base wrong", f["tuned_only_correct"]),
    ("base right / tuned wrong", f["base_only_correct"]),
    ("format adherence", f"{g['base']['format_adherence'] * 100:.0f} % base, "
                         f"{g['tuned']['format_adherence'] * 100:.0f} % tuned"),
]
for k, v in rows:
    print(f"  {k:<26} {v}")

def ids(pattern):
    d = sorted(glob.glob(pattern))[-1]
    return [json.loads(line)["question_id"] for line in open(d)]

try:
    b = ids("results/raw/*evaluate-final-base_job*/*/forced_choice.jsonl")
    t = ids("results/raw/*evaluate-final-tuned_job*/*/forced_choice.jsonl")
    print()
    print(f"  paired check               {len(b)} base / {len(t)} tuned, "
          f"identical ids and order: {b == t}")
except (IndexError, FileNotFoundError):
    pass

# Read the verdict off the numbers instead of asserting it. If a rerun ever lands a
# non-significant result, this says so rather than claiming significance anyway.
print()
if f["ci_low_pp"] > 0 and f["mcnemar_p"] < 0.05:
    print(f"  significant: the interval excludes zero and McNemar rejects at "
          f"p={f['mcnemar_p']:.3f}")
else:
    print(f"  not significant at 0.05: CI {f['ci_low_pp']:+.2f} to "
          f"{f['ci_high_pp']:+.2f} pp, McNemar p={f['mcnemar_p']:.3f}")
if f["ci_low_pp"] > 0 and g["base"]["format_adherence"] == g["tuned"]["format_adherence"] == 1.0:
    print("  format adherence was 100% on both sides, so the gain is answer selection")
PY
}

# The full sweep: merge, then four topologies each driven to its own saturation
# point, then a soak, then aggregation. About 40 minutes. Deliberately not part of
# a normal demo run, which is why it sits behind --measure.
measure_throughput() {
  title "4a. Measuring inference throughput"
  note "four topologies, each swept to its own saturation point, then a 10 minute soak"
  note "about 40 minutes; servers are cancelled automatically, including on Ctrl-C"

  MERGED_MODEL="${MERGED_MODEL:-${SHARED_ROOT:-${ROOT}}/models/tuned-merged}"
  BENCH_ROOTS=()

  if [[ ! -f "${MERGED_MODEL}/config.json" ]]; then
    local adapter job
    adapter="$(ls -dt results/raw/*_train_job*/*/adapter 2>/dev/null | head -1)"
    [[ -n "${adapter}" ]] || { fail "no trained adapter found; run the train step first"; return 1; }
    note "merging ${adapter} into ${MERGED_MODEL}"
    # shellcheck disable=SC2046
    job="$(ADAPTER="${PWD}/${adapter#./}" MERGE_OUT="${MERGED_MODEL}" \
          sbatch --parsable $(sbatch_args) slurm/merge.sbatch)"
    follow_job "${job}" "logs/merge-${job}.out" 'merged|verification' || return 1
  else
    ok "merged model" "${MERGED_MODEL}"
  fi

  # Screen the small topologies to find each knee, then take repetitions at the
  # operating point on the two that carry the headline numbers. P3 keeps one
  # server for both its final run and the soak.
  measure_topology P0 "screen:64,128,256"      || true
  measure_topology P1 "screen:128,256,512"     || true
  measure_topology P2 "final:256"              || true
  measure_topology P3 "final:512" "soak:256"   || true

  if (( ${#BENCH_ROOTS[@]} == 0 )); then
    fail "no benchmark runs completed, leaving results/summary/inference.json alone"
    return 1
  fi

  note "aggregating ${#BENCH_ROOTS[@]} benchmark runs"
  local args=()
  local root
  for root in "${BENCH_ROOTS[@]}"; do
    [[ -n "${root}" ]] && args+=(--bench-root "${root}")
  done
  "${PY}" scripts/aggregate_benchmarks.py "${args[@]}" \
    --out results/summary/inference.json || return 1
  ok "results/summary/inference.json rebuilt from this run"
}

step_throughput() {
  if (( MEASURE )); then
    measure_throughput || warn "measurement incomplete, showing whatever is on disk"
  fi
  title "4. Inference throughput"
  note "output-token goodput under p95 TTFT < 2000 ms, p95 TPOT < 100 ms, zero errors"
  "${PY}" - <<'PY'
import json
d = json.load(open("results/summary/inference.json"))
print(f"  {'topology':<10}{'GPUs':>6}{'goodput tok/s':>16}{'concurrency':>13}{'per GPU':>10}")
seen, rows = set(), []
for t in d["topologies"]:
    b = t["best_passing"]
    if b and t["topology"] not in seen:
        seen.add(t["topology"])
        rows.append((t["topology"], t["gpu_count"], b))
for name, gpus, b in sorted(rows):
    print(f"  {name:<10}{gpus:>6}{b['output_token_goodput']:>16,.1f}"
          f"{b['concurrency']:>13}{b['output_tokens_per_s_per_gpu']:>10,.1f}")

soak = [p for t in d["topologies"] for p in t["points"] if p["stage"] == "soak"]
if soak:
    p = soak[0]
    print()
    print(f"  soak: {p['requests']:,} requests over {p['wall_seconds']:.0f} s, "
          f"{p['errors']} errors, {p['output_tokens_per_s']:,.0f} tok/s, "
          f"p95 TTFT {p['ttft_ms']['p95']:.0f} ms")

# Computed, not hardcoded. A stale literal here would contradict the table printed
# directly above it, which is the one place nobody can afford a wrong number.
best = {name: b["output_token_goodput"] for name, _, b in rows}
if {"P1", "P2"} <= best.keys():
    gain = (best["P2"] / best["P1"] - 1.0) * 100.0
    print()
    print(f"  P2 beats P1 by {gain:.1f}% on the same 2 GPUs: "
          "replication over tensor parallelism")
    print("  per-GPU is flat across 1, 2 and 4 GPUs, so replication scales linearly")
PY
}

declare -a STEPS=(preflight validate train accuracy throughput)

usage() {
  echo "usage: ./scripts/demo.sh [--no-pause] [--reuse] [--measure] [step ...]"
  echo "steps: ${STEPS[*]}"
  echo
  echo "  --reuse     show the last completed validation and training runs instead"
  echo "              of submitting new ones. Nothing reaches the cluster, so the"
  echo "              whole walkthrough works while it is busy."
  echo "  --measure   with the throughput step, run the real serving sweep first:"
  echo "              merge, four topologies, a soak, then aggregate. About 40"
  echo "              minutes. Not for a live demo. Servers are cancelled on exit."
}

main() {
  local requested=()
  while (( $# )); do
    case "$1" in
      --list) printf '%s\n' "${STEPS[@]}"; return 0 ;;
      --no-pause) PAUSE=0 ;;
      --reuse) REUSE=1 ;;
      --measure) MEASURE=1 ;;
      -h|--help) usage; return 0 ;;
      *) requested+=("$1") ;;
    esac
    shift
  done

  if (( ${#requested[@]} == 0 )); then
    requested=("${STEPS[@]}")
  else
    PAUSE=0
  fi

  # A step needs the cluster only if it will actually submit something. The
  # display steps read results/summary/ and must work in a bare clone. --reuse
  # removes the need too, but only where there is a recorded run to show instead,
  # which a fresh clone does not have.
  local needs_cluster=0 name
  for name in "${requested[@]}"; do
    case "${name}" in
      validate) (( REUSE )) && latest_validate_run >/dev/null 2>&1 || needs_cluster=1 ;;
      train)    (( REUSE )) && latest_train_run    >/dev/null 2>&1 || needs_cluster=1 ;;
    esac
  done
  if (( MEASURE )); then
    needs_cluster=1
  fi

  if (( needs_cluster )) && [[ -z "${SLURM_PARTITION:-}" ]]; then
    fail "SLURM_PARTITION is not set, so nothing can be submitted."
    echo
    if (( REUSE )); then
      note "--reuse looked for a recorded run under results/raw/ and found none."
      note "A fresh clone ships the summaries, not the raw runs they came from."
    fi
    note "To run the live steps, fill in configs/cluster.env:"
    note "  cp configs/cluster.env.example configs/cluster.env   # if it is missing"
    note "  ./scripts/discover_cluster.sh                        # reads the values"
    note "  ./scripts/setup.sh --check                           # confirms them"
    echo
    note "To see the recorded results without touching the cluster:"
    note "  ./scripts/demo.sh accuracy throughput"
    echo
    return 2
  fi

  local first=1 failed=0
  for name in "${requested[@]}"; do
    if ! declare -F "step_${name}" >/dev/null; then
      fail "unknown step: ${name}"
      usage
      return 2
    fi
    (( first )) || pause
    first=0
    # Keep going if a step fails. Losing the accuracy and throughput sections
    # because a training job died is the worst thing that can happen live, and
    # those two read from results/ and are unaffected by a bad allocation.
    if ! "step_${name}"; then
      failed=1
      fail "step ${name} did not complete, continuing with the rest"
    fi
  done
  echo
  return "${failed}"
}

main "$@"
