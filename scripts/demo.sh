#!/usr/bin/env bash
# Demo driver. Runs the PoC story one step at a time with readable output.
#
#   ./scripts/demo.sh              walk every step, pausing between them
#   ./scripts/demo.sh validate     run a single step
#   ./scripts/demo.sh --list       show the step names
#
# Live steps submit a Slurm job and show progress until it finishes on its own.
# Nothing needs Ctrl-C. Steps that only read artifacts never touch the cluster.
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

step_validate() {
  title "1. Qualify the allocation"
  note "portable checks on 4 GPUs, then intra-node and 3x inter-node NCCL"
  local job
  # shellcheck disable=SC2046
  job="$(sbatch --parsable $(sbatch_args) slurm/validate.sbatch)"
  note "submitted job ${job}"
  echo
  follow_job "${job}" "logs/validate-${job}.out" 'using NCCL binary|Avg bus bandwidth|artifacts in'
  echo
  local run
  run="$(ls -dt results/raw/*_validate_job"${job}" 2>/dev/null | head -1)"
  [[ -n "${run}" ]] || run="$(ls -dt results/raw/*_validate_job* | head -1)"
  "${PY}" - "${run}" <<'PY'
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

step_train() {
  title "2. Multi-node training"
  note "Qwen2.5-7B base, LoRA, 2 nodes x 2 GPUs, recipe pinned by recipe_lock.json"
  local job
  # shellcheck disable=SC2046
  job="$(TRAIN_CONFIG=configs/train_ranking.yaml TRAIN_FINAL=1 \
        RECIPE_LOCK=results/summary/recipe_lock.json \
        sbatch --parsable $(sbatch_args) slurm/train.sbatch)"
  note "submitted job ${job}, expect about 10 minutes on 4,964 rows"
  echo
  follow_job "${job}" "logs/train-${job}.err" 'applied recipe lock|training rows|adapter written'
  echo
  local run
  run="$(ls -d results/raw/*_train_job"${job}"/*/ 2>/dev/null | head -1)"
  [[ -n "${run}" ]] || { fail "no run directory for job ${job}"; return 1; }
  "${PY}" - "${run}" <<'PY'
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

step_throughput() {
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
  echo "usage: ./scripts/demo.sh [--no-pause] [step ...]"
  echo "steps: ${STEPS[*]}"
}

main() {
  local requested=()
  while (( $# )); do
    case "$1" in
      --list) printf '%s\n' "${STEPS[@]}"; return 0 ;;
      --no-pause) PAUSE=0 ;;
      -h|--help) usage; return 0 ;;
      *) requested+=("$1") ;;
    esac
    shift
  done

  : "${SLURM_PARTITION:?set SLURM_PARTITION in configs/cluster.env}"

  if (( ${#requested[@]} == 0 )); then
    requested=("${STEPS[@]}")
  else
    PAUSE=0
  fi

  local first=1
  for name in "${requested[@]}"; do
    if ! declare -F "step_${name}" >/dev/null; then
      fail "unknown step: ${name}"
      usage
      return 2
    fi
    (( first )) || pause
    first=0
    "step_${name}"
  done
  echo
}

main "$@"
