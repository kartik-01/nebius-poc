#!/usr/bin/env bash
# Drive vllm bench serve against every endpoint in endpoints.json.
# For multi-replica topologies, one bench process runs per base_url with a
# deterministic prompt shard so fleets stay comparable to single-replica runs.
set -Eeuo pipefail

ENDPOINTS_JSON="${1:?}"
OUT="${2:?}"
STAGE="${3:?}"
MODEL_NAME="${4:-merged}"
PLAN="${OUT}/bench_plan.json"

python3 - "${ENDPOINTS_JSON}" "${OUT}" "${STAGE}" "${MODEL_NAME}" "${PLAN}" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path

endpoints = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
stage = sys.argv[3]
model = sys.argv[4]
plan = json.load(open(sys.argv[5]))

base_urls = plan["base_urls"] or endpoints.get("base_urls") or []
if not base_urls:
    raise SystemExit("no base_urls in endpoints/plan")

concurrencies = plan["concurrency"]
reps = int(plan["repetitions"])
input_tokens = int(plan["input_tokens"])
output_tokens = int(plan["output_tokens"])
warmup = int(plan["warmup_requests"])
num_replicas = len(base_urls)

def prompts_for(conc: int) -> int:
    """Requests per replica for one concurrency point.

    Each replica gets floor(fleet / n) prompts so P2/P3 do not inflate fleet
    traffic relative to P0. The fleet total scales with load because a fixed
    count turns high-concurrency runs into a single wave that measures ramp-up
    instead of steady state; eight batches per replica, floored at 256 total.
    """
    soak_prompts = int(plan.get("soak_prompts") or 0)
    if stage == "soak" and soak_prompts:
        fleet = soak_prompts
    else:
        fleet = max(256, conc * 8)
    return max(fleet // num_replicas, warmup + 8)

# The benchmark CLI changes shape across vLLM releases: 0.8.5 has no --backend and
# no --save-detailed, while newer builds have both. Probe once and only pass flags
# this build actually accepts, so the same script works on a customer's version.
help_text = subprocess.run(
    ["vllm", "bench", "serve", "--help"],
    capture_output=True, text=True, check=False,
).stdout


def supports(flag: str) -> bool:
    return flag in help_text


jobs = []
for conc in concurrencies:
    for rep in range(1, reps + 1):
        for index, base_url in enumerate(base_urls):
            tag = f"{stage}_c{conc}_r{rep}_e{index}"
            result_dir = out / tag
            result_dir.mkdir(parents=True, exist_ok=True)
            seed = 10_000 + conc * 100 + rep * 10 + index
            num_prompts = prompts_for(conc)
            # endpoints.json stores OpenAI-style base URLs ending in /v1, but the
            # bench client wants the server root and adds --endpoint itself.
            server_root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
            cmd = [
                "vllm", "bench", "serve",
                "--base-url", server_root,
                "--model", model,
                "--endpoint", "/v1/completions",
                "--dataset-name", "random",
                "--random-input-len", str(input_tokens),
                "--random-output-len", str(output_tokens),
                "--num-prompts", str(num_prompts),
                "--max-concurrency", str(max(1, conc // num_replicas)),
                "--seed", str(seed),
                "--save-result",
                "--result-dir", str(result_dir),
                "--result-filename", "bench.json",
            ]
            if supports("--backend"):
                cmd += ["--backend", "vllm"]
            # Defaults report p99 only; the goodput guardrails need p50 and p95 too.
            if supports("--percentile-metrics"):
                cmd += ["--percentile-metrics", "ttft,tpot,itl,e2el"]
            if supports("--metric-percentiles"):
                cmd += ["--metric-percentiles", "50,95,99"]
            detailed_cmd = cmd + ["--save-detailed"] if supports("--save-detailed") else list(cmd)
            meta = {
                "tag": tag,
                "stage": stage,
                "concurrency": conc,
                "repetition": rep,
                "endpoint_index": index,
                "base_url": base_url,
                "num_prompts": num_prompts,
                "max_concurrency": max(1, conc // num_replicas),
                "seed": seed,
                "command": detailed_cmd,
            }
            (result_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            jobs.append((detailed_cmd, cmd, result_dir, meta))

# Screening: sequential per concurrency so the serve job is not thrashed.
# Within a concurrency, launch one process per replica together.
from collections import defaultdict
groups = defaultdict(list)
for job in jobs:
    meta = job[3]
    groups[(meta["concurrency"], meta["repetition"])].append(job)

for key in sorted(groups):
    batch = groups[key]
    procs = []
    for detailed_cmd, fallback_cmd, result_dir, meta in batch:
        log_path = result_dir / "bench.log"
        print(f"starting {meta['tag']} -> {meta['base_url']}", flush=True)
        handle = open(log_path, "w")
        try:
            proc = subprocess.Popen(detailed_cmd, stdout=handle, stderr=subprocess.STDOUT)
        except Exception:
            handle.close()
            handle = open(log_path, "w")
            proc = subprocess.Popen(fallback_cmd, stdout=handle, stderr=subprocess.STDOUT)
        procs.append((proc, handle, meta, result_dir))

    if stage == "soak":
        # The soak is sized via soak_prompts to finish on its own, because the
        # client only writes bench.json at the end. This is a safety net for a run
        # that overshoots: give it the full window plus slack, then stop it.
        deadline = time.time() + int(plan.get("soak_minutes") or 10) * 60 * 2
        while time.time() < deadline and any(proc.poll() is None for proc, *_ in procs):
            time.sleep(5)
        for proc, handle, meta, result_dir in procs:
            if proc.poll() is None:
                print(f"soak overran its window; stopping {meta['tag']}", flush=True)
                proc.terminate()
    for proc, handle, meta, result_dir in procs:
        rc = proc.wait()
        handle.close()
        (result_dir / "exit_code.txt").write_text(str(rc) + "\n")
        if rc != 0 and stage != "soak":
            raise SystemExit(f"bench failed for {meta['tag']} rc={rc}")

print(f"completed {len(jobs)} bench shards under {out}")
PY
