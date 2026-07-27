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

# Keep total request count comparable across topologies: each replica gets
# floor(total / n) prompts so P2/P3 do not inflate fleet traffic.
base_prompts = 256
prompts_per_replica = max(base_prompts // num_replicas, warmup + 8)

jobs = []
for conc in concurrencies:
    for rep in range(1, reps + 1):
        for index, base_url in enumerate(base_urls):
            tag = f"{stage}_c{conc}_r{rep}_e{index}"
            result_dir = out / tag
            result_dir.mkdir(parents=True, exist_ok=True)
            seed = 10_000 + conc * 100 + rep * 10 + index
            cmd = [
                "vllm", "bench", "serve",
                "--backend", "vllm",
                "--base-url", base_url,
                "--model", model,
                "--endpoint", "/v1/completions",
                "--dataset-name", "random",
                "--random-input-len", str(input_tokens),
                "--random-output-len", str(output_tokens),
                "--num-prompts", str(prompts_per_replica),
                "--max-concurrency", str(max(1, conc // num_replicas)),
                "--seed", str(seed),
                "--save-result",
                "--result-dir", str(result_dir),
                "--result-filename", "bench.json",
            ]
            # Newer vLLM builds accept --save-detailed; keep best-effort.
            detailed_cmd = cmd + ["--save-detailed"]
            meta = {
                "tag": tag,
                "stage": stage,
                "concurrency": conc,
                "repetition": rep,
                "endpoint_index": index,
                "base_url": base_url,
                "num_prompts": prompts_per_replica,
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
        soak_seconds = int(plan.get("soak_minutes") or 10) * 60
        time.sleep(soak_seconds)
        for proc, handle, meta, result_dir in procs:
            proc.terminate()
    for proc, handle, meta, result_dir in procs:
        rc = proc.wait()
        handle.close()
        (result_dir / "exit_code.txt").write_text(str(rc) + "\n")
        if rc != 0 and stage != "soak":
            raise SystemExit(f"bench failed for {meta['tag']} rc={rc}")

print(f"completed {len(jobs)} bench shards under {out}")
PY
