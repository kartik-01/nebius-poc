#!/usr/bin/env bash
# Launch local vLLM replicas for one Slurm node. Called from slurm/serve.sbatch.
set -Eeuo pipefail

: "${OUT:?}"
: "${TOPOLOGY:?}"
: "${MODEL_PATH:?}"
: "${REPLICAS_PER_NODE:?}"
: "${TP_SIZE:?}"
: "${BASE_PORT:?}"
: "${DTYPE:?}"
: "${MAX_MODEL_LEN:?}"
: "${GPU_MEM_UTIL:?}"
: "${MAX_NUM_SEQS:?}"
: "${MAX_BATCHED_TOKENS:?}"

NODE_RANK="${SLURM_NODEID:-0}"
HOST="$(hostname -s)"
ENDPOINT_FILE="${OUT}/endpoints-${HOST}.jsonl"
PIDS=()

: > "${ENDPOINT_FILE}"

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

for replica in $(seq 0 $((REPLICAS_PER_NODE - 1))); do
  PORT=$((BASE_PORT + replica))
  START_GPU=$((replica * TP_SIZE))
  END_GPU=$((START_GPU + TP_SIZE - 1))
  GPU_LIST="$(seq -s, "${START_GPU}" "${END_GPU}")"
  LOG="${OUT}/vllm-${HOST}-r${replica}.log"

  echo "node=${HOST} replica=${replica} gpus=${GPU_LIST} port=${PORT}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
      >"${LOG}" 2>&1 &
  PIDS+=($!)

  printf '{"topology":"%s","node":"%s","node_rank":%s,"replica":%s,"host":"%s","port":%s,"tensor_parallel_size":%s,"cuda_visible_devices":"%s"}\n' \
    "${TOPOLOGY}" "${HOST}" "${NODE_RANK}" "${replica}" "${HOST}" "${PORT}" "${TP_SIZE}" "${GPU_LIST}" \
    >> "${ENDPOINT_FILE}"
done

python3 - "${ENDPOINT_FILE}" "${OUT}" <<'PY'
import json, socket, sys, time
from pathlib import Path

endpoints = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
out = Path(sys.argv[2])
deadline = time.time() + 600
for ep in endpoints:
    while time.time() < deadline:
        try:
            with socket.create_connection((ep["host"], int(ep["port"])), timeout=2):
                break
        except OSError:
            time.sleep(2)
    else:
        raise SystemExit(f"server never became ready: {ep}")
(out / f"ready-{endpoints[0]['host']}").write_text("ok\n")
print(f"ready {len(endpoints)} endpoints on {endpoints[0]['host']}")
PY

if [[ "${NODE_RANK}" == "0" ]]; then
  python3 - "${OUT}" "${TOPOLOGY}" <<'PY'
import json, sys, time
from pathlib import Path

out = Path(sys.argv[1])
topology = sys.argv[2]
expect_nodes = int(__import__("os").environ.get("SLURM_NNODES", "1"))
deadline = time.time() + 180
while time.time() < deadline:
    parts = sorted(out.glob("endpoints-*.jsonl"))
    ready = list(out.glob("ready-*"))
    if len(parts) >= expect_nodes and len(ready) >= expect_nodes:
        break
    time.sleep(2)
else:
    raise SystemExit("timed out waiting for all nodes to publish endpoints")

rows = []
for path in parts:
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
payload = {
    "topology": topology,
    "endpoints": rows,
    "base_urls": [f"http://{row['host']}:{row['port']}/v1" for row in rows],
}
(out / "endpoints.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
fi

echo "servers up; sleeping until job is cancelled"
sleep infinity
