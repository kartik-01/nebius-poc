#!/usr/bin/env bash
# Shared helpers for Slurm jobs that use stock Enroot images + a mounted repo.
# shellcheck shell=bash

nebius_poc_container_python_setup() {
  # Repo is mounted at /workspace. Stock pytorch images already ship torch + numpy +
  # pyyaml; we only need the LoRA/HF stack from the wheelhouse.
  export PYTHONPATH="/workspace/src:/workspace${PYTHONPATH:+:${PYTHONPATH}}"

  if python -c "import peft, transformers, datasets, accelerate" >/dev/null 2>&1; then
    return 0
  fi

  local wheels="/workspace/containers/images/wheels"
  if [[ ! -d "${wheels}" ]] || ! compgen -G "${wheels}/*" >/dev/null; then
    echo "LoRA deps missing and no wheelhouse at ${wheels}" >&2
    return 1
  fi

  # --no-deps: do not replace the image's torch/cuda stack. Pull only the Python
  # packages we need, then fill their pure/manylinux deps from the same wheelhouse.
  python -m pip install --no-index --find-links="${wheels}" --no-deps \
    accelerate datasets peft transformers huggingface_hub safetensors tokenizers \
    >/tmp/nebius-pip.log 2>&1 \
    || {
      echo "pip install (no-deps) from wheelhouse failed; see /tmp/nebius-pip.log" >&2
      tail -n 80 /tmp/nebius-pip.log >&2 || true
      return 1
    }

  python -m pip install --no-index --find-links="${wheels}" \
    pyarrow dill multiprocess xxhash pandas fsspec packaging filelock \
    regex requests tqdm typing_extensions psutil pyyaml numpy \
    click httpx httpcore h11 anyio certifi charset_normalizer idna urllib3 \
    hf_xet python_dateutil six \
    aiohttp aiosignal attrs frozenlist multidict propcache yarl aiohappyeyeballs \
    >/tmp/nebius-pip-deps.log 2>&1 \
    || {
      echo "pip install deps from wheelhouse failed; see /tmp/nebius-pip-deps.log" >&2
      tail -n 80 /tmp/nebius-pip-deps.log >&2 || true
      return 1
    }

  python -c "import peft, transformers, datasets, accelerate" \
    || {
      echo "LoRA stack still not importable after wheelhouse install" >&2
      return 1
    }
}
