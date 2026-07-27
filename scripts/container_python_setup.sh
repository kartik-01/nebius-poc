#!/usr/bin/env bash
# Shared helpers for Slurm jobs that use stock Enroot images + a mounted repo.
# shellcheck shell=bash

nebius_poc_container_python_setup() {
  # Repo is mounted at /workspace. Stock pytorch images lack peft/transformers;
  # install from the pre-downloaded wheelhouse when present.
  export PYTHONPATH="/workspace/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
  local wheels="/workspace/containers/images/wheels"
  if [[ -d "${wheels}" ]] && compgen -G "${wheels}/*" >/dev/null; then
    python -m pip install --no-index --find-links="${wheels}" \
      accelerate datasets numpy peft pyyaml transformers >/tmp/nebius-pip.log 2>&1 \
      || {
        echo "pip install from wheelhouse failed; see /tmp/nebius-pip.log" >&2
        tail -n 50 /tmp/nebius-pip.log >&2 || true
        return 1
      }
  fi
}
