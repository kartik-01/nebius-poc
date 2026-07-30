#!/bin/bash
# Print the allocated GPU information on each node. This is the smallest useful job
# on the cluster and the first one to run: it confirms the account, partition and QoS
# are usable, that both nodes schedule, and that each task sees the GPUs it was given.
#
# Nothing here needs a container, the repo, or configs/cluster.env, so it also works
# as a bare check before any of the setup in docs/RUNBOOK.md is done.
#
# Usage:
#   sbatch --partition=<partition> [--account=<account>] [--qos=<qos>] slurm/gpu_info.sbatch
#   # or, with configs/cluster.env already filled:
#   source configs/cluster.env
#   sbatch --partition="$SLURM_PARTITION" ${SLURM_ACCOUNT:+--account=$SLURM_ACCOUNT} \
#     ${SLURM_QOS:+--qos=$SLURM_QOS} slurm/gpu_info.sbatch
#
#SBATCH --job-name=nebius-gpu-info
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-info-%j.out
#SBATCH --error=logs/gpu-info-%j.err

set -Eeuo pipefail

mkdir -p "${SLURM_SUBMIT_DIR:-$(pwd)}/logs"

echo "job ${SLURM_JOB_ID} on nodes ${SLURM_JOB_NODELIST}"
echo "nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS} gpus_per_node=2"
echo

# One task per node, so each line of output is that node reporting its own allocation.
# --gpus-per-node here matches the directive above; asking for GPUs per task instead
# gives each task a single device and hides the rest of the allocation.
srun --ntasks-per-node=1 --gpus-per-node=2 bash -c '
  echo "=== $(hostname) ==="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader
  echo
'
