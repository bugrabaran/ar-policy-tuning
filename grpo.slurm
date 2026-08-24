#!/bin/bash -l
#SBATCH --job-name=ar-policy-tuning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export REPO_DIR
export NNODES="${SLURM_NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

cd "$REPO_DIR"
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c 'NODE_RANK="$SLURM_PROCID" bash "$REPO_DIR/train.sh"'
