#!/bin/bash
#SBATCH --account=naiss2026-4-1390-gpu
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00

# Trains one config; the array index is the run (seed) index.
# Usage (from the repository root):
#   sbatch --array=0-9 scripts/train.sh <gin_path> <save_prefix>

GIN_PATH=$1
SAVE_PREFIX=$2

apptainer exec --nv --bind /nobackup /nobackup/proj/disk/naiss2025-22-438/apps/container_gpu.sif python3 train.py \
    --gin_path "${GIN_PATH}" \
    --save_path "${SAVE_PREFIX}_${SLURM_ARRAY_TASK_ID}"
