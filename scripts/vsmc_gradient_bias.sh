#!/bin/bash
#SBATCH --account=naiss2026-4-1390-cpu
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=vsmc_gradient_bias_%j.out

# Paper appendix "Comparison against unbiased VSMC gradients" (CPU only).
# Takes about a minute on 32 cores.
# Usage (from the repository root): sbatch scripts/vsmc_gradient_bias.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
apptainer exec --bind /nobackup /nobackup/proj/disk/naiss2025-22-438/apps/container.sif \
    python3 -m appendix.vsmc_gradient_bias
