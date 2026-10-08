#!/bin/bash
#SBATCH --account=naiss2026-4-1390-cpu
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=dsr_bias_variance_%j.out

# Paper appendix "Bias and Variance of DSR Estimates" (CPU only).
# Usage (from the repository root): sbatch scripts/dsr_bias_variance.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
apptainer exec --bind /nobackup /nobackup/proj/disk/naiss2025-22-438/apps/container.sif \
    python3 -m appendix.dsr_bias_variance
