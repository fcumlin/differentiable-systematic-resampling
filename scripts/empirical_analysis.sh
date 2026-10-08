#!/bin/bash
#SBATCH --account=naiss2025-22-438
#SBATCH -C NOGPU
#SBATCH --time=15:00:00

# Paper appendix "Empirical Analysis of the DSR Relaxation" and Fig. 1.
# Usage (from the repository root): sbatch scripts/empirical_analysis.sh

RUN="apptainer exec ../danse/pytorch_training.sif python3 -m appendix.empirical_analysis"

${RUN}.transport_maps
${RUN}.transport_maps --taus 0.1 --output T_heatmap_combined_N25.png
${RUN}.relaxation_metrics --output relaxation_metrics.json
${RUN}.plot_relaxation_metrics --results relaxation_metrics.json
