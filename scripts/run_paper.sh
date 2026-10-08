#!/bin/bash
# Submits every training run in the paper, NUM_RUNS (default 10) per config.
#
#   configs/<dataset>/<method>_<smnr>.gin  -> runs/<dataset>/<PREFIX><method>_<smnr>_<run>
#   appendix/ablations/<study>/<name>.gin  -> runs/ablations/<study>/<PREFIX><name>_<run>
#
# Usage (from the repository root):
#   bash scripts/run_paper.sh              # everything
#   bash scripts/run_paper.sh configs/lorenz63/dsr_10db.gin ...   # a subset
#   NUM_RUNS=3 PREFIX=TEST_ bash scripts/run_paper.sh            # test runs

NUM_RUNS=${NUM_RUNS:-10}
PREFIX=${PREFIX:-}

if [ "$#" -gt 0 ]; then
    GIN_PATHS=("$@")
else
    GIN_PATHS=($(find configs appendix/ablations -name '*.gin' | sort))
fi

for GIN in "${GIN_PATHS[@]}"; do
    NAME="${GIN%.gin}"
    SAVE="runs/${NAME#configs/}"
    SAVE="${SAVE/runs\/appendix\//runs/}"
    SAVE="$(dirname "${SAVE}")/${PREFIX}$(basename "${SAVE}")"
    echo "Submitting: ${GIN} -> ${SAVE}_[0-$((NUM_RUNS - 1))]"
    sbatch --array=0-$((NUM_RUNS - 1)) scripts/train.sh "${GIN}" "${SAVE}"
done
