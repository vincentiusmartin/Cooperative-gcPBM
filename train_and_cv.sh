#!/bin/bash
#
#
# TO RUN: sbatch train_and_cv.sh

# the end value must be equal to one less than the number of runs
#SBATCH --array=0-55%16
#SBATCH --mail-type=END
#SBATCH --mail-user=<email here>

outdir="" # specify path for data output (must already exist)

models=("support_vector_regression"  "random_forest_regression")

experiments=("ets1_ets1" "ets1_runx1")

feature_sets=(
"distance"
"affinity"
"orientation"
"distance,affinity"
"distance,orientation"
"distance,affinity,orientation"
"distance,affinity,orientation,shape_in,shape_out"
"distance,affinity,orientation,sequence_in,sequence_out"
)


# for loop over models, feature sets, and models
args=()
for model in "${models[@]}"; do
  for experiment in "${experiments[@]}"; do
    for feature_set in "${feature_sets[@]}"; do
      args+=("${experiment} ${model} ${feature_set}")
    done
  done
done

# echo ${#args[@]}

# Ensure paths  to appropriate python installation and python file are correct
srun python cooperativity-prediction/automate_grid_search.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${args[${SLURM_ARRAY_TASK_ID}]}"
