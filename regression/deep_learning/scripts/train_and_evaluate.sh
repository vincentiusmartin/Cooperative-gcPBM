#!/bin/bash
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh <gridsearch config file> <data config file>
#
# SLURM parameters:
#SBATCH --mail-type=END

gridsearch_config=$1
data_config=$2

# make output directory if it does not already exist
outdir="$HOME/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

# copy the search grid into the output directory so we know what search produced these results
cp "$gridsearch_config" "$outdir/search-config.json"

# use "retrieve_grid_array" script to get arguments for model training experiment
args=()
IFS=";" read -r -a args <<< "$(python ./retrieve_grid_array.py "$outdir/search-config.json")"

echo "${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ./experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_config}" ${args[${SLURM_ARRAY_TASK_ID}]}
