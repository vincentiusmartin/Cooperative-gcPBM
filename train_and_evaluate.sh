#!/bin/bash
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh <gridsearch config file> \
# <data config file> <python executable>
#
# SLURM parameters:
#SBATCH --array=0-383%32
#SBATCH --exclude=linux[41-60]
#SBATCH --mail-type=END

gridsearch_config=$1
data_config=$2
# it would be better to output the parameter ranges for each of these searches into a file that goes
# into the respective file.
# Should these parameter ranges go in a python file with a dictionary that I import?
# that would make it easier to find all possible combinations because I could just use itertools.
outdir="$HOME/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

cp "$gridsearch_config" "$outdir"

args=()
IFS=";" read -r -a args <<< "$(/home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python \
 ./retrieve_grid_array.py "$gridsearch_config")"

echo "${args[${SLURM_ARRAY_TASK_ID}]}"
#  paths to appropriate python installation and python file should be substituted for the first two
# arguments
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python \
/home/users/kap52/dl_cooperativity/experiment.py "${SLURM_ARRAY_TASK_ID}" "${outdir}" \
"${data_config}" ${args[${SLURM_ARRAY_TASK_ID}]}
