#!/bin/bash
#
#
# TO RUN: sbatch -p compsci-gpu train_and_evaluate.sh

# the end value must be equal to one less than the number of runs
#SBATCH --array=0-168%24
#SBATCH --mail-type=END
#ASDFSBATCH --mail-user=

outdir="$HOME/$(date +"%Y_%m_%d_%I_%M_%p")" # specify path for data output (must already exist)

data_path=""

architectures=("one_layer_cnn") # "1_conv_layer_max_pool" "2_conv_layer")

experiments=("ets1_ets1" "ets1_runx1")  # 2

kernel_sizes=(4 8 12 16 20 24 28)  # 7

mers=(1 2 3)  # 3

# 2 * 7 * 4 * 3 = 168

# for loop over models, feature sets, and models
args=()
for experiment in "${experiments[@]}"; do
  for architecture in "${architectures[@]}"; do
    for mer in "${mers[@]}"; do
      for kernel_size in "${kernel_sizes[@]}"; do
        args+=("${experiment} ${architecture} ${mer} ${kernel_size}")
      done
    done
  done
done

#echo ${#args[@]}

# Use correct paths to appropriate python installation and python file
srun python dl_cooperativity/experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]}
