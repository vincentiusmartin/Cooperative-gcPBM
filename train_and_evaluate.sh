#!/bin/bash
#
#
# TO RUN: sbatch -p compsci-gpu train_and_evaluate.sh

# the end value must be equal to one less than the number of runs
#SBATCH --array=0-1%24
#SBATCH --mail-type=END
#SBATCH --output=dl.out

outdir="/home/users/kap52/${date +%m-%d-%Y-%H-%M}"
mkdir -p $outdir

data_path="/usr/xtmp/kpinheiro/data"

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
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python /home/users/kap52/dl_cooperativity/experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]}
