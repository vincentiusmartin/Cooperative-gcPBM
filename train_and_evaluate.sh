#!/bin/bash
#
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh

# 2 * 4 * 4 * 4 * 2 = 256
# the end value must be equal to one less than the number of runs
#SBATCH --array=0-7%8
#SBATCH --mail-type=END
#SBATCH --output=dl.out


outdir="/home/users/kap52/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

data_path="/usr/xtmp/kpinheiro/data"

architectures=( "two_layer_cnn" )  # 1
# "multi_input_one_layer_cnn" "multi_input_two_layer_cnn") #"one_layer_cnn" "two_layer_cnn")  # 2

experiments=( "ets1_ets1" )  # ( "ets1_runx1" )

kernel_sizes=(8) #kernel_sizes=(4 8 12 16)  # 4

kernel2_sizes=(4)  # (4 8 12 16) # 4

kernel3_sizes=(4 8 12 16) # 4

mers=(3)  # 2

batch_sizes=(1 2 4 8 16 32 64 128)

# for loop over models, feature sets, and models
args=()
for experiment in "${experiments[@]}"; do
  for architecture in "${architectures[@]}"; do
    for mer in "${mers[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for kernel_size in "${kernel_sizes[@]}"; do
          if [ "${architecture}" = "two_layer_cnn" ] || [ "${architecture}" = "multi_input_two_layer_cnn" ] || [ "${architecture}" = "three_layer_cnn" ]
          then
            for kernel2_size in "${kernel2_sizes[@]}"; do
              if [ "${architecture}" = "three_layer_cnn" ]
              then
                for kernel3_size in "${kernel3_sizes[@]}"; do
                  args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size} ${kernel2_size} ${kernel3_size}")
                done
              else
              args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size} ${kernel2_size}")
              fi
            done
          else
          args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size}")
          fi
        done
      done
    done
  done
done

#echo ${#args[@]}

# Use correct paths to appropriate python installation and python file
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python /home/users/kap52/dl_cooperativity/experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]}
