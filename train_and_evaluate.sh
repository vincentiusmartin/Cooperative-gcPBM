#!/bin/bash
#
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:2 train_and_evaluate.sh

# the end value must be equal to one less than the number of runs
#SBATCH --array=0-287%50
#SBATCH --mail-type=END
#SBATCH --output=dl.out


outdir="/home/users/kap52/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

data_path="/usr/xtmp/kpinheiro/data"

architectures=( "one_layer_cnn" "two_layer_cnn" )  # 2

experiments=( "ets1_runx1" "ets1_ets1" ) # 2

kernel_sizes=(4 8 12 16 20 24)  # 6

kernel2_sizes=(4 8 12 16 20 24) # 6

kernel3_sizes=(4 8 12 16) # 4

kernel4_sizes=(4 8 12 16) # 4

mers=(1 2)  # 2

batch_sizes=(32)

# for loop over models, feature sets, and models
args=()
for experiment in "${experiments[@]}"; do
  for architecture in "${architectures[@]}"; do
    for mer in "${mers[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for kernel_size in "${kernel_sizes[@]}"; do
          if [ "${architecture}" = "two_layer_cnn" ] || [ "${architecture}" = "multi_input_two_layer_cnn" ] || [ "${architecture}" = "three_layer_cnn" ] || [ "${architecture}" = "four_layer_cnn" ]
          then
            for kernel2_size in "${kernel2_sizes[@]}"; do
              if [ "${architecture}" = "three_layer_cnn" ] || [ "${architecture}" = "four_layer_cnn" ]
              then
                for kernel3_size in "${kernel3_sizes[@]}"; do
                  if [ "${architecture}" = "four_layer_cnn" ]
                  then
                    for kernel4_size in "${kernel4_sizes[@]}"; do
                      args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size},${kernel2_size},${kernel3_size},${kernel4_size}")
                    done
                  else
                    args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size},${kernel2_size},${kernel3_size}")
                  fi
                done
              else
              args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size},${kernel2_size}")
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

#extra_features=""
extra_features="site1_score,site2_score"
# Use correct paths to appropriate python installation and python file
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python /home/users/kap52/dl_cooperativity/experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]} "${extra_features}"
