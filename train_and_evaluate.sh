#!/bin/bash
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh
#
# SLURM parameters:
#SBATCH --array=0-383%32
#SBATCH --exclude=linux[41-60]
#SBATCH --mail-type=END


outdir="<insert path here>/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

data_path="<insert path to data directory here>" #  eg. "/usr/xtmp/kpinheiro/data"

architectures=( "three_layer_cnn" )

experiments=("ets1_ets1" "ets1_runx1")

kernel_sizes=(3 7 11 15)

kernel2_sizes=(3 7 11 15)

kernel3_sizes=(3 7 11 15)

kernel4_sizes=(3 7 11 15)

kernel5_sizes=(3 7 11 15)

mers=(1 2 3)

batch_sizes=(32)

# simplify enumeration of feature combinations
args=()
for experiment in "${experiments[@]}"; do
  for architecture in "${architectures[@]}"; do
    for mer in "${mers[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for kernel_size in "${kernel_sizes[@]}"; do
          if [ "${architecture}" = "two_layer_cnn" ] || [ "${architecture}" = "three_layer_cnn" ] || [ "${architecture}" = "four_layer_cnn" ] || [ "${architecture}" = "five_layer_cnn" ]
          then
            for kernel2_size in "${kernel2_sizes[@]}"; do
              if [ "${architecture}" = "three_layer_cnn" ] || [ "${architecture}" = "four_layer_cnn" ] || [ "${architecture}" = "five_layer_cnn" ]
              then
                for kernel3_size in "${kernel3_sizes[@]}"; do
                  if [ "${architecture}" = "four_layer_cnn" ] || [ "${architecture}" = "five_layer_cnn" ]
                  then
                    for kernel4_size in "${kernel4_sizes[@]}"; do
                      if [ "${architecture}" = "five_layer_cnn" ]
                      then
                        for kernel5_size in "${kernel5_sizes[@]}"; do
                          args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size},${kernel2_size},${kernel3_size},${kernel4_size},${kernel5_size}")
                        done
                      else
                        args+=("${experiment} ${architecture} ${mer} ${batch_size} ${kernel_size},${kernel2_size},${kernel3_size},${kernel4_size}")
                      fi
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

extra_features="site1_score,site2_score"
#  paths to appropriate python installation and python file should be substituted for the first two
# arguments
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python /home/users/kap52/dl_cooperativity/experiment.py \
"${SLURM_ARRAY_TASK_ID}" "${outdir}" "${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]} "${extra_features}"
