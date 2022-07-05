#!/bin/bash
#
# TO RUN:
# sbatch -p compsci-gpu --gres=gpu:1 train_and_evaluate.sh
#
# SLURM parameters:
#SBATCH --array=0-383%32
#SBATCH --exclude=linux[41-60]
#SBATCH --mail-type=END

# it would be better to output the parameter ranges for each of these searches into a file that goes
# into the respective file.
# Should these parameter ranges go in a python file with a dictionary that I import?
# that would make it easier to find all possible combinations because I could just use itertools.
outdir="$HOME/id_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$outdir"

data_path="<insert path to data directory here>" #  eg. "/usr/xtmp/kpinheiro/data"

num_layers=(3)

experiments=("ets1_ets1" "ets1_runx1")

kernel_sizes=(3 7 11 15)

kernel2_sizes=(3 7 11 15)

kernel3_sizes=(3 7 11 15)

kernel4_sizes=(3 7 11 15)

kernel5_sizes=(3 7 11 15)

mers=(1 2 3)

batch_sizes=(32)

include_affinities="false"

# simplify enumeration of feature combinations
args=()
for experiment in "${experiments[@]}"; do
  for num_layer in "${num_layers[@]}"; do
    for mer in "${mers[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        for l1_width in "${kernel_sizes[@]}"; do
          if (( num_layer > 1)); then
            for l2_width in "${kernel2_sizes[@]}"; do
              if (( num_layer > 2)); then
                for l3_width in "${kernel3_sizes[@]}"; do
                  if (( num_layer > 3)); then
                    for l4_width in "${kernel4_sizes[@]}"; do
                      if (( num_layer > 4)); then
                        for l5_width in "${kernel5_sizes[@]}"; do
                          args+=("${experiment} ${num_layer} ${mer} ${batch_size} \
                          ${l1_width},${l2_width},${l3_width},${l4_width},${l5_width}")
                        done
                      else
                        args+=("${experiment} ${num_layer} ${mer} ${batch_size} \
                        ${l1_width},${l2_width},${l3_width},${l4_width}")
                      fi
                    done
                  else
                    args+=("${experiment} ${num_layer} ${mer} ${batch_size} \
                    ${l1_width},${l2_width},${l3_width}")
                  fi
                done
              else
              args+=("${experiment} ${num_layer} ${mer} ${batch_size} ${l1_width},${l2_width}")
              fi
            done
          else
          args+=("${experiment} ${num_layer} ${mer} ${batch_size} ${l1_width}")
          fi
        done
      done
    done
  done
done

echo "${args[${SLURM_ARRAY_TASK_ID}]}"
#  paths to appropriate python installation and python file should be substituted for the first two
# arguments
srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python \
/home/users/kap52/dl_cooperativity/experiment.py \ "${SLURM_ARRAY_TASK_ID}" "${outdir}" \
"${data_path}" ${args[${SLURM_ARRAY_TASK_ID}]} "${include_affinities}"
