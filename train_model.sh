#!/bin/bash

#SBATCH --exclude=linux[41-60]
#SBATCH --mail-type=END
#SBATCH --output=dl2.out

srun /home/users/kap52/miniconda3/envs/dl_cooperativity/bin/python \
/home/users/kap52/dl_cooperativity/train_model.py