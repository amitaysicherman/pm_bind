#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-20



python predict_all.py --split_index $(($SLURM_ARRAY_TASK_ID - 1))