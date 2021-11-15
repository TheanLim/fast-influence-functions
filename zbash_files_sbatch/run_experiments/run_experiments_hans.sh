#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40Gb
#SBATCH --time=8:00:00
#SBATCH --output=logs/run_experiments_hans_1_log.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

source activate firstDL
module load anaconda3/3.7
module load cuda/11.0

python -c'import torch; print(torch.cuda.is_available())'
python run_experiments.py hans-augmentation