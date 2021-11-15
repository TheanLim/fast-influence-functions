#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40Gb
#SBATCH --time=5:00:00
#SBATCH --output=logs/run_experiments_recall_at_m_log.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

source activate firstDL
module load anaconda3/3.7
module load cuda/11.0

python recall_at_m.py