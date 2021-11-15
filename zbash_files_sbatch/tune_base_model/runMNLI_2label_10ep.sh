#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40Gb
#SBATCH --time=8:00:00
#SBATCH --output=mnli_2label_10ep_log.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

source activate firstDL
module load anaconda3/3.7
bash scripts/run_MNLI_2label_xLaunch_10ep.sh