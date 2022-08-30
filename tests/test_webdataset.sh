#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=test_wds
#SBATCH --output=test_wds_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/mugs/tests/test_webdataset_2.py

echo "Done"
