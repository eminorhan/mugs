#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:30:00
#SBATCH --job-name=mugs_load_test
#SBATCH --output=mugs_load_test_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u /scratch/eo41/mugs/load_mugs_model.py

echo "Done"
