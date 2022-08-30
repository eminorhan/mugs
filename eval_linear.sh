#!/bin/bash

##SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=mugs_eval_linear
#SBATCH --output=mugs_eval_linear_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

srun python -u /scratch/eo41/mugs/eval_linear.py \
	--arch "vit_large" \
	--pretrained_weights "/scratch/eo41/mugs/models_vitl/say_5fps_vitl16_checkpoint.pth" \
	--checkpoint_key "student" \
	--batch_size_per_gpu 2048 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/mugs/evals/imagenet" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val"
		
echo "Done"
