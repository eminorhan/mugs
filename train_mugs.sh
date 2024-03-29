#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=350GB
#SBATCH --time=48:00:00
#SBATCH --job-name=mugs_train
#SBATCH --output=mugs_train_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

#srun python -u /scratch/eo41/mugs/train_mugs.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--batch_size_per_gpu 64 \
#	--num_workers 8 \
#	--freeze_last_layer 0 \
#	--lr 0.0005 \
#	--min_lr 0.0005 \
#	--output_dir "/scratch/eo41/mugs/models_vitb16" \
#	--data_path "/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar" \
#	--save_prefix "say_5fps_vitb16"

#srun python -u /scratch/eo41/mugs/train_mugs.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--batch_size_per_gpu 64 \
#	--num_workers 8 \
#	--freeze_last_layer 0 \
#	--lr 0.0005 \
#	--min_lr 0.0005 \
#	--output_dir "/scratch/eo41/mugs/models_vitb16" \
#	--data_path "/scratch/eo41/data/saycam/S_5fps_300s_{000000..000003}.tar" \
#	--save_prefix "s_5fps_vitb16"

#srun python -u /scratch/eo41/mugs/train_mugs.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--batch_size_per_gpu 64 \
#	--num_workers 4 \
#	--freeze_last_layer 0 \
#	--lr 0.0005 \
#	--min_lr 0.0005 \
#	--output_dir "/scratch/eo41/mugs/models_vitb16" \
#	--data_path "/scratch/eo41/data/saycam/A_5fps_300s_{000000..000002}.tar" \
#	--save_prefix "a_5fps_vitb16"

srun python -u /scratch/eo41/mugs/train_mugs.py \
	--use_fp16 false \
	--arch "vit_base" \
	--batch_size_per_gpu 64 \
	--num_workers 4 \
	--freeze_last_layer 0 \
	--lr 0.0005 \
	--min_lr 0.0005 \
	--output_dir "/scratch/eo41/mugs/models_vitb16" \
	--data_path "/scratch/eo41/data/saycam/Y_5fps_300s_{000000..000002}.tar" \
	--save_prefix "y_5fps_vitb16"
		
echo "Done"
