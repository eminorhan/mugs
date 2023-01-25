#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --time=00:59:00
#SBATCH --job-name=eval_video_seg
#SBATCH --output=eval_video_seg_%A_%a.out
#SBATCH --array=0-11

module purge
module load cuda/11.6.2

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say s a y say s a y say s a y)
ARCHS=(vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)
PATCHES=(16 16 16 16 16 16 16 16 16 16 16 16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
PATCH=${PATCHES[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH
echo $PATCH

srun python -u /scratch/eo41/mugs/eval_video_segmentation.py \
	--arch ${ARCH} \
	--patch_size ${PATCH} \
	--data_path "/vast/eo41/data/davis-2017/DAVIS/" \
	--output_dir "/scratch/eo41/mugs/evals/davis-2017/${SUBJECT}_${MODEL}" \
	--pretrained_weights "/scratch/eo41/mugs/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_${MODEL}

echo "Done"
