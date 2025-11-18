#!/bin/bash -l
#SBATCH --job-name=task0_ucf
#SBATCH --partition=course_gpu
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --account=2025-fall-ds-677-amr239-ak3535
#SBATCH --qos=course
#SBATCH --time=6:00:00
#SBATCH --output=./job_logs/ucf101/task0/%x.%j.out
#SBATCH --error=./job_logs/ucf101/task0/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# module load Miniforge3
conda activate ds_project
module load CUDA

TRAIN_CONFIG_FILE="config/train_configs/UCF101/train_task0.yml"
OUTPUT_FOLDER="model_save/ucf101/task0"

python train_task_0.py \
    --config "$TRAIN_CONFIG_FILE" \
    --save_folder "$OUTPUT_FOLDER"