#!/bin/bash -l
#SBATCH --job-name=tasksall_ucf
#SBATCH --partition=course_gpu
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --account=2025-fall-ds-677-amr239-ak3535
#SBATCH --qos=course
#SBATCH --time=20:00:00
#SBATCH --output=./job_logs/ucf101/all/%x.%j.out
#SBATCH --error=./job_logs/ucf101/all/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G

module load Miniforge3
conda activate ds_project
module load CUDA

TRAIN_CONFIG_FILE="config/train_configs/UCF101/train_task0.yml"
OUTPUT_FOLDER="model_save/final_trial"

python train_task_0.py \
    --config "$TRAIN_CONFIG_FILE" \
    --save_folder "$OUTPUT_FOLDER"

TRAIN_CONFIG_FILE="config/train_configs/UCF101/train_task1_unfreeze_ft.yml"

python fine_tune.py \
    --config "$TRAIN_CONFIG_FILE" \
    --save_folder "$OUTPUT_FOLDER"

EVAL_CONFIG_FILE="config/eval_configs/UCF101/eval_task1_unfreeze.yml"

python evaluation.py \
    --config "$EVAL_CONFIG_FILE" \
    --save_results "$OUTPUT_FOLDER"

TRAIN_CONFIG_FILE="config/train_configs/UCF101/train_task2_unfreeze_ft.yml"

python fine_tune.py \
    --config "$TRAIN_CONFIG_FILE" \
    --save_folder "$OUTPUT_FOLDER"

EVAL_CONFIG_FILE="config/eval_configs/UCF101/eval_task2_unfreeze.yml"

python evaluation.py \
    --config "$EVAL_CONFIG_FILE" \
    --save_results "$OUTPUT_FOLDER"