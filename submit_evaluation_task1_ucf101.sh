#!/bin/bash -l
#SBATCH --job-name=eval_ucf
#SBATCH --partition=course_gpu
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --account=2025-fall-ds-677-amr239-ak3535
#SBATCH --qos=course
#SBATCH --time=6:00:00
#SBATCH --output=./job_logs/ucf101/eval/task1/%x.%j.out
#SBATCH --error=./job_logs/ucf101/eval/task1/%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G

# module load Miniforge3
conda activate ds_project
module load CUDA

EVAL_CONFIG_FILE="config/eval_configs/UCF101/eval_task1.yml"
OUTPUT_FOLDER="model_save/new_trial"

python evaluation.py \
    --config "$EVAL_CONFIG_FILE" \
    --save_results "$OUTPUT_FOLDER"
