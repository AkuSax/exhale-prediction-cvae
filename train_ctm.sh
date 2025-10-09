#!/bin/bash
#SBATCH --job-name=train_cycletransmorph
#SBATCH --output=logs/cycletransmorph_%j.out
#SBATCH --error=logs/cycletransmorph_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exhale_pred

python train_ctm.py \
    --data_dir /path/to/your/processed_data \
    --save_dir ./checkpoints/cycletransmorph \
    --epochs 200 \
    --batch_size 1 \
    --lr 1e-4 \
    --img_size 128 \
    --lambda_reg 1.0 \
    --lambda_cycle 1.0 \
    --save_interval 10