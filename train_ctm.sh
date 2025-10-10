#!/bin/bash

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0

# Run the training script using torchrun
torchrun --nproc_per_node=2 --nnodes=1 train_ctm.py \
    --data_dir "/scratch/ak9250/Exhale_Prediction_Data" \
    --epochs 200 \
    --batch_size 2 \
    --n_workers 8 \
    --val_interval 5 \
    --lr 1e-4 \
    --beta1 0.5 \
    --reg_lambda 0.02 \
    --cycle_lambda 0.1 \
    --debug \
    --dataset_fraction 0.2