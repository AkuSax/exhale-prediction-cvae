#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

torchrun --nproc_per_node=2 train_ctm.py \
    --data_dir "/mnt/hot/public/Akul/exhale_pred_data" \
    --save_dir "model_runs/ctm_debug" \
    --epochs 3 \
    --batch_size 1 \
    --n_workers 4 \
    --val_interval 1 \
    --lr 1e-4 \
    --dataset_fraction 0.01 \
    --alpha 2.0 \
    --lambda_cycle 1.0