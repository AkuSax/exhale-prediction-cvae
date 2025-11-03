#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
save_dir="model_runs/ctm_run_4_test"
mkdir -p $save_dir

torchrun --nproc_per_node=4 train_ctm.py \
    --data_dir "/mnt/hot/public/Akul/exhale_pred_data_10pct" \
    --save_dir $save_dir \
    --epochs 50 \
    --batch_size 1 \
    --n_workers 8 \
    --val_interval 2 \
    --lr 1e-4 \
    --dataset_fraction 1 \
    --alpha 2.0 \
    --lambda_cycle 1.0 \
    --lambda_expansion 10.0