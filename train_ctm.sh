#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_SOCKET_IFNAME=ibs9

torchrun --nproc_per_node=4 train_ctm.py \
    --data_dir "/mnt/hot/public/Akul/exhale_pred_data" \
    --save_dir "model_runs/ctm_run_1" \
    --epochs 200 \
    --batch_size 5 \
    --n_workers 8 \
    --val_interval 5 \
    --lr 1e-4 \
    --dataset_fraction 0.2 \
    --alpha 2.0
    --alpha 2.0