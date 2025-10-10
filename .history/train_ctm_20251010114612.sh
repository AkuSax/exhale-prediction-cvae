#!/bin/bash

export NCCL_SOCKET_IFNAME=ibs9

torchrun --nproc_per_node=4 train_ctm.py \
    --data_dir "/mnt/hot/public/Akul/exhale_pred_data" \
    --epochs 200 \
    --batch_size 2 \
    --n_workers 8 \
    --val_interval 5 \
    --lr 1e-4 \
    --dataset_fraction 0.2
