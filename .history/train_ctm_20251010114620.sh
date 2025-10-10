#!/bin/bash

# Make all 4 GPUs visible to the script
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Tell DDP to use the InfiniBand network interface
export NCCL_SOCKET_IFNAME=ibs9

# Launch 4 processes, one for each of the 4 GPUs
torchrun --nproc_per_node=4 train_ctm.py \
    --data_dir "/mnt/hot/public/Akul/exhale_pred_data" \
    --epochs 200 \
    --batch_size 2 \
    --n_workers 8 \
    --val_interval 5 \
    --lr 1e-4 \
    --dataset_fraction 0.2