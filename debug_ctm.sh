#!/bin/bash

torchrun --nproc_per_node=2 train_ctm.py \
    --data_dir /mnt/hot/public/Akul/exhale_pred_data \
    --save_dir ./checkpoints/cycletransmorph_ddp_debug \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --img_size 128 \
    --lambda_reg 1.0 \
    --lambda_cycle 1.0 \
    --save_interval 10 \
    --debug
