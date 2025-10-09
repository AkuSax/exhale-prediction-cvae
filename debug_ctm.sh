#!/bin/bash

python train_ctm.py \
    --data_dir /mnt/hot/public/Akul/exhale_pred_data \
    --save_dir ./checkpoints/cycletransmorph_debug \
    --epochs 500 \
    --batch_size 1 \
    --lr 1e-4 \
    --img_size 128 \
    --lambda_reg 1.0 \
    --lambda_cycle 1.0 \
    --debug