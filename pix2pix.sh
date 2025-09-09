#!/bin/bash
PROJECT_NAME="exhale-prediction-pix2pix"
MODEL_SAVE_DIR="../model_runs/pix2pix_run_1"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=150
# Pix2Pix is also memory-intensive, start with a batch size of 1
BATCH_SIZE=1
LEARNING_RATE=2e-4
# Adam optimizer parameters, common for GANs
BETA1=0.5
BETA2=0.999
# Weight for the L1 reconstruction loss
LAMBDA_L1=100
NUM_WORKERS=4

python train_pix2pix.py \
    --project_name "$PROJECT_NAME" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --b1 "$BETA1" \
    --b2 "$BETA2" \
    --lambda_l1 "$LAMBDA_L1" \
    --num_workers "$NUM_WORKERS"