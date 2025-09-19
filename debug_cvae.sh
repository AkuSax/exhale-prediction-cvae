#!/bin/bash
PROJECT_NAME="exhale-prediction-cvae-debug"
MODEL_SAVE_DIR="./model_runs/cvae_run_debug"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=2
BATCH_SIZE=4
LEARNING_RATE=1e-4
LATENT_DIM=256
CONDITION_SIZE=128
NUM_WORKERS=8 # Increased for better data loading performance
BETA=1.0
LOSS_FN="l1"
VALIDATE_EVERY=1
LOG_IMAGES_EVERY=1
KL_ANNEAL_EPOCHS=2 # Short anneal for a short run
FREE_BITS=0.0
NUM_SAMPLES=100

torchrun --standalone --nproc_per_node=2 train_cvae.py \
    --project_name "$PROJECT_NAME" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --latent_dim "$LATENT_DIM" \
    --condition_size "$CONDITION_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --beta "$BETA" \
    --loss_fn "$LOSS_FN" \
    --validate_every "$VALIDATE_EVERY" \
    --log_images_every "$LOG_IMAGES_EVERY" \
    --kl_anneal_epochs "$KL_ANNEAL_EPOCHS" \
    --free_bits "$FREE_BITS" \
    --num_samples "$NUM_SAMPLES"