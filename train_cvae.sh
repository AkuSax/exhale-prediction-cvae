#!/bin/bash
PROJECT_NAME="exhale-prediction-cvae-baseline"
MODEL_SAVE_DIR="./model_runs/cvae_run_1"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=1e-4
LATENT_DIM=256
CONDITION_SIZE=128
NUM_WORKERS=8

# --- New Arguments ---
VALIDATE_EVERY=5  # Run validation every 5 epochs
LOG_IMAGES_EVERY=10 # Log image samples every 10 epochs

# --- Hyperparameters ---
BETA=1.0
LOSS_FN="l1"

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
    --log_images_every "$LOG_IMAGES_EVERY"