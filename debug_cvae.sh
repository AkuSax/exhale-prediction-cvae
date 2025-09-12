#!/bin/bash
PROJECT_NAME="exhale-prediction-cvae-debug"
MODEL_SAVE_DIR="./model_runs/cvae_run_debug"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=5e-5
LATENT_DIM=256
CONDITION_SIZE=128
NUM_WORKERS=4
NUM_SAMPLES=100

# --- Hyperparameters ---
BETA=1.0
LOSS_FN="l1"
VALIDATE_EVERY=1
LOG_IMAGES_EVERY=1
KL_ANNEAL_EPOCHS=5 # Anneal beta over the first 5 epochs
FREE_BITS=0.05     # Add a small free bits threshold

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
    --num_samples "$NUM_SAMPLES" \
    --kl_anneal_epochs "$KL_ANNEAL_EPOCHS" \
    --free_bits "$FREE_BITS"