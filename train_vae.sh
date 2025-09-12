#!/bin/bash
PROJECT_NAME="exhale-prediction-cvae-kl-annealing"
MODEL_SAVE_DIR="./model_runs/cvae_run_2"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=1e-4
LATENT_DIM=256
CONDITION_SIZE=128
NUM_WORKERS=8

# --- New Arguments ---
VALIDATE_EVERY=5
LOG_IMAGES_EVERY=10
KL_ANNEAL_EPOCHS=25 # Gradually increase beta over the first 25 epochs

# --- Hyperparameters ---
# The final target value for beta after annealing
BETA=0.001
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
    --log_images_every "$LOG_IMAGES_EVERY" \
    --kl_anneal_epochs "$KL_ANNEAL_EPOCHS"