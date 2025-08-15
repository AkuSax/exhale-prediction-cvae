#!/bin/bash
PROJECT_NAME="exhale-prediction-vae"
MODEL_SAVE_DIR="./model_runs/vae_run_7"
DATA_DIR="/hot/Akul/exhale_pred_data"

EPOCHS=400
BATCH_SIZE=4
LEARNING_RATE=1e-4
LATENT_DIM=256
NUM_WORKERS=8
BETA=0.1

torchrun --standalone --nproc_per_node=2 train_vae.py \
    --project_name "$PROJECT_NAME" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --latent_dim "$LATENT_DIM" \
    --num_workers "$NUM_WORKERS" \
    --beta "$BETA"