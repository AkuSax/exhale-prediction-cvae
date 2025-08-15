#!/bin/bash
# Script to configure and run the cVAE training experiment.

PROJECT_NAME="exhale-prediction-cvae"
MODEL_SAVE_DIR="./saved_models_cvae"
DATA_DIR="/hot/Akul/exhale_pred_data"

EPOCHS=150
BATCH_SIZE=4
LEARNING_RATE=1e-4
LATENT_DIM=256
NUM_WORKERS=8

python train_cvae_wandb.py \
    --project_name "$PROJECT_NAME" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --latent_dim "$LATENT_DIM" \
    --num_workers "$NUM_WORKERS"