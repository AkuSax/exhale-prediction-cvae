#!/bin/bash
PROJECT_NAME="exhale-prediction-voxelmorph"
MODEL_SAVE_DIR="../model_runs/voxelmorph_run_1"
DATA_DIR="/mnt/hot/public/Akul/exhale_pred_data"

EPOCHS=200
# VoxelMorph is memory-intensive, start with a batch size of 1
BATCH_SIZE=1
LEARNING_RATE=1e-4
# Alpha is the weight for the DVF smoothness regularizer
ALPHA=0.02
NUM_WORKERS=4

python train_voxelmorph.py \
    --project_name "$PROJECT_NAME" \
    --model_save_dir "$MODEL_SAVE_DIR" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --alpha "$ALPHA" \
    --num_workers "$NUM_WORKERS"