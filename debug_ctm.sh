# Set visible devices
export CUDA_VISIBLE_DEVICES="0,1,2,3"

base_data_dir="/mnt/hot/public/Akul/exhale_pred_data"
base_port=29500

alphas=(2.0 2.0 2.0 2.0)
lambda_expansions=(0.1 1.0 5.0 20.0) # Test 0.0 (baseline), 0.1 (gentle), 1.0 (medium), 5.0 (strong)

echo "Launching ${#alphas[@]} training jobs sequentially"

# --- Loop and Launch Jobs ---
for i in ${!alphas[@]}; do
    alpha_val=${alphas[$i]}
    lambda_exp_val=${lambda_expansions[$i]}
    
    # --- 1. Assign a unique port for this job ---
    port=$((base_port + i))

    # --- 2. Create a unique save directory ---
    save_dir="model_runs/ctm_debug_b${new_batch_size}_alpha_${alpha_val}_exp_${lambda_exp_val}"
    mkdir -p $save_dir
    
    echo "-----------------------------------------------------"
    echo "--- !!! Force-killing any old training processes... !!! ---"
    # --- Cleanup step to prevent zombie processes ---
    pkill -9 -f train_ctm.py
    sleep 5 # Give the OS a moment to release the GPU memory
    
    echo "Starting DEBUG job for alpha=$alpha_val, lambda_expansion=$lambda_exp_val on port=$port"
    echo "Logs: $save_dir/train.log"
    echo "-----------------------------------------------------"

    # --- 3. Launch the job and WAIT for it to finish ---
    # Using 10 epochs and val_interval 2 for a quick debug
    nohup torchrun --nproc_per_node=4 --master_port $port train_ctm.py \
        --data_dir $base_data_dir \
        --save_dir $save_dir \
        --epochs 10 \
        --batch_size 5 \
        --n_workers 8 \
        --val_interval 2 \
        --lr 1e-4 \
        --dataset_fraction 0.05 \
        --alpha $alpha_val \
        --lambda_cycle 1.0 \
        --lambda_expansion $lambda_exp_val > $save_dir/train.log 2>&1
        
    echo "Job for alpha=$alpha_val, lambda_expansion=$lambda_exp_val finished. Starting next job..."
done

# Final cleanup
echo "-----------------------------------------------------"
echo "--- !!! Final cleanup... !!! ---"
pkill -9 -f train_ctm.py
echo "All debug jobs have completed."