# CycleTransMorph: Diffeomorphic Registration for Exhale CT Prediction

This repository implements a deep learning framework for predicting Exhale CT scans from Inhale CT scans (and vice versa) using **CycleTransMorph (CTM)**. The model leverages a SwinUNETR backbone to predict diffeomorphic deformation fields, allowing for topology-preserving image registration and synthesis between respiratory states.

The project handles 3D volumetric NIfTI data, preprocesses it into isotropic numpy arrays, and performs distributed training using PyTorch DDP.

![Ventilation Flythrough](ventilation_flythrough.gif)

## 📂 Project Structure

```text
exhale-prediction-cvae/
├── debug.ipynb               # Debugging notebook
├── debug_ctm.sh              # Shell script for debugging runs
├── dir_lab_evaluation.ipynb  # Notebook for DIR-Lab dataset evaluation
├── environment.yml           # Conda environment definition
├── evaluation.ipynb          # General model evaluation notebook
├── losses.py                 # Custom loss functions (NCC, Grad, InvCons, Volume)
├── models.py                 # CycleTransMorph and SpatialTransformer architectures
├── preprocess_data.py        # Script to preprocess raw CT NIfTI files
├── preprocess_masks.py       # Script to preprocess binary masks
├── train_ctm.py              # Main distributed training script
├── train_ctm.sh              # Shell script to launch training
└── ...
````

## 🛠️ Installation

The project requires Python 3.10 and uses PyTorch with CUDA support. A Conda environment file is provided.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate exhale_pred
```

**Key Dependencies:**

  * PyTorch 2.5.1 + CUDA 12.1
  * MONAI 1.5.0
  * SimpleITK & Nibabel (Data IO)
  * Weights & Biases (Logging)

## 📊 Data Preparation

The training pipeline expects preprocessed `.npy` files (isotropic, normalized, and padded/cropped).

### 1\. Preprocess Images

Use `preprocess_data.py` to convert raw NIfTI (`.nii.gz`) scan pairs into numpy arrays.

```bash
python preprocess_data.py \
    --output_dir "/path/to/processed_data" \
    --subset_frac 1.0 \
    --num_workers 8
```

  * **Input:** Expects folders with `_INSP_image.nii.gz` and `_EXP_image.nii.gz` naming conventions.
  * **Processing:** Resamples to isotropic spacing (\~4.24mm), clips HU to [-1000, 300], and pads/crops to `(128, 128, 128)`.
  * **Output:** Creates `inhale/` and `exhale/` subdirectories with `.npy` files.

### 2\. Preprocess Masks

Ensure you also preprocess the corresponding lung masks using `preprocess_masks.py` (logic similar to data preprocessing but using Nearest Neighbor interpolation). The training script expects masks in `masks/inhale/` and `masks/exhale/`.

## 🚀 Training

Training is performed using `torchrun` for Distributed Data Parallel (DDP) execution. The main entry point is `train_ctm.py`.

### Quick Start

Modify `train_ctm.sh` to point to your data and desired save location, then run:

```bash
bash train_ctm.sh
```

### Command Line Arguments

```bash
torchrun --nproc_per_node=4 train_ctm.py \
    --data_dir "/path/to/processed_data" \
    --save_dir "model_runs/experiment_name" \
    --epochs 200 \
    --batch_size 1 \
    --lr 1e-4 \
    --alpha 2.0 \          # Regularization weight (Gradient Smoothing)
    --lambda_cycle 1.0 \   # Cycle Consistency weight
    --lambda_expansion 10.0 # Volume Conservation weight
```

### Loss Functions

The model optimizes a composite loss function:

1.  **Similarity Loss (NCC):** Ensures the warped image matches the target.
2.  **Regularization Loss:** Penalizes gradients in the deformation field (smoothness).
3.  **Cycle Consistency:** Ensures $T_{A \to B} \circ T_{B \to A} \approx Identity$.
4.  **Volume Conservation:** Penalizes unrealistic expansion/compression of lung volumes.

## 🧠 Model Architecture

The core model is **CycleTransMorph**, defined in `models.py`.

  * **Backbone:** SwinUNETR (Transformer-based U-Net) extracting features from concatenated $(Moving, Fixed)$ images.
  * **Registration Head:** Predicts a Stationary Velocity Field (SVF).
  * **Integration:** Uses a **Scaling and Squaring** layer to integrate the SVF into a Diffeomorphic Deformation Field (DVF).
  * **Spatial Transformer:** Warps the moving image using the computed DVF.

## 📈 Evaluation

Use the provided notebooks for analysis:

  * **`evaluation.ipynb`**: Calculate Dice scores and visualize warped images vs. ground truth.
  * **`dir_lab_evaluation.ipynb`**: Specific evaluation metrics for the DIR-Lab dataset.

Training progress, images, and Jacobian statistics are logged automatically to **Weights & Biases**.
