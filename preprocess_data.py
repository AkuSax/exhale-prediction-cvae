import multiprocessing
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Configuration ---
# Adjust these paths to match your system
RAW_DATA_ROOT = Path("/mnt/hot/public/COPDGene-1")
PROCESSED_DATA_DIR = Path("/mnt/hot/public/Akul/exhale_pred_data")

# Target shape for all scans
TARGET_SHAPE = (128, 128, 128)

# Hounsfield Unit (HU) windowing parameters for lung tissue
# As recommended in the research document [cite: 519]
LUNG_WINDOW_LEVEL = -600
LUNG_WINDOW_WIDTH = 1500
MIN_BOUND = LUNG_WINDOW_LEVEL - (LUNG_WINDOW_WIDTH / 2)
MAX_BOUND = LUNG_WINDOW_LEVEL + (LUNG_WINDOW_WIDTH / 2)

def normalize_scan_hu(scan_data: np.ndarray) -> np.ndarray:
    """
    Robustly normalizes a CT scan using Hounsfield Unit windowing.

    1. Clips the intensity values to a specified HU range for lungs.
    2. Scales the result to a [0, 1] range.
    """
    # Clip the scan to the lung window
    scan_data = np.clip(scan_data, MIN_BOUND, MAX_BOUND)

    # Scale to [0, 1]
    scan_data = (scan_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    return scan_data.astype(np.float32)

def process_patient_pair(patient_id: str):
    """
    Loads, processes (resizes and normalizes), and saves the inhale/exhale
    scan pair for a single patient ID.
    """
    try:
        inhale_path = RAW_DATA_ROOT / f"{patient_id}_INSP_image.nii.gz"
        exhale_path = RAW_DATA_ROOT / f"{patient_id}_EXP_image.nii.gz"

        if not (inhale_path.exists() and exhale_path.exists()):
            return f"Skipped {patient_id}: Missing one or both scan files."

        # --- Process Inhale Scan ---
        inhale_nii = nib.load(inhale_path)
        inhale_data = inhale_nii.get_fdata()

        # Resize the scan
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, inhale_data.shape)]
        inhale_resized = zoom(inhale_data, zoom_factors, order=1) # Linear interpolation

        # Normalize the resized scan
        inhale_normalized = normalize_scan_hu(inhale_resized)

        np.save(PROCESSED_DATA_DIR / "inhale" / f"{patient_id}.npy", inhale_normalized)

        # --- Process Exhale Scan ---
        exhale_nii = nib.load(exhale_path)
        exhale_data = exhale_nii.get_fdata()

        # Resize the scan
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, exhale_data.shape)]
        exhale_resized = zoom(exhale_data, zoom_factors, order=1)

        # Normalize the resized scan
        exhale_normalized = normalize_scan_hu(exhale_resized)

        np.save(PROCESSED_DATA_DIR / "exhale" / f"{patient_id}.npy", exhale_resized)

        return None  # Success
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    """
    Main function to find all patient IDs and process their scans in parallel.
    """
    # Create output directories
    (PROCESSED_DATA_DIR / "inhale").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / "exhale").mkdir(parents=True, exist_ok=True)

    # Find unique patient IDs from filenames
    all_files = list(RAW_DATA_ROOT.glob("*_INSP_image.nii.gz"))
    patient_ids = sorted([f.name.split('_')[0] for f in all_files])

    if not patient_ids:
        print(f"Error: No inhale scans found in {RAW_DATA_ROOT}. Please check the path.")
        return

    print(f"Found {len(patient_ids)} patient scan pairs. Starting preprocessing...")

    # Use a multiprocessing pool to parallelize the work
    # Leave a couple of CPU cores free for system stability
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient_pair, patient_ids), total=len(patient_ids)))

    # Print any errors that occurred during processing
    error_count = 0
    for res in results:
        if res is not None:
            print(res)
            error_count += 1

    print("\n--- Preprocessing Complete ---")
    print(f"Successfully processed: {len(patient_ids) - error_count} pairs.")
    print(f"Failed to process: {error_count} pairs.")

if __name__ == "__main__":
    main()