import multiprocessing
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_ROOT = Path("/mnt/hot/public/COPDGene-1")
PROCESSED_DATA_DIR = Path("/mnt/hot/public/Akul/exhale_pred_data")
TARGET_SHAPE = (128, 128, 128)

def normalize_scan_hu(scan_nii: nib.Nifti1Image) -> np.ndarray:
    """
    Normalizes a CT scan using Hounsfield Unit (HU) windowing.
    Includes checks for non-finite values in data and header.
    """
    slope = 1.0
    intercept = 0.0
    try:
        header_slope = scan_nii.header['scl_slope']
        header_intercept = scan_nii.header['scl_inter']
        
        if np.isfinite(header_slope) and np.isfinite(header_intercept):
            slope = header_slope
            intercept = header_intercept

    except KeyError:
        pass

    scan_data = scan_nii.get_fdata().astype(np.float32)
    
    if not np.all(np.isfinite(scan_data)):
        scan_data = np.nan_to_num(scan_data, nan=-1000.0, posinf=400.0, neginf=-1000.0)

    if slope != 1.0 or intercept != 0.0:
        scan_data = scan_data * slope + intercept

    # --- FINAL FIX: Use a standard 'lung window' for better contrast ---
    min_bound, max_bound = -1000.0, 400.0
    scan_data = np.clip(scan_data, min_bound, max_bound)
    scan_data = (scan_data - min_bound) / (max_bound - min_bound)
    
    return np.clip(scan_data, 0.0, 1.0).astype(np.float32)

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

        inhale_nii = nib.load(inhale_path)
        inhale_normalized = normalize_scan_hu(inhale_nii)
        zoom_factors_inhale = [t / s for t, s in zip(TARGET_SHAPE, inhale_normalized.shape)]
        inhale_resized = zoom(inhale_normalized, zoom_factors_inhale, order=1)
        np.save(PROCESSED_DATA_DIR / "inhale" / f"{patient_id}.npy", inhale_resized)

        exhale_nii = nib.load(exhale_path)
        exhale_normalized = normalize_scan_hu(exhale_nii)
        zoom_factors_exhale = [t / s for t, s in zip(TARGET_SHAPE, exhale_normalized.shape)]
        exhale_resized = zoom(exhale_normalized, zoom_factors_exhale, order=1)
        np.save(PROCESSED_DATA_DIR / "exhale" / f"{patient_id}.npy", exhale_resized)
        
        return None
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    """
    Main function to find all patient IDs and process their scans in parallel.
    """
    (PROCESSED_DATA_DIR / "inhale").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / "exhale").mkdir(parents=True, exist_ok=True)

    all_files = list(RAW_DATA_ROOT.glob("*_INSP_image.nii.gz"))
    patient_ids = sorted([f.name.split('_')[0] for f in all_files])

    if not patient_ids:
        print(f"Error: No inhale scans found in {RAW_DATA_ROOT}. Please check the path.")
        return

    print(f"Found {len(patient_ids)} patient scan pairs. Starting preprocessing...")

    num_processes = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient_pair, patient_ids), total=len(patient_ids)))

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