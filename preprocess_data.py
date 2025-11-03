import argparse
import multiprocessing
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_ROOT = Path("/mnt/hot/public/COPDGene-1")
TARGET_SHAPE = (128, 128, 128)

def normalize_scan(scan_data: np.ndarray) -> np.ndarray:
    """
    Normalizes a CT scan using the standard HU window for lung parenchyma.
    
    The HU window is set to [-1000, 300] to capture the full range of
    lung tissue (-1000 to -400 HU) and surrounding context.
    
    Args:
        scan_data: The raw numpy array of the CT scan.
    """
    # SOTA-recommended bounds for lung CT registration
    min_bound, max_bound = -1000.0, 300.0

    scan_data = np.clip(scan_data, min_bound, max_bound)
    scan_data = (scan_data - min_bound) / (max_bound - min_bound)
    return scan_data.astype(np.float32)

def process_patient_pair(args):
    """
    Loads, processes (resizes then normalizes), and saves the inhale/exhale
    scan pair for a single patient ID.
    
    Takes a tuple of (patient_id, opt) as input for starmap.
    """
    patient_id, opt = args
    
    try:
        inhale_path = RAW_DATA_ROOT / f"{patient_id}_INSP_image.nii.gz"
        exhale_path = RAW_DATA_ROOT / f"{patient_id}_EXP_image.nii.gz"

        if not (inhale_path.exists() and exhale_path.exists()):
            return f"Skipped {patient_id}: Missing one or both scan files."

        # --- Process Inhale Scan ---
        inhale_nii = nib.load(inhale_path)
        inhale_data = inhale_nii.get_fdata().astype(np.float32)
        
        zoom_factors_inhale = [t / s for t, s in zip(TARGET_SHAPE, inhale_data.shape)]
        inhale_resized = zoom(inhale_data, zoom_factors_inhale, order=1)
        
        # Normalize using the fixed SOTA bounds
        inhale_normalized = normalize_scan(inhale_resized)
        np.save(opt.output_dir / "inhale" / f"{patient_id}.npy", inhale_normalized)

        # --- Process Exhale Scan ---
        exhale_nii = nib.load(exhale_path)
        exhale_data = exhale_nii.get_fdata().astype(np.float32)
        zoom_factors_exhale = [t / s for t, s in zip(TARGET_SHAPE, exhale_data.shape)]
        exhale_resized = zoom(exhale_data, zoom_factors_exhale, order=1)
        
        # Normalize using the fixed SOTA bounds
        exhale_normalized = normalize_scan(exhale_resized)
        np.save(opt.output_dir / "exhale" / f"{patient_id}.npy", exhale_normalized)
        
        return None
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    """
    Main function to find all patient IDs and process their scans in parallel.
    """
    parser = argparse.ArgumentParser(description="Preprocess Lung CT Scans")
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        required=True, 
        help="Path to the output processed data directory."
    )
    parser.add_argument(
        '--subset_frac', 
        type=float, 
        default=1.0, 
        help="Fraction of the dataset to process (e.g., 0.1 for 10%)"
    )
    opt = parser.parse_args()

    (opt.output_dir / "inhale").mkdir(parents=True, exist_ok=True)
    (opt.output_dir / "exhale").mkdir(parents=True, exist_ok=True)

    all_files = list(RAW_DATA_ROOT.glob("*_INSP_image.nii.gz"))
    patient_ids = sorted([f.name.split('_')[0] for f in all_files])

    if not patient_ids:
        print(f"Error: No inhale scans found in {RAW_DATA_ROOT}. Please check the path.")
        return

    # --- Subset Logic ---
    if opt.subset_frac < 1.0:
        num_patients = int(len(patient_ids) * opt.subset_frac)
        patient_ids = patient_ids[:num_patients]
        print(f"--- Processing a subset of {num_patients} patients ({opt.subset_frac * 100}%) ---")
    
    print(f"Found {len(patient_ids)} patient scan pairs. Starting preprocessing...")
    print(f"Normalization Bounds: [-1000, 300] HU")
    print(f"Output Directory: {opt.output_dir}")

    # --- Use all available CPU cores ---
    num_processes = multiprocessing.cpu_count()
    print(f"Starting parallel processing with {num_processes} workers.")
    
    # Create argument list for starmap
    process_args = [(pid, opt) for pid in patient_ids]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient_pair, process_args), total=len(patient_ids)))

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