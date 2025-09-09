import random
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import multiprocessing
import os

# Configuration
RAW_DATA_DIR = Path("/hot/COPDGene-1")
OUTPUT_DIR = Path("/hot/Akul/exhale_pred_data")
TARGET_SHAPE_2D = (512, 512) 

def normalize_scan(scan):
    """Normalizes the scan by focusing on the tissue intensity range."""
    tissue_mask = scan > 100
    if not np.any(tissue_mask):
        return np.zeros_like(scan, dtype=np.float32)

    min_bound = np.percentile(scan[tissue_mask], 1)
    max_bound = np.percentile(scan[tissue_mask], 99)
    
    if max_bound == min_bound:
        return np.zeros_like(scan, dtype=np.float32)
        
    scan = np.clip(scan, min_bound, max_bound)
    scan = (scan - min_bound) / (max_bound - min_bound)
    return scan.astype(np.float32)

def process_patient(patient_id):
    """Extracts, processes, and saves all 2D slices for a single patient."""
    try:
        # Process Inhale Scan
        inhale_path = RAW_DATA_DIR / f"{patient_id}_INSP_image.nii.gz"
        if inhale_path.exists():
            inhale_nii = nib.load(inhale_path)
            inhale_data = inhale_nii.get_fdata()
            for i in range(inhale_data.shape[2]):
                slice_2d = inhale_data[:, :, i]
                normalized = normalize_scan(slice_2d)
                zoom_factors = [t / s for t, s in zip(TARGET_SHAPE_2D, normalized.shape)]
                resized = zoom(normalized, zoom_factors, order=1)
                np.save(OUTPUT_DIR / "inhale" / f"{patient_id}_slice_{i:04d}.npy", resized)
        
        # Process Exhale Scan
        exhale_path = RAW_DATA_DIR / f"{patient_id}_EXP_image.nii.gz"
        if exhale_path.exists():
            exhale_nii = nib.load(exhale_path)
            exhale_data = exhale_nii.get_fdata()
            for i in range(exhale_data.shape[2]):
                slice_2d = exhale_data[:, :, i]
                normalized = normalize_scan(slice_2d)
                zoom_factors = [t / s for t, s in zip(TARGET_SHAPE_2D, normalized.shape)]
                resized = zoom(normalized, zoom_factors, order=1)
                np.save(OUTPUT_DIR / "exhale" / f"{patient_id}_slice_{i:04d}.npy", resized)
        
        return None
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    (OUTPUT_DIR / "inhale").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exhale").mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(list(set([f.name.split('_')[0] for f in RAW_DATA_DIR.glob("*_image.nii.gz")])))
    print(f"Found {len(patient_ids)} patients. Processing all of them...")

    num_processes = min(os.cpu_count() - 2, 20)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient, patient_ids), total=len(patient_ids)))

    for res in results:
        if res is not None:
            print(res)
    print("2D slice preprocessing complete.")

if __name__ == "__main__":
    main()