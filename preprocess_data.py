import random
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import multiprocessing
import os

# Configuration for data preprocessing.
DATA_ROOT = Path("/hot/COPDGene-1")
OUTPUT_DIR = Path("/hot/Akul/exhale_pred_data")
NUM_PATIENTS_FOR_TEST = 32
TARGET_SHAPE = (128, 128, 128)

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
    """Loads, processes, and saves the inhale/exhale pair for a single patient."""
    try:
        inhale_path = DATA_ROOT / f"{patient_id}_INSP_image.nii.gz"
        exhale_path = DATA_ROOT / f"{patient_id}_EXP_image.nii.gz"
        
        if not (inhale_path.exists() and exhale_path.exists()):
            return f"Skipped {patient_id}: Missing files."

        # Process Inhale Scan
        inhale_nii = nib.load(inhale_path)
        inhale_data = inhale_nii.get_fdata()
        inhale_normalized = normalize_scan(inhale_data)
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, inhale_normalized.shape)]
        inhale_resized = zoom(inhale_normalized, zoom_factors, order=1)
        np.save(OUTPUT_DIR / "inhale" / f"{patient_id}.npy", inhale_resized)
        
        # Process Exhale Scan
        exhale_nii = nib.load(exhale_path)
        exhale_data = exhale_nii.get_fdata()
        exhale_normalized = normalize_scan(exhale_data)
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, exhale_normalized.shape)]
        exhale_resized = zoom(exhale_normalized, zoom_factors, order=1)
        np.save(OUTPUT_DIR / "exhale" / f"{patient_id}.npy", exhale_resized)
        
        return None # Return None on success
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    (OUTPUT_DIR / "inhale").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "exhale").mkdir(parents=True, exist_ok=True)

    all_files = list(DATA_ROOT.glob("*_image.nii.gz"))
    patient_ids = sorted(list(set([f.name.split('_')[0] for f in all_files])))
    
    if len(patient_ids) < NUM_PATIENTS_FOR_TEST:
        num_to_process = len(patient_ids)
    else:
        num_to_process = NUM_PATIENTS_FOR_TEST
        
    selected_ids = random.sample(patient_ids, num_to_process)
    print(f"Processing {len(selected_ids)} patient scans using multiple cores...")

    # Use a pool of worker processes to parallelize the work
    # Leave a couple of cores free for the OS
    num_processes = min(os.cpu_count() - 2, 20) 
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered for efficiency and wrap with tqdm for a progress bar
        results = list(tqdm(pool.imap_unordered(process_patient, selected_ids), total=len(selected_ids)))

    # Print any errors that occurred during processing
    for res in results:
        if res is not None:
            print(res)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
