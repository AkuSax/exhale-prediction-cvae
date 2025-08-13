# /home/dmic/Fibrosis/Akul/exhale_pred/preprocess_data.py

import os
import random
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# --- Configuration ---
# Corrected data root directory
DATA_ROOT = Path("/hot/COPDGene-1")
OUTPUT_DIR = Path("./processed_data")
NUM_PATIENTS_FOR_TEST = 32 # Using 32 patients for the flash test
TARGET_SHAPE = (128, 128, 128) # Resize all scans to this shape

# --- Lung Windowing & Normalization ---
def normalize_scan(scan):
    """Clips to a lung window and normalizes to [0, 1]."""
    min_bound = -1000.0  # Hounsfield Units for air
    max_bound = 0.0     # Hounsfield Units for soft tissue
    scan = np.clip(scan, min_bound, max_bound)
    scan = (scan - min_bound) / (max_bound - min_bound)
    return scan.astype(np.float32)

def main():
    print("🚀 Starting data preprocessing for the flash test...")
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        (OUTPUT_DIR / "inhale").mkdir()
        (OUTPUT_DIR / "exhale").mkdir()

    # Get unique patient IDs from the filenames
    all_files = list(DATA_ROOT.glob("*_image.nii.gz"))
    patient_ids = sorted(list(set([f.name.split('_')[0] for f in all_files])))
    
    if not patient_ids:
        print(f"❌ Error: No patient IDs found in {DATA_ROOT}. Check the path and filenames.")
        return

    # Select a random subset of patients for the flash test
    selected_ids = random.sample(patient_ids, min(NUM_PATIENTS_FOR_TEST, len(patient_ids)))
    print(f"Found {len(patient_ids)} total patients. Processing a random subset of {len(selected_ids)}.")

    for patient_id in tqdm(selected_ids, desc="Processing Patients"):
        try:
            inhale_path = DATA_ROOT / f"{patient_id}_INSP_image.nii.gz"
            exhale_path = DATA_ROOT / f"{patient_id}_EXP_image.nii.gz"
            
            # Ensure both files exist before proceeding
            if not inhale_path.exists() or not exhale_path.exists():
                print(f"⚠️ Warning: Missing INSP or EXP scan for {patient_id}. Skipping.")
                continue

            # Load and process inhale scan
            inhale_nii = nib.load(inhale_path)
            inhale_data = inhale_nii.get_fdata()
            zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, inhale_data.shape)]
            inhale_resized = zoom(inhale_data, zoom_factors, order=1)
            inhale_normalized = normalize_scan(inhale_resized)
            
            # Load and process exhale scan
            exhale_nii = nib.load(exhale_path)
            exhale_data = exhale_nii.get_fdata()
            zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, exhale_data.shape)]
            exhale_resized = zoom(exhale_data, zoom_factors, order=1)
            exhale_normalized = normalize_scan(exhale_resized)

            # Save as NumPy arrays
            np.save(OUTPUT_DIR / "inhale" / f"{patient_id}.npy", inhale_normalized)
            np.save(OUTPUT_DIR / "exhale" / f"{patient_id}.npy", exhale_normalized)

        except Exception as e:
            print(f"❌ Error processing {patient_id}: {e}")

    print(f"✅ Preprocessing complete. Data saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()