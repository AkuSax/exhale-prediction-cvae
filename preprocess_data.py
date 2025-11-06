import argparse
import multiprocessing
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk  # Import the new library
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_ROOT = Path("/mnt/hot/public/COPDGene-1")
TARGET_SHAPE = (128, 128, 128)
# --- FIX ---
# Define a target spacing based on our analysis to prevent cropping
TARGET_SPACING = (4.2369140625, 4.2369140625, 4.2369140625)
# --- END FIX ---

def normalize_scan(scan_data: np.ndarray) -> np.ndarray:
    """
    Normalizes a CT scan using the standard HU window for lung parenchyma.
    """
    min_bound, max_bound = -1000.0, 300.0
    scan_data = np.clip(scan_data, min_bound, max_bound)
    scan_data = (scan_data - min_bound) / (max_bound - min_bound)
    return scan_data.astype(np.float32)

def resample_and_pad(nii_path: Path, is_mask: bool) -> np.ndarray:
    """
    Loads a NIfTI file, resamples it to a target isotropic spacing,
    and then pads or crops it to a target shape.
    """
    # Load the image using SimpleITK to easily access metadata
    itk_image = sitk.ReadImage(str(nii_path))
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # --- Step 1: Resample to isotropic spacing ---
    # Calculate the new size based on the target spacing
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, TARGET_SPACING)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(itk_image)))) # Use min value for padding

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(itk_image)
    
    # SimpleITK GetArrayFromImage returns (D, H, W), which matches PyTorch
    resampled_data = sitk.GetArrayFromImage(resampled_image)
    
    # --- Step 2: Pad or Crop to target shape ---
    current_shape = resampled_data.shape
    pad_needed = [(max(0, ts - cs)) for ts, cs in zip(TARGET_SHAPE, current_shape)]

    pad_before = [p // 2 for p in pad_needed]
    pad_after = [p - pb for p, pb in zip(pad_needed, pad_before)]

    # Use min value for padding
    pad_value = np.min(resampled_data)
    padded_data = np.pad(
        resampled_data,
        list(zip(pad_before, pad_after)),
        mode='constant',
        constant_values=pad_value,
    )

    # Crop if the padded data is larger than the target shape
    crop_start = [(ps - ts) // 2 for ps, ts in zip(padded_data.shape, TARGET_SHAPE)]
    
    cropped_data = padded_data[
        crop_start[0] : crop_start[0] + TARGET_SHAPE[0],
        crop_start[1] : crop_start[1] + TARGET_SHAPE[1],
        crop_start[2] : crop_start[2] + TARGET_SHAPE[2],
    ]
    
    return cropped_data

def process_patient_pair(args):
    """
    Loads, processes (resamples then normalizes), and saves the inhale/exhale
    scan pair for a single patient ID.
    """
    patient_id, opt = args
    
    try:
        inhale_path = RAW_DATA_ROOT / f"{patient_id}_INSP_image.nii.gz"
        exhale_path = RAW_DATA_ROOT / f"{patient_id}_EXP_image.nii.gz"

        if not (inhale_path.exists() and exhale_path.exists()):
            return f"Skipped {patient_id}: Missing one or both scan files."

        # --- Process Inhale Scan ---
        inhale_iso_data = resample_and_pad(inhale_path, is_mask=False)
        inhale_normalized = normalize_scan(inhale_iso_data)
        np.save(opt.output_dir / "inhale" / f"{patient_id}.npy", inhale_normalized)

        # --- Process Exhale Scan ---
        exhale_iso_data = resample_and_pad(exhale_path, is_mask=False)
        exhale_normalized = normalize_scan(exhale_iso_data)
        np.save(opt.output_dir / "exhale" / f"{patient_id}.npy", exhale_normalized)
        
        return None
    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Preprocess Lung CT Scans with Isotropic Resampling")
    parser.add_argument(
        '--output_dir', type=Path, required=True, help="Path to the output processed data directory."
    )
    parser.add_argument(
        '--subset_frac', type=float, default=1.0, help="Fraction of the dataset to process (e.g., 0.1 for 10%)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers.",
    )
    opt = parser.parse_args()

    (opt.output_dir / "inhale").mkdir(parents=True, exist_ok=True)
    (opt.output_dir / "exhale").mkdir(parents=True, exist_ok=True)

    all_files = list(RAW_DATA_ROOT.glob("*_INSP_image.nii.gz"))
    patient_ids = sorted([f.name.split('_')[0] for f in all_files])

    if not patient_ids:
        print(f"Error: No inhale scans found in {RAW_DATA_ROOT}.")
        return

    if opt.subset_frac < 1.0:
        num_patients = int(len(patient_ids) * opt.subset_frac)
        patient_ids = patient_ids[:num_patients]
        print(f"--- Processing a subset of {num_patients} patients ({opt.subset_frac * 100}%) ---")
    
    print(f"Found {len(patient_ids)} patient scan pairs. Starting preprocessing...")
    print(f"Using Isotropic Target Spacing: {TARGET_SPACING} mm")

    num_processes = opt.num_workers
    print(f"Starting parallel processing with {num_processes} workers.")
    
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