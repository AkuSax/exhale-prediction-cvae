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
# This value must be identical to preprocess_data_FIXED.py
TARGET_SPACING = (4.2369140625, 4.2369140625, 4.2369140625)
# --- END FIX ---

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
    resampler.SetDefaultPixelValue(0) # Pad masks with 0

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(itk_image)
    resampled_data = sitk.GetArrayFromImage(resampled_image).astype(np.uint8)
    
    # --- Step 2: Pad or Crop to target shape ---
    current_shape = resampled_data.shape
    pad_needed = [(max(0, ts - cs)) for ts, cs in zip(TARGET_SHAPE, current_shape)]

    pad_before = [p // 2 for p in pad_needed]
    pad_after = [p - pb for p, pb in zip(pad_needed, pad_before)]

    padded_data = np.pad(
        resampled_data,
        list(zip(pad_before, pad_after)),
        mode='constant',
        constant_values=0,
    )

    crop_start = [(ps - ts) // 2 for ps, ts in zip(padded_data.shape, TARGET_SHAPE)]
    
    cropped_data = padded_data[
        crop_start[0] : crop_start[0] + TARGET_SHAPE[0],
        crop_start[1] : crop_start[1] + TARGET_SHAPE[1],
        crop_start[2] : crop_start[2] + TARGET_SHAPE[2],
    ]
    
    return cropped_data

def process_patient_masks(args):
    """
    Loads, resamples, and saves BOTH inhale and exhale masks.
    """
    patient_id, opt = args
    
    try:
        inhale_mask_path = RAW_DATA_ROOT / f"{patient_id}_INSP_mask.nii.gz"
        exhale_mask_path = RAW_DATA_ROOT / f"{patient_id}_EXP_mask.nii.gz"

        if not inhale_mask_path.exists():
            return f"Skipped {patient_id}: Missing INHALE mask file."
        if not exhale_mask_path.exists():
            return f"Skipped {patient_id}: Missing EXHALE mask file."

        # --- Process Inhale Mask ---
        inhale_iso_mask = resample_and_pad(inhale_mask_path, is_mask=True)
        save_dir_inhale = opt.output_dir / "masks" / "inhale"
        np.save(save_dir_inhale / f"{patient_id}_INSP_mask.npy", inhale_iso_mask)

        # --- Process Exhale Mask ---
        exhale_iso_mask = resample_and_pad(exhale_mask_path, is_mask=True)
        save_dir_exhale = opt.output_dir / "masks" / "exhale"
        np.save(save_dir_exhale / f"{patient_id}_EXP_mask.npy", exhale_iso_mask)
        
        return None
    except Exception as e:
        return f"Error processing {patient_id} masks: {e}"

def main():
    parser = argparse.ArgumentParser(description="Preprocess Lung CT Masks with Isotropic Resampling")
    parser.add_argument(
        '--output_dir', type=Path, required=True, help="Path to the root processed data directory"
    )
    parser.add_argument(
        '--subset_frac', type=float, default=1.0, help="Fraction of the dataset to process"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers.",
    )
    opt = parser.parse_args()

    (opt.output_dir / "masks" / "inhale").mkdir(parents=True, exist_ok=True)
    (opt.output_dir / "masks" / "exhale").mkdir(parents=True, exist_ok=True)

    all_mask_files = list(RAW_DATA_ROOT.glob("*_EXP_mask.nii.gz"))
    patient_ids = sorted([f.name.split('_')[0] for f in all_mask_files])

    if not patient_ids:
        print(f"Error: No exhale masks found in {RAW_DATA_ROOT}.")
        return

    if opt.subset_frac < 1.0:
        num_patients = int(len(patient_ids) * opt.subset_frac)
        patient_ids = patient_ids[:num_patients]
        print(f"--- Processing a subset of {num_patients} patients ({opt.subset_frac * 100}%) ---")

    print(f"Found {len(patient_ids)} patient IDs. Starting mask preprocessing...")
    print(f"Using Isotropic Target Spacing: {TARGET_SPACING} mm")
    
    num_processes = opt.num_workers
    print(f"Starting parallel processing with {num_processes} workers.")
    
    process_args = [(pid, opt) for pid in patient_ids]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient_masks, process_args), total=len(patient_ids)))

    error_count = 0
    for res in results:
        if res is not None:
            print(res)
            error_count += 1

    print("\n--- Mask Preprocessing Complete ---")
    print(f"Successfully processed: {len(patient_ids) - error_count} pairs.")
    print(f"Failed to process: {error_count} pairs.")

if __name__ == "__main__":
    main()