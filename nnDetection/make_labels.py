import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import nibabel as nib

def lps_to_ijk(lps_coords, sitk_image):
    """
    Convert LPS physical space coordinates to voxel indices using the image transform.
    Returns (z, y, x) for direct use with NumPy arrays.
    """
    ijk = sitk_image.TransformPhysicalPointToIndex(lps_coords)
    return ijk[::-1]  # Convert (x, y, z) → (z, y, x) for NumPy

def create_mask(ct_path, json_path, out_path):
    # Load CT scan
    ct_image = sitk.ReadImage(str(ct_path))
    ct_shape = tuple(reversed(ct_image.GetSize()))  # Convert (x, y, z) to (z, y, x)

    # Load ROI annotation
    with open(json_path, "r") as f:
        data = json.load(f)

    roi = data["markups"][0]
    center_lps = np.array(roi["center"])
    size_mm = np.array(roi["size"])
    half_size = size_mm / 2

    # Define bounding box corners in LPS
    min_lps = (center_lps - half_size).tolist()
    max_lps = (center_lps + half_size).tolist()

    # Convert to IJK indices (z, y, x)
    min_index = lps_to_ijk(min_lps, ct_image)
    max_index = lps_to_ijk(max_lps, ct_image)

    # Clamp to image bounds
    min_corner = np.maximum(min_index, [0, 0, 0])
    max_corner = np.minimum(max_index, np.array(ct_shape) - 1)

    # Create empty mask
    mask_array = np.zeros(ct_shape, dtype=np.uint8)

    # Draw bounding box in mask
    zmin, ymin, xmin = min_corner
    zmax, ymax, xmax = max_corner
    mask_array[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1

    # Convert back to ITK image (transpose to x, y, z)
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(ct_image)
    sitk.WriteImage(mask_image, str(out_path))
    print(f"[✓] Saved mask to: {out_path}")


# # === CONFIGURE THESE PATHS ===
# ct_dir = Path("/home/tarobben/scratch/PFO_Complete/")        # Folder with .nii.gz CT scans
# label_dir = Path("/home/tarobben/scratch/PFO_labels_complete/")     # Folder with .json ROI files
# out_dir = Path("/home/tarobben/scratch/nndet/Task001_test/labelsTr/")        # Where to save .nii.gz masks
# out_dir.mkdir(parents=True, exist_ok=True)

# # === BATCH PROCESS ===
# for json_file in label_dir.glob("*.json"):
    
#     base = json_file.stem
#     base =base.replace('.mrk', '')
#     ct_file = ct_dir / f"{base}.nii.gz"
#     if not ct_file.exists():
#         print(f"Missing CT: {ct_file}")
#         continue

#     out_path = out_dir / f"{base}_mask.nii.gz"
#     try:
#         create_mask(ct_file, json_file, out_path)
#     except Exception as e:
#         print(f"Error with {base}: {e}")

def generate_pfo_jsons(label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(label_dir):
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            continue
        if "_mask" not in fname:
            continue

        label_path = os.path.join(label_dir, fname)
        mask = nib.load(label_path).get_fdata()

        # Check if mask has non-zero content
        if np.max(mask) == 0:
            print(f"[!] Skipping {fname} — mask is empty (non-PFO)")
            continue

        # Get unique instance labels > 0
        instance_labels = np.unique(mask)
        instance_labels = instance_labels[instance_labels > 0]

        instances_dict = {str(int(i)): 0 for i in instance_labels}

        case_id = fname.replace("_mask.nii.gz", "").replace("_mask.nii", "")
        json_path = os.path.join(output_dir, f"{case_id}_mask.json")

        with open(json_path, "w") as f:
            json.dump({"instances": instances_dict}, f, indent=2)

        print(f"[✓] Wrote {json_path}")

# generate_pfo_jsons("/scratch/tarobben/nndet/PFO_masks/", "/scratch/tarobben/nndet/PFO_masks/")
        
# Remove _mask
def remove_mask_files(mask_dir):
    for fname in os.listdir(mask_dir):
        if "_mask" in fname:
            new_name = fname.replace('_mask', '')
            old_path = os.path.join(mask_dir, fname)
            new_path = os.path.join(mask_dir, new_name)

            if os.path.exists(new_path):
                print(f"[!] Skipping (already exists): {new_name}")
            else:
                os.rename(old_path, new_path)
                print(f"[✓] Renamed: {fname} → {new_name}")
#remove_mask_files('/scratch/tarobben/nndet/PFO_masks/')

def add_0000(dir):
    for fname in os.listdir(dir):
        if fname.endswith("_70.nii.gz"):
            new_name = fname.replace('_70', '_70_0000')
        elif fname.endswith("_75.nii.gz"):
            new_name = fname.replace('_75', '_75_0000')
        # new_name = fname.replace('_0000', '')
        old_path = os.path.join(dir, fname)
        new_path = os.path.join(dir, new_name)

        if os.path.exists(new_path):
            print(f"[!] Skipping (already exists): {new_name}")
        else:
            os.rename(old_path, new_path)
            print(f"[✓] Renamed: {fname} → {new_name}")
# add_0000('/home/tarobben/scratch/nndet/Task002/raw_splitted/imagesTs/')
# dir = '/home/tarobben/scratch/nndet/Task001_test/imagesTr/'
# for fname in os.listdir(dir):
#     if not fname.endswith("0000.nii.gz"):
#         print(fname)
#     else:
#         print('complete')

# ===============================================================================================
# Non PFO masks
# ===============================================================================================
    
def generate_empty_masks(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_file in image_dir.glob("*.nii*"):
        img = nib.load(str(img_file))
        empty_mask = np.zeros(img.shape, dtype=np.uint8)
        empty_nii = nib.Nifti1Image(empty_mask, img.affine, img.header)
        
        out_path = output_dir / img_file.name
        nib.save(empty_nii, str(out_path))
        print(f"[✓] Created: {out_path}")

    print(f"\nDone! Created {len(list(image_dir.glob('*.nii*')))} empty masks in {output_dir}")

# generate_empty_masks("/home/tarobben/scratch/NO_PFO_CT/", "/home/tarobben/scratch/nndet/Non_PFO_masks/")

def generate_non_pfo_jsons(label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(label_dir):
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            continue
        # if "_mask" not in fname:
        #     continue

        label_path = os.path.join(label_dir, fname)
        mask = nib.load(label_path).get_fdata()

        # Skip if mask is not empty
        if np.max(mask) > 0:
            print(f"[!] Skipping {fname} — mask contains PFO (not empty)")
            continue

        case_id = fname.replace(".nii.gz", "").replace(".nii", "")
        json_path = os.path.join(output_dir, f"{case_id}.json")

        with open(json_path, "w") as f:
            json.dump({"instances": {}}, f, indent=2)

        print(f"[✓] Wrote {json_path}")

# generate_non_pfo_jsons("/scratch/tarobben/nndet/Non_PFO_masks/", "/scratch/tarobben/nndet/Non_PFO_masks/")