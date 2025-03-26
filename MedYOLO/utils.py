import os
import json
import numpy as np
import nibabel as nib

def validate_medyolo_labels(json_folder, nifti_folder, max_distance=1.5):
    """
    Checks MedYOLO label files for out-of-range centers and fully out-of-volume boxes.

    Args:
        json_folder (str): Folder containing .json annotation files.
        nifti_folder (str): Folder containing .nii.gz files (matching names).
        max_distance (float): Acceptable normalized range (default [-0.5, 1.5]).
    """
    invalid_labels = []
    fully_outside = []

    print("🔍 Validating labels...")
    for json_file in os.listdir(json_folder):
        if not json_file.endswith(".json"):
            continue

        base_name = os.path.splitext(json_file)[0]
        nifti_file = f"{base_name}.nii.gz"
        nifti_path = os.path.join(nifti_folder, nifti_file)
        json_path = os.path.join(json_folder, json_file)

        if not os.path.exists(nifti_path):
            print(f"⚠️ Missing NIfTI file for {json_file}")
            continue

        try:
            # Load volume info
            nifti = nib.load(nifti_path)
            shape = nifti.shape  # (Z, X, Y)
            voxel_size = nifti.header.get_zooms()  # (Z, X, Y)
            Z_total = shape[0] * voxel_size[0]
            X_total = shape[1] * voxel_size[1]
            Y_total = shape[2] * voxel_size[2]

            # Load annotation
            with open(json_path, "r") as f:
                data = json.load(f)

            if "center" not in data or "size" not in data:
                print(f"❌ Missing center/size in {json_file}")
                invalid_labels.append(json_file)
                continue

            # Convert from LPS → RAS
            center_lps = data["center"]
            center_ras = [-center_lps[0], -center_lps[1], center_lps[2]]
            size = data["size"]

            # Normalize centers
            Zc = center_ras[0] / Z_total
            Xc = center_ras[1] / X_total
            Yc = center_ras[2] / Y_total

            if any([Zc < -max_distance, Zc > max_distance,
                    Xc < -max_distance, Xc > max_distance,
                    Yc < -max_distance, Yc > max_distance]):
                print(f"❌ Out-of-range center in {json_file}: Z={Zc:.2f}, X={Xc:.2f}, Y={Yc:.2f}")
                invalid_labels.append(json_file)

            # Calculate real-world bounding box extents
            z_min = center_ras[0] - size[0] / 2
            z_max = center_ras[0] + size[0] / 2
            x_min = center_ras[1] - size[1] / 2
            x_max = center_ras[1] + size[1] / 2
            y_min = center_ras[2] - size[2] / 2
            y_max = center_ras[2] + size[2] / 2

            if (z_max < 0 or z_min > Z_total or
                x_max < 0 or x_min > X_total or
                y_max < 0 or y_min > Y_total):
                print(f"🚫 Box fully outside volume: {json_file}")
                fully_outside.append(json_file)

        except Exception as e:
            print(f"⚠️ Error reading {json_file}: {e}")
            invalid_labels.append(json_file)

    print("\n📋 Validation Summary:")
    print(f"  🔴 Invalid centers: {len(invalid_labels)} files")
    print(f"  ⚠️  Boxes fully outside scan: {len(fully_outside)} files")

# Example usage:
# validate_medyolo_labels(
#     json_folder="/scratch/tarobben/MedYOLO/PFO_labels_annotated/",
#     nifti_folder="/scratch/tarobben/PFO_CT/"
# )




def check_label_center_inside_volume(json_folder, nifti_folder):
    """
    Verifies that each label center (converted to RAS) lies within the actual
    physical bounds of the image volume using the affine transformation.
    """
    outside_cases = []

    for json_file in os.listdir(json_folder):
        if not json_file.endswith(".json"):
            continue

        base_name = os.path.splitext(json_file)[0]
        nifti_file = f"{base_name}.nii.gz"
        json_path = os.path.join(json_folder, json_file)
        nifti_path = os.path.join(nifti_folder, nifti_file)

        if not os.path.exists(nifti_path):
            print(f"❌ NIfTI file not found for {json_file}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if "center" not in data:
                print(f"⚠️ Missing center in {json_file}, skipping...")
                continue

            center_lps = data["center"]
            center_ras = np.array([-center_lps[0], -center_lps[1], center_lps[2]])

            nifti = nib.load(nifti_path)
            shape = nifti.shape
            affine = nifti.affine

            # Get all 8 corner voxel coordinates
            corners_vox = np.array([
                [0, 0, 0],
                [0, 0, shape[2]-1],
                [0, shape[1]-1, 0],
                [0, shape[1]-1, shape[2]-1],
                [shape[0]-1, 0, 0],
                [shape[0]-1, 0, shape[2]-1],
                [shape[0]-1, shape[1]-1, 0],
                [shape[0]-1, shape[1]-1, shape[2]-1],
            ])
            corners_mm = nib.affines.apply_affine(affine, corners_vox)

            Z_min, X_min, Y_min = corners_mm.min(axis=0)
            Z_max, X_max, Y_max = corners_mm.max(axis=0)

            Zc, Xc, Yc = center_ras
            print(f"file: {json_file}")
            print(f"    Center (RAS): Z={Zc:.2f}, X={Xc:.2f}, Y={Yc:.2f}")
            print(f"    Volume Z=({Z_min:.2f}, {Z_max:.2f}), X=({X_min:.2f}, {X_max:.2f}), Y=({Y_min:.2f}, {Y_max:.2f})")

            if not (Z_min <= Zc <= Z_max and X_min <= Xc <= X_max and Y_min <= Yc <= Y_max):
                print(f"🚫 {json_file} center outside volume:")
                print(f"    Center (RAS): Z={Zc:.2f}, X={Xc:.2f}, Y={Yc:.2f}")
                print(f"    Volume Z=({Z_min:.2f}, {Z_max:.2f}), X=({X_min:.2f}, {X_max:.2f}), Y=({Y_min:.2f}, {Y_max:.2f})")
                outside_cases.append(json_file)

        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")

    print("\n✅ Check complete.")
    print(f"Total cases outside volume: {len(outside_cases)}")


check_label_center_inside_volume("/scratch/tarobben/MedYOLO/PFO_labels_annotated/", "/scratch/tarobben/PFO_CT/")