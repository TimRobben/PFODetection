import os
import json 
import numpy as np
import nibabel as nib


def convert_json_to_medyolo(nifti_folder, json_folder, output_folder):
    """
    Reads JSON files corresponding to NIfTI images, normalizes the bounding box data using
    physical coordinates from affine matrix (with full volume corner checking), and saves them in MedYOLO label format.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for nifti_file in os.listdir(nifti_folder):
        if not nifti_file.endswith(".nii.gz"):
            continue

        patient_id = nifti_file.replace(".nii.gz", "")
        json_file = f"{patient_id}.mrk.json"
        nifti_path = os.path.join(nifti_folder, nifti_file)
        json_path = os.path.join(json_folder, json_file)

        if not os.path.exists(json_path):
            print(f"❌ JSON file missing for {nifti_file}, skipping...")
            continue

        # Load NIfTI and compute volume bounds
        nifti = nib.load(nifti_path)
        shape = nifti.shape
        affine = nifti.affine
        print('origin', affine[:3,3])

        # Compute 8 corners in mm space
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
        print('vox',corners_vox)
        corners_mm = nib.affines.apply_affine(affine, corners_vox)
        print('affine', affine)
        print('mm', corners_mm)
        X_min, Y_min, Z_min = corners_mm.min(axis=0)
        X_max, Y_max, Z_max = corners_mm.max(axis=0)
        print('min',X_min,Y_min, Z_min)
        print('max', X_max, Y_max, Z_max)

        # Load JSON
        with open(json_path, "r") as file:
            data = json.load(file)

        if "center" not in data or "size" not in data:
            print(f"❌ Missing center or size in {json_file}, skipping...")
            continue

        center = data["center"]
        size = data["size"]
        # Debug + sort bounds in case affine flips signs
        Z_min, Z_max = min(Z_min, Z_max), max(Z_min, Z_max)
        X_min, X_max = min(X_min, X_max), max(X_min, X_max)
        Y_min, Y_max = min(Y_min, Y_max), max(Y_min, Y_max)

        # Y_min=Y_min*-1
        # Y_max=Y_max*-1
        # print(X_min,X_max)

        # X_min=X_min*-1
        # X_max=X_max*-1
        # print('y',Y_min,Y_max)
        # print('x', X_min,X_max)
        X_min, X_max = -X_max, -X_min
        Y_min, Y_max = -Y_max, -Y_min
        print('x',X_min,X_max)
        print('y',Y_min,Y_max)
        print('z',Z_min, Z_max)
        print('center', center)
        # Normalize center using real bounds
        X_Center = (center[0] - X_min) / (X_max - X_min)
        Y_Center = (center[1] - Y_min) / (Y_max - Y_min)
        Z_Center = (center[2] - Z_min) / (Z_max - Z_min)
        # print(f"   🔎 Normalized center: Z={Z_Center:.2f}, X={X_Center:.2f}, Y={Y_Center:.2f}\n")
        print('norm center', X_Center,Y_Center,Z_Center)

        # Check if center is within extended MedYOLO range
        # if not (-0.5 <= Z_Center <= 1.5 and -0.5 <= X_Center <= 1.5 and -0.5 <= Y_Center <= 1.5):
        #     print(f"⚠️ Skipping {json_file} due to out-of-range normalized center:")
        #     print(f"   📌 Center mm: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
        #     print(f"   🧊 Volume mm bounds:")
        #     print(f"      Z = ({Z_min:.2f}, {Z_max:.2f})")
        #     print(f"      X = ({X_min:.2f}, {X_max:.2f})")
        #     print(f"      Y = ({Y_min:.2f}, {Y_max:.2f})")
        #     print(f"   🔎 Normalized center: Z={Z_Center:.2f}, X={X_Center:.2f}, Y={Y_Center:.2f}\n")
        #     continue
        # # Normalize size
        print('xyz',size)
        Z_Length = size[2] / (Z_max - Z_min)
        X_Length = size[0] / (X_max - X_min)
        Y_Length = size[1] / (Y_max - Y_min)
        
        # Save label
        label_str = f"1 {Z_Center:.6f} {X_Center:.6f} {Y_Center:.6f} {Z_Length:.6f} {X_Length:.6f} {Y_Length:.6f}\n"
        print(label_str)
        output_txt_path = os.path.join(output_folder, f"{patient_id}.txt")

        with open(output_txt_path, "w") as txt_file:
            txt_file.write(label_str)

        print(f"✅ Converted {json_file} → {output_txt_path}")
        print(f"   📌 Center mm: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
        print(f"   🧊 Volume mm bounds:")
        print(f"      Z = ({Z_min:.2f}, {Z_max:.2f})")
        print(f"      X = ({X_min:.2f}, {X_max:.2f})")
        print(f"      Y = ({Y_min:.2f}, {Y_max:.2f})")
        print(f"   🔎 Normalized center: Z={Z_Center:.2f}, X={X_Center:.2f}, Y={Y_Center:.2f}\n")

# Run the function with your paths
convert_json_to_medyolo(
    nifti_folder="/home/tarobben/scratch/PFO_Complete/",
    json_folder="/home/tarobben/scratch/MedYOLO/PFO_labels_annotated/",
    output_folder="/home/tarobben/scratch/MedYOLO/PFO_labels_MedYOLO/"
)