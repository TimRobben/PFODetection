import os
import json 
import numpy as np
# import nibabel as nib

def rename_json_files(json_folder):
    """
    Rename .mrk.json files to match the corresponding NIfTI file format.
    Example: patientnumber.mrk.json -> patientnumber_70.json
    """

    # List all files in the directory
    for filename in os.listdir(json_folder):
        if filename.endswith("_70_70.mrk_70.mrk.json"):  # Only process .mrk.json files
            # Extract patient number (everything before .mrk.json)
            patient_number = filename.replace("_70_70.mrk_70.mrk.json", "")

            # New filename
            new_filename = f"{patient_number}_70.mrk.json"
            old_path = os.path.join(json_folder, filename)
            new_path = os.path.join(json_folder, new_filename)

            # Rename file
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {filename} -> {new_filename}")

# rename_json_files("/scratch/tarobben/PFO_labels_complete/")  # Change this to the actual folder path

def clean_json_files(json_folder):
    """
    Extracts only the necessary fields (center & size) from .json files and deletes everything else.
    Overwrites the original JSON files with cleaned versions.
    """

    for filename in os.listdir(json_folder):
        if filename.endswith("_70.mrk.json") or filename.endswith("_0000.mrk.json"):  # Only process renamed JSON files
            json_path = os.path.join(json_folder, filename)

            # Load the original JSON file
            with open(json_path, "r") as file:
                data = json.load(file)

            # Extract only necessary information
            if "markups" in data and len(data["markups"]) > 0:
                markup = data["markups"][0]  # Assume only one markup per file

                cleaned_data = {
                    "center": markup.get("center", []),
                    "size": markup.get("size", [])
                }

                # Overwrite the original file with the cleaned data
                with open(json_path, "w") as file:
                    json.dump(cleaned_data, file, indent=4)

                print(f"✅ Cleaned: {filename}")

clean_json_files("/home/tarobben/scratch/MedYOLO/PFO_labels_annotated/")  # Change this to the actual folder path

def check_nifti_sizes(nifti_folder):
    """
    Reads all NIfTI files in a folder and prints their total physical dimensions in mm.
    """
    for filename in os.listdir(nifti_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):  # Check for NIfTI files
            nifti_path = os.path.join(nifti_folder, filename)
            
            # Load NIfTI file
            nifti = nib.load(nifti_path)
            shape = nifti.shape  # Image shape (Z, X, Y)
            voxel_size = nifti.header.get_zooms()  # Voxel size in mm

            # Compute full image dimensions in mm
            Z_total = shape[0] * voxel_size[0]  # Depth
            X_total = shape[1] * voxel_size[1]  # Width
            Y_total = shape[2] * voxel_size[2]  # Height

            print(f"📂 {filename}:")
            print(f"   - Shape (voxels): {shape}")
            print(f"   - Voxel size (mm): {voxel_size}")
            print(f"   - Full size (mm): Z={Z_total:.2f}, X={X_total:.2f}, Y={Y_total:.2f}\n")


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
        json_file = f"{patient_id}.json"
        nifti_path = os.path.join(nifti_folder, nifti_file)
        json_path = os.path.join(json_folder, json_file)

        if not os.path.exists(json_path):
            print(f"❌ JSON file missing for {nifti_file}, skipping...")
            continue

        # Load NIfTI and compute volume bounds
        nifti = nib.load(nifti_path)
        shape = nifti.shape
        affine = nifti.affine

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
        corners_mm = nib.affines.apply_affine(affine, corners_vox)
        # print(corners_mm)
        X_min, Y_min, Z_min = corners_mm.min(axis=0)
        X_max, Y_max, Z_max = corners_mm.max(axis=0)

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
        # X_min, X_max = -X_min, -X_max
        # Y_min, Y_max = -Y_min, -Y_max
        # Normalize center using real bounds
        X_Center = (-center[0] - X_min) / (X_max - X_min)
        Y_Center = (-center[1] - Y_min) / (Y_max - Y_min)
        Z_Center = (center[2] - Z_min) / (Z_max - Z_min)
        # print(f"   🔎 Normalized center: Z={Z_Center:.2f}, X={X_Center:.2f}, Y={Y_Center:.2f}\n")
        


        # Check if center is within extended MedYOLO range
        if not (-0.5 <= Z_Center <= 1.5 and -0.5 <= X_Center <= 1.5 and -0.5 <= Y_Center <= 1.5):
            print(f"⚠️ Skipping {json_file} due to out-of-range normalized center:")
            print(f"   📌 Center mm: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
            print(f"   🧊 Volume mm bounds:")
            print(f"      Z = ({Z_min:.2f}, {Z_max:.2f})")
            print(f"      X = ({X_min:.2f}, {X_max:.2f})")
            print(f"      Y = ({Y_min:.2f}, {Y_max:.2f})")
            print(f"   🔎 Normalized center: Z={Z_Center:.2f}, X={X_Center:.2f}, Y={Y_Center:.2f}\n")
            continue
        # Normalize size
        Z_Length = size[0] / (Z_max - Z_min)
        X_Length = size[1] / (X_max - X_min)
        Y_Length = size[2] / (Y_max - Y_min)
        
        # Save label
        label_str = f"1 {Z_Center:.6f} {X_Center:.6f} {Y_Center:.6f} {Z_Length:.6f} {X_Length:.6f} {Y_Length:.6f}\n"
        print(label_str)
        output_txt_path = os.path.join(output_folder, f"{patient_id}.txt")

        with open(output_txt_path, "w") as txt_file:
            txt_file.write(label_str)

        print(f"✅ Converted {json_file} → {output_txt_path}")


# Run the function with your paths
# convert_json_to_medyolo(
#     nifti_folder="/scratch/tarobben/PFO_CT/",
#     json_folder="/scratch/tarobben/MedYOLO/PFO_labels_annotated/",
#     output_folder="/scratch/tarobben/MedYOLO/PFO_labels_MedYOLO/"
# )

# check_nifti_sizes("/scratch/tarobben/PFO_CT/")  # Change to your actual folder path

# rename_json_files("/scratch/tarobben/MedYOLO/All_data_formatted/")  # Change this to the actual folder path


# ====================================================================================================================================
# NO PFO 
# ====================================================================================================================================

import os
import shutil
import pandas as pd

def copy_no_pfo_nifti(xlsx_path, nifti_folder, output_folder, column_name, sheet_name=0):
    """
    Copies NIfTI files where the selected column contains 'No' to a new folder.
    Handles NIfTI filenames with variable suffixes (e.g., 65, 70, 75).

    Args:
        xlsx_path (str): Path to the Excel (.xlsx) file containing patient data.
        nifti_folder (str): Folder containing NIfTI files.
        output_folder (str): Destination folder for "No PFO" cases.
        column_name (str): Name of the column to check for 'No'.
        sheet_name (int or str): Sheet name or index (default is 0, the first sheet).
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load Excel file
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Filter patients with "No" in the specified column (case-insensitive)
    no_pfo_patients = df[df[column_name].astype(str).str.lower() == "no"]

    for index, row in no_pfo_patients.iterrows():
        patient_id = str(row["Imaging pseudo ID"]).strip()  

        # Find all possible NIfTI files for this patient (any suffix number)
        possible_files = [f for f in os.listdir(nifti_folder) if f.startswith(f"{patient_id}_") and f.endswith(".nii.gz")]
        missing_files = []
        if not possible_files:
            print(f"❌ No matching NIfTI files found for patient {patient_id}, skipping...")
            missing_files.append(patient_id)
            continue

        for nifti_filename in possible_files:
            nifti_path = os.path.join(nifti_folder, nifti_filename)
            new_path = os.path.join(output_folder, nifti_filename)

            shutil.copy2(nifti_path, new_path)  # Copy instead of move
            print(f"✅ Copied: {nifti_filename} → {output_folder}")
    print(missing_files)



def create_no_pfo_labels(nifti_folder, label_output_folder):
    """
    Creates empty MedYOLO label files for all NIfTI files in a folder.
    Ensures the label filenames exactly match the NIfTI filenames, except for the extension.

    Args:
        nifti_folder (str): Path to the folder containing NIfTI (.nii.gz) files.
        label_output_folder (str): Path where the empty label files should be saved.
    """

    # Create label output folder if it doesn't exist
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder)

    for nifti_file in os.listdir(nifti_folder):
        if nifti_file.endswith(".nii.gz"):  # Process only NIfTI files
            # Replace .nii.gz with .txt to create the matching label filename
            label_filename = nifti_file.replace(".nii.gz", ".txt")
            label_path = os.path.join(label_output_folder, label_filename)

            # Create an empty label file
            with open(label_path, "w") as label_file:
                # Uncomment below if you want a placeholder instead of an empty file
                # label_file.write("0 0 0 0 0 0 0\n")
                pass

            print(f"✅ Created label file: {label_filename}")


# create_no_pfo_labels(
#     nifti_folder="/scratch/tarobben/NO_PFO_CT/",  # Folder containing copied "No PFO" images
#     label_output_folder="/scratch/tarobben/MedYOLO/NO_PFO_labels_MedYOLO/"  # Folder to save the labels
# )

# copy_no_pfo_nifti(
#     xlsx_path="/scratch/tarobben/MTHdata_imagingID_with_scans.xlsx",
#     nifti_folder="/scratch/tarobben/CT_scans_original/",
#     output_folder="/scratch/tarobben/NO_PFO_CT/",
#     column_name="CTA_HEART_PFO"  # Adjust to match the column in your CSV
# )
