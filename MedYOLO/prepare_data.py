import os
import json 
import nibabel as nib

def rename_json_files(json_folder):
    """
    Rename .mrk.json files to match the corresponding NIfTI file format.
    Example: patientnumber.mrk.json -> patientnumber_70.json
    """

    # List all files in the directory
    for filename in os.listdir(json_folder):
        if filename.endswith(".mrk.json"):  # Only process .mrk.json files
            # Extract patient number (everything before .mrk.json)
            patient_number = filename.replace(".mrk.json", "")

            # New filename
            new_filename = f"{patient_number}_70.json"
            old_path = os.path.join(json_folder, filename)
            new_path = os.path.join(json_folder, new_filename)

            # Rename file
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {filename} -> {new_filename}")


def clean_json_files(json_folder):
    """
    Extracts only the necessary fields (center & size) from .json files and deletes everything else.
    Overwrites the original JSON files with cleaned versions.
    """

    for filename in os.listdir(json_folder):
        if filename.endswith("_70.json"):  # Only process renamed JSON files
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
    Reads JSON files corresponding to NIfTI images, normalizes the bounding box data, 
    and saves them in MedYOLO label format.

    Args:
        nifti_folder (str): Path to the folder containing NIfTI (.nii.gz) files.
        json_folder (str): Path to the folder containing corresponding JSON (.json) files.
        output_folder (str): Path where the MedYOLO labels (.txt) will be saved.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    for nifti_file in os.listdir(nifti_folder):
        if nifti_file.endswith(".nii.gz"):  # Process only NIfTI files
            patient_id = nifti_file.replace("_70.nii.gz", "")  # Extract patient ID
            json_file = f"{patient_id}_70.json"  # Expected JSON filename
            json_path = os.path.join(json_folder, json_file)
            nifti_path = os.path.join(nifti_folder, nifti_file)

            if not os.path.exists(json_path):
                print(f"❌ JSON file missing for {nifti_file}, skipping...")
                continue

            # Load NIfTI file to get image dimensions
            nifti = nib.load(nifti_path)
            shape = nifti.shape  # (Z, X, Y)
            voxel_size = nifti.header.get_zooms()  # (Z, X, Y) voxel size in mm

            # Compute full image dimensions in mm
            Z_total = shape[0] * voxel_size[0]
            X_total = shape[1] * voxel_size[1]
            Y_total = shape[2] * voxel_size[2]

            # Load JSON file
            with open(json_path, "r") as file:
                data = json.load(file)

            if "center" not in data or "size" not in data:
                print(f"❌ Missing center or size in {json_file}, skipping...")
                continue

            center = data["center"]  # Center in mm (Z, X, Y)
            size = data["size"]  # Size in mm (Z-Length, X-Length, Y-Length)

            # Normalize coordinates
            Z_Center = max(-0.5, min(1.5, center[0] / Z_total))
            X_Center = max(-0.5, min(1.5, center[1] / X_total))
            Y_Center = max(-0.5, min(1.5, center[2] / Y_total))

            Z_Length = size[0] / Z_total
            X_Length = size[1] / X_total
            Y_Length = size[2] / Y_total

            # Save in MedYOLO format
            medyolo_label = f"1 {Z_Center:.6f} {X_Center:.6f} {Y_Center:.6f} {Z_Length:.6f} {X_Length:.6f} {Y_Length:.6f}\n"
            output_txt_path = os.path.join(output_folder, f"{patient_id}_70.txt")

            with open(output_txt_path, "w") as txt_file:
                txt_file.write(medyolo_label)

            print(f"✅ Converted {json_file} → {output_txt_path}")


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

import os

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


create_no_pfo_labels(
    nifti_folder="/scratch/tarobben/NO_PFO_CT/",  # Folder containing copied "No PFO" images
    label_output_folder="/scratch/tarobben/MedYOLO/NO_PFO_labels_MedYOLO/"  # Folder to save the labels
)

# copy_no_pfo_nifti(
#     xlsx_path="/scratch/tarobben/MTHdata_imagingID_with_scans.xlsx",
#     nifti_folder="/scratch/tarobben/CT_scans_original/",
#     output_folder="/scratch/tarobben/NO_PFO_CT/",
#     column_name="CTA_HEART_PFO"  # Adjust to match the column in your CSV
# )
# convert_json_to_medyolo(
#     nifti_folder="/scratch/tarobben/PFO_CT/",
#     json_folder="/scratch/tarobben/MedYOLO/PFO_labels_annotated/",
#     output_folder="/scratch/tarobben/MedYOLO/PFO_labels_MedYOLO/"
# )
# check_nifti_sizes("/scratch/tarobben/PFO_CT/")  # Change to your actual folder path
# clean_json_files("/scratch/tarobben/MedYOLO/All_data_formatted/")  # Change this to the actual folder path
# rename_json_files("/scratch/tarobben/MedYOLO/All_data_formatted/")  # Change this to the actual folder path
