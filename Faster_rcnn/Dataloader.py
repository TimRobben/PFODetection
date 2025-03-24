import torch
import torchvision
import detectron2

import subprocess
import os

def convert_nifti_to_png_med2image(nifti_file, output_dir, axis=2):
    """
    Convert a NIfTI file to PNG images using med2image as a Python module.
    
    Parameters:
      nifti_file (str): Path to the input NIfTI file (.nii or .nii.gz).
      output_dir (str): Directory where the PNG images will be saved.
      axis (int): The axis along which to slice the volume (default is 2, typically axial).
      
    This function calls med2image using 'python -m med2image' to bypass PATH limitations.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the output file pattern
    output_pattern = os.path.join(output_dir, "slice_%03d.png")
    
    # Construct the command using the module invocation
    command = [
        "python", "-m", "med2image",
        "-i", nifti_file,
        "-o", output_pattern,
        "--axis", str(axis),
        "--format", "png",
        "--force"  # Overwrite output files if they already exist
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Conversion complete. PNG files saved in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print("An error occurred during conversion:", e)

# Example usage:
# convert_nifti_to_png_med2image("path/to/your_image.nii.gz", "output/png_slices")


def get_dataset_CT(img_dir):
    dataset_dicts = []



    return dataset_dicts