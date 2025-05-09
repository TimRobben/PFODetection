import os
import pickle
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import json

def convert_predictions_to_segmentations(pred_dir, output_dir):
    """
    Convert all nnDetection prediction .pkl files in a directory into 3D label masks with predicted boxes.

    Parameters:
    - pred_dir: Path to directory containing .pkl prediction files.
    - output_dir: Path to directory to save .nii.gz output segmentations.
    """
    os.makedirs(output_dir, exist_ok=True)
    pkl_files = glob(os.path.join(pred_dir, "*.pkl"))

    for pkl_file in tqdm(pkl_files, desc="Converting predictions"):
        with open(pkl_file, 'rb') as f:
            pred = pickle.load(f)

        pred_boxes = pred['pred_boxes']
        origin = pred['itk_origin']
        spacing = pred['itk_spacing']
        direction = pred['itk_direction']
        size = pred['original_size_of_raw_data']

        # Handle size: if scalar, assume it's a cube
        if isinstance(size, (int, float)):
            size = [size] * 3
        elif isinstance(size, np.ndarray):
            size = size.tolist()

        # Create empty image
        seg = sitk.Image(size, sitk.sitkUInt8)
        seg.SetOrigin(origin)
        seg.SetSpacing(spacing)
        seg.SetDirection(direction)

        # Draw boxes into segmentation
        for box in pred_boxes:
            x1, x2, y1, y2, z1, z2 = map(int, box)
            for z in range(z1, z2):
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
                            seg[x, y, z] = 1

        # Save the segmentation
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
        out_path = os.path.join(output_dir, f"{base_name}_boxes.nii.gz")
        sitk.WriteImage(seg, out_path)

# convert_predictions_to_segmentations(
#     pred_dir="/home/tarobben/scratch/nndet/Task001_model/Task001_test/RetinaUNetV001_D3V001_3d/consolidated/val_predictions/",
#     output_dir="/home/tarobben/scratch/nndet/Task001_model/"
# )

def convert_predictions_to_slicer_json(prediction_dir, output_dir, score_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(prediction_dir):
        if not file.endswith(".pkl"):
            continue

        pred_path = os.path.join(prediction_dir, file)
        with open(pred_path, "rb") as f:
            data = pickle.load(f)

        boxes = data.get("pred_boxes", [])
        scores = data.get("pred_scores", [])
        spacing = np.array(data.get("itk_spacing", [1.0, 1.0, 1.0]), dtype=float)
        origin = np.array(data.get("itk_origin", [0.0, 0.0, 0.0]), dtype=float)

        if len(boxes) == 0 or len(scores) == 0:
            print(f"{file}: No predictions.")
            continue

        valid_indices = [i for i, s in enumerate(scores) if float(s) >= score_threshold]
        if not valid_indices:
            print(f"{file}: No boxes above threshold.")
            continue

        for i in valid_indices:
            box = boxes[i].astype(float)
            score = float(scores[i])

            # Convert voxel coordinates to physical LPS space
            # box = [x1, y1, z1, x2, y2, z2]
            p1 = box[:3] * spacing + origin
            p2 = box[3:] * spacing + origin

            center = ((p1 + p2) / 2).tolist()
            size = np.abs(p2 - p1).tolist()

            roi = {
                    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
                    "markups": [{
                        "type": "ROI",
                        "coordinateSystem": "LPS",
                        "coordinateUnits": "mm",
                        "locked": False,
                        "fixedNumberOfControlPoints": False,
                        "labelFormat": "%N-%d",
                        "lastUsedControlPointNumber": 1,
                        "roiType": "Box",
                        "center": center,
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "size": size,
                        "insideOut": False,
                        "controlPoints": [{
                            "id": "1",
                            "label": f"{file[:-4]}-{i+1} ({score:.2f})",
                            "description": f"Confidence score: {score:.4f}",
                            "associatedNodeID": "vtkMRMLScalarVolumeNode1",
                            "position": center,
                            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined"
                        }],
                        "measurements": [{
                            "name": "volume",
                            "enabled": False,
                            "units": "cm3",
                            "printFormat": "%-#4.4g%s"
                        }],
                        "display": {
                            "visibility": False,
                            "opacity": 1.0,
                            "color": [0.4, 1.0, 1.0],
                            "selectedColor": [0.45, 0.45, 0.45],
                            "activeColor": [0.4, 1.0, 0.0],
                            "propertiesLabelVisibility": True,
                            "pointLabelsVisibility": False,
                            "textScale": 3.0,
                            "glyphType": "Sphere3D",
                            "glyphScale": 3.0,
                            "glyphSize": 5.0,
                            "useGlyphScale": True,
                            "sliceProjection": False,
                            "sliceProjectionUseFiducialColor": True,
                            "sliceProjectionOutlinedBehindSlicePlane": False,
                            "sliceProjectionColor": [1.0, 1.0, 1.0],
                            "sliceProjectionOpacity": 0.6,
                            "lineThickness": 0.2,
                            "lineColorFadingStart": 1.0,
                            "lineColorFadingEnd": 10.0,
                            "lineColorFadingSaturation": 1.0,
                            "lineColorFadingHueOffset": 0.0,
                            "handlesInteractive": True,
                            "translationHandleVisibility": True,
                            "rotationHandleVisibility": False,
                            "scaleHandleVisibility": True,
                            "interactionHandleScale": 3.0,
                            "snapMode": "toVisibleSurface"
                        }
                    }]
                }


            out_path = os.path.join(output_dir, f"{file[:-4]}_{i}.mrk.json")
            with open(out_path, "w") as f_out:
                json.dump(roi, f_out, indent=4)

        print(f"{file}: {len(valid_indices)} predictions converted.")


convert_predictions_to_slicer_json("/home/tarobben/scratch/nndet/Task001_model/Task001_test/RetinaUNetV001_D3V001_3d/consolidated/test_predictions/",
output_dir="/home/tarobben/scratch/nndet/Task001_model/pred", score_threshold=0.9)
        
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_all_folds_losses(folds_root_dir):
    """
    Plot training and validation losses for all folds in one figure.
    Each fold must have `ml_runs.txt` and `val_losses.txt` in its folder.
    
    Args:
        folds_root_dir (str): Path to the directory containing all fold folders.
    """
    plt.figure(figsize=(12, 6))
    
    for fold_name in sorted(os.listdir(folds_root_dir)):
        fold_path = os.path.join(folds_root_dir, fold_name)
        if not os.path.isdir(fold_path):
            continue
        
        # Paths to training and validation loss files
        train_file = os.path.join(fold_path, "train_loss")
        val_file = os.path.join(fold_path, "val_loss")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            print(f"Skipping {fold_name}, missing files.")
            continue
        
        # Read training losses
        train_epochs = []
        train_losses = []
        with open(train_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    train_losses.append(float(parts[1]))
                    train_epochs.append(int(parts[2]))

        # Convert iteration count to actual epochs
        train_epochs = np.array(train_epochs) // 2500  # Assuming 2500 iters per epoch
        train_losses = np.array(train_losses)

        # Read validation losses
        val_losses = []
        with open(val_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    val_losses.append(float(parts[1]))

        val_epochs = list(range(len(val_losses)))

        # Plot both
        plt.plot(train_epochs, train_losses, '--', alpha=0.6, label=f"{fold_name} Train")
        plt.plot(val_epochs, val_losses, '-', label=f"{fold_name} Val")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses for All Folds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot_all_folds_losses("/home/tarobben/scratch/nndet/Task001_model/Losses/")
