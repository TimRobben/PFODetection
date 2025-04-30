import os
import json
import numpy as np
import nibabel as nib
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nibabel.orientations import aff2axcodes

def world_to_voxel(coord_mm, affine):
    # print('coord_mm', coord_mm)
    coord_mm = np.append(coord_mm, 1)  # make homogeneous
    # print('coord_mm', coord_mm)
    # print(affine)
    # print(aff2axcodes(affine))
    voxel = np.linalg.inv(affine).dot(coord_mm)
    # print('voxel',voxel)
    # print('other voxel', voxel)
    return voxel[:3]  # return only x, y, z in voxel space

def process_all_ct_scans(ct_folder, ann_folder, output_slices_folder, output_json_path, slicing_axis=2, roi_category_id=0):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": roi_category_id, "name": "ROI", "supercategory": "none"}
        ]
    }
    global_image_id = 1
    global_annotation_id = 1

    ct_files = [f for f in os.listdir(ct_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

    for ct_filename in sorted(ct_files):
        ct_path = os.path.join(ct_folder, ct_filename)
        ct_basename = os.path.splitext(os.path.splitext(ct_filename)[0])[0]
        ann_path = os.path.join(ann_folder, ct_basename + ".mrk.json")

        roi_defined = False
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann_json = json.load(f)
            if len(ann_json.get("markups", [])) > 0:
                roi = ann_json["markups"][0]
                roi_defined = True
                center = np.array(roi["center"])
                size = np.array(roi["size"])
                orientation = np.array(roi["orientation"]).reshape(3, 3)
                half_size = size / 2.0
                corners_local = np.array([[dx, dy, dz]
                                          for dx in (-half_size[0], half_size[0])
                                          for dy in (-half_size[1], half_size[1])
                                          for dz in (-half_size[2], half_size[2])])
                
                # corners_world = center + corners_local.dot(orientation.T)
                corners_world = center + corners_local
                roi_axis_vals = corners_world[:, slicing_axis]
                roi_axis_min = np.min(roi_axis_vals)
                roi_axis_max = np.max(roi_axis_vals)
                
                plane_axes = [ax for ax in range(3) if ax != slicing_axis]
                roi_plane_min = np.min(corners_world[:, plane_axes], axis=0)
                roi_plane_max = np.max(corners_world[:, plane_axes], axis=0)
                # print(center, size, orientation)
                print(corners_local)
                print(corners_world)
                # print("axis",roi_axis_min,roi_axis_max)
                print(roi_plane_min,roi_plane_max)
        
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()
        
        orig_spacing = ct_img.header.get_zooms()[:3]
        orig_origin = ct_img.affine[:3, 3]

        if slicing_axis != 0:
            ct_data = np.moveaxis(ct_data, slicing_axis, 0)
            spacing = [orig_spacing[slicing_axis]] + [orig_spacing[i] for i in range(3) if i != slicing_axis]
            origin = [orig_origin[slicing_axis]] + [orig_origin[i] for i in range(3) if i != slicing_axis]
        else:
            spacing = orig_spacing
            origin = orig_origin

        spacing = np.array(spacing)
        origin = np.array(origin)
        # origin = origin[::-1]
        num_slices = ct_data.shape[0]
        slice_height = ct_data.shape[1]
        slice_width = ct_data.shape[2]
        # print(ct_img.header)
        # print('orig_origin',orig_origin)
        # print('orig_spacing',orig_spacing)
        # print('spacing',spacing)
        # print('origin',origin)
        # print('axial',num_slices)
        # print('sagital',slice_height)
        # print('coronal',slice_width)
        ct_output_folder = os.path.join(output_slices_folder, ct_basename)
        os.makedirs(ct_output_folder, exist_ok=True)
        count = 1
        for i in range(num_slices):
            slice_phys = origin[0] + i * spacing[0]
            slice_img = ct_data[i, :, :]
            slice_norm = ((slice_img - np.min(slice_img)) / (np.ptp(slice_img) + 1e-8) * 255).astype(np.uint8)
            slice_corrected = np.fliplr(np.rot90(slice_norm, k=-1)) 
            slice_filename = f"{ct_basename}_slice_{i:03d}.png"
            slice_filepath = os.path.join(ct_output_folder, slice_filename)
            imageio.imwrite(slice_filepath, slice_corrected)
            
            # print(slice_phys)
            # print(slice_img)
            image_entry = {
                "id": global_image_id-1,
                "file_name": slice_filepath,
                "width": slice_width,
                "height": slice_height
            }
            coco["images"].append(image_entry)
            # print(roi_defined, roi_axis_min ,'<=', slice_phys ,'<=', roi_axis_max)
        
            if roi_defined and (roi_axis_min <= slice_phys <= roi_axis_max):
                
                x0_mm = min(roi_plane_min[0], roi_plane_max[0])
                x1_mm = max(roi_plane_min[0], roi_plane_max[0])
                y0_mm = min(roi_plane_min[1], roi_plane_max[1])
                y1_mm = max(roi_plane_min[1], roi_plane_max[1])
                # print('x0', x0_mm)
                # print('x1',x1_mm)
                # print('y0', y0_mm)
                # print('y1', y1_mm)
                # print('slice_phys',slice_phys)
                v0 = world_to_voxel([-x0_mm, -y0_mm, slice_phys], ct_img.affine)
                v1 = world_to_voxel([-x1_mm, -y1_mm, slice_phys], ct_img.affine)
                # print('v0',v0)
                # print('v1', v1)
                
                if slicing_axis == 2:  # axial
                    x0_px, y0_px = v0[0], v0[1]
                    x1_px, y1_px = v1[0], v1[1]
                elif slicing_axis == 0:  # sagittal
                    x0_px, y0_px = v0[1], v0[2]
                    x1_px, y1_px = v1[1], v1[2]
                elif slicing_axis == 1:  # coronal
                    x0_px, y0_px = v0[0], v0[2]
                    x1_px, y1_px = v1[0], v1[2]


            
                bbox_x = float(min(x0_px, x1_px))
                bbox_y = float(min(y0_px, y1_px))
                bbox_width = float(abs(x1_px - x0_px))
                bbox_height = float(abs(y1_px - y0_px))
                
                # print("Expected pixel X range:", (x0_mm - origin[1]) / spacing[1], (x1_mm - origin[1]) / spacing[1])
                # print("Expected pixel Y range:", (y0_mm - origin[2]) / spacing[2], (y1_mm - origin[2]) / spacing[2])
                # print("Image shape (H x W):", ct_data.shape[1], ct_data.shape[2])
                # print("v0:", v0)
                # print("v1:", v1)
                # print("bbox_y before flip:", bbox_y)
                # print("bbox_y after flip:", slice_height - bbox_y - bbox_height)
                area = bbox_width * bbox_height

                annotation_entry = {
                    "id": global_annotation_id,
                    "image_id": global_image_id-1,
                    "category_id": roi_category_id,
                    "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0
                }
                coco["annotations"].append(annotation_entry)
                global_annotation_id += 1

            global_image_id += 1

        print(f"Processed CT scan '{ct_basename}': {num_slices} slices saved in {ct_output_folder}")

    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"Combined COCO JSON saved to {output_json_path}")
# Example usage:
if __name__ == "__main__":
    ct_folder = "/home/tarobben/scratch/MedYOLO/nifticheck/yolo/"         # Folder with complete CT scans (.nii/.nii.gz)
    ann_folder = "/home/tarobben/scratch/PFO_labels_complete/"     # Folder with corresponding annotation JSON files
    output_slices_folder = "/home/tarobben/scratch/MedYOLO/nifticheck/yolo/"   # Where subfolders for each CT will be created
    output_json_path = "/home/tarobben/scratch/MedYOLO/nifticheck/yolo/"  # Combined COCO-style JSON file for all slices
    # slicing_axis: 0 for axial, 1 for coronal, 2 for sagittal (default is 0)
    process_all_ct_scans(ct_folder, ann_folder, output_slices_folder, output_json_path, slicing_axis=2)


def show_slice_with_bbox(png_path, bbox, rotation_k=-1):
    """
    Displays a rotated PNG slice and overlays the original bounding box (no rotation).
    
    Parameters:
        png_path (str): Path to the PNG slice.
        bbox (list): Bounding box in [x, y, width, height] format.
        rotation_k (int): Number of 90° counter-clockwise rotations to apply to image only.
    """
    image = imageio.imread(png_path)
    # rotated_image = np.rot90(image, k=rotation_k)  # Rotate image only
    # flipped_image = np.fliplr(rotated_image)
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    # ax.imshow(rotated_image, cmap='gray')

    x, y, w, h = bbox  # Bbox remains unchanged
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.title(f"Rotated Image (k={rotation_k}) with Original BBox")
    plt.axis('off')
    plt.show()

# Example usage:
# Replace with the actual values you're checking
png_path = "/home/tarobben/scratch/RCNN/Test2/1027_70/1027_70_slice_143.png"
bbox = [
                141.90692625821887,
                212.2792274073289,
                33.2467532467532,
                26.597402597402606
            ]
# show_slice_with_bbox(png_path, bbox)
