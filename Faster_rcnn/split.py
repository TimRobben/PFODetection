import json
import random
import os
from collections import defaultdict

def extract_scan_id(file_name):
    return file_name.split("/")[-2]  # e.g., CT001 from CT001/CT001_slice_012.png

def scan_split_coco(json_path, out_train, out_val, val_ratio=0.2, seed=42):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    # Group images by scan ID
    scans = defaultdict(list)
    for img in images:
        scan_id = extract_scan_id(img["file_name"])
        scans[scan_id].append(img)

    print(f"Total scans: {len(scans)}")

    # Split scan IDs randomly
    all_scan_ids = list(scans.keys())
    random.seed(seed)
    random.shuffle(all_scan_ids)
    val_size = int(len(all_scan_ids) * val_ratio)
    val_scan_ids = set(all_scan_ids[:val_size])
    train_scan_ids = set(all_scan_ids[val_size:])

    # Gather images for each split
    train_images = []
    val_images = []

    for sid in train_scan_ids:
        train_images.extend(scans[sid])
    for sid in val_scan_ids:
        val_images.extend(scans[sid])

    # Get image IDs
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    # Filter annotations by split
    train_anns = [ann for ann in annotations if ann["image_id"] in train_ids]
    val_anns = [ann for ann in annotations if ann["image_id"] in val_ids]

    # Helper to write out new JSON
    def save_json(path, imgs, anns):
        with open(path, "w") as f:
            json.dump({
                "images": imgs,
                "annotations": anns,
                "categories": categories
            }, f, indent=4)
        print(f"Wrote {len(imgs)} images and {len(anns)} annotations to {path}")

    save_json(out_train, train_images, train_anns)
    save_json(out_val, val_images, val_anns)

# 🔧 Example usage:
# scan_split_coco(
#     "/home/tarobben/scratch/RCNN/PFO_png/pfo_coco.json",
#     "/home/tarobben/scratch/RCNN/pfo_train.json",
#     "/home/tarobben/scratch/RCNN/pfo_val.json",
#     val_ratio=0.2
# )

def overfit():
    with open("/home/tarobben/scratch/RCNN/pfo_train.json") as f:
        data = json.load(f)

    # Only keep images that have annotations
    image_ids_with_anns = set(ann["image_id"] for ann in data["annotations"])
    annotated_images = [img for img in data["images"] if img["id"] in image_ids_with_anns]

    # Sample 10 examples
    sampled_images = random.sample(annotated_images, 10)
    sampled_ids = set(img["id"] for img in sampled_images)

    sampled_anns = [ann for ann in data["annotations"] if ann["image_id"] in sampled_ids]

    mini_data = {
        "images": sampled_images,
        "annotations": sampled_anns,
        "categories": data["categories"]
    }

    with open("/home/tarobben/scratch/RCNN/output_pfo/pfo_tiny.json", "w") as f:
        json.dump(mini_data, f, indent=4)

overfit()