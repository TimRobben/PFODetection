import json
import random
import os
from collections import defaultdict

def extract_scan_id(file_name):
    return file_name.split("/")[0]  # e.g., CT001 from CT001/CT001_slice_012.png

def stratified_split_by_scan(json_path, out_train, out_val, val_ratio=0.2, seed=42):
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

    # Determine if a scan is positive (has at least one annotation)
    scan_labels = {}
    for scan_id, imgs in scans.items():
        has_pfo = any(img["id"] in anns_by_image for img in imgs)
        scan_labels[scan_id] = int(has_pfo)

    # Separate scans by label
    pos_scans = [sid for sid, label in scan_labels.items() if label == 1]
    neg_scans = [sid for sid, label in scan_labels.items() if label == 0]

    print(f"Total scans: {len(scans)}")
    print(f"Positive scans: {len(pos_scans)}")
    print(f"Negative scans: {len(neg_scans)}")

    # Split function
    def split_scans(scan_list):
        random.shuffle(scan_list)
        val_size = int(len(scan_list) * val_ratio)
        return scan_list[val_size:], scan_list[:val_size]

    random.seed(seed)
    train_pos, val_pos = split_scans(pos_scans)
    train_neg, val_neg = split_scans(neg_scans)

    train_scans = set(train_pos + train_neg)
    val_scans = set(val_pos + val_neg)


    train_images = []
    val_images = [] 
    for sid in train_scans:
        train_images.extend(scans[sid])
    for sid in val_scans:
        val_images.extend(scans[sid])

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_anns = [ann for ann in annotations if ann["image_id"] in train_ids]
    val_anns = [ann for ann in annotations if ann["image_id"] in val_ids]

    def save_json(path, imgs, anns):
        with open(path, "w") as f:
            json.dump({
                "images": imgs,
                "annotations": anns,
                "categories": categories
            }, f, indent=4)
        print(f"Wrote {len(imgs)} images to {path}")

    save_json(out_train, train_images, train_anns)
    save_json(out_val, val_images, val_anns)

# Example usage:
stratified_split_by_scan(
    "/home/tarobben/scratch/RCNN/PFO_png/pfo_coco.json",
    "/home/tarobben/scratch/RCNN/pfo_train.json",
    "/home/tarobben/scratch/RCNN/pfo_val.json",
    val_ratio=0.2
)
