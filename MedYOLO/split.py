import os
import shutil
import random
from pathlib import Path
import yaml

def prepare_medyolo_split(
    pfo_image_dir,
    pfo_label_dir,
    no_pfo_image_dir,
    no_pfo_label_dir,
    output_base_dir,
    n_pfo_total=18,
    n_no_pfo_total=72,
    val_ratio=0.2,
    seed=42
):
    """
    Prepares a MedYOLO-compatible dataset split with specified number of PFO and No PFO cases.

    Args:
        pfo_image_dir (str): Path to PFO .nii.gz files.
        pfo_label_dir (str): Path to PFO .txt label files.
        no_pfo_image_dir (str): Path to No PFO .nii.gz files.
        no_pfo_label_dir (str): Path to No PFO .txt label files.
        output_base_dir (str): Path where the new split will be saved.
        n_pfo_total (int): Number of PFO samples to use.
        n_no_pfo_total (int): Number of No PFO samples to use.
        val_ratio (float): Ratio of data to use for validation (e.g., 0.2 = 20%).
        seed (int): Random seed for reproducibility.
    """

    random.seed(seed)

    # Collect files
    pfo_images = sorted([f for f in os.listdir(pfo_image_dir) if f.endswith(".nii.gz")])
    no_pfo_images = sorted([f for f in os.listdir(no_pfo_image_dir) if f.endswith(".nii.gz")])

    # Subsample
    selected_pfo = random.sample(pfo_images, min(n_pfo_total, len(pfo_images)))
    selected_no_pfo = random.sample(no_pfo_images, min(n_no_pfo_total, len(no_pfo_images)))

    # Combine and shuffle
    all_samples = [(f, 'pfo') for f in selected_pfo] + [(f, 'no_pfo') for f in selected_no_pfo]
    random.shuffle(all_samples)

    # Split
    n_val = int(len(all_samples) * val_ratio)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    def copy_samples(samples, split):
        for fname, label in samples:
            src_img = os.path.join(pfo_image_dir if label == 'pfo' else no_pfo_image_dir, fname)
            src_lbl = os.path.join(pfo_label_dir if label == 'pfo' else no_pfo_label_dir, fname.replace(".nii.gz", ".txt"))

            dst_img = os.path.join(output_base_dir, 'images', split, fname)
            dst_lbl = os.path.join(output_base_dir, 'labels', split, fname.replace(".nii.gz", ".txt"))

            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    copy_samples(train_samples, 'train')
    copy_samples(val_samples, 'val')

    print(f"✅ Prepared MedYOLO split at: {output_base_dir}")
    print(f"   → Train: {len(train_samples)} images")
    print(f"   → Val: {len(val_samples)} images")


def write_medyolo_yaml(output_base_dir, yaml_path='/home/tarobben/Documents/PFODetection/MedYOLO/MedYOLO-main/data/Test.yaml'):
    """
    Writes a MedYOLO-compatible data.yaml file.

    Args:
        output_base_dir (str): The base output directory that contains images/train and images/val.
        yaml_path (str): Optional path to save the YAML file. Defaults to <output_base_dir>/data.yaml
    """
    if yaml_path is None:
        yaml_path = os.path.join(output_base_dir, "data.yaml")

    data = {
        "train": os.path.join(output_base_dir, "images", "train"),
        "val": os.path.join(output_base_dir, "images", "val"),
        "nc": 2,
        "names": ["no_pfo", "pfo"]
    }
    print(data)
    # Write to YAML file
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=True)


    print(f"📄 data.yaml written to: {yaml_path}")


# Example of how you'd add this to your pipeline:
write_medyolo_yaml("/scratch/tarobben/MedYOLO/Test/")


# prepare_medyolo_split(
#     pfo_image_dir="/scratch/tarobben/PFO_CT/",
#     pfo_label_dir="/scratch/tarobben/MedYOLO/PFO_labels_MedYOLO/",
#     no_pfo_image_dir="/scratch/tarobben/NO_PFO_CT/",
#     no_pfo_label_dir="/scratch/tarobben/MedYOLO/NO_PFO_labels_MedYOLO/",
#     output_base_dir="/scratch/tarobben/MedYOLO/Test/",
#     n_pfo_total=18,
#     n_no_pfo_total=72,
#     val_ratio=0.2
# )

