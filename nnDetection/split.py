import os
import shutil
import random

def balanced_train_test_split(
    pfo_img_dir,
    nonpfo_img_dir,
    pfo_label_dir,
    nonpfo_label_dir,
    output_dir,
    test_ratio=0.2,
    seed=42
):
    random.seed(seed)

    # Get lists
    pfo_files = sorted([f for f in os.listdir(pfo_img_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
    nonpfo_files = sorted([f for f in os.listdir(nonpfo_img_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

    random.shuffle(pfo_files)
    random.shuffle(nonpfo_files)

    # Calculate how many PFOs to use for test
    num_pfo_test = max(1, int(len(pfo_files) * test_ratio))
    num_pfo_train = len(pfo_files) - num_pfo_test

    # Sample matching number of non-PFOs
    nonpfo_test = random.sample(nonpfo_files, num_pfo_test * 16)
    remaining_nonpfo = [f for f in nonpfo_files if f not in nonpfo_test]
    nonpfo_train = random.sample(remaining_nonpfo, num_pfo_train)

    # Split sets
    pfo_train = pfo_files[num_pfo_test:]
    pfo_test = pfo_files[:num_pfo_test]

    train = [(pfo_img_dir, pfo_label_dir, f) for f in pfo_train] + \
            [(nonpfo_img_dir, nonpfo_label_dir, f) for f in nonpfo_train]

    test = [(pfo_img_dir, pfo_label_dir, f) for f in pfo_test] + \
           [(nonpfo_img_dir, nonpfo_label_dir, f) for f in nonpfo_test]

    print(f"[INFO] Train: {len(train)} (PFO: {len(pfo_train)}, non-PFO: {len(nonpfo_train)})")
    print(f"[INFO] Test: {len(test)} (PFO: {len(pfo_test)}, non-PFO: {len(nonpfo_test)})")

    for split_name, split_list in zip(["Tr", "Ts"], [train, test]):
        img_out = os.path.join(output_dir, f"images{split_name}")
        lbl_out = os.path.join(output_dir, f"labels{split_name}")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_dir, lbl_dir, fname in split_list:
            case_id = fname.replace(".nii.gz", "").replace(".nii", "")
            label_name = f"{case_id}.nii.gz"
            json_name = label_name.replace(".nii.gz", ".json").replace(".nii", ".json")

            shutil.copy(os.path.join(img_dir, fname), os.path.join(img_out, fname))

            label_path = os.path.join(lbl_dir, label_name)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(lbl_out, label_name))

            json_path = os.path.join(lbl_dir, json_name)
            if os.path.exists(json_path):
                shutil.copy(json_path, os.path.join(lbl_out, json_name))

    print("[✓] Balanced stratified split done.")


# balanced_train_test_split("/scratch/tarobben/PFO_Complete/", "/scratch/tarobben/NO_PFO_CT/", "/scratch/tarobben/nndet/PFO_masks/", "/scratch/tarobben/nndet/Non_PFO_masks/", "/scratch/tarobben/nndet/Task002/", test_ratio=0.2)


def validate_image_label_pairs(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz")]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]

    # Get case IDs (strip off _0000.nii.gz for images, .nii.gz for labels)
    image_ids = set([f.replace("_0000.nii.gz", "") for f in image_files])
    label_ids = set([f.replace(".nii.gz", "") for f in label_files])

    missing_labels = image_ids - label_ids
    missing_images = label_ids - image_ids

    print("✅ Matched pairs:", len(image_ids & label_ids))
    
    if missing_labels:
        print("❌ Images with no matching labels:")
        for f in sorted(missing_labels):
            print(f"  {f}_0000.nii.gz")

    if missing_images:
        print("❌ Labels with no matching images:")
        for f in sorted(missing_images):
            print(f"  {f}.nii.gz")

    if not missing_labels and not missing_images:
        print("🎉 All image/label pairs match!")

# Replace these with your actual paths
validate_image_label_pairs(
    images_dir="/home/tarobben/scratch/nndet/Task002/raw_splitted/imagesTr/",
    labels_dir="/home/tarobben/scratch/nndet/Task002/raw_splitted/labelsTr/"
)
