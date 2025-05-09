import os, shutil, random, yaml
from pathlib import Path

def prepare_medyolo_split(
        pfo_img_dir, pfo_lbl_dir,
        neg_img_dir, neg_lbl_dir,
        out_dir,
        n_train_dup=4,         # duplicate factor applied **only** to train PFOs
        n_val_pfo=4,           # keep these original PFOs for val
        n_train_neg=80,        # negatives kept for train
        n_val_neg=80,          # negatives kept for val  (reflecting prevalence)
        seed=42):

    random.seed(seed)
    out_dir = Path(out_dir)

    # ---------- collect file stems ----------
    pfo_files = sorted(f for f in os.listdir(pfo_img_dir) if f.endswith('.nii.gz'))
    neg_files = sorted(f for f in os.listdir(neg_img_dir) if f.endswith('.nii.gz'))

    assert len(pfo_files) >= n_val_pfo, "Not enough PFO cases!"

    # ---------- split positives ----------
    val_pfo = random.sample(pfo_files, n_val_pfo)
    train_pfo = [f for f in pfo_files if f not in val_pfo]

    # duplicate train PFOs k‑times
    train_pfo_dup = train_pfo * n_train_dup

    # ---------- sample negatives ----------
    random.shuffle(neg_files)
    train_neg = neg_files[:n_train_neg]
    val_neg   = neg_files[n_train_neg:n_train_neg+n_val_neg]

    # ---------- helper -----------
    def copy_set(file_list, split, label_dir, img_dir, make_unique=True):
        for idx, stem in enumerate(file_list):
            if make_unique:
                # e.g. 1160_70_dup2.nii.gz
                dst_stem = stem.replace('.nii.gz', f'_dup{idx}.nii.gz')
            else:
                dst_stem = stem
            src_img = Path(img_dir)/stem
            src_lbl = Path(label_dir)/stem.replace('.nii.gz','.txt')

            dst_img = out_dir/'images'/split/dst_stem
            dst_lbl = out_dir/'labels'/split/dst_stem.replace('.nii.gz','.txt')
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)                        # or os.link for hard‑link
            shutil.copy2(src_lbl, dst_lbl)

 # ---------- copy files ----------
    copy_set(train_pfo_dup, 'train', pfo_lbl_dir, pfo_img_dir)
    copy_set(train_neg,     'train', neg_lbl_dir, neg_img_dir)
    copy_set(val_pfo,       'val',   pfo_lbl_dir, pfo_img_dir)
    copy_set(val_neg,       'val',   neg_lbl_dir, neg_img_dir)

    print(f"Train  : {len(train_pfo_dup)} PFO  + {len(train_neg)} No‑PFO")
    print(f"Val    : {len(val_pfo)} PFO  + {len(val_neg)} No‑PFO")
    print("✅ dataset written to", out_dir)
    # ---------- write YAML ----------
    data = {
        'train': str(out_dir/'images'/'train'),
        'val'  : str(out_dir/'images'/'val'),
        'nc'   : 2,
        'names': ['no_pfo', 'pfo']
    }
    yaml_path = out_dir/'data.yaml'
    with open(yaml_path,'w') as f: yaml.safe_dump(data, f, sort_keys=False)
    print('data yaml saved to ', yaml_path)

    
    # ---------------- usage ----------------
prepare_medyolo_split(
    pfo_img_dir = "/scratch/tarobben/PFO_Complete/",
    pfo_lbl_dir = "/scratch/tarobben/MedYOLO/PFO_labels_MedYOLO/",
    neg_img_dir = "/scratch/tarobben/NO_PFO_CT/",
    neg_lbl_dir = "/scratch/tarobben/MedYOLO/NO_PFO_labels_MedYOLO/",
    out_dir     = "/scratch/tarobben/MedYOLO/Test/",
    n_train_dup = 4,
    n_val_pfo   = 4,
    n_train_neg = 60,
    n_val_neg   = 60
)
