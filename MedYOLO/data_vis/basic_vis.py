import os, glob, numpy as np, nibabel as nib, pandas as pd

def summarize_pfo_vs_nopfo(
        pfo_img_dir : str,
        pfo_lab_dir : str,
        nopfo_img_dir: str,
        save_csv     = True,
):
    """
    Quick‐and‐dirty audit for a **binary** MedYOLO dataset that is stored in two
    completely separate trees:

    ├── PFO_images/   (*.nii.gz)
    ├── PFO_labels/   (*.txt  with boxes – class 1)
    ├── NO_PFO_images/(*.nii.gz)                # -- no boxes
    └── (no labels for NO-PFO)

    • Parses every bounding-box in `pfo_lab_dir`
    • Converts d-w-h from normalised YOLO units → millimetres
    • Gathers slice-thickness stats for *both* sets
    • Prints a tidy summary and (optionally) writes two CSVs

    Returns
    -------
    tuple(summary_df, slice_df)
        summary_df : bbox size stats for class 1 (PFO)
        slice_df   : slice-thickness distribution for both groups
    """

    # ---------- helper ----------
    def nii_meta(path):
        nii   = nib.load(path)
        shp   = np.array(nii.shape)              # (Z,X,Y)
        zoom  = np.array(nii.header.get_zooms()) # mm
        return shp, zoom, float(zoom[0])         # slice-thickness

    # ---------- scan PFO ----------
    rec = []                 # rows = one bounding-box
    used_imgs = set()        # to log slice dz once per image
    slice_rows = []          # slice-thickness rows

    for lab in glob.glob(os.path.join(pfo_lab_dir, "*.txt")):
        stem   = os.path.basename(lab)[:-4]
        img    = os.path.join(pfo_img_dir, stem + ".nii.gz")
        if not os.path.exists(img):
            print(f"⚠️  missing image for label {lab}")
            continue
        shape, vox, dz = nii_meta(img)
        if stem not in used_imgs:
            slice_rows.append(dict(group="PFO", image=stem, dz_mm=dz))
            used_imgs.add(stem)

        with open(lab) as f:
            for ln in (l.strip() for l in f if l.strip()):
                prt = ln.split()
                if len(prt) != 7:
                    print(f"❌ malformed: {lab} → {ln}")
                    continue
                _, zc, xc, yc, dl, wl, hl = map(float, prt)
                d_mm = dl*shape[0]*vox[0]
                w_mm = wl*shape[1]*vox[1]
                h_mm = hl*shape[2]*vox[2]
                rec.append(dict(cls=1, image=stem, d_mm=d_mm, w_mm=w_mm, h_mm=h_mm))

    # ---------- scan NO-PFO for slice thickness only ----------
    for img in glob.glob(os.path.join(nopfo_img_dir, "*.nii*")):
        stem = os.path.basename(img).split(".nii")[0]
        _, _, dz = nii_meta(img)
        slice_rows.append(dict(group="NO_PFO", image=stem, dz_mm=dz))

    # ---------- pandas summaries ----------
    if not rec:
        raise RuntimeError("No PFO boxes parsed – check label paths")

    df_boxes   = pd.DataFrame(rec)
    df_slices  = pd.DataFrame(slice_rows)

    summary = (df_boxes
               .agg({'d_mm':['count','mean','min','max'],
                     'w_mm':['mean','min','max'],
                     'h_mm':['mean','min','max']})
               .T.rename(columns={'count':'n_boxes'}))

    slice_stats = (df_slices
                   .groupby("group")['dz_mm']
                   .agg(['count','median','min','max'])
                   .rename(columns={'count':'n_imgs','dz_mm':'slice_thk_mm'}))

    # ---------- print ----------
    print("\n===== PFO bounding-box size stats (mm) =====")
    print(summary.to_string(float_format="%.2f"))
    print("\n===== Slice-thickness distribution  =====")
    print(slice_stats.to_string(float_format="%.3f"))

    # ---------- save ----------
    if save_csv:
        summary.to_csv(os.path.join(pfo_lab_dir, "pfo_bbox_stats.csv"))
        slice_stats.to_csv(os.path.join(pfo_lab_dir, "slice_thickness_stats.csv"))
        print("\n✅ CSVs saved next to PFO label folder")

    return summary, slice_stats



# ---------------- example call -----------------
if __name__ == "__main__":
    summarize_pfo_vs_nopfo(
        pfo_img_dir   = "/home/tarobben/scratch/PFO_Complete",
        pfo_lab_dir   = "/home/tarobben/scratch/MedYOLO/PFO_labels_MedYOLO/",
        nopfo_img_dir = "/home/tarobben/scratch/NO_PFO_CT",
    )
