#!/bin/bash
#SBATCH --job-name=Medyolo
#SBATCH --cpus-per-task=8  # Adjust CPU allocation
#SBATCH --mem=100G  # Memory per FEAT job
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00  # Adjust runtime
#SBATCH --nice=10  # Higher priority


# Activate the BART environment using its path
eval "$('/opt/amc/devel/miniconda3-4.8.2/bin/conda' 'shell.bash' 'hook')"
conda activate /home/tarobben/.conda/envs/medyolo   # path or env‑name

cd /home/tarobben/Documents/PFODetection/MedYOLO/MedYOLO-main
# Run your Python script
#python slurm.py
python train.py --data Test.yaml --adam --norm CT --epochs 1000 --patience 200 --cfg models3D/yolo3D_custom.yaml --batch-size 8 --imgsz 350 --image-weights 
