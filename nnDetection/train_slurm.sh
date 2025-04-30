#!/bin/bash
#SBATCH --job-name=nndet_train_fold3
#SBATCH --output=logs/nndet_fold3_%j.out
#SBATCH --error=logs/nndet_fold3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00


# Load Conda
source /opt/amc/devel/miniconda3-4.8.2/etc/profile.d/conda.sh
conda activate nndet_venv

# Set required nnDetection environment variables
export OMP_NUM_THREADS=1
export det_data=/home/tarobben/data/
export det_models=/home/tarobben/scratch/nndet/Task001_model/
export det_num_threads=8
export det_verbose=1

# Start training on fold 3
nndet_train Task001_test -o exp.fold=3 --sweep 
