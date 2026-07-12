#!/bin/bash
#SBATCH -J diagnose-cnn
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH -t 0:30:00
#SBATCH -o logs/diagnose-%j.out
#SBATCH -e logs/diagnose-%j.err
#SBATCH --mail-type=END,FAIL

# Run from the repo root on the cluster: sbatch scripts/diagnose_cnn.slurm
# Trains the CNN on the V100, scores validation, and writes diagnostic plots +
# summary to plots/diagnostics/. Also saves outputs/cnn/model.pt so a later
# `python scripts/diagnose_cnn.py --load` can re-plot without retraining.

mkdir -p logs

module load nvidia/latest

source ts-anomaly-env/bin/activate

python -u -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -u scripts/diagnose_cnn.py
