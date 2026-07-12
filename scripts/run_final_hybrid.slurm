#!/bin/bash
#SBATCH -J final-hybrid-anomaly
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH -t 1:30:00
#SBATCH -o logs/final_hybrid-%j.out
#SBATCH -e logs/final_hybrid-%j.err
#SBATCH --mail-type=END,FAIL

# Run from the repo root on the cluster: sbatch scripts/run_final_hybrid.slurm
# Note: the V100 only accelerates the CNN half. LOF is sklearn (CPU-only) and
# on the wide cross-sensor feature set it is the likely bottleneck - the 8 CPU
# cores (n_jobs=-1) and RAM matter more for it than the GPU. Walltime is 1h30
# (vs 1h for run_hybrid) to leave headroom for the LOF fit/scoring.

mkdir -p logs

module load nvidia/latest

source ts-anomaly-env/bin/activate

python -u -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# -u: unbuffered stdout, so `tail -f` on the log file shows live progress
# instead of nothing until the buffer flushes or the job exits.
python -u -m src.pipelines.run_final_hybrid
