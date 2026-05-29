#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 08:00:00
#SBATCH -J fsnet-ncqcqp
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /orcd/scratch/orcd/008/khain/FSNet
echo "=== FSNet on REAL multimodal benchmark: nonconvex QCQP $(date) on $SLURM_NODELIST ==="
nvidia-smi -L

# Single FSNet, trained to convergence (per_sample_lbfgs=0; the =1 arm diverges in
# training and is eval-only). hdim/layers/epochs = repo defaults.
python main.py \
  --method FSNet \
  --prob_type nonconvex \
  --prob_name qcqp \
  --hidden_dim 1024 \
  --num_layers 4 \
  --seed 0

echo "=== done $(date) ==="
