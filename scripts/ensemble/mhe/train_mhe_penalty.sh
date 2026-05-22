#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 03:00:00
#SBATCH -J mhe-train-pen
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
cd /orcd/scratch/orcd/008/khain/FSNet

SEED=${SEED:-0}
NUM_HEADS=${NUM_HEADS:-5}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
# Match the existing vanilla ens5 baseline (eq=10, ineq=10) for apples-to-apples
echo "=== MHE penalty seed=$SEED M=$NUM_HEADS hdim=$HIDDEN_DIM $(date) on $SLURM_NODELIST ==="

python -c "
import yaml
with open('configs/default.yaml') as f: c = yaml.safe_load(f)
c['penalty']['eq_pen_weight'] = 10.0
c['penalty']['ineq_pen_weight'] = 10.0
with open('/tmp/mhe_pen_cfg_${SLURM_JOB_ID}.yaml','w') as f: yaml.dump(c, f)
"

python main.py \
  --config /tmp/mhe_pen_cfg_${SLURM_JOB_ID}.yaml \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MultiHeadMLP \
  --mhe_num_heads $NUM_HEADS \
  --hidden_dim $HIDDEN_DIM \
  --num_layers 4 \
  --seed $SEED

echo "=== done $(date) ==="
