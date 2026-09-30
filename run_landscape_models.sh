#!/bin/bash
#SBATCH -J lsModels
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-9

# Models for the loss-landscape comparison (SOCP 100-50-50-10000).
# All share seed, init, data (7000 optimal labels), and optimizer settings
# (lr 3e-4, dropout 0, 3000 epochs, cosine), picked by run_landscape_screen.sh
# as the lowest own-training-loss setting for both sup_pen and penalty.
# Dropout is 0 so the trained weights are a minimum of the eval-mode loss
# that the landscapes plot.
#
#  idx  name              method      rho    penalty
#   0   sl_small          sup_pen     10     l2
#   1   sl_mid            sup_pen     1e3    l2
#   2   sl_high           sup_pen     1e5    l2
#   3   sl_fs_small       sup_pen_fs  10     l2    (FS layer in the training loop)
#   4   ssl_small         penalty     10     l2
#   5   sl_l1_small       sup_pen     10     l1
#   6   sl_l1_high        sup_pen     1e5    l1
#   7   ssl_l1_small      penalty     10     l1
#   8   ssl_l1_high       penalty     1e5    l1    (trains on the merit itself)
#   9   sl_fs_l1_small    sup_pen_fs  10     l1

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

METHODS=(sup_pen sup_pen sup_pen sup_pen_fs penalty sup_pen sup_pen penalty penalty sup_pen_fs)
RHOS=(10.0 1000.0 100000.0 10.0 10.0 10.0 100000.0 10.0 100000.0 10.0)
PENS=(l2 l2 l2 l2 l2 l1 l1 l1 l1 l1)
i=${SLURM_ARRAY_TASK_ID}

python main.py \
    --method ${METHODS[$i]} \
    --prob_type nonsmooth_nonconvex --prob_name socp \
    --seed ${SEED:-0} --train_size 7000 \
    --num_epochs ${EPOCHS:-3000} --lr ${LR:-3e-4} --lr_schedule cosine --eta_min 1e-6 \
    --dropout 0.0 --pen_type ${PENS[$i]} \
    --eq_pen_weight ${RHOS[$i]} --ineq_pen_weight ${RHOS[$i]}
