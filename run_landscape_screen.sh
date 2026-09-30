#!/bin/bash
#SBATCH -J lsScreen
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-11

# Hyperparameter screen for the landscape models: fit quality of SL (sup_pen)
# and SSL (penalty) at rho=10 before retraining the full model set.
#   method x lr {3e-4, 1e-3, 3e-3} x dropout {0, 0.1}; 3000 epochs, 7000 labels.
# Score with screen_landscape_fit.py.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

METHODS=(sup_pen penalty)
LRS=(3e-4 1e-3 3e-3)
DROPS=(0.0 0.1)
i=${SLURM_ARRAY_TASK_ID}
M=${METHODS[$((i / 6))]}
LR=${LRS[$(((i / 2) % 3))]}
DO=${DROPS[$((i % 2))]}

python main.py \
    --method $M \
    --prob_type nonsmooth_nonconvex --prob_name socp \
    --seed 0 --train_size 7000 \
    --num_epochs ${EPOCHS:-3000} --lr $LR --lr_schedule cosine --eta_min 1e-6 \
    --dropout $DO --eq_pen_weight 10.0 --ineq_pen_weight 10.0
