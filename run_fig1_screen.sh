#!/bin/bash
#SBATCH -J fig1Screen
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7

# lr screen for the four Fig. 1 models, all with the L1 penalty (lr 3e-4 already run
# by run_landscape_models.sh). Pick per model the lr with the lowest own training loss.
#   M1 sup_pen rho 10 | M2 sup_pen rho 1e5 | M3 sup_pen_fs rho 10 | M4 penalty rho 10
#   idx = 2 * model + (0: lr 1e-4, 1: lr 1e-3)

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

METHODS=(sup_pen sup_pen sup_pen_fs penalty)
RHOS=(10.0 100000.0 10.0 10.0)
LRS=(1e-4 1e-3)
i=${SLURM_ARRAY_TASK_ID}
m=$((i / 2))

python main.py \
    --method ${METHODS[$m]} \
    --prob_type nonsmooth_nonconvex --prob_name socp \
    --seed 0 --train_size 7000 \
    --num_epochs 3000 --lr ${LRS[$((i % 2))]} --lr_schedule cosine --eta_min 1e-6 \
    --dropout 0.0 --pen_type l1 \
    --eq_pen_weight ${RHOS[$m]} --ineq_pen_weight ${RHOS[$m]}
