#!/bin/bash
#SBATCH -J moreSeeds
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Seeds 3-9 of M1, M2, M4 (L1 penalty, seed-0 selected lr). Index = 3 * (seed - 3) + model.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
METHODS=(sup_pen sup_pen penalty)
RHOS=(10.0 100000.0 10.0)
LRS=(3e-4 1e-4 1e-3)
i=${SLURM_ARRAY_TASK_ID}; SEED=$((i / 3 + 3)); m=$((i % 3))
python main.py --method ${METHODS[$m]} --prob_type nonsmooth_nonconvex --prob_name socp \
    --seed $SEED --train_size 7000 --num_epochs 3000 --lr ${LRS[$m]} --lr_schedule cosine --eta_min 1e-6 \
    --dropout 0.0 --pen_type l1 --eq_pen_weight ${RHOS[$m]} --ineq_pen_weight ${RHOS[$m]}
