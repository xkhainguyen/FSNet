#!/bin/bash
#SBATCH -J seedsGen
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Train one seed (= array index) of MODEL: M1 (SL rho 10, lr 3e-4), M2 (SL rho 1e5, lr 1e-4),
# M2m (SL rho 1e5, lr 3e-4: M2 at M1's lr), M4 (SSL rho 10, lr 1e-3). L1 penalty, 3000 epochs.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
case $MODEL in
    M1)  A="--method sup_pen --lr 3e-4 --eq_pen_weight 10.0 --ineq_pen_weight 10.0" ;;
    M2)  A="--method sup_pen --lr 1e-4 --eq_pen_weight 100000.0 --ineq_pen_weight 100000.0" ;;
    M2m) A="--method sup_pen --lr 3e-4 --eq_pen_weight 100000.0 --ineq_pen_weight 100000.0" ;;
    M4)  A="--method penalty --lr 1e-3 --eq_pen_weight 10.0 --ineq_pen_weight 10.0" ;;
esac
python main.py $A --prob_type nonsmooth_nonconvex --prob_name socp --seed $SLURM_ARRAY_TASK_ID \
    --train_size 7000 --num_epochs 3000 --lr_schedule cosine --eta_min 1e-6 --dropout 0.0 --pen_type l1
