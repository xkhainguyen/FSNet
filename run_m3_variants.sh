#!/bin/bash
#SBATCH -J m3var
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-8
# M3 variants that make FS do real work (sup_pen_fs, L1 penalty type, lr 3e-4, 3000 epochs).
#   a: rho = 0                 b: rho = 10, FS truncated (10 iters, backprop 5)
#   c: rho = 0 + 5 ||FS(y) - y||^2
# Index = 3 * variant + seed.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
i=${SLURM_ARRAY_TASK_ID}; v=$((i / 3)); SEED=$((i % 3))
case $v in
    0) EXTRA="--eq_pen_weight 0.0 --ineq_pen_weight 0.0" ;;
    1) EXTRA="--eq_pen_weight 10.0 --ineq_pen_weight 10.0 --fs_max_iter 10 --max_diff_iter 5" ;;
    2) EXTRA="--eq_pen_weight 0.0 --ineq_pen_weight 0.0 --dist_weight 5.0" ;;
esac
python main.py --method sup_pen_fs --prob_type nonsmooth_nonconvex --prob_name socp \
    --seed $SEED --train_size 7000 --num_epochs 3000 --lr 3e-4 --lr_schedule cosine --eta_min 1e-6 \
    --dropout 0.0 --pen_type l1 $EXTRA
