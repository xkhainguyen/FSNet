#!/bin/bash
#SBATCH -J m3bPlanes
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-1
# M3b (SL + truncated FS, 10 iters): 0 = held-out plane through seeds 0,1,2 (81x81, margin 1.0);
# 1 = held-out random plane around seed 0 (radius 1, 51x51). FS evaluated with the same 10 iterations.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
ck() { ls -d $D/*_MLP_sup_pen_fs_seed$1_nepochs3000_lr0.0003_trainsize7000_obj0.1_eq10.0_ineq10.0_fsit10_diff5_penl1_dropout0.0_* \
        | while read r; do [ -f $r/model.pt ] && echo $r/model.pt; done | tail -1; }
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    python compute_landscape_compare.py --mode plane --split test --fs_max_iter 10 --xnum 81 --n_eval 1000 --plane_margin 1.0 \
        --ckpt s0=$(ck 0) --ckpt s1=$(ck 1) --ckpt s2=$(ck 2) --out figures/landscape/fig1/seedplane_wide_test_M3b.npz
else
    python compute_landscape_compare.py --mode random --split test --fs_max_iter 10 --xnum 51 --range -1 1 --n_eval 1000 \
        --ckpt M3b=$(ck 0) --out figures/landscape/fig1/random_test_r1.0_M3b.npz
fi
