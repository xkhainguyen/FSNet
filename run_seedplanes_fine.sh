#!/bin/bash
#SBATCH -J seedPlanesFine
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-3
# Multiple local minima check for M1 / M2 (SL, L1 penalty), held-out test split.
#   0: plane through M1 seeds 0,1,2    1: plane through M2 seeds 0,1,2
#   2: M1 random plane zoomed to +-0.01 3: M2 random plane zoomed to +-0.01
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
OUT=figures/landscape/fig1
ck() {  # $1 = weights string, $2 = lr string, $3 = seed
    ls -d $D/*_MLP_sup_pen_seed$3_nepochs3000_lr$2_trainsize7000_$1_penl1_dropout0.0_lrschedcosine_etamin1e-06 \
        | while read r; do [ -f $r/model.pt ] && echo $r/model.pt; done | tail -1
}
W1=obj0.1_eq10.0_ineq10.0; W2=obj0.1_eq100000.0_ineq100000.0
case $SLURM_ARRAY_TASK_ID in
    0) python compute_landscape_compare.py --mode plane --split test --no_fs --xnum 121 --n_eval 1000 --plane_margin 0.5 \
           --ckpt s0=$(ck $W1 0.0003 0) --ckpt s1=$(ck $W1 0.0003 1) --ckpt s2=$(ck $W1 0.0003 2) --out $OUT/seedplane_fine_test_M1.npz ;;
    1) python compute_landscape_compare.py --mode plane --split test --no_fs --xnum 121 --n_eval 1000 --plane_margin 0.5 \
           --ckpt s0=$(ck $W2 0.0001 0) --ckpt s1=$(ck $W2 0.0001 1) --ckpt s2=$(ck $W2 0.0001 2) --out $OUT/seedplane_fine_test_M2.npz ;;
    2) python compute_landscape_compare.py --mode random --split test --no_fs --xnum 41 --range -0.01 0.01 --n_eval 1000 \
           --ckpt M1=$(ck $W1 0.0003 0) --out $OUT/random_test_r0.01_M1.npz ;;
    3) python compute_landscape_compare.py --mode random --split test --no_fs --xnum 41 --range -0.01 0.01 --n_eval 1000 \
           --ckpt M2=$(ck $W2 0.0001 0) --out $OUT/random_test_r0.01_M2.npz ;;
esac
