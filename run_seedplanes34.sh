#!/bin/bash
#SBATCH -J seedPlanes34
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-1
# Plane through seeds 0,1,2 of M3 (SL + FS, needs FS) and M4 (SSL), held-out, 81x81.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
OUT=figures/landscape/fig1
ck() {  # $1 method, $2 weights, $3 lr, $4 seed
    ls -d $D/*_MLP_$1_seed$4_nepochs3000_lr$3_trainsize7000_$2_penl1_dropout0.0_lrschedcosine_etamin1e-06 \
        | while read r; do [ -f $r/model.pt ] && echo $r/model.pt; done | tail -1
}
case $SLURM_ARRAY_TASK_ID in
    0) M=M3; A=(sup_pen_fs obj0.1_eq10.0_ineq10.0 0.0003); FS="" ;;
    1) M=M4; A=(penalty obj1.0_eq10.0_ineq10.0 0.001); FS="--no_fs" ;;
esac
python compute_landscape_compare.py --mode plane --split test $FS --xnum 81 --n_eval 1000 --plane_margin 0.5 \
    --ckpt s0=$(ck ${A[@]} 0) --ckpt s1=$(ck ${A[@]} 1) --ckpt s2=$(ck ${A[@]} 2) --out $OUT/seedplane_fine_test_$M.npz
