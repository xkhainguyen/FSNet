#!/bin/bash
#SBATCH -J widePlanes
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-2
# Wide (margin 1.0), fine (151x151) held-out planes through seeds 0,1,2 of M1, M2, M4.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
NAMES=(M1 M2 M4); METHODS=(sup_pen sup_pen penalty)
WSTR=(obj0.1_eq10.0_ineq10.0 obj0.1_eq100000.0_ineq100000.0 obj1.0_eq10.0_ineq10.0); LRSTR=(0.0003 0.0001 0.001)
m=$SLURM_ARRAY_TASK_ID
ck() { ls -d $D/*_MLP_${METHODS[$m]}_seed$1_nepochs3000_lr${LRSTR[$m]}_trainsize7000_${WSTR[$m]}_penl1_dropout0.0_lrschedcosine_etamin1e-06 \
        | while read r; do [ -f $r/model.pt ] && echo $r/model.pt; done | tail -1; }
# TRIPLE="a b c" picks the seeds (default 0 1 2); array index = 3 * triple_slot + model when TRIPLES is set
T=(${TRIPLE:-0 1 2}); TAG=wide$([ "${TRIPLE:-0 1 2}" = "0 1 2" ] || echo _t${T[0]}${T[1]}${T[2]})
python compute_landscape_compare.py --mode plane --split test --no_fs --xnum 151 --n_eval 1000 --plane_margin 1.0 \
    --ckpt s${T[0]}=$(ck ${T[0]}) --ckpt s${T[1]}=$(ck ${T[1]}) --ckpt s${T[2]}=$(ck ${T[2]}) \
    --out figures/landscape/fig1/seedplane_${TAG}_test_${NAMES[$m]}.npz
