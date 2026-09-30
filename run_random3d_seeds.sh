#!/bin/bash
#SBATCH -J rand3dSeeds
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Held-out random-direction planes (radius 1, 51x51) of M1-M4 for seeds 0-2.
# Array index = 4 * seed + model. FS is computed only for M3 (its own loss needs it).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
OUT=figures/landscape/fig1
NAMES=(M1 M2 M3 M4)
METHODS=(sup_pen sup_pen sup_pen_fs penalty)
WSTR=(obj0.1_eq10.0_ineq10.0 obj0.1_eq100000.0_ineq100000.0 obj0.1_eq10.0_ineq10.0 obj1.0_eq10.0_ineq10.0)
LRSTR=(0.0003 0.0001 0.0003 0.001)
i=${SLURM_ARRAY_TASK_ID}; SEED=$((i / 4)); m=$((i % 4)); n=${NAMES[$m]}
pat=${METHODS[$m]}_seed${SEED}_nepochs3000_lr${LRSTR[$m]}_trainsize7000_${WSTR[$m]}_penl1_dropout0.0_lrschedcosine_etamin1e-06
run=$(ls -d $D/*_MLP_$pat | while read r; do [ -f $r/model.pt ] && echo $r; done | tail -1)
[ -n "$run" ] || { echo "missing $pat"; exit 1; }
SFX=$([ "$SEED" = 0 ] || echo _s$SEED)
FS=$([ "$n" = M3 ] || echo --no_fs)
python compute_landscape_compare.py --mode random --ckpt $n=$run/model.pt --split test $FS \
    --xnum 51 --range -1 1 --n_eval 1000 --out $OUT/random_test_r1.0_$n$SFX.npz
