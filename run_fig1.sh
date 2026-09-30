#!/bin/bash
#SBATCH -J fig1
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-11

# Fig. 1: held-out (test split) landscapes of the four models chosen by select_fig1.py.
#   tasks 0-7:  random filter-normalized plane, model = i % 4, radius = (0.1, 0.5)[i / 4]
#   tasks 8-11: Hessian stiffness / kink of the own loss on the test split, model = i - 8

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
OUT=figures/landscape/fig1
mkdir -p logs $OUT

NAMES=(M1 M2 M3 M4)
RADII=(0.1 0.5)
i=${SLURM_ARRAY_TASK_ID}
n=${NAMES[$((i % 4))]}
ckpt=$(python -c "import json; print(json.load(open('$OUT/ckpts.json'))['$n']['ckpt'])")
echo "$n: $ckpt"

if [ "$i" -lt 8 ]; then
    R=${RADII[$((i / 4))]}
    python compute_landscape_compare.py --mode random --ckpt $n=$ckpt --split test \
        --xnum 41 --range -$R $R --n_eval 1000 --out $OUT/random_test_r${R}_$n.npz
else
    python hessian_spectrum.py --ckpt $n=$ckpt --split test --out $OUT/hessian_test_$n.json
fi
