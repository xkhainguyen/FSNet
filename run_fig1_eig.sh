#!/bin/bash
#SBATCH -J fig1Eig
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-3

# Fig. 1 (conditioning view): for each model, the top Hessian eigenvector of its own
# loss on the test split, then held-out landscapes on the plane (top eigenvector,
# random filter-normalized direction), both scaled to the same weight distance.
# An ill-conditioned loss shows as a long narrow valley, a well-conditioned one as a round bowl.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
OUT=figures/landscape/fig1
mkdir -p logs $OUT

NAMES=(M1 M2 M3 M4)
n=${NAMES[$SLURM_ARRAY_TASK_ID]}
ckpt=$(python -c "import json; print(json.load(open('$OUT/ckpts.json'))['$n']['ckpt'])")
echo "$n: $ckpt"

python hessian_spectrum.py --ckpt $n=$ckpt --split test --out $OUT/hessian_test_$n.json --save_vec $OUT/topvec_test_$n.pt
for R in 0.1 0.5; do
    python compute_landscape_compare.py --mode random --ckpt $n=$ckpt --split test --dir_vec $OUT/topvec_test_$n.pt \
        --xnum 41 --range -$R $R --n_eval 1000 --out $OUT/eig_test_r${R}_$n.npz
done
