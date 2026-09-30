#!/bin/bash
#SBATCH -J m1m2r1
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-1
# M1 / M2 held-out random-direction planes at radius 1 (the earlier paper figure's range).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
NAMES=(M1 M2); n=${NAMES[$SLURM_ARRAY_TASK_ID]}
ckpt=$(python -c "import json; print(json.load(open('figures/landscape/fig1/ckpts.json'))['$n']['ckpt'])")
python compute_landscape_compare.py --mode random --ckpt $n=$ckpt --split test --no_fs \
    --xnum 51 --range -1 1 --n_eval 1000 --out figures/landscape/fig1/random_test_r1.0_$n.npz
