#!/bin/bash
#SBATCH -J alignedPlanes
#SBATCH -p mit_normal,mit_preemptable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-2
# Held-out wide plane (151x151, margin 1.0) through seed 0 and permutation-aligned seeds 1, 2
# (align_seeds.py), for M1, M2, M4. Compare with seedplane_wide_test_* (unaligned).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 LAB_THREADS=8
cd ~/FSNet
NAMES=(M1 M2 M4); n=${NAMES[$SLURM_ARRAY_TASK_ID]}
A=figures/landscape/fig1/aligned
T=(${TRIPLE:-0 1 2}); TAG=aligned$([ "${TRIPLE:-0 1 2}" = "0 1 2" ] || echo _t${T[0]}${T[1]}${T[2]})
python compute_landscape_compare.py --mode plane --split test --no_fs --xnum 151 --n_eval 1000 --plane_margin 1.0 \
    --ckpt s${T[0]}=$A/${n}_s${T[0]}.pt --ckpt s${T[1]}=$A/${n}_s${T[1]}.pt --ckpt s${T[2]}=$A/${n}_s${T[2]}.pt \
    --out figures/landscape/fig1/seedplane_${TAG}_test_$n.npz
