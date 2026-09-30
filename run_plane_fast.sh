#!/bin/bash
# Held-out wide plane through 3 aligned seeds with the FS layer, split over several GPUs.
# Each array task evaluates every N-th row (--shard) with grouped FS solves (--group 16), then a
# dependent job merges the shards. Same grid as run_alignedplanes.sh (151x151, margin 1.0).
#
#   MODEL=M3f SEEDS="0 1 2" N=4 bash run_plane_fast.sh
# (submits the array and the merge job; ~9 GPU-min per plane on H200, / N wall-clock)
set -e
MODEL=${MODEL:-M3f}; SEEDS=(${SEEDS:-0 1 2}); N=${N:-4}
TAG=aligned$([ "${SEEDS[*]}" = "0 1 2" ] || echo _t${SEEDS[0]}${SEEDS[1]}${SEEDS[2]})
OUT=figures/landscape/fig1/seedplane_${TAG}_test_${MODEL}.npz
A=figures/landscape/fig1/aligned
PRE='source ~/.bashrc; conda activate ml4opt; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; cd ~/FSNet'
jid=$(sbatch --parsable -J plane${MODEL} -p pi_donti_gpu,mit_normal_gpu,mit_preemptable -G h200:1 \
    --cpus-per-task=4 --mem=32G -t 01:00:00 --requeue --array=0-$((N - 1)) \
    -o logs/%x-%A_%a.out -e logs/%x-%A_%a.err --wrap "$PRE; python compute_landscape_compare.py --mode plane \
    --split test --xnum 151 --n_eval 1000 --plane_margin 1.0 --ckpt s${SEEDS[0]}=$A/${MODEL}_s${SEEDS[0]}.pt \
    --ckpt s${SEEDS[1]}=$A/${MODEL}_s${SEEDS[1]}.pt --ckpt s${SEEDS[2]}=$A/${MODEL}_s${SEEDS[2]}.pt \
    --shard \$SLURM_ARRAY_TASK_ID/$N --group 16 --out ${OUT%.npz}_shard\$SLURM_ARRAY_TASK_ID.npz")
mid=$(sbatch --parsable -J merge${MODEL} -p mit_normal,mit_preemptable -c 1 --mem=8G -t 00:10:00 \
    --dependency=afterok:$jid -o logs/%x-%j.out -e logs/%x-%j.err \
    --wrap "$PRE; python merge_shards.py --out $OUT --n $N")
echo "array $jid ($N shards) -> merge $mid -> $OUT"
