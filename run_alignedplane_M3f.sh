#!/bin/bash
#SBATCH -J planeM3f
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=node2119
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Held-out wide plane through permutation-aligned M3f seeds 0, 1, 2 with the FS layer
# (same grid as run_alignedplanes.sh: 151x151, margin 1.0), for the 4-method many-minima figure.
# 6 row shards in parallel on one GPU, then merged. --group 16: 16 grid points per grouped FS solve
# (same numbers as the per-point loop; ~7x faster on an H200, whose fp64 is fast).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
cd ~/FSNet
A=figures/landscape/fig1/aligned
F=figures/landscape/fig1/seedplane_aligned_test_M3f
N=6
for k in $(seq 0 $((N - 1))); do
    python compute_landscape_compare.py --mode plane --split test --xnum 151 --n_eval 1000 --plane_margin 1.0 \
        --ckpt s0=$A/M3f_s0.pt --ckpt s1=$A/M3f_s1.pt --ckpt s2=$A/M3f_s2.pt --shard $k/$N --group 16 \
        --out ${F}_shard$k.npz > logs/planeM3f_shard$k.log 2>&1 &
done
wait
python - <<PY
import numpy as np
parts = [dict(np.load(f"${F}_shard{k}.npz", allow_pickle=True)) for k in range($N)]
out = dict(parts[0])
for key in [k for k in out if k.startswith("plane/")]:
    Z = out[key].copy()
    for p in parts[1:]:
        Z = np.where(np.isnan(Z), p[key], Z)
    out[key] = Z
    if np.isnan(Z).any(): print(key, "NaN points after merge:", int(np.isnan(Z).sum()))
np.savez("${F}.npz", **out)
print("merged ${F}.npz")
PY
python make_manyminima4.py && cp -f figures/landscape/fig1/fig1_manyminima4.png paper_figs/fig1/
