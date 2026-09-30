#!/bin/bash
#SBATCH -J curvedsheet
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH --exclude=node2119
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Curved (Bezier) 10-seed sheets for M1, M2, M3f, M4 (weight-matched seeds, connect_sheet.py),
# evaluated with the same settings as the straight sheets sheet10_wm_test_*.npz (run_actalign.sh).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
cd ~/FSNet
F=figures/landscape/fig1
for m in M1 M2 M3f M4; do
    python connect_sheet.py --model $m --layout tri10 --out $F/mids_tri10_$m.pt > logs/curved_fit_$m.log 2>&1 &
done
wait
for m in M1 M2 M3f M4; do
    fs=""; [ $m = M3f ] && fs="--fs"
    python compute_sheet.py --model $m --layout tri10 $fs --n 121 --margin 0.15 --mids $F/mids_tri10_$m.pt \
        --out $F/sheet10_curved_test_$m.npz > logs/curved_sheet_$m.log 2>&1 &
done
wait
for al in wm curved; do
    for m in M1 M2 M3f M4; do
        python sheet_stats.py --model $m --files $F/sheet10_${al}_test_$m.npz | sed "s/^| $m/| $m $al/"
    done
done
python make_sheet_fig_M3f.py --tag curved
