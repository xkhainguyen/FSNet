#!/bin/bash
#SBATCH -J m3cand
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# After the rho 1 / 0.8 M3 runs: quality table, align seeds 0-2, FS-evaluated 3-seed sheets, figure.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
cd ~/FSNet
python m3_candidates.py
python align_seeds.py --models M1 M3r1 M3r08 --nseeds 3
python compute_sheet.py --model M1 --layout tri3 --n 101 --margin 0.1 --out figures/landscape/fig1/sheet_tri3_test_M1.npz &
python compute_sheet.py --model M3r1 --layout tri3 --fs --n 101 --margin 0.1 --out figures/landscape/fig1/sheet_tri3_test_M3r1.npz &
python compute_sheet.py --model M3r08 --layout tri3 --fs --n 101 --margin 0.1 --out figures/landscape/fig1/sheet_tri3_test_M3r08.npz &
wait
python make_m3_sheets.py && cp figures/landscape/fig1/fig1_m3_rho_sheets.png paper_figs/fig1/
