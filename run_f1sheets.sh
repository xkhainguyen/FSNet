#!/bin/bash
#SBATCH -J f1sheets
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# All 16 hex19 sheets (4 models x {orderings 0,1,2 on test, ordering 0 on train}) in one job,
# 4 processes at a time on the same GPU (one job slot under the per-user submit limit).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd ~/FSNet
for m in M1 M2 M2m M4; do
    for c in "0 test" "1 test" "2 test" "0 train"; do
        read o sp <<< "$c"
        python compute_sheet.py --model $m --layout hex19 --order $o --split $sp \
            --out figures/landscape/fig1/sheet_hex19_o${o}_${sp}_$m.npz > logs/f1sheet_${m}_o${o}_${sp}.log 2>&1 &
    done
    wait
done
ls figures/landscape/fig1/sheet_hex19_* | wc -l
