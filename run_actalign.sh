#!/bin/bash
#SBATCH -J actalign
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=node2119
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Activation-matching alignment vs weight matching on the 10-seed tri10 sheets (M1, M2, M3f, M4),
# identical sheet settings for both alignments.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
cd ~/FSNet
F=figures/landscape/fig1
python align_seeds.py --method act --models M1 M2 M3f M4 --nseeds 10
for al in wm act; do
    dir=$F/aligned; [ $al = act ] && dir=$F/aligned_act
    for m in M1 M2 M3f M4; do
        fs=""; [ $m = M3f ] && fs="--fs"
        python compute_sheet.py --model $m --layout tri10 $fs --n 121 --margin 0.15 --aligned_dir $dir \
            --out $F/sheet10_${al}_test_$m.npz > logs/actalign_${al}_$m.log 2>&1 &
    done
done
wait
for al in wm act; do
    for m in M1 M2 M3f M4; do
        python sheet_stats.py --model $m --files $F/sheet10_${al}_test_$m.npz | sed "s/^| $m/| $m $al/"
    done
done
