#!/bin/bash
#SBATCH -J connect
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-2
# Curved-path (Bezier) connectivity for MODEL, seed pairs (0,1), (0,2), (1,2) = array index.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p figures/landscape/fig1/connect
P=("0 1" "0 2" "1 2"); read a b <<< "${P[$SLURM_ARRAY_TASK_ID]}"
A=$(python -c "from fig1_common import ckpt; print(ckpt('$MODEL', $a))")
B=$(python -c "from fig1_common import ckpt; print(ckpt('$MODEL', $b))")
python connect_seeds.py --a $A --b $B --out figures/landscape/fig1/connect/${MODEL}_s${a}_s${b}.json
