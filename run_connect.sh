#!/bin/bash
#SBATCH -J connect
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-26
# Mode connectivity (connect_seeds.py) between seed 0 and seeds 1-9 of M1, M2, M4.
# Index = 9 * model + (partner seed - 1).
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
OUT=figures/landscape/fig1/connect
mkdir -p $OUT
NAMES=(M1 M2 M4)
METHODS=(sup_pen sup_pen penalty)
WSTR=(obj0.1_eq10.0_ineq10.0 obj0.1_eq100000.0_ineq100000.0 obj1.0_eq10.0_ineq10.0)
LRSTR=(0.0003 0.0001 0.001)
i=${SLURM_ARRAY_TASK_ID}; m=$((i / 9)); B=$((i % 9 + 1))
ck() { ls -d $D/*_MLP_${METHODS[$m]}_seed$1_nepochs3000_lr${LRSTR[$m]}_trainsize7000_${WSTR[$m]}_penl1_dropout0.0_lrschedcosine_etamin1e-06 \
        | while read r; do [ -f $r/model.pt ] && echo $r/model.pt; done | tail -1; }
python connect_seeds.py --a $(ck 0) --b $(ck $B) --out $OUT/${NAMES[$m]}_s0_s$B.json
