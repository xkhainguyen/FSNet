#!/bin/bash
#SBATCH -J lsHess
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-8

# Hessian conditioning / smoothness (hessian_spectrum.py) of each
# run_landscape_models.sh model's own training loss and of the merit.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
HDIR=figures/landscape/hessian$([ "${SEED:-0}" = 0 ] || echo _s${SEED})
mkdir -p logs $HDIR

D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
SEED=${SEED:-0}
PRE=seed${SEED}_nepochs3000_lr0.0003_trainsize7000
POST=dropout0.0_lrschedcosine_etamin1e-06
NAMES=(sl_small sl_high sl_fs_small ssl_small sl_l1_small sl_l1_high ssl_l1_small ssl_l1_high sl_fs_l1_small)
PATTERNS=(
    sup_pen_${PRE}_obj0.1_eq10.0_ineq10.0_${POST}
    sup_pen_${PRE}_obj0.1_eq100000.0_ineq100000.0_${POST}
    sup_pen_fs_${PRE}_obj0.1_eq10.0_ineq10.0_${POST}
    penalty_${PRE}_obj1.0_eq10.0_ineq10.0_${POST}
    sup_pen_${PRE}_obj0.1_eq10.0_ineq10.0_penl1_${POST}
    sup_pen_${PRE}_obj0.1_eq100000.0_ineq100000.0_penl1_${POST}
    penalty_${PRE}_obj1.0_eq10.0_ineq10.0_penl1_${POST}
    penalty_${PRE}_obj1.0_eq100000.0_ineq100000.0_penl1_${POST}
    sup_pen_fs_${PRE}_obj0.1_eq10.0_ineq10.0_penl1_${POST}
)
i=${SLURM_ARRAY_TASK_ID}
run=$(ls -d $D/*_MLP_${PATTERNS[$i]} | while read r; do [ -f $r/model.pt ] && echo $r; done | tail -1)
python hessian_spectrum.py --ckpt ${NAMES[$i]}=$run/model.pt \
    --out $HDIR/${NAMES[$i]}.json
