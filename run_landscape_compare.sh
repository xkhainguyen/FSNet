#!/bin/bash
#SBATCH -J lsCompare
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-13

# Loss landscapes for the run_landscape_models.sh models.
#   tasks 0-8:  random 2D plane around each model (shared direction seeds), radius RANGE
#   task 9:     plane through sl_small, sl_high, ssl_small
#   task 10:    plane through sl_small, sl_fs_small, ssl_small
#   task 11:    plane through sl_small, ssl_small, ssl_l1_high
#   task 12:    random plane around sl_fs_l1_small
#   task 13:    plane through sl_small, sl_fs_small, ssl_l1_high
# Checkpoints: newest run of each model; override with CKPT_<NAME> (upper case).

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs figures/landscape

XNUM=${XNUM:-41}
RANGE=${RANGE:-0.1}
NEVAL=${NEVAL:-1000}
SFXS=$([ "${SEED:-0}" = 0 ] || echo _s${SEED})
SPLIT=${SPLIT:-train}
SFXS=${SFXS}$([ "$SPLIT" = train ] || echo _${SPLIT})
TAG=${TAG:-v2${SFXS}_r${RANGE}_n${XNUM}}

D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
SEED=${SEED:-0}
PRE=seed${SEED}_nepochs3000_lr0.0003_trainsize7000
POST=dropout0.0_lrschedcosine_etamin1e-06
NAMES=(sl_small sl_mid sl_high sl_fs_small ssl_small sl_l1_small sl_l1_high ssl_l1_small ssl_l1_high sl_fs_l1_small)
PATTERNS=(
    sup_pen_${PRE}_obj0.1_eq10.0_ineq10.0_${POST}
    sup_pen_${PRE}_obj0.1_eq1000.0_ineq1000.0_${POST}
    sup_pen_${PRE}_obj0.1_eq100000.0_ineq100000.0_${POST}
    sup_pen_fs_${PRE}_obj0.1_eq10.0_ineq10.0_${POST}
    penalty_${PRE}_obj1.0_eq10.0_ineq10.0_${POST}
    sup_pen_${PRE}_obj0.1_eq10.0_ineq10.0_penl1_${POST}
    sup_pen_${PRE}_obj0.1_eq100000.0_ineq100000.0_penl1_${POST}
    penalty_${PRE}_obj1.0_eq10.0_ineq10.0_penl1_${POST}
    penalty_${PRE}_obj1.0_eq100000.0_ineq100000.0_penl1_${POST}
    sup_pen_fs_${PRE}_obj0.1_eq10.0_ineq10.0_penl1_${POST}
)
declare -A CK
for k in "${!NAMES[@]}"; do
    n=${NAMES[$k]}
    override=CKPT_${n^^}
    run=$(ls -d $D/*_MLP_${PATTERNS[$k]} 2>/dev/null | while read r; do [ -f $r/model.pt ] && echo $r; done | tail -1)
    CK[$n]=${!override:-$run/model.pt}
done

i=${SLURM_ARRAY_TASK_ID}
need() { for n in "$@"; do [ -f "${CK[$n]}" ] || { echo "missing checkpoint for $n: ${CK[$n]}"; exit 1; }; echo "ckpt $n: ${CK[$n]}"; done; }

if [ "$i" -lt 9 ] || [ "$i" -eq 12 ]; then
    n=${NAMES[$([ "$i" -eq 12 ] && echo 9 || echo $i)]}
    need $n
    python compute_landscape_compare.py --mode random --ckpt $n=${CK[$n]} \
        --xnum $XNUM --range -$RANGE $RANGE --n_eval $NEVAL --split $SPLIT \
        --out figures/landscape/random_${TAG}_${n}.npz
else
    case $i in
        9)  P=(sl_small sl_high ssl_small);        OUT=plane_high ;;
        10) P=(sl_small sl_fs_small ssl_small);    OUT=plane_fs ;;
        11) P=(sl_small ssl_small ssl_l1_high);    OUT=plane_l1 ;;
        13) P=(sl_small sl_fs_small ssl_l1_high);  OUT=plane_fsmerit ;;
    esac
    need ${P[@]}
    python compute_landscape_compare.py --mode plane \
        --ckpt ${P[0]}=${CK[${P[0]}]} --ckpt ${P[1]}=${CK[${P[1]}]} --ckpt ${P[2]}=${CK[${P[2]}]} \
        --xnum $XNUM --n_eval $NEVAL --split $SPLIT \
        --out figures/landscape/${OUT}_v2${SFXS}_n${XNUM}.npz
fi
