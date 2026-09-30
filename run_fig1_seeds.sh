#!/bin/bash
#SBATCH -J fig1Seeds
#SBATCH -p pi_donti_gpu,mit_normal_gpu,mit_preemptable
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# Fig. 1 seed replication. MODE=train: train one (seed, model) with the lr chosen by
# select_fig1.py on seed 0. MODE=eval: held-out Hessian + top-eigenvector planes for it.
# Array index = 4 * (seed - 1) + model_index, seeds 1-2, models M1-M4.

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs figures/landscape/fig1

NAMES=(M1 M2 M3 M4)
METHODS=(sup_pen sup_pen sup_pen_fs penalty)
RHOS=(10.0 100000.0 10.0 10.0)
WSTR=(obj0.1_eq10.0_ineq10.0 obj0.1_eq100000.0_ineq100000.0 obj0.1_eq10.0_ineq10.0 obj1.0_eq10.0_ineq10.0)
LRS=(3e-4 1e-4 3e-4 1e-3)
LRSTR=(0.0003 0.0001 0.0003 0.001)
i=${SLURM_ARRAY_TASK_ID}
SEED=$((i / 4 + 1)); m=$((i % 4)); n=${NAMES[$m]}

if [ "$MODE" = train ]; then
    python main.py --method ${METHODS[$m]} --prob_type nonsmooth_nonconvex --prob_name socp \
        --seed $SEED --train_size 7000 --num_epochs 3000 --lr ${LRS[$m]} --lr_schedule cosine --eta_min 1e-6 \
        --dropout 0.0 --pen_type l1 --eq_pen_weight ${RHOS[$m]} --ineq_pen_weight ${RHOS[$m]}
else
    D=results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000
    pat=${METHODS[$m]}_seed${SEED}_nepochs3000_lr${LRSTR[$m]}_trainsize7000_${WSTR[$m]}_penl1_dropout0.0_lrschedcosine_etamin1e-06
    run=$(ls -d $D/*_MLP_$pat | while read r; do [ -f $r/model.pt ] && echo $r; done | tail -1)
    [ -n "$run" ] || { echo "missing $pat"; exit 1; }
    OUT=figures/landscape/fig1; T=${n}_s${SEED}
    python hessian_spectrum.py --ckpt $n=$run/model.pt --split test --out $OUT/hessian_test_$T.json --save_vec $OUT/topvec_test_$T.pt
    for R in 0.1 0.5; do
        python compute_landscape_compare.py --mode random --ckpt $n=$run/model.pt --split test --dir_vec $OUT/topvec_test_$T.pt \
            --xnum 41 --range -$R $R --n_eval 1000 --out $OUT/eig_test_r${R}_$T.npz
    done
fi
