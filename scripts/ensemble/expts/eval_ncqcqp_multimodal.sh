#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 01:30:00
#SBATCH -J eval-ncqcqp
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

set -e
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd /orcd/scratch/orcd/008/khain/FSNet

# Run dir is passed as $1 (the converged FSNet nonconvex-qcqp training dir).
RUN_DIR="$1"
echo "=== Convergence-controlled perturbation eval on REAL multimodal (nonconvex QCQP) ==="
echo "run dir: $RUN_DIR"

# The diagnostic: does perturbation help when BOTH the network is converged
# (it is — this is the final checkpoint) AND the repair is run to convergence?
#
# Two repair regimes x K-perturbation:
#   - UNDER-CONVERGED repair: default max_iter, per_sample_lbfgs=0 (legacy OR-convergence)
#   - CONVERGED repair:       repair_max_iter=200, per_sample_lbfgs=1 (batch-invariant AND-convergence)
# If a K=100 gain survives the CONVERGED-repair regime on this multimodal problem,
# that is the genuine multimodal benefit on a real benchmark. If it collapses
# (like the convex frameworks), the negative result extends to real multimodal.

for PS in 0 1; do
  for RMAX in 50 200; do
    for K in 1 100; do
      for EPS in 0.0 0.1; do
        if [ "$K" = "1" ] && [ "$EPS" != "0.0" ]; then continue; fi
        echo ""
        echo ">>> per_sample_lbfgs=$PS  repair_max_iter=$RMAX  K=$K  eps=$EPS"
        python eval.py --run_dir "$RUN_DIR" \
          --ensemble_post post --ensemble_agg best_merit \
          --per_sample_lbfgs $PS \
          --repair_max_iter $RMAX \
          --inference_perturb_k $K --inference_perturb_eps $EPS \
          2>&1 | grep -E "Obj:|Opt Gap|Eq Vio|Ineq Vio|Merit:" || true
      done
    done
  done
done

echo "=== done $(date) ==="
