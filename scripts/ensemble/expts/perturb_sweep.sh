#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J perturb-sweep
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Idea 2: free-ensemble via perturbed repair restarts.
# Sweep K (members) and eps (perturbation scale) on single-model checkpoints.
# Aggregation is best_merit (post-style on perturbed candidates).
run_perturb() {
  local NAME=$1 DIR=$2 K=$3 EPS=$4 DIST=$5
  echo "------ PERTURB $NAME K=$K eps=$EPS dist=$DIST ------"
  python eval.py --run_dir "$DIR" \
     --inference_perturb_k $K --inference_perturb_eps $EPS --inference_perturb_dist $DIST \
     --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -25
}

for cfg in "PEN_SINGLE $PEN_SINGLE" "FSNET_SINGLE $FSNET_SINGLE" ; do
  NAME=$(echo $cfg | cut -d' ' -f1)
  DIR=$(echo $cfg | cut -d' ' -f2)
  echo "######## $NAME ########"
  # K=1 is the original single model (no perturbation) — covered by audit
  for K in 5 10 20 ; do
    for EPS in 0.01 0.05 0.1 0.2 ; do
      run_perturb "$NAME" "$DIR" $K $EPS gauss
    done
  done
  # Also try antithetic at promising eps for variance reduction
  for K in 10 20 ; do
    for EPS in 0.05 0.1 ; do
      run_perturb "$NAME" "$DIR" $K $EPS antithetic
    done
  done
done

echo "=== perturb-sweep done $(date) ==="
