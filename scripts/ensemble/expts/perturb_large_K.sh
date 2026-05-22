#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J perturb-largeK
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# Push K higher to find saturation.
for K in 50 100 ; do
  for EPS in 0.05 0.1 ; do
    echo "------ FSNET single (hdim=2048) K=$K eps=$EPS ------"
    python eval.py --run_dir "$FSNET_SINGLE" \
       --inference_perturb_k $K --inference_perturb_eps $EPS \
       --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -20
    echo "------ FSNET member_0 (hdim=1024) K=$K eps=$EPS ------"
    python eval.py --checkpoints "$FSNET_ENS5_VAN/members/member_0.pt" \
       --inference_perturb_k $K --inference_perturb_eps $EPS \
       --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -20
  done
done

echo "=== perturb-largeK done $(date) ==="
