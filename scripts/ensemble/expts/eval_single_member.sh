#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J eval-single-mem
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# member_0.pt of ens5_vanilla = single model with same config as the ensemble.
# Use it as the apples-to-apples single baseline.
PEN_MEMBER0="$PEN_ENS5_VAN/members/member_0.pt"
FSNET_MEMBER0="$FSNET_ENS5_VAN/members/member_0.pt"

echo "######## SINGLE-MODEL (penalty, member_0 of ens5_vanilla, eq=10 ineq=10) ########"
python eval.py --checkpoints "$PEN_MEMBER0" --test_batch_sizes 256 2>&1 | tail -25

echo "######## SINGLE-MODEL (FSNet, member_0 of ens5_vanilla) ########"
python eval.py --checkpoints "$FSNET_MEMBER0" --test_batch_sizes 256 2>&1 | tail -25

# Single + skip_repair
echo "######## PEN single + skip_repair ########"
python eval.py --checkpoints "$PEN_MEMBER0" --skip_repair --test_batch_sizes 256 2>&1 | tail -25

echo "######## FSNET single + skip_repair ########"
python eval.py --checkpoints "$FSNET_MEMBER0" --skip_repair --test_batch_sizes 256 2>&1 | tail -25

# Single + perturbation sweep (FSNet, since penalty doesn't have repair → perturbation is moot)
for K in 5 10 20 ; do
  for EPS in 0.01 0.05 0.1 0.2 ; do
    echo "------ FSNET single perturb K=$K eps=$EPS ------"
    python eval.py --checkpoints "$FSNET_MEMBER0" \
       --inference_perturb_k $K --inference_perturb_eps $EPS \
       --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -25
  done
done

# Repair max_iter sweep on FSNet single (Idea 3)
for IT in 1 5 10 20 50 100 ; do
  echo "------ FSNET single repair_max_iter=$IT ------"
  python eval.py --checkpoints "$FSNET_MEMBER0" --repair_max_iter $IT --test_batch_sizes 256 2>&1 | tail -25
done

echo "=== eval-single-member done $(date) ==="
