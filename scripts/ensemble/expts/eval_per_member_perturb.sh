#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 00:30:00
#SBATCH -J per-mem-perturb
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# 1) Eval each member of FSNet ens5_vanilla alone (no perturb)
for i in 0 1 2 3 4 ; do
  echo "######## FSNET member_$i alone ########"
  python eval.py --checkpoints "$FSNET_ENS5_VAN/members/member_$i.pt" \
     --test_batch_sizes 256 2>&1 | tail -20
done

# 2) Eval each member of FSNet ens5_vanilla with perturbation K=20 eps=0.1
for i in 0 1 2 3 4 ; do
  echo "######## FSNET member_$i + perturb K=20 eps=0.1 ########"
  python eval.py --checkpoints "$FSNET_ENS5_VAN/members/member_$i.pt" \
     --inference_perturb_k 20 --inference_perturb_eps 0.1 \
     --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -20
done

echo "=== done $(date) ==="
