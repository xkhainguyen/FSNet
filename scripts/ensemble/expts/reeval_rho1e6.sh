#!/bin/bash
#SBATCH -p pi_donti_gpu
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -t 02:00:00
#SBATCH -J reeval-rho1e6
#SBATCH -A mit_general
#SBATCH -o /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.out
#SBATCH -e /orcd/scratch/orcd/008/khain/FSNet/logs/%x-%j.err

source /orcd/scratch/orcd/008/khain/FSNet/scripts/ensemble/expts/_common.sh

# ============================================================
# BASELINES — re-eval at ρ=1e6 unified
# ============================================================
echo "######## Single FSNet hdim=2048 ########"
python eval.py --run_dir "$FSNET_SINGLE" --test_batch_sizes 256 2>&1 | tail -15

echo "######## Single penalty hdim=2048 ########"
python eval.py --run_dir "$PEN_SINGLE" --test_batch_sizes 256 2>&1 | tail -15

echo "######## VE M=5 FSNet — post + best_{obj,merit} ########"
for AGG in best_obj best_merit ; do
  python eval.py --run_dir "$FSNET_ENS5_VAN" --ensemble_post post --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
done

echo "######## FGE FSNet — post + best_{obj,merit} ########"
for AGG in best_obj best_merit ; do
  python eval.py --run_dir "$FSNET_ENS5_FGE" --ensemble_post post --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
done

echo "######## VE penalty — pre + best_{obj,merit} ########"
for AGG in best_obj best_merit ; do
  python eval.py --run_dir "$PEN_ENS5_VAN"  --ensemble_post pre --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
done
echo "######## VE ens20 penalty — pre + best_{obj,merit} ########"
for AGG in best_obj best_merit ; do
  python eval.py --run_dir "$PEN_ENS20_VAN" --ensemble_post pre --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
done

# ============================================================
# IDEA 2 — Perturbation sweep at ρ=1e6
# ============================================================
echo "######## Perturbation on FSNet single (hdim=2048) ########"
for K in 5 10 20 50 100 ; do
  for EPS in 0.05 0.1 ; do
    echo "------ K=$K eps=$EPS ------"
    python eval.py --run_dir "$FSNET_SINGLE" \
      --inference_perturb_k $K --inference_perturb_eps $EPS \
      --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15
  done
done

echo "######## Perturbation on FSNet member_0 (hdim=1024, apples-to-apples) ########"
for K in 5 10 20 50 100 ; do
  for EPS in 0.05 0.1 ; do
    echo "------ K=$K eps=$EPS ------"
    python eval.py --checkpoints "$FSNET_ENS5_VAN/members/member_0.pt" \
      --inference_perturb_k $K --inference_perturb_eps $EPS \
      --ensemble_agg best_merit --test_batch_sizes 256 2>&1 | tail -15
  done
done

# ============================================================
# IDEA 1 — MHE re-eval at ρ=1e6
# ============================================================
echo "######## MHE FSNet — post + best_{obj,merit} ########"
for D in $(ls -d $P_NSC/202605*FSNet*_mhe5_seed* 2>/dev/null) ; do
  [ ! -f "$D/model.pt" ] && continue
  echo "=== $(basename $D) ==="
  for AGG in best_obj best_merit ; do
    python eval.py --run_dir "$D" --ensemble_post post --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
  done
done

echo "######## MHE penalty — pre + best_{obj,merit} ########"
for D in $(ls -d $P_NSC/202605*penalty_e1000_lr1e-04_n7000_hdim1024_mhe*_seed* 2>/dev/null) ; do
  [ ! -f "$D/model.pt" ] && continue
  echo "=== $(basename $D) ==="
  for AGG in best_obj best_merit ; do
    python eval.py --run_dir "$D" --ensemble_post pre --ensemble_agg $AGG --test_batch_sizes 256 2>&1 | tail -15
  done
done

echo "=== reeval-rho1e6 done $(date) ==="
