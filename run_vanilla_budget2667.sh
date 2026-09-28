#!/bin/bash
#SBATCH -J vanSSL2667
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH -G l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH --exclude=node4104
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7%4

# Match Ours total offline budget (~480s) using 0.18 s/ep:
#   480 / 0.18 = 2667 epochs

source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd ~/FSNet
mkdir -p logs

METHODS=(penalty penalty penalty penalty adaptive_penalty adaptive_penalty adaptive_penalty adaptive_penalty)
SEEDS=(0 1 2 3 0 1 2 3)
NUM_EPOCHS=2667

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo " Job started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Job ID: $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo " Node: $SLURM_NODELIST  GPU: ${CUDA_VISIBLE_DEVICES:-unknown}"
echo " Method: $METHOD  Seed: $SEED  Epochs: $NUM_EPOCHS"
echo " Target: 480/0.18 = 2667 epochs"
echo "=============================================="

python main.py \
    --method "$METHOD" \
    --prob_type nonsmooth_nonconvex \
    --prob_name socp \
    --lr 0.0001 \
    --seed "$SEED" \
    --num_epochs "$NUM_EPOCHS" \
    --train_size 7000

echo "=============================================="
echo " Job finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
