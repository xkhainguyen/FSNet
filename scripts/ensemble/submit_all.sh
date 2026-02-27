#!/bin/bash
# Submit all ensemble experiments with SLURM dependency chaining.
# Usage: bash scripts/ensemble/submit_all.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting ensemble experiments..."

# Training jobs (run in parallel on separate GPUs)
JID_VANILLA=$(sbatch --parsable "$DIR/exp_vanilla.sh")
echo "  exp_vanilla  -> $JID_VANILLA"

JID_FGE=$(sbatch --parsable "$DIR/exp_fge.sh")
echo "  exp_fge      -> $JID_FGE"

JID_FSNET=$(sbatch --parsable "$DIR/exp_fsnet.sh")
echo "  exp_fsnet    -> $JID_FSNET"

# Eval job (runs after vanilla + fsnet finish successfully)
JID_EVAL=$(sbatch --parsable --dependency=afterok:${JID_VANILLA}:${JID_FSNET} "$DIR/exp_eval.sh")
echo "  exp_eval     -> $JID_EVAL  (after $JID_VANILLA, $JID_FSNET)"

echo ""
echo "All submitted. Monitor with:  squeue -u \$USER"
echo "Cancel all:  scancel $JID_VANILLA $JID_FGE $JID_FSNET $JID_EVAL"
