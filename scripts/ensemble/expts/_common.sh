#!/bin/bash
# NOTE: source ~/.bashrc BEFORE set -e — bashrc may return non-zero from
# helper scripts (e.g. conda init) and would otherwise abort the job.
source ~/.bashrc
conda activate ml4opt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
set -e
cd /orcd/scratch/orcd/008/khain/FSNet
mkdir -p logs

P_NSC="results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
PEN_SINGLE="$P_NSC/20260417-114439_penalty_seed0_e1000_lr1e-04_n7000"
FSNET_SINGLE="$P_NSC/20260315-181820_FSNet_seed0_e300_lr1e-04_n7000"
PEN_ENS5_VAN="$P_NSC/20260316-003102_penalty_seed0_e1000_lr1e-04_n7000_ens5_vanilla_pre"
PEN_ENS5_FGE_search=$(ls -d $P_NSC/*penalty_seed*ens5_fge_pre 2>/dev/null | head -1)
FSNET_ENS5_VAN="$P_NSC/20260315-172824_FSNet_seed0_e300_lr1e-04_n7000_ens5_vanilla_pre"
FSNET_ENS5_FGE="$P_NSC/20260316-004220_FSNet_seed0_e600_lr1e-04_n7000_ens5_fge_pre"
PEN_ENS20_VAN="$P_NSC/20260315-175720_penalty_seed0_e1000_lr1e-04_n7000_ens20_vanilla_pre"
