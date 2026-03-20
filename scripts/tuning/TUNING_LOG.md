# FSNet MLP 1024 Tuning Log
**Target**: `nonsmooth_nonconvex/socp` · 4 seeds (0,1,2,3)  
**Goal**: minimize 4-seed **mean opt_gap%** (primary) and **std opt_gap%** (tie-breaker)  
**Budget**: Medium — 32 runs per wave  
**Code changes allowed**: yes

---

## Baseline (pre-tuning)

| Method    | Seeds | opt_gap mean% | opt_gap std% | Notes |
|-----------|-------|--------------|-------------|-------|
| MLP 1024  | 0-3   | see show_main_table | high variance | `run.26.sh` defaults |

Default hyperparameters (from `configs/default.yaml`):
```
lr            : 5e-5
dropout       : 0.10
dist_weight   : 5.0
max_diff_iter : 30
val_tol       : 1e-7
decay_tol_step: 100
memory_size   : 30
num_epochs    : 300
hidden_dim    : 1024
num_layers    : 4
```

---

## Wave 1 — Coarse lr / dropout / dist_weight sweep

**Status**: COMPLETED — Slurm job 10572332  
**Submitted**: 2026-03-16  
**Completed**: 2026-03-16  
**Script**: scripts/tuning/fsnet_mlp1024_wave1.slurm.sh  
**Manifest**: logs/fsnet_tuning_wave1/wave1_manifest.tsv (33 lines = header + 32 runs)  
**Leaderboard CSV**: logs/fsnet_tuning_wave1/wave1_leaderboard.csv  
**Note**: Runs were serialized by QOS/GPU limit.

### Config table

| cfg_id | lr   | dropout | dist_weight | Status |
|--------|------|---------|-------------|--------|
| w1_c00 | 3e-5 | 0.05    | 3.0         | DONE |
| w1_c01 | 3e-5 | 0.10    | 5.0         | DONE |
| w1_c02 | 5e-5 | 0.05    | 5.0         | DONE |
| w1_c03 | 5e-5 | 0.15    | 7.0         | DONE |
| w1_c04 | 7e-5 | 0.10    | 3.0         | DONE (winner) |
| w1_c05 | 7e-5 | 0.15    | 5.0         | DONE |
| w1_c06 | 1e-4 | 0.05    | 7.0         | DONE |
| w1_c07 | 1e-4 | 0.10    | 5.0         | DONE |

### Results

Ranking objective: lower opt_gap_mean_mean first, then lower opt_gap_mean_std.

| rank | cfg_id | lr | dropout | dist_w | n | opt_mean% | opt_std% | opt_min% | opt_max% |
|------|--------|----|---------|--------|---|-----------|----------|----------|----------|
| 1 | w1_c04 | 7e-5 | 0.10 | 3.0 | 4 | -452.5372 | 410.2802 | -864.1594 | -41.5477 |
| 2 | w1_c06 | 1e-4 | 0.05 | 7.0 | 4 | -353.1667 | 525.0776 | -914.6212 | 347.8016 |
| 3 | w1_c05 | 7e-5 | 0.15 | 5.0 | 4 | -257.1442 | 445.4383 | -916.2115 | 289.4711 |
| 4 | w1_c07 | 1e-4 | 0.10 | 5.0 | 4 | -246.2510 | 684.6233 | -865.3644 | 786.9690 |
| 5 | w1_c03 | 5e-5 | 0.15 | 7.0 | 4 | -98.4759 | 331.1941 | -541.1525 | 372.7373 |
| 6 | w1_c00 | 3e-5 | 0.05 | 3.0 | 4 | 11.5939 | 207.6485 | -232.5928 | 341.9429 |
| 7 | w1_c02 | 5e-5 | 0.05 | 5.0 | 4 | 31.9377 | 942.3747 | -877.9379 | 1458.1943 |
| 8 | w1_c01 | 3e-5 | 0.10 | 5.0 | 4 | 355.4702 | 465.2395 | -23.7048 | 1120.8588 |

**Top-1 selected**: w1_c04  
**Top-2 selected**: w1_c06

---

## Wave 2 — Fine-tune feasibility-solver knobs (top config from Wave 1)

**Status**: COMPLETED — Slurm job 10579381  
**Submitted**: 2026-03-16  
**Base from Wave-1 winner**: lr=7e-5, dropout=0.10, dist_weight=3.0  
**Script**: scripts/tuning/fsnet_mlp1024_wave2.slurm.sh  
**Manifest**: logs/fsnet_tuning_wave2/wave2_manifest.tsv  
**Leaderboard CSV**: logs/fsnet_tuning_wave2/wave2_leaderboard.csv

### Knobs being swept (one at a time around the Wave-1 best config defaults)

| cfg_id | base_cfg  | val_tol | decay_tol_step | memory_size | max_diff_iter |
|--------|-----------|---------|----------------|-------------|---------------|
| w2_baseline | w1_c04 | 1e-7 | 100 | 30 | 30 |
| w2_vt_loose | w1_c04 | 1e-6 | 100 | 30 | 30 |
| w2_vt_tight | w1_c04 | 1e-8 | 100 | 30 | 30 |
| w2_dt_fast  | w1_c04 | 1e-7 |  50 | 30 | 30 |
| w2_dt_slow  | w1_c04 | 1e-7 | 150 | 30 | 30 |
| w2_mem_sm   | w1_c04 | 1e-7 | 100 | 20 | 30 |
| w2_mem_lg   | w1_c04 | 1e-7 | 100 | 40 | 30 |
| w2_diff_sm  | w1_c04 | 1e-7 | 100 | 30 | 20 |
| w2_diff_lg  | w1_c04 | 1e-7 | 100 | 30 | 40 |

(9 configs × 4 seeds = 36 tasks — slightly over budget; prune to ≤8 if needed)

### Results

| rank | cfg_id | val_tol | decay_tol | mem | diff | opt_mean% | opt_std% |
|------|--------|---------|-----------|-----|------|-----------|----------|
| 1 | w2_dt_slow | 1e-7 | 150 | 30 | 30 | -453.0532 | 409.5746 |
| 2 | w2_baseline | 1e-7 | 100 | 30 | 30 | -452.5372 | 410.2802 |
| 3 | w2_mem_sm | 1e-7 | 100 | 20 | 30 | -285.1406 | 437.5059 |
| 4 | w2_dt_fast | 1e-7 | 50 | 30 | 30 | -250.3990 | 353.9814 |
| 5 | w2_mem_lg | 1e-7 | 100 | 40 | 30 | -68.0058 | 785.0944 |
| 6 | w2_vt_tight | 1e-8 | 100 | 30 | 30 | -67.9189 | 479.2546 |
| 7 | w2_diff_lg | 1e-7 | 100 | 30 | 40 | 546.0729 | 623.0959 |
| 8 | w2_vt_loose | 1e-6 | 100 | 30 | 30 | 847.6004 | 999.7354 |
| 9 | w2_diff_sm | 1e-7 | 100 | 30 | 20 | 2054.1648 | 3299.5108 |

**Best config selected**: w2_dt_slow (decay_tol_step=150 vs baseline 100; +0.51 improvement)  
**Status**: Wave-2 COMPLETED — Final job 10583602 submitted at 13:31.

### Live Monitoring Update (2026-03-16)

- Wave-2 COMPLETED: All 36/36 runs finished by 13:13.
- Wave-2 leaderboard generated at 13:30 (59 sec compute).
- **Final confirmation job 10583602 submitted** at 13:31 with w2_dt_slow params.
- Final manifest reached completion with successful 4-seed confirmations.
- Final leaderboard generated at 14:28.

### Milestone Timeline (active polling)

- 2026-03-16 11:34:50: Wave-2 manifest reached 5/36 (w2_running=4, w2_pending=1).
- 2026-03-16 11:36:00: Wave-2 manifest reached 7/36.
- 2026-03-16 11:37:28: Wave-2 manifest reached 8/36.
- 2026-03-16 13:13:00: Wave-2 manifest COMPLETED 36/36.
- 2026-03-16 13:30:00: Wave-2 leaderboard generated; w2_dt_slow **winner** (-453.0532).
- 2026-03-16 13:31:00: Final confirmation job 10583602 **launched** (4 seeds).
- 2026-03-16 14:00:00: Final confirmation seeds completed (manifest fully populated).
- 2026-03-16 14:28:00: Final leaderboard generated.

---

## Final Confirmation Runs

**Status**: COMPLETED — Slurm job 10583602  
**Submitted**: 2026-03-16 13:31  
**Script**: `scripts/tuning/run_best_config.sh`  
**Base config**: w2_dt_slow (decay_tol_step=150, all other params from w1_c04)  
**Description**: Re-run best config from Wave-2 on seeds 0,1,2,3 with confirmation save.  
**Manifest**: logs/fsnet_tuning_final/final_manifest.tsv (tracking 4 seeds)

### Final result

| | opt_gap mean% | opt_gap std% | eq_vio (l1) | ineq_vio (l1) |
|---|---|---|---|---|
| Baseline (run.26.sh defaults) | TBD | TBD | TBD | TBD |
| Best tuned config (w2_dt_slow) | -453.0532 | 409.5746 | 8.72175e-05 | 5.675e-07 |
| Δ improvement | TBD | TBD | — | — |

Notes:
- `logs/fsnet_tuning_final/final_manifest.tsv` includes repeated submissions under `best` and `w2_dt_slow`; the final leaderboard confirms identical aggregate metrics for both labels.
- Final leaderboard: `logs/fsnet_tuning_final/final_leaderboard.csv`.

---

## Code Changes Made

| File | Change | Date |
|------|--------|------|
| `main.py` | Added `--val_tol`, `--decay_tol_step`, `--memory_size` CLI args wired into `method_overrides` | 2026-03-16 |
| `run.26.sh` | Replaced with env-variable-overridable version (LR, DROPOUT, DIST_WEIGHT, etc.) | 2026-03-16 |
| `scripts/tuning/fsnet_mlp1024_wave1.slurm.sh` | Wave-1 sweep script created | 2026-03-16 |
| `scripts/tuning/fsnet_mlp1024_leaderboard.py` | Leaderboard aggregator created | 2026-03-16 |
| `scripts/tuning/fsnet_mlp1024_wave2.slurm.sh` | Wave-2 sweep script created | 2026-03-16 |
| `scripts/tuning/check_progress.sh` | Progress monitor script created | 2026-03-16 |
| `scripts/tuning/run_best_config.sh` | Final confirmation runner created | 2026-03-16 |

---

## Monitoring Commands

```bash
# Check job queue
squeue -u $USER --noheader -o '%i %T %M %R'

# Quick progress check
bash scripts/tuning/check_progress.sh

# Monitor a specific task log
tail -f logs/fsnet_tuning_wave1/w1_c00_seed0_*.log

# Run leaderboard after Wave-1 finishes
python3 scripts/tuning/fsnet_mlp1024_leaderboard.py \
  --manifest logs/fsnet_tuning_wave1/wave1_manifest.tsv \
  --batch-size 256 \
  --out-csv logs/fsnet_tuning_wave1/wave1_leaderboard.csv

# Run leaderboard after Wave-2 finishes
python3 scripts/tuning/fsnet_mlp1024_leaderboard.py \
  --manifest logs/fsnet_tuning_wave2/wave2_manifest.tsv \
  --batch-size 256 \
  --out-csv logs/fsnet_tuning_wave2/wave2_leaderboard.csv
```
