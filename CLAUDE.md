# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to start

`README.md` has the full user-facing surface (CLI flags, training/eval workflows, project layout). This file adds **the architectural picture you can't get by reading any single file**, plus working-style guidance specific to this codebase.

## Working style

This is research code, not a framework. Optimize for **small, targeted changes** that preserve existing experiment behavior unless the user explicitly asks for a refactor.

- Always check `git status` before editing — the worktree often has ongoing experiment changes.
- Prefer config-/CLI-driven changes over hard-coded experiment values.
- Keep imports simple and local; this codebase doesn't have a framework abstraction layer to slot into.
- Do not mass-reformat notebooks or mass-rewrite historical experiment scripts just to make the tree cleaner.
- If you see unrelated local modifications in the worktree, work around them — don't revert them.
- When in doubt about whether the user wants a quick research fix or a cleaner reusable implementation, ask.

When you change training behavior, keep `main.py` / `configs/default.yaml` / `README.md` aligned. When you change evaluation behavior, verify both the single-checkpoint and ensemble paths in `eval.py`.

## Validation

Choose the lightest validation that actually covers the change:

- Doc-only: no runtime validation needed.
- CLI/config plumbing: run a `--help` invocation or a short targeted command.
- Training/evaluation logic: the `--num_epochs 5 --train_size 500` smoke pattern below usually runs in ~20 s.
- If a full sweep is too expensive, state what you verified and what remains unverified.

There is **no dedicated test suite** in this repo.

## Architecture: the four composable axes

Every experiment in this repo is a choice on four orthogonal axes. Most code lives at the intersection.

1. **Method** (`--method`): defines the training-time loss and the test-time repair.
   - Method dispatch in `utils/trainer.py:compute_batch_loss` (~L457) and `utils/evaluator.py:_post_process_predictions` (~L172).
   - **The repair step matters more than the loss** for FSNet on these problems: full L-BFGS at `max_iter=50` reduces merit by ~31000× vs `skip_repair=True`. The NN is essentially a warm start for L-BFGS.
   - **`penalty` has identity repair** (pass-through in `_post_process_predictions`). So for penalty: `--ensemble_post pre` and `--ensemble_post post` are equivalent; `--inference_perturb_k` is a no-op; `--skip_repair` and `--repair_max_iter` are no-ops.

2. **Network** (`--network`): `MLP`, `SampledContextMLPv{1,2}`, `LocalContextMLPv{1,2}`, `MoE`, `MultiHeadMLP`. Defined in `models/neural_networks.py`, wired into `create_model` (`utils/trainer.py:create_model`, ~L280).
   - `EnsembleMLP` is the wrapper used when `--ensemble_size > 1` (vanilla or FGE).
   - `MultiHeadMLP` (added in commit `8ef818b`) is a single-network alternative that exposes the same `forward_all(x) -> (M,B,out)` contract as `EnsembleMLP`, so it routes through the existing ensemble eval path automatically.

3. **Ensemble training mode** (`--ensemble_mode`): `vanilla` (M independent seeds, sequential) or `fge`. Implementation in `utils/trainer.py:_train_vanilla_ensemble` / `_train_fge_ensemble` (~L1316, L1352). FGE is confirmed worse than vanilla on locked-down eval; if vanilla and FGE numbers disagree dramatically, treat the FGE result as the suspect one.

4. **Eval-time aggregation**: `--ensemble_post {pre, post}` × `--ensemble_agg {mean, median, best_obj, best_merit}` × (optional) `--inference_perturb_k K --inference_perturb_eps ε`. Implementation in `utils/evaluator.py:_get_final_prediction` and `_aggregate_predictions` (~L191, L350).

## The merit weight: a subtle gotcha (now resolved)

Before commit `8ef818b`, the codebase used **two different ρ for merit**: the `best_merit` selector used ρ=1e5, while the reported `merit_mean` (`_compute_merit`) used ρ=1e6. Both are now unified at **ρ=1e6** in `utils/evaluator.py` (lines 108, 294, 377, 391). When comparing to old eval numbers from before `8ef818b`, the unified-ρ versions can be 10–25% lower for the same checkpoints.

## Training/eval data flow (FSNet, the most representative method)

```
[X, Y_true] ─► NN forward ─► Y_pred ─► opt_problem.scale(Y_pred) ─► Y_scaled
                                                                      │
                                                                      ▼
                                              hybrid_lbfgs_solve (training-time, differentiable)
                                              nondiff_lbfgs_solve (eval-time, no grad)
                                                                      │  (L-BFGS minimizes
                                                                      │   ‖eq_resid‖² + ‖ineq_resid‖²
                                                                      │   — pure feasibility,
                                                                      │   no obj term)
                                                                      ▼
                                                                   Y_final
                                                                      │
                                                      ┌───────────────┴───────────────┐
                                                      ▼                                ▼
                                          loss = obj_w·obj(Y_final)              compute_batch_metrics
                                               + dist_w·‖Y_final - Y_scaled‖²    → merit, opt_gap, ...
                                               (penalty terms if pre-repair
                                                feasibility was too poor)
```

Key consequence: at eval, `best_merit` selection over K post-repaired candidates is effectively `best_obj` selection, because the L-BFGS makes all K candidates approximately feasible. The Merit reduction across K shows up in EqL1 (not Obj) because of the 1e6 weighting.

## Config plumbing

`main.py:create_parser` does a three-layer merge:

1. `configs/default.yaml` → flat per-method dicts (`config[method]`) via `_merge_defaults`.
2. Nested-config normalization for MoE / context models / MultiHeadMLP via `_normalize_moe_config`, `_normalize_context_configs` (and inline for MHE).
3. CLI overrides applied at the top level (`config[key]`), at the nested-config level (`config['MoE'][key]`), and at the method level (`config[method][key]`).

Method-specific knobs like `lr`, `num_epochs`, `scale`, `dist_weight`, `max_iter`, `memory_size`, `val_tol` live under `config[method]`, not at the top level. Tuning hyperparameters per method requires writing to the right namespace.

## Eval has two checkpoint paths

- `eval.py --run_dir <dir>`: auto-discovers `members/member_*.pt` (ensemble) or `model.pt` (single). Loads the model class from the saved config's `network` field. Use this for everything except ad-hoc ensembles.
- `eval.py --checkpoints a.pt b.pt c.pt`: combines multiple single-model checkpoints into an `EnsembleMLP` at eval time. The saved `ensemble_size` inside each `.pt` is **ignored** in favor of `len(checkpoints)`.

**Quirk for MHE checkpoints**: `MultiHeadMLP` is trained via the single-model path (`train()` in trainer.py), so the saved config has `ensemble_size=1`. The eval save directory name will not contain `_ens5`; instead the source identifier is in the `_first_checkpoint_path` config field. When aggregating MHE eval results from CSV, filter by source-dir pattern rather than `ensemble_size`.

## Run-name convention (training)

Built in `utils/trainer.py:load_instance` (~L137). The suffix grows with config:

```
{timestamp}_{method}_e{num_epochs}_lr{lr}_n{train_size}_hdim{hidden_dim}
  [_ens{M}_{mode}_{post}]      # ensemble
  [_ctx{N} | _ctxv2k{K}e{E} | _localctxv2d{D}]   # context model
  [_mhe{M}h{head_hidden_dim}]  # MultiHeadMLP
  [_moe{N}k{K}_temp{T}_noise{S}]               # MoE
  _seed{seed}
```

Use this when greping `results/` for matching configurations.

## Common commands

```bash
# Activate the tracked env
conda activate ml4opt

# This is required on the pi_donti_gpu partition (RTX/H200 nodes) — without
# it, scipy fails to import with a GLIBCXX_3.4.26 error:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Smoke checks (no GPU needed)
python main.py --help
python eval.py --help

# Smoke training (small, ~20s on CPU/GPU)
python main.py --method penalty --prob_type nonsmooth_nonconvex --prob_name socp \
               --num_epochs 5 --train_size 500 --seed 0

# Eval one checkpoint with the unified-ρ best_merit selector
python eval.py --run_dir results/.../some_run --ensemble_post post --ensemble_agg best_merit

# Apply perturbation eval to any single-model FSNet checkpoint (zero retraining)
python eval.py --run_dir results/.../some_fsnet_run \
               --inference_perturb_k 20 --inference_perturb_eps 0.1 \
               --ensemble_agg best_merit

# Ablate repair contribution
python eval.py --run_dir results/.../some_run --skip_repair
python eval.py --run_dir results/.../some_run --repair_max_iter 20
```

Treat each method × network × ensemble × aggregation combination as its own integration surface; verify changes with the smallest config that exercises the path you changed.

## SLURM patterns in this repo

- Existing ensemble SLURM jobs (the "official" infrastructure): `scripts/ensemble/exp_*.sh`, dispatched by `scripts/ensemble/submit_all.sh`. Use `mit_normal_gpu,mit_preemptable` with `--gres=gpu:l40s:1`.
- Recent overnight experiments: `scripts/ensemble/{mhe,expts}/*.sh`. Use `pi_donti_gpu` with `--gres=gpu:rtx_pro_6000:1` (or `h200:1`). Account is `mit_general`. Note the **4-GPU-per-user QoS limit** on `pi_donti_gpu`.
- **Empirical: L40S is faster than RTX Pro 6000 for FSNet training** (L-BFGS-bound, kernel-launch-limited, not FLOPS-limited). Prefer L40S when available.

Do not casually change `#SBATCH` directives, job names, or checkpoint paths in experiment scripts.

## Where the recent work lives

- `scripts/ensemble/mhe/`: training scripts for `MultiHeadMLP`.
- `scripts/ensemble/expts/`: experiment scripts + analysis from the 2026-05-22 advisor-prep session. `MEETING_PITCH.md` / `FINAL_SUMMARY.md` / `RESULTS_DRAFT.md` document the findings (MHE, perturbation, repair-ablation).
- `archive/`: superseded scripts and notebooks. Don't touch unless explicitly archaeology-related.

## Things to be careful with

- **The `default` `--ensemble_post pre --ensemble_agg mean`** in `eval.py` is almost never what you want for FSNet ensembles — it averages raw NN outputs, which is far from feasibility, and then runs repair once on the average. Always use `--ensemble_post post --ensemble_agg best_merit` for FSNet ensemble eval.
- **`results/` is gitignored**, but `results/` paths get baked into the run-name string and `_first_checkpoint_path` config field. Eval tagging assumes the original training-time path structure exists; if you move runs, eval tags can become misleading.
- **`datasets/*` are tracked binaries (~22 MB each)**. Don't regenerate casually; the `random2025` seed is the canonical dataset prefix.
