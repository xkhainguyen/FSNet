## Neural Solver Ensemble

A benchmark for learning-based approaches to constrained optimization. Trains neural networks to predict solutions to parametric optimization problems (QP, QCQP, SOCP) across convex, nonconvex, and nonsmooth settings. Supports multiple solution methods, deep ensembles, and flexible evaluation.

## Installation

```bash
pip install -r requirements.txt
pip install wandb  # optional, for experiment tracking
```

## Methods

| Method | Flag | Description |
|---|---|---|
| FSNet | `--method FSNet` | Feasibility-Seeking Neural Network with differentiable L-BFGS |
| Penalty | `--method penalty` | Fixed penalty weights on constraint violations |
| Adaptive Penalty | `--method adaptive_penalty` | Dynamically increasing penalty weights |
| DC3 | `--method DC3` | Deep Constraint Completion and Correction |
| Projection | `--method projection` | QP projection onto feasible set (QP only) |
| Supervised | `--method sup` | Huber loss against ground-truth solutions |
| Supervised + Penalty | `--method sup_pen` | Supervised loss with penalty regularization |
| Supervised Partial | `--method sup_partial` | Supervised on partial variables (with completion) |
| S3Net | `--method S3Net` | Supervised + feasibility-seeking hybrid |
| Semi-supervised | `--method semi` | Half-batch supervised + half-batch self-supervised |

## Problems

|  | QP | QCQP | SOCP |
|---|---|---|---|
| **Convex** | `--prob_type convex --prob_name qp` | `--prob_type convex --prob_name qcqp` | `--prob_type convex --prob_name socp` |
| **Nonconvex** | `--prob_type nonconvex --prob_name qp` | `--prob_type nonconvex --prob_name qcqp` | `--prob_type nonconvex --prob_name socp` |
| **Nonsmooth Nonconvex** | `--prob_type nonsmooth_nonconvex --prob_name qp` | `--prob_type nonsmooth_nonconvex --prob_name qcqp` | `--prob_type nonsmooth_nonconvex --prob_name socp` |

## Training

### Single model

```bash
python main.py \
    --method FSNet \
    --prob_type convex \
    --prob_name qp
```

### Context-augmented model

```bash
python main.py \
    --method FSNet \
    --prob_type convex \
    --prob_name qp \
    --network SampledContextMLPv1 \
    --context_num_points 16
```

`SampledContextMLPv1` augments `x` with context features computed at a fixed bank of sampled reference `y` points. The current v1 context includes the objective scalar plus the full equality and inequality residual vectors for each sampled reference point, flattened directly into the predictor input.
The sampled reference bank is derived deterministically from the main run `--seed`.

`SampledContextMLPv2` uses the same sampled per-point features, but encodes each sampled point first and mean-pools the encoded context before fusing it with `x`.

`LocalContextMLPv1` uses a two-stage local refinement design: it predicts a coarse `y0` from `x`, computes local objective and constraint features at that coarse prediction, and then predicts the final `y` from `[x, y0, local_structure(x, y0)]`.

`LocalContextMLPv2` keeps the same local two-stage idea, but predicts a bounded residual correction on top of the coarse prediction and adds a coarse-stage auxiliary penalty loss during training.

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | YAML config file |
| `--seed` | `2025` | Random seed |
| `--train_size` | `7000` | Training set size |
| `--network` | `MLP` | `MLP`, `SampledContextMLPv1`, `SampledContextMLPv2`, `LocalContextMLPv1`, `LocalContextMLPv2`, `MultiHeadMLP`, or `MoE` |
| `--lr` | (from config) | Learning rate |
| `--num_epochs` | (from config) | Number of training epochs |
| `--hidden_dim` | `1024` | MLP hidden dimension |
| `--num_layers` | `4` | Number of hidden layers |
| `--dropout` | `0.1` | Dropout rate |
| `--init` | `default` | Initialization scheme for the selected method. `output_center` is label-free and initializes output-layer weights with small Xavier plus zero output bias. `mean_bias` additionally uses train labels to set output bias from the normalized train-set mean. |
| `--init_gain` | `0.2` | Xavier gain for init schemes that initialize output-layer weights. |
| `--context_num_points` | `16` | Number of sampled reference `y` points for `SampledContextMLPv1` |
| `--context_encoder_dim` | `128` | Point-encoder hidden size for `SampledContextMLPv2` |
| `--local_delta_scale` | `0.2` | Max residual correction scale for `LocalContextMLPv2` |
| `--local_coarse_loss_weight` | `0.5` | Weight on coarse-stage penalty loss for `LocalContextMLPv2` |
| `--mhe_num_heads` | `5` | Number of heads for `--network MultiHeadMLP` |
| `--mhe_head_hidden_dim` | `hidden_dim//4` | Head hidden width for `MultiHeadMLP` |
| `--checkpoint` | `None` | Resume from a saved `.pt` file |
| `--save_intermediate` | `False` | Save model at each validation step |

See `configs/default.yaml` for full per-method hyperparameters, MoE settings, and `SampledContextMLPv1` context settings.

## Deep Ensembles

Train multiple models as an ensemble with `--ensemble_size > 1`.

### Vanilla ensemble (independent initializations)

Trains M models from different random seeds and combines them:

```bash
python main.py \
    --method FSNet \
    --prob_type convex --prob_name qp \
    --ensemble_size 5 --ensemble_mode vanilla
```

### Fast Geometric Ensembling (FGE)

Pre-trains a single model, then collects M snapshots using cyclical learning rate:

```bash
python main.py \
    --method FSNet \
    --prob_type convex --prob_name qp \
    --ensemble_size 5 --ensemble_mode fge \
    --fge_pretrain_ratio 0.8
```

| Flag | Default | Description |
|---|---|---|
| `--ensemble_size` | `1` | Number of ensemble members (1 = single model) |
| `--ensemble_mode` | `vanilla` | `vanilla` (independent inits) or `fge` (snapshot ensemble) |
| `--fge_pretrain_ratio` | `0.8` | Fraction of epochs for pre-training before FGE snapshot collection |
| `--fge_lr_max` | (base lr) | Peak LR during FGE cyclical phase |

### Multi-Head Ensemble (MHE)

A single network with one shared trunk and M independent heads. Each head produces a candidate solution; all candidates are repaired separately and aggregated via `best_merit` at evaluation. No learned router (unlike `MoE`). Typically much smaller (~28% the parameters of an M-member vanilla ensemble of the same width).

```bash
python main.py \
    --method FSNet \
    --prob_type nonsmooth_nonconvex --prob_name socp \
    --network MultiHeadMLP \
    --mhe_num_heads 5 \
    --hidden_dim 1024 --num_layers 4
```

Notes:

- For **FSNet/DC3/projection** (methods with a non-trivial repair), the in-loop repair during training pushes each head's prediction toward a different feasible local optimum, giving the heads functional diversity. MHE matches or beats vanilla ens5 quality at 28% the parameters.
- For **penalty** (no test-time repair), heads on the shared trunk stay too similar — MHE trains faster but Merit lags vanilla. A diversity loss between heads (e.g., functional repulsion) is needed to close the gap.

### Ensemble evaluation modes

Control how ensemble members are combined at evaluation time:

| Flag | Values | Description |
|---|---|---|
| `--ensemble_post` | `pre` (default), `post` | `pre` = average NN outputs then post-process once. `post` = post-process each member then aggregate. |
| `--ensemble_agg` | `mean` (default), `median`, `best_obj`, `best_merit` | Aggregation strategy. `best_obj` picks the member with the lowest objective per sample. `best_merit` picks the member with the best merit (`obj + 1e6·(eq_viol_L1 + ineq_viol_L1)`). |

```bash
# Fast: average then post-process once
python main.py ... --ensemble_post pre --ensemble_agg mean

# High quality: post-process each member, pick the best per sample
python main.py ... --ensemble_post post --ensemble_agg best_merit
```

**Merit weight**: Both the reported `merit_mean` (`_compute_merit`) and the `best_merit` aggregation selector use the same weight `ρ = 1e6` on equality and inequality L1 violations. Prior to commit `8ef818b` the selector used `ρ = 1e5`; old eval numbers from that period may differ slightly.

## Evaluation

Use `eval.py` to evaluate saved checkpoints without retraining.

### Single model

```bash
python eval.py --checkpoints results/.../model.pt
```

### Saved ensemble

```bash
python eval.py --checkpoints results/.../ensemble_model.pt
```

### Ad-hoc ensemble from independently trained models

Combine any number of single-model checkpoints into an ensemble at evaluation time:

```bash
python eval.py \
    --checkpoints model_seed0.pt model_seed1.pt model_seed2.pt \
    --ensemble_post post \
    --ensemble_agg best_merit
```

### Inference-time perturbation (multi-restart repair)

Turn any single-model checkpoint into a K-restart "free ensemble" by perturbing its raw NN output K times and running the repair operator (L-BFGS for FSNet) on each, then aggregating via `best_merit`. Zero retraining cost. Only meaningful for methods with a non-trivial repair step (FSNet, DC3, projection) — for `penalty` (identity repair) the trick is a no-op.

```bash
python eval.py --checkpoints results/.../model.pt \
    --inference_perturb_k 20 \
    --inference_perturb_eps 0.1 \
    --ensemble_agg best_merit
```

| Flag | Default | Description |
|---|---|---|
| `--inference_perturb_k` | `0` (disabled) | Number of perturbed L-BFGS restarts. K=20 ε=0.1 typically beats vanilla M=5 ensembles on FSNet. |
| `--inference_perturb_eps` | `0.05` | Standard deviation of the additive noise on the raw NN output. Sweet spot ε≈0.05–0.1 for FSNet on SOCP. |
| `--inference_perturb_dist` | `gauss` | `gauss`, `antithetic` (deterministic ±z pairs for variance reduction), or `sphere` (radially uniform). |
| `--inference_perturb_keep_original` | `1` | Include the unperturbed prediction as restart 0 (1=yes, 0=no). |

### Repair-step ablation

Quantify the contribution of the per-sample repair (L-BFGS for FSNet, projection for DC3, identity for penalty) on a saved checkpoint:

```bash
# Skip repair entirely; evaluate the raw NN output
python eval.py --checkpoints results/.../model.pt --skip_repair

# Override the L-BFGS iteration budget
python eval.py --checkpoints results/.../model.pt --repair_max_iter 20
```

| Flag | Default | Description |
|---|---|---|
| `--skip_repair` | off | Bypass the repair step; evaluate the raw scaled NN output directly. |
| `--repair_max_iter` | (from config) | Override the L-BFGS `max_iter` at eval time. Reducing this is destructive — on FSNet, dropping from 50 → 20 inflates merit ~90× per restart. |

### Eval flags

| Flag | Default | Description |
|---|---|---|
| `--run_dir` | — | Training run directory. Auto-discovers `members/member_*.pt` (ensemble) or `model.pt`. |
| `--checkpoints` | — | One or more `.pt` checkpoint paths. Multiple are combined into an ad-hoc ensemble. (Either `--run_dir` or `--checkpoints` is required.) |
| `--config` | (from checkpoint) | Override YAML config |
| `--ensemble_size` | (from checkpoint) | Subsample to first N members |
| `--ensemble_post` | `pre` | `pre` or `post` (see above) |
| `--ensemble_agg` | `mean` | `mean`, `median`, `best_obj`, `best_merit` |
| `--skip_repair` | off | See "Repair-step ablation" |
| `--repair_max_iter` | (from config) | See "Repair-step ablation" |
| `--inference_perturb_k` | `0` | See "Inference-time perturbation" |
| `--inference_perturb_eps` | `0.05` | See "Inference-time perturbation" |
| `--inference_perturb_dist` | `gauss` | See "Inference-time perturbation" |
| `--test_batch_sizes` | (from config) | Override test batch sizes |

## Weights & Biases

Add `--wandb` to any training or evaluation command to log metrics to W&B:

```bash
# Training
python main.py --method FSNet --prob_type convex --prob_name qp \
    --wandb --wandb_project my-project --wandb_tags convex qp

# Evaluation
python eval.py --checkpoints model.pt \
    --wandb --wandb_project my-project-eval
```

Logged metrics include per-epoch training loss, constraint violations, optimality gap, learning rate, validation metrics, and final test results. Trained models are saved as W&B artifacts.

| Flag | Default | Description |
|---|---|---|
| `--wandb` | off | Enable W&B logging |
| `--wandb_project` | `FSNet` | W&B project name |
| `--wandb_entity` | `None` | W&B team or user |
| `--wandb_run_name` | (auto) | Custom run name |
| `--wandb_tags` | `None` | Tags for the run |

## Project Structure

```
.
├── main.py                    # Training entry point
├── eval.py                    # Evaluation entry point
├── configs/
│   └── default.yaml           # Default hyperparameters for all methods
├── models/
│   └── neural_networks.py     # MLP, EnsembleMLP, MultiHeadMLP, MoE, context-augmented variants
├── utils/
│   ├── trainer.py             # Training loop, loss functions, ensemble training
│   ├── evaluator.py           # Evaluation, ensemble aggregation, perturbation, repair ablation
│   ├── optimization_utils.py  # Problem definitions (QP, QCQP, SOCP variants)
│   └── lbfgs.py               # Differentiable / hybrid L-BFGS solver
├── scripts/
│   ├── ensemble/              # Ensemble experiment drivers
│   │   ├── exp_*.sh           # SLURM drivers for vanilla / FGE / penalty / FSNet
│   │   ├── mhe/               # MultiHeadMLP training + smoke-test scripts
│   │   └── expts/             # Audit, perturbation, repair-ablation, MHE-eval scripts + analysis
│   ├── tuning/                # Hyperparameter tuning orchestration
│   ├── landscape/             # Compute-side: input/weight → merit landscape probes
│   ├── analysis/              # FSNet winner-region analyses, checkpoint diagnostics
│   └── visualization/         # Plotting notebooks (winner regions, training curves, etc.)
├── datasets/
│   ├── convex/                # Generated convex problem datasets
│   ├── nonconvex/             # Generated nonconvex problem datasets
│   └── nonsmooth_nonconvex/   # Generated nonsmooth nonconvex datasets
├── archive/                   # Stale or superseded scripts/notebooks (kept for history)
│   ├── scripts/
│   └── notebooks/
└── results/                   # Saved models and evaluation results (gitignored)
```
