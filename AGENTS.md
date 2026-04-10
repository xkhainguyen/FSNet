# AGENTS.md

## Purpose

This repository contains research code for learning-based solvers for parametric constrained optimization problems. The main workflows are:

- training models with `main.py`
- evaluating saved checkpoints with `eval.py`
- generating datasets under `datasets/`
- running experiment sweeps with shell or Slurm scripts under the repo root and `scripts/`
- analyzing results in notebooks

Agents working in this repo should optimize for small, targeted changes that preserve existing experiment behavior unless the user explicitly asks for a broader refactor.

## Repository Map

- `main.py`: primary training entry point
- `eval.py`: evaluation entry point for single checkpoints, saved ensembles, and ad hoc ensembles
- `configs/default.yaml`: shared defaults plus per-method overrides
- `models/neural_networks.py`: model definitions such as `MLP`, `EnsembleMLP`, and `MixtureOfExperts`
- `utils/trainer.py`: dataset loading, training loop, model creation, checkpointing
- `utils/evaluator.py`: evaluation logic and ensemble aggregation
- `utils/optimization_utils.py`: optimization problem definitions and utilities
- `utils/lbfgs.py`: differentiable or hybrid L-BFGS pieces used by FSNet-style methods
- `datasets/`: generated datasets and dataset creation scripts
- `scripts/`: analysis utilities, plotting helpers, timing scripts, ensemble helpers, and tuning scripts
- `scripts/tuning/`: Slurm-oriented tuning workflow and tuning log
- `piml/` and top-level `*.ipynb`: exploratory notebooks and analysis
- `results/`: generated experiment artifacts, usually not meant for source control

## Environment And Setup

Prefer the tracked Conda environment:

```bash
conda env create -f ml4opt.yml
conda activate ml4opt
```

Notes:

- `README.md` still mentions `requirements.txt`, but the checked-in environment file is `ml4opt.yml`.
- `wandb` is optional and only needed when logging experiments.
- This repo is often used on GPU and HPC systems; do not assume local CPU-only runs are representative.

## Common Commands

Train a model:

```bash
python main.py --method FSNet --prob_type convex --prob_name qp
```

Evaluate a checkpoint:

```bash
python eval.py --checkpoints path/to/model.pt
```

Evaluate from a run directory:

```bash
python eval.py --run_dir path/to/run_dir
```

Example tuning or orchestration assets live in:

- `run.*.sh`
- `sweep_*.sh`
- `scripts/ensemble/`
- `scripts/tuning/`

When changing command-line behavior, update `README.md` if the user-facing workflow changes.

## Working Style For Agents

- Start by reading `README.md`, `main.py`, `eval.py`, and any directly affected utility files.
- Check `git status` before editing. The worktree may already contain ongoing experiment changes.
- Prefer minimal edits that preserve backward compatibility for existing scripts and checkpoints.
- Favor config- or CLI-driven changes over hard-coded experiment-specific values.
- Keep research scripts and notebooks lightweight; do not mass-reformat notebooks unless explicitly asked.
- Avoid touching generated datasets or large result artifacts unless the task is specifically about them.
- Do not delete or rewrite historical experiment scripts just to make the tree cleaner.

## Editing Guidance

- If a change affects training arguments, keep `main.py`, `configs/default.yaml`, and any related docs aligned.
- If a change affects evaluation behavior, check both single-checkpoint and ensemble paths in `eval.py`.
- Preserve compatibility with existing checkpoint structure when possible. Old runs may rely on stored config keys.
- Prefer adding narrowly scoped helpers over large architectural rewrites.
- Keep imports simple and local conventions consistent; this codebase is pragmatic research code, not a framework.

## Validation Expectations

Choose the lightest validation that still meaningfully covers the change:

- for doc-only changes, no runtime validation is needed
- for CLI or config plumbing, run a help command or a short targeted invocation if feasible
- for training or evaluation logic, prefer a small targeted smoke test over a full sweep
- if a full run is too expensive, explain what you did verify and what remains unverified

Useful checks include:

```bash
python main.py --help
python eval.py --help
```

There is no obvious dedicated unit test suite in this repository, so do not claim broad test coverage unless you actually ran relevant scripts.

## HPC And Sweep Notes

- Many shell scripts are Slurm-oriented and may assume cluster-specific resources such as GPUs and queue settings.
- Do not casually change `#SBATCH` directives, job names, or checkpoint paths in experiment scripts unless the task calls for it.
- Tuning workflow details are documented in `scripts/tuning/TUNING_LOG.md`; use that file to understand recent experiment conventions.

## Data And Artifact Safety

- Treat files under `datasets/` as data assets first and code second.
- Treat `results/`, `logs/`, `figures/`, and `wandb/` as generated outputs.
- Do not commit generated outputs or large binaries unless the user explicitly requests it.
- Be careful with notebooks and copied notebooks such as `* copy.ipynb`; they may contain active exploratory work.

## When Unsure

- Ask whether the user wants a research-only quick fix or a cleaner reusable implementation if the tradeoff is non-obvious.
- If you see unrelated local modifications, work around them instead of reverting them.
- Leave concise notes in your final response about assumptions, validation performed, and any expensive steps you intentionally skipped.
