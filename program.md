# ML4OPT Autoresearch Program

This file defines how to run autonomous autoresearch for this repository.

Before starting, read these references first:

- `AGENTS.md` for repository workflow, editing boundaries, validation expectations, and artifact safety.
- `FEATURE_AUGMENTATION.md` for the current LocalContextMLPv2 direction, benchmark order, metrics, and first tuning knobs.
- `README.md`, `main.py`, `eval.py`, `models/neural_networks.py`, `utils/trainer.py`, `utils/evaluator.py`, and `configs/default.yaml` for the live implementation details.

`AGENTS.md` and `FEATURE_AUGMENTATION.md` are authoritative for this autoresearch run. Follow them before making decisions.

This file also inherits the generic operational rules from
`third-party/autoresearch/program.md`. When there is a conflict, this
repository-specific file overrides the generic version.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag. Propose a tag based on today's date. The branch `autoresearch/<tag>` must not already exist.
2. Create the branch from the current working branch: `git checkout -b autoresearch/<tag>`.
3. Check `git status` before editing. Do not revert unrelated changes.
4. Read the reference files listed above, especially `AGENTS.md` and `FEATURE_AUGMENTATION.md`.
5. Confirm the tracked Conda environment is available:
   - `conda env create -f ml4opt.yml`
   - `conda activate ml4opt`
6. Verify the target dataset exists under `datasets/nonsmooth_nonconvex/socp/` by checking that the dataset path resolved by `utils/trainer.py` is present.
7. Initialize an untracked TSV file named `autoresearch_results.tsv` with the header row defined below.
8. Confirm setup looks good, then begin experimentation.

Generated outputs under `results/`, `logs/`, `wandb/`, `figures/`, and similar directories are not source artifacts and must not be committed unless the user explicitly asks for that.

The first execution in a fresh autoresearch loop should establish the relevant
baseline before trying modifications.

## Scope

The current autoresearch target is narrow and fixed:

- Problem slice: `nonsmooth_nonconvex/socp`
- Architecture target: `LocalContextMLPv2`
- Active experimentation method: `penalty`
- Future compatibility target: `FSNet`

Only run autoresearch experiments with `penalty` for now. Keep all LocalContextMLPv2 changes compatible with later `FSNet` use, but do not spend the active loop on FSNet runs unless the user explicitly expands the scope.

Each experiment should be treated as a single-GPU run unless the user explicitly
changes that contract.

## Allowed Edits

Default edit scope is narrow. Prefer small, targeted changes that improve LocalContextMLPv2 behavior without changing unrelated experiment behavior.

Primary editable files:

- `models/neural_networks.py`
- `utils/trainer.py`
- `configs/default.yaml`

Conditionally editable files:

- `main.py` only if CLI or config plumbing is required for an experiment
- `eval.py` only if a minimal evaluation or metric-extraction improvement is required
- `README.md` only if user-facing CLI behavior changes

Do not edit:

- `datasets/`
- `results/`, `logs/`, `wandb/`, `figures/`
- historical tuning scripts unless the task explicitly becomes sweep orchestration
- unrelated notebooks or exploratory files
- `third-party/autoresearch/program.md`

If a change affects CLI or config behavior, keep `main.py`, `configs/default.yaml`, and docs aligned.

Do not install new packages or add dependencies as part of autoresearch. Use
the environment and dependencies already tracked by this repository.

## Baselines

The first two runs must establish unchanged penalty baselines for the target slice.

Penalty MLP baseline:

```bash
python main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MLP
```

Penalty LocalContextMLPv2 baseline:

```bash
python main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2
```

Use repo defaults unless the user explicitly changes the benchmark contract.

Do not run FSNet as a baseline in the initial loop. FSNet is a later compatibility target after penalty-only improvements have stabilized.

The first changed run after baselines should still be conservative and easy to
interpret.

## Evaluation Rules

Do not invent a new evaluator. Use the repository's saved `test_summary.yaml` outputs and existing `eval.py` flow.

Use batch size `256` as the canonical comparison point by default, matching the existing tuning conventions.

### Penalty ranking

For `penalty`, rank experiments by:

1. lower `merit_mean` 
2. lower `eq_violation_l1_mean`
3. lower `ineq_violation_l1_mean`
4. lower `opt_gap_mean` only if feasibility did not regress

This follows `FEATURE_AUGMENTATION.md`: objective gains alone are not enough if merit or feasibility gets worse.

A LocalContextMLPv2 change is only interesting if it improves against both penalty baselines:

- the plain `MLP` penalty baseline
- the existing `LocalContextMLPv2` penalty baseline

### Future FSNet check

Do not use FSNet for the active search loop.

FSNet compatibility requirement:

- avoid LocalContextMLPv2 changes that assume penalty-only loss semantics or break the standard training and evaluation interface
- keep model input as `x` and output as normalized `y`
- keep checkpoint and config behavior compatible with the current FSNet pipeline where practical

If the user later promotes an idea to FSNet validation, rank FSNet runs by:

1. lower `opt_gap_mean`
2. lower `eq_violation_l1_mean + ineq_violation_l1_mean`
3. lower `merit_mean`
4. lower `avg_inference_time`

## Research Heuristics

The optimization target is the repository's existing evaluation outputs. Within
that fixed benchmark contract:

- prefer lower merit and feasibility violations over objective-only wins
- treat VRAM as a soft constraint; moderate increases are acceptable only when
  the metric gain is meaningful
- prefer simpler changes when results are otherwise comparable

Simplicity is a real selection criterion. A tiny gain is not worth carrying if
it requires brittle or ugly complexity. Equal performance with cleaner code is a
win and should be kept.

## Logging Results

Log each completed experiment to `autoresearch_results.tsv`. Use tab-separated values, not commas.

Header:

```tsv
method	network	commit	batch_size	opt_gap_mean	eq_violation_l1_mean	ineq_violation_l1_mean	merit_mean	avg_inference_time	status	description	save_dir	eval_summary	run_log
```

Fields:

1. `method`: normally `penalty` for the active loop; reserve `FSNet` rows for later compatibility checks
2. `network`: `MLP` or `LocalContextMLPv2`
3. `commit`: short git hash
4. `batch_size`: normally `256`
5. `opt_gap_mean`
6. `eq_violation_l1_mean`
7. `ineq_violation_l1_mean`
8. `merit_mean`
9. `avg_inference_time`
10. `status`: `keep`, `discard`, or `crash`
11. `description`: short summary of the idea tested
12. `save_dir`: training run directory
13. `eval_summary`: path to `test_summary.yaml`
14. `run_log`: path to the captured training log

Use `0` or a clear placeholder for missing metrics on crashes.

## Experiment Loop

The experiment loop is autonomous once started.

LOOP FOREVER:

1. Start from the current best kept `penalty` + `LocalContextMLPv2` commit.
2. Inspect the current git branch and commit before changing code.
3. Make one targeted hypothesis change.
4. Commit the change.
5. Run the target penalty experiment and capture logs to a file.
6. Read the resulting `test_summary.yaml` and extract the batch-size-256 metrics.
7. Append a row to `autoresearch_results.tsv`.
8. Keep the commit only if it improves the current best penalty `LocalContextMLPv2` result.
9. Reject any change that still loses to the penalty `MLP` baseline.
10. If it does not improve the penalty leaderboard, revert to the prior best commit and continue.

After a promising penalty improvement is found, preserve FSNet compatibility by reviewing whether the change still fits the existing FSNet training and evaluation interface. Do not spend the active loop on FSNet runs unless the user explicitly asks for that next stage.

Do not stop to ask whether to continue once the loop is underway. Keep
iterating until interrupted by the user.

If ideas stop working, continue searching for the next reasonable hypothesis by
re-reading the scoped files, revisiting near-misses, and trying combinations or
larger but still targeted architecture changes.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Commands

Run training with full log capture. Avoid flooding context with training output.

Penalty MLP baseline:

```bash
python main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network MLP \
  > run.log 2>&1
```

Penalty LocalContextMLPv2:

```bash
python main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2 \
  > run.log 2>&1
```

Future FSNet compatibility check, only after penalty improvements stabilize:

```bash
python main.py \
  --method FSNet \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2 \
  > run.log 2>&1
```

If training completed but the metrics need to be regenerated or verified:

```bash
python eval.py --run_dir path/to/run_dir
```

After each run, read `test_summary.yaml` from the run directory and extract the `256` batch-size section. If the summary is missing or malformed, use `eval.py --run_dir` as a recovery path before classifying the run.

Keep training output redirected to log files. Do not stream long-running logs
into the interaction context.

## Runtime Rules

Treat the repository's normal training runtime as fixed experimental budget
territory unless the user changes it. If a run materially exceeds the expected
runtime envelope or appears stuck, stop it, classify it as failure, and move on.

If a completed run does not produce the expected summary artifacts, inspect the
log first. Use the evaluator as a recovery path only when the training run seems
to have finished successfully.

## First Knobs To Tune

Per `FEATURE_AUGMENTATION.md`, start with LocalContextMLPv2-specific knobs before broader model tuning:

1. `local_delta_scale`
   - baseline `0.2`
   - first conservative test `0.1`
2. `local_coarse_loss_weight`
   - baseline `0.5`
   - first stronger-feasibility test `1.0`

Only after testing those should you broaden into:

- hidden dimension
- dropout
- other training hyperparameters

Use `penalty` as the only active iteration loop for now. Once a penalty improvement is credible relative to both the penalty MLP baseline and the penalty LocalContextMLPv2 baseline, move to a later FSNet compatibility check.

## Later Hypotheses

If the first knob sweep is exhausted, continue with ideas suggested in `FEATURE_AUGMENTATION.md`:

- add gradient-based local features
- add a small gating mechanism over local feature groups
- try residual or multi-head trunk variants
- shift more effort from explicit feature augmentation toward loss shaping for `penalty`

Prefer simpler changes when gains are similar.

## Crashes And Failures

If a run crashes:

1. Check whether it is a simple implementation mistake.
2. If the fix is obvious and local, fix it and retry once or twice.
3. If the idea is fundamentally bad or unstable, log it as `crash` or `discard` and move on.

If a run completes but fails to beat the current best penalty result, discard it and revert to the prior best commit.

Do not claim broad wins from a single seed unless the user later asks for multi-seed confirmation.

If a crash is clearly due to a trivial implementation mistake, fix it and retry
once or twice. If the idea itself is unstable, log it and move on rather than
spending many retries on it.

## Guardrails

- Prefer targeted changes over broad refactors.
- Preserve compatibility with existing checkpoints where practical.
- Do not rewrite historical scripts just to clean up the tree.
- Do not modify generated artifacts.
- Keep the benchmark contract fixed unless the user explicitly changes it.
- Use `AGENTS.md` and `FEATURE_AUGMENTATION.md` whenever a tradeoff is unclear.
- Do not commit the experiment TSV; leave it as an untracked working artifact.
