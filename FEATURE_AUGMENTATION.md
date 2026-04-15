# FEATURE_AUGMENTATION

## Current Status

This repo now has four feature-augmentation variants:

- `ContextMLPv1`: fixed sampled context, flattened directly into the predictor input
- `ContextMLPv2`: fixed sampled context, encoded per sampled point and mean-pooled
- `LocalContextMLPv1`: local structure features computed around the model's coarse prediction
- `LocalContextMLPv2`: local residual refinement with a coarse-stage auxiliary penalty loss

All of them keep the outer training and evaluation interface unchanged:

- model input is still `x`
- model output is still normalized `y`
- losses, post-processing, and evaluator logic still fit the existing pipeline

## What We Learned

The central hypothesis still makes sense:

- raw `x` alone may not separate optimization regimes or winner regions well enough

But the early variants showed that adding structure is not enough by itself. The structure also has to be aligned with the final training objective.

Observed behavior:

- `ContextMLPv1` and `ContextMLPv2` did not help enough for `penalty`
- `LocalContextMLPv1` improved raw objective and opt gap, but hurt feasibility too much

That last result is important. It suggests local structure is useful, but the model needs stronger pressure to keep feasibility under control.

## Current Recommended Direction: LocalContextMLPv2

`LocalContextMLPv2` is the current best candidate.

### Stage 1: coarse solution

Predict a coarse normalized solution:

- `y0 = coarse_mlp(x)`

### Stage 2: local structure at the coarse point

Compute local features at `y0`:

- `obj_fn(y0)`
- full equality residual vector `eq_resid(x, y0)`
- full inequality residual vector `ineq_resid(x, y0)`

Normalize these local features with `LayerNorm`.

### Stage 3: residual correction

Predict a bounded residual correction:

- `delta_y = delta_head([x, y0, local_features])`
- `y_hat = clamp(y0 + alpha * delta_y, 0, 1)`

where `alpha = local_delta_scale`.

### Extra training signal

`LocalContextMLPv2` also adds an auxiliary penalty loss on the coarse prediction `y0` during training.

This is intended to stop the coarse stage from chasing objective improvements while becoming too infeasible.

## Why V2 Is Better Than LocalContextMLPv1

`LocalContextMLPv1` replaced the coarse prediction completely, which made it easy for the refinement stage to move to lower-objective but much less feasible regions.

`LocalContextMLPv2` is more conservative:

- it keeps the coarse solution as the anchor
- it only predicts a bounded correction
- it penalizes the coarse solution directly during training

This should make it easier to preserve feasibility while still exploiting useful local structure.

## Model Summary

### ContextMLPv1

- fixed sampled `y_ref`
- full residual vectors at sampled points
- flatten all features
- concatenate with `x`

### ContextMLPv2

- fixed sampled `y_ref`
- full residual vectors at sampled points
- encode each sampled point
- mean-pool encoded context
- concatenate with `x`

### LocalContextMLPv1

- coarse prediction `y0`
- compute objective and residual features at `y0`
- predict a full replacement solution

### LocalContextMLPv2

- coarse prediction `y0`
- compute objective and residual features at `y0`
- predict a bounded residual correction
- auxiliary coarse-stage penalty loss during training

## Recommended Benchmark Order

Prefer testing on `nonsmooth_nonconvex/socp`.

Because `penalty` is fast, use it as the primary iteration loop:

1. `MLP`
2. `LocalContextMLPv2`
3. `LocalContextMLPv1` only as an ablation
4. `ContextMLPv2` only if needed for comparison
5. `ContextMLPv1` only as an older baseline

Then validate the best local model on `FSNet`.

## Metrics To Compare

Track at minimum:

- `opt_gap_mean`
- `merit_mean`
- `eq_violation_l1_mean`
- `ineq_violation_l1_mean`
- training time per epoch
- evaluation time

For `penalty`, prioritize:

- lower `merit_mean`
- lower `eq_violation_l1_mean`
- lower `ineq_violation_l1_mean`
- lower `opt_gap_mean` if feasibility does not regress

This is important: better objective alone is not enough if merit gets worse.

## First Knobs To Tune For LocalContextMLPv2

1. `local_delta_scale`
   - smaller values keep the correction conservative
   - start with `0.2`, then try `0.1`
2. `local_coarse_loss_weight`
   - larger values enforce better coarse feasibility
   - start with `0.5`, then try `1.0`
3. hidden dimension and dropout only after the two local-specific knobs are tested

## Next Iterations If LocalContextMLPv2 Helps

1. Add gradient-based local features:
   - `eq_grad(x, y0)`
   - `ineq_grad(x, y0)`
2. Add a small gating mechanism over local feature groups.
3. Reuse the local structure branch for ensemble routing.
4. Let the correction scale depend on confidence or feasibility.

## Next Iterations If LocalContextMLPv2 Still Does Not Help

1. Add gradients, not just residuals.
2. Predict local merit / feasibility auxiliaries jointly.
3. Use a residual or multi-head trunk without explicit handcrafted structure.
4. Shift more effort from feature augmentation to loss shaping for `penalty`.

## Current Usage

Recommended fast test command:

```bash
python3 main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2
```

More conservative variant:

```bash
python3 main.py \
  --method penalty \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2 \
  --local_delta_scale 0.1 \
  --local_coarse_loss_weight 1.0
```

FSNet comparison command:

```bash
python3 main.py \
  --method FSNet \
  --prob_type nonsmooth_nonconvex \
  --prob_name socp \
  --network LocalContextMLPv2
```

For fair comparison, keep all non-context hyperparameters fixed relative to the vanilla baseline.

## Experiment Report: 2026-04-10

All results below are from:

- problem: `nonsmooth_nonconvex/socp`
- size: `SOCPProblem-100-50-50-10000`
- primary focus: `penalty`
- secondary checks: `FSNet`

The metrics cited below use the `512` batch-size section from each `test_summary.yaml`.

### Baseline References

Vanilla `penalty`, 1000 epochs:

- run: `20260410-120653_penalty_seed2025_e1000_lr1e-04_n7000`
- `objective = 19.5725`
- `opt_gap_mean = 12.6813`
- `eq_violation_l1_mean = 0.64167`
- `ineq_violation_l1_mean = 0.00657`
- `merit_mean = 648255.31`

Vanilla `penalty`, 2000 epochs:

- run: `20260410-123710_penalty_seed2025_e2000_lr1e-04_n7000`
- `objective = 19.3956`
- `opt_gap_mean = 11.6621`
- `eq_violation_l1_mean = 0.58873`
- `ineq_violation_l1_mean = 0.00618`
- `merit_mean = 594930.15`

Vanilla `FSNet`, 300 epochs:

- run: `20260410-121309_FSNet_seed2025_e300_lr1e-04_n7000`
- `objective = 16.4747`
- `opt_gap_mean = -5.1522`
- `eq_violation_l1_mean = 5.594e-05`
- `ineq_violation_l1_mean = 4.2e-07`
- `merit_mean = 72.84`

These runs are the main reference points for the experiments below.

### ContextMLPv1

Penalty with fixed sampled context and flattened full residual vectors did not beat vanilla penalty.

Representative runs:

- `20260410-115637_penalty_seed2025_e1000_lr1e-04_n7000_ctx16`
- `20260410-122204_penalty_seed2025_e1000_lr1e-04_n7000_ctx16`
- `20260410-122047_penalty_seed2025_e2000_lr1e-04_n7000_ctx16`

Observed pattern:

- objective became much worse than vanilla penalty
- merit also became worse
- increasing epochs to 2000 did not rescue the method

Example:

- `20260410-122204...ctx16`: `opt_gap_mean = 43.9522`, `merit_mean = 662108.46`

Conclusion:

- fixed sampled global context + raw flattening was not a good architecture for `penalty`

### ContextMLPv2

Encoding sampled points before pooling was cleaner than v1, but still did not beat vanilla penalty.

Representative runs:

- `20260410-124555_penalty_seed2025_e1000_lr1e-04_n7000_ctxv2k4e128`
- `20260410-124613_penalty_seed2025_e1000_lr1e-04_n7000_ctxv2k16e128`

Observed pattern:

- better than `ContextMLPv1` in some cases, but still worse than vanilla penalty
- changing from `k=4` to `k=16` barely changed the result

Examples:

- `...ctxv2k4e128`: `opt_gap_mean = 18.4792`, `merit_mean = 728632.85`
- `...ctxv2k16e128`: `opt_gap_mean = 18.4852`, `merit_mean = 728556.25`

Conclusion:

- fixed sampled context was still too global and not aligned enough with the local structure that matters for `penalty`

### LocalContextMLPv1

This variant used a coarse prediction and local structure at that prediction, then replaced the output with a refined prediction.

Representative runs:

- `20260410-130349_penalty_seed2025_e1000_lr1e-04_n7000_localctxv1`
- `20260410-135530_penalty_seed2025_e1000_lr1e-04_n7000_localctxv1`

Observed pattern:

- objective and opt gap could improve dramatically
- but feasibility often degraded too much
- the result was poor merit despite seemingly strong objective numbers

Examples:

- `20260410-130349...localctxv1`: `objective = 17.5023`, `opt_gap_mean = 0.7781`
  but `eq_violation_l1_mean = 0.85027`, `ineq_violation_l1_mean = 0.01952`, `merit_mean = 869804.30`

Conclusion:

- local structure around the current prediction is useful
- but replacing the coarse prediction outright makes it too easy to trade away feasibility for objective

### LocalContextMLPv2

This variant kept the coarse prediction as an anchor, predicted a bounded residual correction, and added a coarse-stage auxiliary penalty loss.

Representative runs with default penalty weights:

- `20260410-131915_penalty_seed2025_e1000_lr1e-04_n7000_localctxv2d0.2`

Observed pattern:

- the added coarse-stage regularization changed behavior substantially
- but the first default-weight result was still clearly worse than vanilla penalty

Example:

- `...localctxv2d0.2`: `objective = 23.6022`, `opt_gap_mean = 35.9563`, `merit_mean = 783662.80`

Training behavior from the corresponding log:

- very large infeasibility at the beginning
- rapid early reduction in gross violations
- continued objective improvement afterward
- but still the wrong final tradeoff under the default penalty weights

Conclusion:

- the architecture trained as intended
- but under the original penalty weights, it still did not produce the best merit

### Effect Of Stronger Constraint Weights

Later on 2026-04-10, experiments increased `eq_pen_weight` and `ineq_pen_weight` to:

- `eq_pen_weight = 100`
- `ineq_pen_weight = 50`

for both vanilla `penalty` and `LocalContextMLPv2`.

Representative paired runs:

- vanilla: `20260410-135401_penalty_seed2025_e1000_lr1e-04_n7000`
- local: `20260410-135936_penalty_seed2025_e1000_lr1e-04_n7000_localctxv2d0.2`

Observed pattern:

- stronger feasibility weights helped both methods relative to the original low-penalty setting
- the local model especially benefited from higher constraint weights
- however, the local model still did not produce a decisive best-overall result

Pair summary:

- vanilla strong-penalty: `objective = 29.1594`, `opt_gap_mean = 67.9841`, `merit_mean = 225155.91`
- local strong-penalty: `objective = 22.2482`, `opt_gap_mean = 28.1313`, `merit_mean = 158002.73`

This suggests:

- local structure can help, but only when feasibility pressure is increased substantially
- the architecture changes the loss tradeoff enough that old default penalty weights are no longer appropriate
- comparisons should be done best-vs-best after tuning penalty weights, not only under the old defaults

Another pair later in the day:

- vanilla strong-penalty: `20260410-140646...` gave `merit_mean = 673181.65`
- local strong-penalty: `20260410-140639...` gave `merit_mean = 169334.61`

That pair appears qualitatively similar but is much worse in absolute objective/opt-gap terms, so the exact settings or initialization path likely differed in a way that made those runs less competitive overall.

### FSNet Checks

The original fixed-context idea was not uniformly bad. On FSNet, `ContextMLPv1` helped a bit relative to vanilla:

- vanilla FSNet: `objective = 16.4747`, `opt_gap_mean = -5.1522`, `merit_mean = 72.84`
- `ContextMLPv1`: `objective = 16.3003`, `opt_gap_mean = -6.1552`, `merit_mean = 82.19`

Interpretation:

- structure augmentation can help initialization-sensitive methods like FSNet
- but the same approach does not automatically transfer to fast `penalty`

`LocalContextMLPv1` did not help FSNet:

- `20260410-130600_FSNet_seed2025_e300_lr1e-04_n7000_localctxv1`
- `objective = 21.6022`, `opt_gap_mean = 24.4474`, `merit_mean = 47.03`

So the local-replacement idea that was problematic for `penalty` also hurt FSNet.

### Overall Conclusions From 2026-04-10

1. The motivating idea is still plausible:
   raw `x` alone likely does not capture all the structure needed for winner separation or routing.

2. Fixed sampled global context was not effective for `penalty`.

3. Local context near the current prediction is more promising than random sampled context.

4. However, local structure must be tied tightly to feasibility control.

5. Increasing constraint weights helped the local residual model more than it helped the original default setup.

6. The fairest next comparison is:
   tuned vanilla `penalty` vs tuned `LocalContextMLPv2`, under matched penalty-weight tuning budget.

7. Architecture conclusions should be based on merit and feasibility, not objective alone.

### Recommended Next Step After 2026-04-10

Do not add another new context architecture immediately.

Instead:

1. Tune `eq_pen_weight` and `ineq_pen_weight` for both:
   - vanilla `penalty`
   - `LocalContextMLPv2`
2. Keep the tuning budget matched.
3. Compare best-vs-best on:
   - `merit_mean`
   - `eq_violation_l1_mean`
   - `ineq_violation_l1_mean`
   - `opt_gap_mean`
4. Only if `LocalContextMLPv2` wins under tuned settings should more effort go into richer local features such as gradients or routing heads.
