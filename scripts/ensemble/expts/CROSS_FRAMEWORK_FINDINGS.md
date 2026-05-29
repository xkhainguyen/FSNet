# Cross-framework verification: when does multi-start / "free ensembling" actually help in L2O?

**Question.** On amortized constrained optimization, pre-repair multi-start ("free
ensembling" — perturb the NN output K times and run repair on each, then pick the
best by merit) was originally reported as a near-zero-cost ensemble alternative,
giving ~60% merit reduction at K=100 on FSNet over single-shot inference. Is this
gain *real* (a genuine functional-diversity benefit) or a *repair-operator
under-convergence artifact* (the multi-start papers over a badly-tuned solver)?

**Verdict (5 frameworks tested).** The gain has **three distinct sources**, and
they must be disentangled before claiming an ensembling result:

1. **Under-converged iterative repair on a convex objective** (FSNet/L-BFGS,
   DC3/grad-SGD, Πnet/ADMM) → the gain is an **artifact**. It collapses to
   numerical noise once the repair runs to convergence; fixing the repair
   budget is 10–20× cheaper and reproduces the same numbers. *Most published
   "free ensemble" results on convex benchmarks live here.*
2. **Direction-dependent repair** (Bisection-Projection / radial bisection) →
   gain is **real but small** (~0.4% objective); survives full convergence
   because the repaired point depends on approach direction.
3. **Nonconvex objective and/or non-Euclidean closed-form correction**
   (HardNet) → gain is **real and large** (2× objective improvement) and
   cannot be a convergence artifact (no iterative solver). **But the mechanism
   is not yet isolated** — it could be the nonconvex `pᵀsin(y)` objective, the
   non-Euclidean pinv/ReLU correction (HardNet-Aff is NOT the true Euclidean
   projection), or undertraining. See §5 and the **Recipe** at the end for the
   controls that would disentangle these; they have not been run.

A convex objective with a converged Euclidean projection — the most common
benchmark setup — gets **exactly zero** genuine benefit from perturbation.

Frameworks: FSNet (this repo, L-BFGS), DC3 (this repo, grad-SGD), Πnet (Terpin
et al. 2025, ADMM), Bisection-Projection (Liang et al. 2025, radial bisection),
HardNet (Min & Azizan, NeurIPS 2024, closed-form affine projection).

---

## 1. FSNet (this repo) — L-BFGS repair

The FSNet repair (`utils/lbfgs.py:nondiff_lbfgs_solve`) minimises
`||eq_resid||² + ||ineq_resid||²` via L-BFGS. The original convergence check was
"val OR grad", scalar-reduced across the batch — so the loop could early-exit
as soon as *any* per-sample val or grad satisfied tolerance, leaving other
samples under-converged. We added a `per_sample_lbfgs` flag (default False)
that enables (a) per-sample line search and (b) val-AND-grad per-sample
convergence.

| Config (nonsmooth SOCP, seed 0)             | per_sample=0 | per_sample=1 |
|---------------------------------------------|--------------|--------------|
| Single, K=1 baseline                        | Merit 84.7   | Merit 16.5   |
| Single, K=100 ε=0.10                        | Merit 30.5   | Merit 16.5   |
| Perturbation gain (relative)                | **−64%**     | **0%**       |
| VE M=5 ensemble, best_merit, K=1            | Merit 29.7   | Merit 16.4   |
| VE M=5 ensemble, K=100 ε=0.10               | Merit 14.6   | Merit 16.4   |

Same checkpoint, different convergence-check semantics. With the corrected
per-sample convergence, perturbation gives zero benefit. The previously-reported
gain was the L-BFGS scalar early-exit leaving samples un-converged.

## 2. DC3 (this repo) — `grad_steps` repair

The DC3 repair (`utils/optimization_utils.py:grad_steps`) runs a fixed
`max_corr_steps=30` iterations of momentum-SGD with fixed `corr_lr=1e-6` on
the inequality penalty. There is **no convergence check at all** — the loop
always runs exactly 30 iterations regardless of feasibility.

**Sweep A — budget:** No perturbation, sweep `max_corr_steps` from 30 to 3000.

| max_corr_steps | IneqL1 | Merit |
|----------------|--------|-------|
| 30 (default)   | 5.7e-4 | 590.6 |
| 100            | 5.3e-4 | 556.4 |
| 300            | 4.5e-4 | 470.6 |
| 1000           | 2.5e-4 | 270.9 |
| 3000           | 5.2e-5 | 76.2  |

Just letting `grad_steps` run longer gives a 7.7× merit reduction with no
perturbation, no architectural change, no retraining.

**Sweep B — learning rate:** No perturbation, default budget=30, vary `corr_lr`.

| corr_lr | IneqL1 | Merit |
|---------|--------|-------|
| 1e-6 (default) | 5.7e-4 | 590.6 |
| 1e-5    | 4.5e-4 | 472.7 |
| 1e-4    | 4.1e-5 | 65.4  |
| 1e-3    | **0.0** | **24.4** ← exact feasibility |

Just bumping `corr_lr` from 1e-6 to 1e-3 gives a 24× merit reduction and
exact feasibility. Compare: the reported K=100 ε=0.01 perturbation result on
DC3+SOCP was Merit 24.25 — **essentially the same number reached by fixing
the lr alone**. The perturbation was papering over a 1000× under-sized step.

## 3. Πnet (vendored at `third-party/pinet`) — ADMM projection repair

The Πnet repair is Douglas–Rachford ADMM projecting onto
`{y : Ay = b, lb ≤ Cy ≤ ub}`. Defaults are `n_iter_test=100` (max iter) with
`tol=1e-3` (early-exit on max constraint violation < tol).

Trained one HCNN on the DC3 nonconvex QP100 benchmark (50 epochs,
features=[200,200], seed 42) using their `benchmark_config_default.yaml`.
At test time, computed the trunk output (pre-projection), replicated K times,
added ε·N(0,I) noise, ran ADMM projection at varying `n_iter_test`, picked
best by Merit = obj + 1e6·(EqMax + IneqMax).

| n_iter_test | K=1 baseline | best K-perturb | reduction |
|-------------|--------------|----------------|-----------|
| 10          | 7925.6       | -9.32 (K=100 ε=0.10)  | ~100%     |
| 50          | 1629.2       | 369.1 (K=100 ε=0.10)  | 77%       |
| **100 (default)** | **116.5** | **44.6 (K=100 ε=0.10)** | **62%** |
| 500 (5×)    | -11.5563     | -11.5587 (K=20 ε=0.01) | 0.02%    |
| 1000        | -11.5566     | -11.5590              | 0.02%     |
| 3000        | -11.5566     | -11.5590              | 0.02%     |

Same shape. At default budget the perturbation gain is huge (62%) and is almost
entirely closing IneqMax (from 1.28e-4 → 5.6e-5); at 5× budget IneqMax drops
to 1e-10 (the projection has effectively converged), and the perturbation gain
shrinks to within numerical noise.

**Head-to-head: just-bump-budget vs perturb-at-default (Πnet, DC3 nonconvex QP100, seed 42):**

| Strategy                                                | Merit       |
|---------------------------------------------------------|-------------|
| Default budget (n_iter_test=100), K=1 baseline          | 116.52      |
| Default budget, K=100 ε=0.10 perturb (best in grid)     | 44.64       |
| **n_iter_test=500, K=1 baseline (no perturb)**          | **-11.56**  |
| n_iter_test=1000, K=1 baseline                          | -11.56      |

Just increasing the ADMM budget 5× (no perturbation, no architectural change)
beats the best K=100 perturbation result by >50 absolute merit units. The same
pattern as DC3: bumping the repair budget alone dominates multi-start.

---

## 4. Bisection-Projection (vendored at `third-party/Bisection-Projection`) — radial bisection

BP (Liang et al. 2025) uses a structurally DIFFERENT repair operator:
**radial bisection from a predicted feasible interior point to an infeasible
candidate**. Unlike Euclidean projection (L-BFGS, ADMM) or gradient correction
(DC3), bisection follows a 1D line from IP to candidate and finds the boundary
intersection on that line. Bisection converges geometrically (rate 0.9 → 30
iters gives 1e-30 alpha gap), so there's no under-convergence failure mode by
construction.

Pretrained QP100-50-50 model from BP repo. IP per sample computed via cvxpy
feasibility solve (not Y_star, to avoid trivial collapse). Bisection in PARTIAL
space (so equality `Ay=x` is preserved by `complete_partial` at every step).
Metrics on the 70/1024 infeasible samples (where bisection actually runs):

| K | ε    | Obj_inf  | IneqL1_inf | Merit_inf |
|---|------|----------|------------|-----------|
| 1 | 0.00 | -0.2834  | 7.0e-6     | **6.74**  |
| 20  | 0.01 | -0.2838 | 0.0 | -0.2838 |
| 100 | 0.01 | -0.2839 | 0.0 | -0.2839 |
| 100 | 0.05 | **-0.2844** | **0.0** | **-0.2844** |
| 100 | 0.10 | -0.2841 | 0.0 | -0.2841 |
| 100 | 1.00 | -0.2607 | 0.0 | -0.2607 (eps too large) |

**Bisection budget sweep (K=100, ε=0.10):**

| n_steps | Obj_inf | IneqL1_inf | Merit_inf |
|---------|---------|------------|-----------|
| 3       | -0.2840 | 0.0        | -0.2840   |
| 30      | -0.2841 | 0.0        | -0.2841   |
| 300     | -0.2841 | 3e-6       | +2.71 (overshoots) |

Two distinctive properties on BP that the other three frameworks lack:

1. **Perturbation gives a real merit gain (6.74 → -0.28)**, and it's NOT a
   budget artifact. n_steps=3 already converges. Bumping n_steps further
   doesn't tighten the result (and actually hurts at n_steps≥100 because
   alpha approaches 1 and the perturbed candidate's tiny eq residual surfaces).
2. **The "gain" is mostly feasibility-tightening (IneqL1 7e-6 → 0), not
   objective improvement.** K=1 gives Obj=-0.2834, best perturb gives
   Obj=-0.2844 — a 0.0010 improvement. The dramatic merit drop (6.74 → -0.28)
   is because IneqL1=7e-6 hits with weight 1e6.

So BP is the one framework where multi-start exploits the repair *geometry*
(radial vs Euclidean). Different perturbed candidates define different rays
from IP, which hit different boundary points. The gain is **moderate**
(objective improvement ~0.4%) but **real** (survives full bisection
convergence).

## 5. HardNet (vendored at `third-party/hardnet`) — closed-form affine projection

HardNet-Aff (Min & Azizan, NeurIPS 2024) appends a **closed-form, one-shot**
correction: `proj(f,x) = f + pinv(A)·(ReLU(bl−Af) − ReLU(Af−bu))`. There is no
iterative solver — no max_iter, no tolerance, no convergence loop. So the
"under-convergence artifact" explanation is structurally impossible here.

> **Caveat on the operator.** This is NOT the Euclidean projection onto
> `{bl ≤ Af ≤ bu}` (that would be a QP — which is what the sibling file
> `hardnet_cvx.py` does via a `CvxpyLayer`). HardNet-Aff is a single pinv/ReLU
> step that is feasible-by-construction under HardNet-Aff's assumptions, but is
> **direction-dependent**: different `f` map to different feasible points (same
> property as BP's radial bisection). So this framework is NOT a clean
> "Euclidean + converged" control.

HardNet's `opt` benchmark is the DC3 **nonconvex-objective** QP100:
`min ½yᵀQy + pᵀsin(y) s.t. Ay≤b, Cy=x` — the only nonconvex *objective* tested
(FSNet/DC3/Πnet/BP all had convex objectives).

Trained 200 epochs (50 warmup, no projection), seed 42, on RTX Pro 6000. The
correction makes every candidate feasible (EqL1≈3e-13, IneqL1≈2e-15) regardless
of K, so any merit change is **pure objective improvement**:

| K | ε    | Obj      | IneqL1   | Merit    |
|---|------|----------|----------|----------|
| 1 | 0.00 | -2.7766  | 1.7e-15  | -2.7766  |
| 5   | 0.10 | -3.9924 | 1.8e-15 | -3.9924 |
| 20  | 0.10 | -4.9953 | 1.7e-15 | -4.9953 |
| 100 | 0.05 | -4.8784 | 1.6e-15 | -4.8784 |
| **100** | **0.10** | **-5.5970** | **1.7e-15** | **-5.5970** |
| 100 | 0.30 | -4.9517 | 2.2e-15 | -4.9517 |
| 100 | 1.00 | +8.2872 | 1.4e-14 | +8.2872 (ε too large) |

K=1 → K=100 (ε=0.10) **doubles the objective improvement** (-2.78 → -5.60), with
feasibility unchanged at machine precision. Clear sweet-spot ε≈0.10. This is a
**real gain that cannot be a repair-convergence artifact** (no iterative solver).

**⚠ Mechanism NOT yet isolated.** Three explanations remain confounded; we
asserted (a) prematurely and have NOT run the control that distinguishes them:

- **(a) nonconvex objective** (`pᵀsin(y)`) → multistart finds deeper basins.
- **(b) direction-dependent correction** (the pinv/ReLU map is not Euclidean —
  see caveat) → multistart genuinely diversifies the feasible point, à la BP.
- **(c) undertraining** — at epoch 200 the valid objective was *still* dropping
  fast (−0.7 → −2.7 over the final 20 epochs). A suboptimal NN point means
  perturbation just explores around a not-yet-converged prediction.

The control that disentangles them (see **Recipe** below) was not run. Until it
is, the honest statement is: *HardNet shows a real, large perturbation gain on a
nonconvex-objective problem; the mechanism is one or more of {nonconvex obj,
non-Euclidean correction, undertraining} and has not been isolated.*

---

## Synthesis

| Framework | Repair operator | Objective | Source of perturbation gain |
|-----------|----------------|-----------|------------------------------|
| FSNet     | L-BFGS (Euclidean, iterative) | convex | **Artifact** — under-converged; fixed by `per_sample_lbfgs=1` |
| DC3       | momentum-SGD on ineq penalty (iterative) | convex | **Artifact** — under-tuned; `corr_lr=1e-3` matches perturb |
| Πnet      | Douglas–Rachford ADMM (Euclidean, iterative) | convex | **Artifact** — under-budgeted; `n_iter_test=500+` matches perturb |
| BP        | radial bisection (non-Euclidean, iterative) | convex | **Real (small)** — repair geometry; ≈0.4% obj, mostly feasibility-tightening |
| HardNet   | one-shot pinv/ReLU correction (closed-form, *non-Euclidean*) | **nonconvex** | **Real (large), mechanism unconfirmed** — 2× obj; one or more of {nonconvex obj, non-Euclidean correction, undertraining} |

The "free ensemble" gain has (at least) **three candidate sources**, which must
be disentangled before claiming an ensembling result:

1. **Under-converged repair** (FSNet, DC3, Πnet — convex obj + iterative
   Euclidean projection). The "gain" is an artifact. Fixing the repair budget
   is 10–20× cheaper and gives the same numbers. *Most published "free
   ensemble" results on convex problems are here.* — **confirmed** via
   budget-sweep controls collapsing the gain.

2. **Direction-dependent repair** (BP — radial bisection; HardNet's pinv/ReLU
   correction is also in this class). Genuine but small for BP; survives full
   convergence because the repaired point depends on approach direction.

3. **Nonconvex objective** (HardNet's `pᵀsin(y)`). Classical multi-start global
   optimization. *Plausible but not isolated from (2)/(undertraining) yet.*

Practical rule for L2O ensembling: **perturbation/ensembling helps iff (a) the
repair is under-converged [fix the repair instead], (b) the repair is
direction-dependent, or (c) the objective is nonconvex.** (a) is firmly
established; (b) is established for BP; (c) is supported by HardNet but the
HardNet result also admits (b) and undertraining as explanations — see Recipe.
A convex objective with a *converged Euclidean* projection — the most common
benchmark setup — gets exactly zero genuine benefit.

---

## Recipe: isolate the HardNet mechanism (not yet run)

Goal: determine whether the HardNet K=100 gain (−2.78 → −5.60) comes from the
nonconvex objective (c), the non-Euclidean correction (b), or undertraining.

Three controls, each flipping ONE factor, same seed/arch/epochs, same K×ε grid:

1. **Convex-objective control (tests c).** Re-run HardNet-Aff with the objective
   swapped `½yᵀQy + pᵀsin(y)` → `½yᵀQy + pᵀy` (drop the sin; keep Q PSD so the
   objective is convex). Same pinv/ReLU correction.
   - Gain *vanishes* ⇒ the gain was the nonconvex objective (c).
   - Gain *persists* ⇒ it's the non-Euclidean correction (b), not (c).
   Implementation: add a `--convex_obj` flag to `hardnet_verify.py` that
   monkeypatches `data.evaluate`/`get_train_loss` to drop the `sin`.

2. **Euclidean-projection control (tests b).** Re-run with HardNet-**Cvx**
   (`hardnet_cvx.py`, the true QP projection via CvxpyLayer) instead of
   HardNet-Aff, keeping the nonconvex objective.
   - Gain *shrinks toward zero* ⇒ part of the Aff gain was the non-Euclidean
     correction (b).
   - Gain *unchanged* ⇒ (b) is not contributing; consistent with (c).
   (Slow — CvxpyLayer solves a QP per sample — so evaluate on a 128-sample
   subset.)

3. **Convergence control (tests undertraining).** Re-run HardNet-Aff for
   1000–2000 epochs (until valid objective plateaus), then repeat the grid.
   - Gain *shrinks* as training converges ⇒ much of it was exploring around a
     suboptimal NN point (undertraining).
   - Gain *stable* ⇒ undertraining is not the driver.

Decision table after running all three:

| (1) convex-obj gain | (2) Euclidean gain | (3) converged gain | Conclusion |
|---------------------|--------------------|--------------------|------------|
| ~0 | — | stable | Pure nonconvex-objective effect (c) — the clean positive result |
| persists | shrinks | stable | Non-Euclidean correction (b), like BP |
| any | any | shrinks→0 | Undertraining artifact — not a real ensembling win |

Only the top row would justify the headline "nonconvex objective ⇒ genuine
multi-start win." Run order: (3) is cheapest and most likely to undercut the
result, so run it first; then (1); then (2) on a subset.

## Reproducibility

- FSNet/DC3 evidence: scripts under `scripts/ensemble/expts/` (paths in commit
  history; specifically `verify_dc3_unaffected.sh`, `dc3_steps_sweep.sh`,
  `sanity_per_sample.sh`).
- Πnet evidence: `scripts/ensemble/expts/pinet/pinet_verify.{py,sh}`. The repo
  is vendored at `third-party/pinet` (commit pinned by clone; see `git -C
  third-party/pinet log -1`).
- BP evidence: `scripts/ensemble/expts/bp/bp_verify.{py,sh}`. Repo vendored at
  `third-party/Bisection-Projection`.
- HardNet evidence: `scripts/ensemble/expts/hardnet/hardnet_verify.{py,sh}`,
  log `logs/hardnet-verify-14694446.out`. Repo vendored at `third-party/hardnet`.
- Run logs: `logs/pinet-verify-14689578.out` (Πnet QP, seed 42),
  `logs/dc3-steps-sweep-14690284.{out,err}` (DC3 grad_steps budget/lr sweep),
  `logs/dc3-unaff-14687899.out` (DC3 unaffected by L-BFGS per_sample flag).
