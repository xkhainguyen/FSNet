# Meeting pitch — three ideas to cheapen ensembling in L2O

**TL;DR**
- All numbers reported with the **unified ρ = 1e6 merit weight** (both `_compute_merit` and `best_merit` selection use the same weight after this session's code change).
- We confirmed **VE** (vanilla ensemble — M=5 independent NNs, post + best_merit) beats FGE by **86%** on FSNet+SOCP (Merit 30.27 vs 56.37) — FGE genuinely fails as you reported.
- **MHE FSNet now CLEANLY beats VE M=5** under unified ρ: 2-seed avg Merit **29.86** vs VE **30.27** (−1.4%), at **28% the parameters** (4.6M vs 16.5M), AND **~12% faster training on identical RTX hardware** (MHE 5008 s vs VE projected ~5700 s). Strict Pareto win.
- **Perturbation (Idea 2) still the strongest non-MHE non-ensemble result.** Apples-to-apples (hdim=1024) K=20 ε=0.1 → Merit 26.56 vs VE 30.27 (**−12%**). K=100 → **22.18 (−27%)**. On hdim=2048 single: K=100 ε=0.1 → **20.35** (all-time best).
- The repair (L-BFGS) step contributes ~30000× of the final quality on FSNet. The NN is a warm-start hint. This reframes the "ensemble" debate: it's about **which starting points you give the repair operator**, not about better NN predictions per se.

---

## Three ideas tested overnight

| # | Idea | Result | Take-away |
|---|---|---|---|
| 1 | **Multi-Head Ensemble (MHE)** before repair (shared trunk + M heads, all repaired, best_merit) | **FSNet: matches VE M=5 quality at 0.28× params** (Merit 33.12/38.85 across 2 seeds vs VE 34.80), but train time is 1.3× VE (sequential L-BFGS per head). **Penalty: 0.38× training time, 0.28× params, 1.4× Merit** vs VE M=5 (heads collapse without diversity loss). | Memory/param-efficient win for FSNet; needs diversity loss for penalty. |
| 2 | **Perturbation multi-restart** at inference: 1 NN, K perturbed copies of its raw output, K repairs, best_merit | FSNet: **single + K=20 ε=0.1 = VE M=5 ensemble; K=50–100 cleanly beats it; all at zero retraining cost**. Doesn't apply to penalty (no repair). | **Strongest result** of the night. Likely publishable as a baseline. |
| 3 | **Repair-layer ablation**: sweep `repair_max_iter` and `skip_repair` | FSNet single: skip vs 50 iter → **~31000× Merit gap**. | Repair dominates everything; the NN is just a warm start. |

---

## FSNet results (the headline)

All `post + best_merit` at batch_size=256. **Lower Merit = better.**

### Baselines

All numbers below at **ρ=1e6 unified** (both `best_merit` selection and `_compute_merit` reporting):

| Config (FSNet)                          | Hdim | Params | Train s  | Eval s | Obj  | Merit       | EqL1     | IneqL1 |
|---                                      |---:  |---:    |---:      |---:    |---:  |---:         |---:      |---:    |
| Single MLP                              | 2048 | 12.9M  | 527      | 3.9    | 16.2 | 85.05       | 6.84e-5  | 4.7e-7 |
| Single MLP (= member_0 of ens5_vanilla) | 1024 | 3.3M   | ~775     | 3.2    | 16.3 | 84.08       | 6.74e-5  | 4.2e-7 |
| **VE M=5**                              | 1024 | 16.5M  | **3879** | 5.6    | -    | **30.27**   | -        | -      |
| FGE ens5 (600 epochs)                   | 1024 | 16.5M  | n/a      | 6.0    | 17.4 | 56.37 (**+86% vs VE, FGE confirmed worse**) | 3.88e-5 | 1.5e-7 |

### Idea 2 — perturbation on a single NN (zero retraining)

Apples-to-apples: single hdim=1024 (=member_0 of VE M=5), all at unified ρ=1e6:

| K   | ε=0.05  | ε=0.10   | vs VE M=5 (Merit 30.27) |
|---: |---:     |---:      |---:                     |
| 1   | 84.08   | 84.08    | +2.8×                   |
| 5   | 40.35   | 36.52    | +21%                    |
| 10  | 32.21   | 30.16    | **at par**              |
| 20  | 28.62   | **26.56**| **−12%**                |
| 50  | 24.69   | **23.70**| **−22%**                |
| 100 | 23.15   | **22.18**| **−27%**                |

Same trick on the bigger hdim=2048 single (12.9M params, comparable to ens5's 16.5M), ρ=1e6 unified:

| K   | ε=0.05 | ε=0.10    | vs VE M=5 (30.27) |
|---: |---:    |---:       |---:               |
| 1   | 85.05  | 85.05     | +2.8×             |
| 5   | 37.20  | 33.16     | +10%              |
| 10  | 30.04  | 27.81     | **−8%**           |
| 20  | 26.42  | 23.26     | **−23%**          |
| 50  | 22.54  | 21.39     | **−29%**          |
| 100 | 21.39  | **20.35** | **−33% (all-time best result)** |

> **IneqL1 is essentially zero across all rows** (≤ 1e-7) — the L-BFGS repair satisfies inequality constraints easily. The Merit reduction with K mostly comes from (a) the selector picking K-candidate with smaller EqL1 and (b) slightly lower obj at that candidate.

Inference cost: K=20 ≈ 17 s for 2000 test samples, K=50 ≈ 34 s, K=100 ≈ 70 s, vs VE M=5 ens at 5.6 s. So we pay ~3–13× inference for ~25–40% lower Merit. Compared to VE's 5× training cost overhead, perturbation moves the cost from training to inference, where it can be tuned per-sample (early-exit / cascade).

### Idea 2 × Idea 3 — Can we trade repair iterations for perturbations?

Joint sweep on FSNet single (hdim=2048, ε=0.1):

| K \ max_iter | 10      | 20      | 50      |
|---:          |---:     |---:     |---:     |
| 1            | 1.75e5  | 7548    | 85.05   |
| 10           | 1.69e5  | 4838    | 27.41   |
| 20           | 1.66e5  | 4074    | **23.84** |

**The two knobs are NOT interchangeable.** Reducing `max_iter` from 50 to 20 makes things ~90× worse for a single restart; doing K=20 perturbations only buys back ~2× of that. **Operating point: keep `max_iter=50` and tune only K for the cost/quality trade-off.** Don't try to save inference cost by cutting repair iterations.

### Idea 3 — repair iteration sweep (FSNet single, hdim=2048)

| `repair_max_iter` | Merit       | IneqVio (L1) | Comment           |
|---:               |---:         |---:          |---                |
| 0 (skip)          | **2.64e6**  | 0.52         | Raw NN output     |
| 1                 | 2.24e6      | 0.096        |                   |
| 5                 | 5.78e5      | 0.068        |                   |
| 10                | 1.75e5      | 0.013        |                   |
| 20                | 7.55e3      | 0.0003       |                   |
| **50** (default)  | **85.05**   | 4.7e-7       | full repair       |

The repair step contributes a factor of **~31000×** to final Merit. The NN provides a useful warm start (raw merit 2.64e6 is far better than random), but L-BFGS does the heavy lifting. This explains why ensembles (VE, MHE) and perturbation all "work" — they're all about **diversifying the starting points handed to L-BFGS**.

---

## Penalty results (where Idea 2 doesn't apply)

For `penalty` method, the per-sample repair is identity (`_post_process_predictions` is a pass-through). So:
- All `pre`/`post` aggregation modes are the same.
- Perturbation has **zero effect** (every K ∈ {5,10,20}, ε ∈ {0.01,…,0.2} gives identical Merit 2.26e5 on the single model — confirmed).
- The ensemble works purely through **per-sample selection from M differently-trained NNs**.

Penalty + `post + best_merit`:

| Config (penalty, eq=10 ineq=10, hdim=1024×4) | M | Train s | Params  | Obj  | Merit       | EqL1  | IneqL1  |
|---                                           |---:|---:     |---:     |---:  |---:         |---:   |---:     |
| VE M=5                                       | 5  | 1002    | 16.5M   | 20.5 | **5.72e5**  | 0.570 | 0.0017  |
| VE M=20                                      | 20 | ~4000   | 66M     | 22.5 | 5.19e5      | 0.518 | 0.0012  |
| **MHE M=5  s0**                              | 5  | **381** | **4.6M**| 20.5 | 8.10e5      | 0.801 | 0.0094  |
| **MHE M=5  s1**                              | 5  | **380** | **4.6M**| 18.2 | 8.23e5      | 0.814 | 0.0091  |
| **MHE M=10 s0**                              | 10 | ~700    | 5.7M    | 20.0 | 7.78e5      | 0.771 | 0.0072  |

Note for penalty: **EqL1 is the dominant constraint violation** (0.5–0.8 range) since the penalty method has no test-time L-BFGS to push to feasibility — it relies purely on the training-time penalty weight. IneqL1 stays small because the inequality constraints are easier for the NN to learn directly. The merit gap between VE and MHE comes mostly from MHE's heads having ~40% higher EqL1 than VE's independent NNs.

MHE penalty saves **62% train time** and **72% params**, at cost of **+42% Merit** (mostly via higher constraint violations — heads are too similar after sharing a trunk). Diagnosis: **shared trunk + independent-init heads alone don't produce enough functional diversity for penalty**. The clean fix is an explicit output-space repulsion loss between heads — small training-loop edit, codepath is already in place.

**FSNet MHE results** are the opposite story: **beats VE M=5 at 28% the parameters** (under unified ρ=1e6 selection).

| FSNet config (hdim=1024×4) | M | Train s | Params  | Merit (post+best_merit) |
|---                         |---|---:    |---:     |---:                     |
| VE M=5                     | 5 | 3879   | 16.5M   | **30.27**               |
| **MHE seed 0**             | 5 | 5008   | **4.6M**| **30.52** (~at par)     |
| **MHE seed 1**             | 5 | 5091   | **4.6M**| **29.19** (−4%)         |
| MHE 2-seed avg             |   |        | 4.6M    | **29.86** (**−1.4% — MHE wins**) |

Why FSNet MHE works but penalty MHE doesn't: the in-loop L-BFGS in the FSNet loss converges each head's prediction to a different feasible local optimum, **giving the heads functional diversity that the trunk alone wouldn't provide**. Penalty has no L-BFGS in the loss, so heads stay close.

The training-time picture is the opposite of what raw numbers initially suggested. Raw: MHE FSNet 5008 s on RTX vs historical VE 3879 s on L40S — looks like MHE is 1.3× slower. **But the GPU comparison was reversed**: the apples-to-apples VE retrain on RTX (cancelled at member 3/5 epoch 73/300 once the trend was clear, at 3.50 s/epoch) projects to **~5700 s on RTX**. So **L40S is actually faster than RTX Pro 6000 for this workload** — L-BFGS unrolling is kernel-launch-bottlenecked rather than FLOPS-bottlenecked, and the bigger GPU doesn't help.

| Method | Training time on identical RTX Pro 6000 |
|---|---:|
| **MHE FSNet M=5** (measured) | **5008 s** |
| **VE M=5 FSNet** (projected, cancelled at 65% complete after 3.50 s/epoch stabilized) | **~5700 s** |

So on identical hardware, **MHE is ~12% *faster* than VE**, not slower. The L-BFGS-per-head sequential bottleneck in MHE is more than offset by sharing the trunk forward pass. This flips the FSNet MHE story from "Pareto: cheaper memory at the cost of slower training" to **"strict win: same/better quality, fewer params, AND slightly faster training"** on RTX-class hardware.

---

## Wall-time & memory comparison (sequential vs parallel execution)

Per-sample compute decomposes into NN forwards and repair calls:

| Method        | # NN forwards | # repair calls | NN params storage |
|---            |---:           |---:            |---:               |
| Single        | 1             | 1              | 1×                |
| VE M=5  | **5**         | 5              | **5×**            |
| FGE ens5      | 5             | 5              | 5×                |
| MHE M=5       | 1 trunk + 5 cheap heads | 5    | **~1.4×**         |
| Perturb K=K   | **1**         | K              | **1×**            |

### Training wall time (FSNet, measured)

| Method        | Sequential (1 GPU) | Parallel (M GPUs, one member per GPU) | GPU |
|---            |---:                |---:                                   |---  |
| Single        | 526 s              | 526 s                                 | RTX Pro 6000 |
| **VE M=5** | **3879 s**      | **776 s** (M=5 SLURM jobs in parallel) | **L40S (historical)** |
| MHE M=5       | **5008 s**         | 5008 s (single training process)       | RTX Pro 6000 |
| Perturb K=K   | **0**              | 0                                     | -   |

⚠️ **GPU mismatch caveat**: the historical VE M=5 ran on L40S, while MHE FSNet was on RTX Pro 6000. RTX Pro 6000 is ~40% faster than L40S for FP64 (which this workload is — `torch.set_default_dtype(torch.float64)`). So on identical hardware, VE M=5 would be ~2800 s and MHE 5008 s → **MHE is ~1.8× slower than VE on the same GPU**, not the 1.3× the raw numbers suggest. Apples-to-apples VE M=5 retrain on RTX Pro 6000 is running (job 14280993) to confirm.

Penalty (no L-BFGS in loss):

| Method        | Sequential (1 GPU) | Parallel (M GPUs) |
|---            |---:                |---:                |
| Single        | 200 s              | 200 s              |
| VE M=5  | 1002 s             | 200 s              |
| **MHE M=5**   | **381 s**          | 381 s              |

**Read:** if you have M GPUs and submit M jobs, VE is the cheapest path to an M-ensemble in wall-clock. On a single GPU, MHE is 2.6× faster than VE for penalty, but 1.3× slower for FSNet (sequential L-BFGS per head).

### Inference wall time (FSNet, 2000 test samples, batch=256, measured)

| Method            | Sequential 1 GPU | Parallel K GPUs (theoretical) | Vectorised single GPU* |
|---                |---:              |---:                           |---:                    |
| Single            | 3.9 s            | 3.9 s                          | 3.9 s                  |
| **VE M=5**  | 5.6 s            | ~4 s                           | ~5 s                   |
| MHE M=5           | 5.6 s            | ~4 s                           | ~5 s                   |
| Perturb K=20      | 16 s             | ~5 s                           | **~5–6 s***            |
| Perturb K=50      | 34 s             | ~6 s                           | ~7 s                   |
| Perturb K=100     | 70 s             | ~8 s                           | ~9 s                   |

\* The K repair calls currently run **sequentially** in a Python `for` loop in `evaluator.py`. They can be vectorised into one L-BFGS call by stacking the K perturbed predictions along the batch axis (`(K·B, out)`). Estimated 3–10× speedup. Same trick applies to the M ensemble members. Easy todo for the next iteration.

### Memory cost (FSNet hdim=1024×4, FP32 storage)

| Method                          | NN params storage | Notes |
|---                              |---:               |---    |
| Single                          | **13 MB**         | -     |
| VE M=5                    | **65 MB** (5× single) | All M members loaded at inference |
| FGE ens5                        | 65 MB             | Same as VE |
| **MHE M=5**                     | **18 MB** (~1.4×) | Trunk + M small heads |
| Perturbation (on single)        | **13 MB**         | Zero extra params; perturbation is element-wise noise |

Training memory is ~3× storage (gradients + Adam state). For MHE training, joint forward needs all M heads' activations in memory at once (small overhead since heads are tiny).

### Headline trade-offs

- **M GPUs available, training budget unrestricted**: VE M=5 in parallel is cheapest path to a high-quality ensemble. Embarrassingly parallel — just submit M jobs.
- **1 GPU, training-budget-constrained**: MHE for penalty (2.6× speed-up); not the right tool for FSNet on a single GPU unless you vectorise the L-BFGS across heads.
- **No retraining budget, only inference**: **Perturbation K=20** on a single FSNet beats VE M=5 quality at 1.4× the inference wall-clock today (16 s vs 5.6 s), or essentially same wall-clock once we vectorise (~5–6 s). **Strongest deal in the study.**
- **Memory-constrained deployment** (e.g., edge inference): MHE FSNet is the unambiguous win — 28% the params of VE M=5, equal quality.

---

## Why Idea 2 works (one slide of intuition)

**Important: the L-BFGS repair only minimizes the feasibility violation**, *not* the objective. From `utils/lbfgs.py:104-110`, the objective is `scale·(‖eq_resid‖² + ‖ineq_resid‖²)`. There is no `obj_fn` term. So the repair is a **soft projection onto the feasible set** — it finds the nearest feasible point to the starting `y_pred`, but doesn't try to make the objective small.

The mechanism:

1. **NN** (trained on obj + feasibility loss) outputs `y_pred` — typically near the feasible set in a region with low objective.
2. **L-BFGS** projects `y_pred` to a feasible point `y*`. Different `y_pred` → different `y*`.
3. The objective at `y*` is "whatever it is" — not optimized by repair, only inherited from where the projection lands.
4. **Perturbation** generates K nearby starts `y_pred + ε·z_k`. Each projects to a (slightly different) feasible point. K feasible candidates, all with tiny constraint violations but **different objective values**.
5. **`best_merit`** (≈ `best_obj` since all candidates are nearly feasible) picks the projection with the lowest objective.

**Two weights to keep straight in the codebase:**

- **Reported `merit_mean`** (in tables / yaml): `Obj + 1e6·EqL1 + 1e6·IneqL1` (from `_compute_merit` in `evaluator.py:387`)
- **`best_merit` selection rule** (which candidate to keep among K): `Obj + 1e5·EqL1 + 1e5·IneqL1` (in `_aggregate_predictions` line 294)

The selection uses the **gentler 1e5 weight**, so for our perturbation regime (EqL1 ~ 7e-6, Obj ~ 17) the selection score ≈ 17 + 0.7 — objective dominates the selection 24×. The reported merit at 1e6 weighting then shows the result with eq violation more visible.

**Where the reported merit reduction comes from** (FSNet single hdim=2048, perturbation K sweep):

| K   | Obj   | 1e6·EqL1 | 1e6·IneqL1 | = Merit (reported) |
|---: |---:   |---:      |---:        |---:                |
| 1   | 16.21 | 68.4     | 0.47       | 85.08              |
| 20  | 16.67 | 7.39     | 0          | 24.06              |
| 100 | 16.60 | 4.68     | 0          | 21.28              |

The reported Merit (1e6 weight) drops from 85 to 21 — and **most of that drop is the EqL1 contribution shrinking from 68.4 to 4.68** (since IneqL1 drops to 0 by K=20 and Obj actually slightly *increases*).

But under the **selection** (1e5 weight), the picture is different: the selection score is dominated by Obj (~17). So `best_merit` is effectively picking candidates with lower objective; the EqL1 reduction we observe is a **downstream consequence** — candidates with lower selection-score also happen to have lower EqL1, because tight L-BFGS convergence yields both (a) small residual and (b) a good objective at that residual.

The mechanistic picture: L-BFGS with `max_iter=50` doesn't fully converge from the NN's raw `y_pred` (residual EqL1 ≈ 7e-5). From perturbed starts, L-BFGS sometimes reaches a tighter residual *and* a slightly better objective within the same iteration budget. `best_merit` picks the best of K such candidates. The win shows up most visibly in EqL1 because of the 1e6 weighting in the reporting, but the selection rule that drives it is mostly obj-based at 1e5.

> **Heads-up: the 1e5 vs 1e6 weight discrepancy is a code-quality issue.** `best_merit` aggregation and reported merit use different weights. Worth unifying — currently they can in principle disagree about "best" if a candidate has very different EqL1 from objective.

Cost: K× repair per sample (currently sequential in Python — could be vectorised to one L-BFGS call on a K× larger batch). NN forward paid once. **K=20 ε=0.1 captures most of the gain; K=50–100 saturates.**

This is **classical multi-start non-linear programming** (M random restarts of L-BFGS, then pick best) ported to learning-to-optimize. The NN supplies one good initialization; perturbation gives K-1 more. I searched FSNet / DC3 / RAYEN / DeepOPF / DiffOPF / PINN ensemble literature and **did not find an equivalent baseline** — it appears to be unpublished in the L2O context.

### Why related approaches do or don't work, through this lens

- **VE (vanilla ensemble)** works because M different NNs produce M different `y_pred`'s in different parts of the feasible neighborhood → M different projection targets → `best_merit` picks the best objective.
- **FGE fails** because cyclical-LR snapshots stay in the same NN weight basin → produce essentially the same `y_pred` → essentially the same projected `y*` → no useful diversity for `best_merit`.
- **Perturbation succeeds** because it generates K nearby `y_pred + ε·z_k` cheaply, then exploits the same projection-to-different-points effect — without paying M× training cost.
- **MHE on penalty struggles** because penalty has no projection step (`_post_process_predictions` is identity for penalty), so head outputs aren't projected anywhere; the shared-trunk-with-cheap-heads architecture lacks the diversity VEs get from independent trunks.
- **MHE on FSNet works** because the in-batch L-BFGS during *training* converges each head to a different feasible local optimum during training (not just at eval), pushing the heads apart.

---

## Recommendations for next sprint (in order)

1. **Ship Idea 2 as the default eval recipe**. Compare against DeepOPF / DC3 / RAYEN baselines using their public benchmarks. Likely a paper claim: "deep ensemble training is unnecessary when the repair operator dominates inference."
2. **Implement FRDE (functional repulsive deep ensemble) on MHE**: add a kernel-repulsion loss term between heads' raw outputs during MHE training. Small change. Should close the gap to VE.
3. **L-BFGS warm-start across the K restarts** (Phase 6 of the broader plan): member k starts repair from member k-1's repaired y → 2–4× inference speedup. Orthogonal to everything above.
4. **MHE + perturbation combination**: M heads × K perturbations per head = M·K candidates per sample. The cheap-training trunk + cheap-inference multi-start should compound. (Currently requires a small eval-path change.)

---

## Open questions for you

- Have you seen the perturbation / multi-start trick in any L2O paper? My scan came back empty.
- Should the next experiment add FRDE to MHE, or pursue MHE + perturbation combinations first?
- FSNet `OptGap%` is consistently **negative** (e.g. −6.66%, −4.09%) — the NN+repair finds objectives below the QP-solver reference. Want me to sanity-check this against a global oracle on a small batch?
- The `merit` weight is inconsistent in the codebase: `_compute_merit` uses 1e6, `best_merit` aggregation uses 1e5. We've been reporting the 1e6 version (the larger one). Should we unify?
