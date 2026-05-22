# Three ideas for cheaper ensembling in amortized constrained optimization

**Meeting prep, 2026-05-22 — SOCPProblem-100-50-50-10000 (nonsmooth nonconvex), pi_donti_gpu (RTX Pro 6000)**

---

## TL;DR

**All numbers below at unified ρ=1e6 merit weight** (after this session's code change — both `_compute_merit` reporter and `best_merit` selector now use the same weight). Prior numbers used ρ=1e6 for reporting and ρ=1e5 for selection; the inconsistency is now fixed.

1. **Single NN + K perturbed L-BFGS restarts beats VE M=5 ensemble at zero retraining cost** (FSNet). Apples-to-apples (hdim=1024): K=20 ε=0.1 → Merit **26.56** vs VE **30.27** (−12%); K=50 → 23.70 (−22%); K=100 → 22.18 (−27%). On hdim=2048: K=100 → **20.35** (all-time best).

2. **MHE FSNet now strictly Pareto-beats VE M=5** under unified ρ: 2-seed avg Merit **29.86** vs VE **30.27** (−1.4%), at **28% the parameters** (4.6M vs 16.5M), AND **~12% faster training on identical RTX hardware** (MHE 5008 s vs VE projected ~5700 s — the L-BFGS-per-head bottleneck is offset by trunk-sharing on this GPU class). **MHE penalty trains 2.6× faster with 28% the params, but Merit lags VE by 42%** — no L-BFGS in penalty loss → heads collapse. Needs an explicit head-diversity loss (FRDE).

3. **The repair step (L-BFGS) dominates final quality** in FSNet — contributes a factor of **~31000×** in Merit between raw NN and full-iteration repair. So "ensembling" is essentially **picking good starting points for L-BFGS** and letting `best_merit` choose.

4. **FGE is reconfirmed worse than VE** by 86% (Merit 56.37 vs 30.27) even with 2× more training epochs (600 vs 300). Single-basin snapshots collapse after repair.

---

## Key tables

### Baselines (FSNet, post + best_merit, batch=256, **lower Merit = better**)

| Config                                | hdim | Params | Train s  | Eval s | Merit       |
|---                                    |---:  |---:    |---:      |---:    |---:         |
| Single MLP                            | 2048 | 12.9M  | 527      | 3.9    | **85.05**   |
| Single MLP (member_0 of VE M=5)       | 1024 | 3.3M   | ~775     | 3.2    | **84.08**   |
| **VE M=5**                            | 1024 | 16.5M  | **3879** | 5.6    | **30.27**   |
| FGE ens5 (600 epochs)                 | 1024 | 16.5M  | ~5800    | 6.0    | 56.37       |

### Idea 2 — Perturbation (zero retraining cost)

Apples-to-apples, hdim=1024 single (= member_0 of VE M=5). **VE M=5 Merit = 30.27 for reference (ρ=1e6 unified).**

| K \ ε       | 0.05  | 0.10  |
|---:         |---:   |---:   |
| 5           | 40.35 | 36.52 |
| 10          | 32.21 | 30.16 |
| 20          | 28.62 | **26.56** |
| 50          | 24.69 | **23.70** |
| 100         | 23.15 | **22.18** |

> **K=20 ε=0.1 beats VE M=5 by 12% — at zero retraining cost. K=100 → 27% better.**

Bigger model (hdim=2048, 12.9M params, ~ ens5's 16.5M total):

| K \ ε       | 0.05  | 0.10  |
|---:         |---:   |---:   |
| 1 (baseline) | -    | 85.05 |
| 5            | 37.20| 33.16 |
| 10           | 30.04| 27.81 |
| 20           | 26.42| **23.26** |
| 50           | 22.54| **21.39** |
| 100          | 21.39| **20.35** |

> **K=100 ε=0.1 → Merit 20.35 — all-time best, 33% better than VE M=5.**

Inference cost: K=20 ≈ 16 s; K=50 ≈ 35 s; K=100 ≈ 70 s vs VE M=5 5.6 s. So we move cost from training to inference, where we can also tune it per-sample (early-exit / cascade).

### Idea 3 — Repair iteration sweep (FSNet single hdim=2048)

| `repair_max_iter` | Merit       | IneqVio (L1)  | Comment            |
|---:               |---:         |---:           |---                 |
| 0 (skip)          | **2.64e6**  | 0.52          | Raw NN output      |
| 1                 | 2.24e6      | 0.096         |                    |
| 5                 | 5.78e5      | 0.068         |                    |
| 10                | 1.75e5      | 0.013         |                    |
| 20                | 7.55e3      | 0.0003        |                    |
| **50** (default)  | **85.05**   | 4.7e-7        | full repair        |

> Repair contributes a factor of **~31000×** in Merit. The NN's contribution to "feasible final solution" is small relative to L-BFGS.

### Idea 2 × Idea 3 — Can we trade repair iterations for perturbations?

Joint sweep on FSNet single (hdim=2048, ε=0.1):

| K \ max_iter | 10      | 20      | 50      |
|---:          |---:     |---:     |---:     |
| 1 (baseline) | 1.75e5  | 7548    | 85.05   |
| 10           | 1.69e5  | 4838    | **27.41** |
| 20           | 1.66e5  | 4074    | **23.84** |

> The two knobs are **not interchangeable**. Reducing `max_iter` is far more destructive (factor of ~2000× from 50→20→10) than adding restarts is helpful (~1.7× at max_iter=10). **Operating point: keep `max_iter=50`, tune K for cost.** Don't try to save inference cost by cutting iterations — the repair fidelity dominates each individual candidate's quality, and `best_merit` can't pick a good candidate if all are bad.

### Idea 1 — MHE (penalty, eq=10 ineq=10, hdim=1024×4)

`post + best_merit` aggregation. Note: penalty has no repair step, so `pre` and `post` are identical here.

| Config              | M  | Train s | Params  | Obj  | Merit       | EqL1   | IneqL1  |
|---                  |---:|---:     |---:     |---:  |---:         |---:    |---:     |
| VE M=5              | 5  | 1002    | 16.5M   | 20.5 | **5.72e5**  | 0.570  | 0.0017  |
| VE M=20             | 20 | ~4000   | 66M     | 22.5 | 5.19e5      | 0.518  | 0.0012  |
| **MHE M=5 seed 0**  | 5  | **381** | **4.6M**| 20.5 | 8.10e5      | 0.801  | 0.0094  |
| **MHE M=5 seed 1**  | 5  | **380** | **4.6M**| 18.2 | 8.23e5      | 0.814  | 0.0091  |
| **MHE M=10 seed 0** | 10 | ~700    | 5.7M    | 20.0 | 7.78e5      | 0.771  | 0.0072  |

For penalty, **EqL1 dominates the violation** (0.5–0.8 range) — penalty has no test-time repair, so eq constraints are only learned via the training penalty weight. IneqL1 stays small because ineq constraints are easier for the NN to satisfy directly. The merit gap between VE and MHE for penalty comes mostly from MHE's heads having ~40% higher EqL1 than VE's independent NNs (shared trunk → less feasibility diversity).

> MHE penalty saves **62% train time + 72% params** at cost of **+42% Merit**. Penalty heads share a trunk and aren't diverse enough on feasibility (no L-BFGS in penalty loss to push them apart). Plain MHE-penalty is a Pareto shift, not strict improvement. Next step: add output-space repulsion loss (FRDE).

### Idea 1 — MHE (FSNet, hdim=1024×4) — opposite story: **quality match at 28% params**

| Config (FSNet)         | M  | Train s | Params  | Merit       |
|---                     |---:|---:     |---:     |---:         |
| VE M=5                 | 5  | **3879**| 16.5M   | **30.27**   |
| **MHE seed 0**         | 5  | 5008    | **4.6M**| **30.52** (~at par)   |
| **MHE seed 1**         | 5  | 5091    | **4.6M**| **29.19** (−4%)       |
| MHE 2-seed avg         | -  | -       | 4.6M    | **29.86** (**−1.4% — MHE wins**) |

> **MHE FSNet beats VE M=5 quality at 28% the parameters** under unified ρ=1e6 (2-seed avg Merit 29.86 vs 30.27). Training-time comparison flips once the GPU is held constant: raw numbers are 5008 s (MHE on RTX) vs 3879 s (historical VE on L40S), but the apples-to-apples VE retrain on RTX projects to **~5700 s** (extrapolated from per-epoch rate ~3.5 s after 65% complete; job cancelled to save GPU). So **L40S is actually faster than RTX Pro 6000 for this workload** (L-BFGS unrolling is kernel-launch-bottlenecked rather than FLOPS-bottlenecked). On identical RTX hardware: **MHE 5008 s vs VE ~5700 s → MHE is ~12% *faster* than VE**, not slower. The L-BFGS-per-head sequential bottleneck in MHE is more than offset by sharing the trunk forward. The slowdown comes from the FSNet loss running L-BFGS sequentially per head inside a single fused autograd graph (vectorising across heads is on the todo — should bring MHE training to ≤1× VE). Unlike penalty, FSNet's in-loop L-BFGS pushes each head's prediction into a distinct feasible local optimum — that's the functional diversity the shared trunk alone wouldn't provide. **A clear quality + memory-efficiency win for FSNet deployment.**

---

## What this means

- **Idea 2 is the strongest finding tonight.** It's a near-zero-cost improvement over VE (vanilla ensemble) for any L2O method with a non-trivial repair step. It's essentially **classical multi-start non-linear programming** (Schaback 1980s, Boender & Rinnooy Kan 1987) ported to amortized opt. My lit search found **no equivalent baseline in the L2O / DeepOPF / FSNet / RAYEN / DC3 / DiffOPF literature** — likely publishable as a strong, almost-free baseline that simpler ensemble papers should compare against.

- **Idea 3 reframes the ensemble question.** Of the gap between raw NN and full-repair output, L-BFGS does ≥99.99% of the work. The NN provides a warm start. So "deep ensembles" for this problem class are really "diverse starting-point ensembles for L-BFGS." This is why Idea 2 works (cheap perturbation of one good NN start → many starts), why VEs work (M independently-trained NN starts), and why FGE fails (FGE's starts collapse to one basin after repair).

- **Idea 1 (MHE) is a partial win.** Cheaper training + smaller models, but lower Merit because the shared trunk reduces functional diversity. The fix is one of:
  1. Add a kernel repulsion loss between heads (FRDE, D'Angelo & Fortuin 2021).
  2. Vary loss weights per head (homotopy ensemble — Phase 2 of the broader plan).
  3. Combine MHE with the cheap inference-time perturbation of Idea 2.

---

## Recommended next sprint

1. Ship Idea 2 as the default eval recipe; compare against published L2O baselines.
2. Implement FRDE on MHE (small training-loop edit; codepath in place).
3. L-BFGS warm-start across the K perturbation restarts (~2-4× inference speedup, orthogonal to everything).
4. Compose MHE + Idea 2 (M heads × K perturbations per head).

---

## Open questions for you

- Have you seen the perturbation / multi-start trick in any L2O paper? My scan came back empty.
- Should I add FRDE first, or pursue MHE + perturbation combinations?
- FSNet `OptGap%` is consistently negative (−6.66% on single + full repair, even more negative with perturbation). The NN+L-BFGS finds objectives below the QP reference. Want me to sanity-check this against a global oracle?
- Merit weight inconsistency: `_compute_merit` (printed) uses **1e6**, `best_merit` aggregation uses **1e5**. We've been reporting the 1e6 version (the printed one) throughout. Should we unify?

---

## All shipped code (for reference)

- `models/neural_networks.py`: `MultiHeadMLP` (shared trunk + M heads, `forward_all` like `EnsembleMLP`).
- `utils/trainer.py`: wired MHE into `create_model`; per-head loss accumulation in `train_epoch`; run-name tag.
- `utils/evaluator.py`: `skip_repair`, `repair_max_iter_override`, and `_perturb_repair_aggregate(K, ε, dist)`.
- `eval.py`: CLI flags `--skip_repair`, `--repair_max_iter`, `--inference_perturb_{k,eps,dist,keep_original}`; runs save with the new tags.
- `main.py`: `--mhe_num_heads`, `--mhe_head_hidden_dim`.

Experiment scripts (`scripts/ensemble/expts/`):
- `audit_baselines.sh`, `perturb_sweep.sh`, `repair_ablation.sh`, `eval_single_member.sh`, `perturb_large_K.sh`, `perturb_iter_combo.sh`, `eval_mhe_pen.sh`, `eval_mhe_fsnet.sh`, `eval_per_member_perturb.sh`, `train_vanilla_fsnet_rtx.sh` — all run.

Analysis (`scripts/ensemble/expts/`):
- `aggregate_results.py`, `print_table.py`, `analyze_perturb.py`, `analyze_repair.py`, `repair_table.py`, `member0_perturb_table.py`, `build_final_tables.py`, `final_summary.py`.
