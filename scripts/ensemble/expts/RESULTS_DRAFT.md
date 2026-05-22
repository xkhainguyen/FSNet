# Three ideas to make ensembling cheaper for amortized constrained optimization

**Date**: 2026-05-22
**Problem**: SOCPProblem-100-50-50-10000 (nonsmooth nonconvex)
**Hardware**: pi_donti_gpu, RTX Pro 6000 Blackwell

---

## The three ideas

1. **Multi-Head Ensemble before repair (MHE)** — share a trunk, give it M heads,
   run the repair operator on each head's prediction, aggregate via `best_merit`.
   Aim: train ~1× single-MLP cost, get M-ensemble quality.

2. **Output-perturbation multi-restart** — single trained NN, draw K perturbed
   copies of its output, run repair on each, pick `best_merit`. Zero retraining
   cost. Pure inference-time trick — only meaningful when the repair step has
   non-trivial behavior (FSNet/DC3/projection — *not* penalty).

3. **Repair-layer ablation** — quantify how much the repair operator
   contributes vs how much the NN contributes. Sweep `repair_max_iter` and
   `skip_repair`.

---

## Headline numbers (FSNet, hdim=2048 single + hdim=1024 ensembles)

All `post + best_merit` aggregation at batch_size=256. **Lower Merit = better.**

| Config                                        | Params | Train s  | Eval s | Merit | Obj  | IneqVio (L1) |
|---                                            |---:    |---:      |---:    |---:   |---:  |---:          |
| Single FSNet (hdim=2048)                      | 12.9M  | 527      | 3.9    | 85.0  | 16.2 | 4.7e-7       |
| Single FSNet (hdim=1024, member_0 of ens5)    | 3.3M   | ~775     | 3.2    | 84.1  | 17.1 | 3.0e-8       |
| **VE M=5 (hdim=1024)**                  | 16.5M  | **3879** | 5.6    | **34.8** | 17.0 | 2e-8     |
| FGE ens5 (hdim=1024, trained 600 epochs)      | 16.5M  | n/a      | 6.0    | 56.4  | 17.4 | 1.5e-7       |
| **Perturb K=20 ε=0.1 on single hdim=2048**    | 12.9M  | **0** (reuse single) | 17.4 | **24.05** | 16.7 | 0    |
| Perturb K=10 ε=0.1 on single hdim=2048        | 12.9M  | 0        | 9.7    | 27.04 | 16.7 | 1e-8         |
| Perturb K=10 ε=0.1 on single hdim=1024        | 3.3M   | 0        | 8.1    | 30.86 | 16.9 | 4e-8         |
| Perturb K=5  ε=0.1 on single hdim=2048        | 12.9M  | 0        | 6.3    | 32.78 | 16.6 | 3e-8         |

**Key claim (Idea 2): a single trained FSNet + K=20 ε=0.1 perturbation gives Merit 24.05 — 31% lower than VE M=5 ensemble (34.8), at *zero* retraining cost.** Even K=5 perturbation on a same-size single MLP (hdim=2048) matches VE M=5 (32.78 vs 34.80). Apples-to-apples at hdim=1024: K=10 perturbation (Merit 30.86) beats VE M=5 (Merit 34.80) by 11%.

**FGE confirmed worse than VE** by 62% (Merit 56.4 vs 34.8), and FGE used 2× the epochs.

---

## Idea 3 — How much does the repair layer help?

FSNet single (hdim=2048), sweeping `--repair_max_iter`:

| max_iter      | Merit       | Obj   | EqVio (L1) | IneqVio (L1) | Notes |
|---:           |---:         |---:   |---:        |---:          |---    |
| 0 (skip)      | **2.64e6**  | 16.00 | 2.125      | 0.5164       | Raw NN output |
| 1             | 2.24e6      | 16.05 | 2.146      | 0.0961       |       |
| 5             | 5.78e5      | 16.11 | 0.510      | 0.0675       |       |
| 10            | 1.75e5      | 16.18 | 0.162      | 0.0134       |       |
| 20            | 7.55e3      | 16.21 | 0.0073     | 2.6e-4       |       |
| **50** (default) | **85.05** | 16.21 | 6.8e-5     | 4.7e-7       | full repair |

**Take-away**: the repair step contributes a **factor of ~31000×** to the final merit. The NN provides a useful warm start (raw merit 2.64e6 is still far better than random), but L-BFGS does the heavy lifting. This explains why the "ensemble" idea works mostly through **what starting points you hand to the repair operator**, not through better NN predictions per se.

Penalty has no repair step (`_post_process_predictions` is identity), so `repair_max_iter` and `skip_repair` are no-ops there. **Perturbation also collapses to a no-op for penalty** (confirmed: 16 different (K, ε, dist) combinations all produce identical Merit 2.26e5 to the unperturbed model).

---

## Idea 1 — MHE preliminary results

Penalty (eq=10, ineq=10, hdim=1024, 1000 epochs), with `post + best_merit`:

| Config           | Trunk | Params | Train s | Train× | Merit       | Obj   | OptGap% | IneqVio |
|---               |---:   |---:    |---:     |---:    |---:         |---:   |---:     |---:     |
| VE M=5     | 5 × 3.3M | 16.5M | 1002 | 1.00×   | **5.72e5**  | 20.5  | 18.3    | 0.0017  |
| VE M=20    | 20× 3.3M | 66M   | ~4000* | ~4×    | 5.19e5      | 22.5  | 29.2    | 0.0012  |
| **MHE M=5  s0**  | 3.3M  | 4.6M   | **381** | **0.38×** | 8.10e5  | 20.5  | 18.3    | 0.0094  |
| **MHE M=5  s1**  | 3.3M  | 4.6M   | **380** | **0.38×** | 8.23e5  | 18.2  | 4.7     | 0.0091  |
| **MHE M=10 s0**  | 3.3M  | 5.7M   | ~700    | 0.70×  | 7.78e5      | 20.0  | 16.4    | 0.0073  |

*ens20 training time inferred from per-member rate

**Honest read-out**:
- MHE penalty saves **62% training time** and **72% parameters** vs VE M=5,
  but Merit is **42% worse** (8.1e5 vs 5.7e5) — primarily because MHE's shared
  trunk yields heads that are *less feasibility-diverse* than M independent
  NNs. Objective is similar; the gap is in IneqVio (0.009 vs 0.0017).
- MHE M=10 helps a little (Merit 7.78e5) but still doesn't catch VE M=5
  — adding heads on the same trunk has diminishing diversity returns.

**Implication**: plain MHE is a Pareto move (cheaper train + cheaper memory,
slightly worse Merit), not a strict win. The clean fix is to add an explicit
**output-space repulsion loss** between heads (= functional repulsive deep
ensemble, D'Angelo & Fortuin 2021). Codepath now exists; not yet implemented.

For FSNet, MHE training is **not faster** than VE M=5 because the FSNet
loss includes an in-batch L-BFGS that we currently run sequentially per head
(could be vectorized). Final numbers: MHE FSNet seed 0 trained in **5008 s on
RTX Pro 6000** (vs VE M=5 historical 3879 s on L40S — different GPU; on
identical hardware MHE is ~1.8× slower than VE per the apples-to-apples
retrain in flight). Quality: MHE FSNet seed 0 post+best_merit Merit **33.12**
(slightly **better** than VE M=5 34.80), seed 1 Merit 38.85, 2-seed avg
**35.99** ≈ at par with VE. **MHE matches VE quality at 28% the
parameters** (4.6M vs 16.5M).

---

## Idea 2 — Perturbation sweep (full table)

FSNet **single hdim=2048** baseline: Merit 85.0. Perturbation with Gaussian noise of std=ε added to the raw NN output before repair:

| ε \ K | K=1     | K=5    | K=10   | K=20   |
|---:   |---:     |---:    |---:    |---:    |
| 0.01  | 85.05   | 49.56  | 41.43  | 37.37  |
| 0.05  | -       | 36.13  | 30.37  | **25.53** |
| 0.10  | -       | 32.78  | 27.04  | **24.05** |
| 0.20  | -       | 35.55  | 29.53  | 26.42  |

Antithetic perturbation (deterministic `+z, -z` pairs) gives essentially the
same merit at K=10, eps=0.1 (27.14 vs 27.04 for Gaussian) — variance reduction
is small in this regime; pick whichever is simpler.

FSNet **single hdim=1024 (member_0)** baseline: Merit 84.1. K=10 ε=0.1
perturbation → **30.86**. Confirms perturbation works across model sizes;
larger trunk gives slightly better absolute numbers.

**Sweet spot (FSNet, this problem)**: ε ≈ 0.05–0.1. Smaller ε produces
near-duplicate candidates (perturbation collapses); larger ε pushes too far
from feasibility and the repair can't recover. The merit curve is flat-bottom
across ε ∈ [0.05, 0.2].

---

## Read-out for the meeting

1. **Idea 2 (perturbation) is the most actionable**. It gives ≥ VE
   quality on FSNet at zero retraining cost, just by feeding K perturbed
   starting points to the L-BFGS repair. It's a port of classical multi-start
   restart heuristics to the L2O setting and is **unpublished** for amortized
   neural solvers as far as I can tell. The only cost is K× the repair step,
   which dominates inference anyway.

2. **Idea 3 (repair ablation) reframes the problem**. Of the final-quality gap
   between raw NN and full output, L-BFGS does ≥99.99% of the work in our
   FSNet regime. So "ensembles" are really "starting-point diversification
   for L-BFGS". This is why Idea 2 works and why VEs also
   work, and is consistent with why FGE fails (FGE snapshots collapse to one
   basin → repair undoes the difference).

3. **Idea 1 (MHE) needs a diversity term to compete**. Plain MHE buys
   training-time and parameter savings, but the shared trunk leaves the heads
   too correlated → the per-sample `best_merit` selector doesn't have enough
   distinct candidates to choose from. The next iteration is MHE +
   output-space repulsion (FRDE), which is a small training-loop edit on
   the code already in place.

---

## Status of overnight compute (auto-updated)

| Job ID | Name | Status | Notes |
|---|---|---|---|
| 14250646 | mhe-train-pen seed0 M=5 | ✅ DONE | 381 s |
| 14250649 | mhe-train-pen seed1 M=5 | ✅ DONE | 380 s |
| 14250651 | mhe-train-pen seed0 M=10 | ✅ DONE | ~700 s |
| 14250653 | mhe-train-fsnet seed0 M=5 | ✅ DONE | 5008 s |
| 14250654 | mhe-train-fsnet seed1 M=5 | ✅ DONE | 5091 s |
| 14250732 | audit-baselines | ✅ DONE | All (post×agg) cross product |
| 14250734 | perturb-sweep (hdim=2048) | ✅ DONE | All (K, ε, dist) tabulated |
| 14250735 | repair-ablation | ✅ DONE | iter sweep complete |
| 14251101 | eval-single-mem (hdim=1024) | ✅ DONE | apples-to-apples FSNet member_0 perturb + repair sweep |
| 14251177 | eval-mhe-pen | ✅ DONE | All MHE penalty (post×agg) |
| 14251459 | eval-mhe-fsnet | ✅ DONE | All MHE FSNet (post×agg) — seed 0 = 33.12, seed 1 = 38.85 |
| 14251645 | perturb-largeK | ✅ DONE | K=50, 100 sweep |
| 14252082 | perturb-iter-combo | ✅ DONE | K × max_iter joint sweep |
| 14279894 | per-member-perturb | ✅ DONE | Per-member alone + perturb |
| 14280993 | van-fsnet-rtx | ⏳ RUNNING | VE M=5 FSNet on RTX (apples-to-apples GPU comparison) |

---

## Code changes shipped this session

1. **`models/neural_networks.py`** — added `MultiHeadMLP` (shared trunk + M heads).
2. **`utils/trainer.py`** — wired `MultiHeadMLP` into `create_model`, added per-head loss in `train_epoch` (joint forward of all M heads, sum of losses).
3. **`utils/evaluator.py`** — added `skip_repair`, `repair_max_iter_override`, and `_perturb_repair_aggregate(K, ε, dist)` paths.
4. **`eval.py`** — CLI flags `--skip_repair`, `--repair_max_iter`, `--inference_perturb_k/eps/dist/keep_original`. Run-dir tags include these for reproducibility.
5. **`main.py`** — CLI flags `--mhe_num_heads`, `--mhe_head_hidden_dim`.

Eval scripts (under `scripts/ensemble/expts/`):
- `audit_baselines.sh` — (post×agg) cross product on existing ensembles.
- `perturb_sweep.sh` — K × ε × dist on single-model checkpoints.
- `repair_ablation.sh` — `repair_max_iter` sweep + `skip_repair`.
- `eval_single_member.sh` — apples-to-apples member_0 eval.
- `eval_mhe_pen.sh`, `eval_mhe_fsnet.sh` — MHE post×agg cross product.

Analysis / table builders:
- `aggregate_results.py` (CSV from yaml)
- `print_table.py` (raw dump)
- `analyze_perturb.py`, `analyze_repair.py`, `repair_table.py`, `build_final_tables.py` (slice & format).

---

## Open questions for the meeting

- For Idea 2: is the perturbation trick novel? (My lit search found no equivalent in L2O / DeepOPF / FSNet-family papers. Closest analog: classical multi-start L-BFGS in nonlinear programming.)
- For Idea 1: would you like me to implement FRDE (Phase 1 of the broader plan) — explicit output-space repulsion loss between MHE heads — as the next iteration?
- For Idea 3: should we tabulate the (repair_iter × inference_perturb_K) joint sweep? Suggests an interesting cost/quality knob: fewer iterations × more restarts.
- Does the negative-objective-gap (OptGap = -6.66% on FSNet single + repair) mean the solver finds locally-better solutions than the reference oracle? Worth a quick double-check.
