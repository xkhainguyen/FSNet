# Experiments index — perturbation / "free ensembling" in learning-to-optimize

This directory holds two related bodies of work:

1. **Cross-framework perturbation study** (2026-05-28/29) — the main, self-contained
   investigation. **Start at [`CROSS_FRAMEWORK_FINDINGS.md`](CROSS_FRAMEWORK_FINDINGS.md).**
2. **Prior MHE / ensembling session** (2026-05-22) — `MEETING_PITCH.md`,
   `FINAL_SUMMARY.md`, `RESULTS_DRAFT.md` and the `diagnose_*`, `compare_*`,
   `perturb_*`, `eval_mhe_*` scripts. ⚠ The perturbation claim there
   ("multi-restart beats deep ensemble at zero cost") is **substantially revised**
   by the cross-framework study below — most of that gain was a repair/network
   under-convergence artifact.

---

## The cross-framework study in one paragraph

**Question:** is the FSNet "free ensemble" perturbation gain (perturb the NN
output K times, repair each, keep best-merit) a real functional-diversity benefit
or a repair-operator under-convergence artifact? **Answer (6 testbeds): the large
gains are convergence artifacts** — under-converged repair *or* undertrained
network — and at full convergence every genuine residual is ~1% of the objective.
No testbed shows a large convergence-surviving ensemble win. The robust deliverable
is the negative result plus a **convergence-control diagnostic**: push repair
budget *and* network training (with LR decay) to convergence before measuring any
ensemble gain; whatever survives is the real part.

## Testbeds and where they live

| # | Testbed | Repair | Result | Scripts |
|---|---------|--------|--------|---------|
| 1 | FSNet (this repo) | L-BFGS (Euclidean, iterative) | artifact — fixed by `per_sample_lbfgs=1` | `verify_dc3_unaffected.sh`, `sanity_per_sample.sh`, `../../..` core in `utils/lbfgs.py` |
| 2 | DC3 (this repo) | grad-SGD on ineq penalty | artifact — `corr_lr 1e-6→1e-3` matches perturb | `dc3_steps_sweep.sh` |
| 3 | Πnet (`third-party/pinet`) | Douglas–Rachford ADMM | artifact — `n_iter_test 100→500` matches perturb | `pinet/pinet_verify.{py,sh}`, `pinet/pinet_verify_seed0.sh` |
| 4 | Bisection-Projection (`third-party/Bisection-Projection`) | radial bisection (non-Euclidean) | real but small (~0.4%) — repair geometry | `bp/bp_verify.{py,sh}` |
| 5 | HardNet (`third-party/hardnet`) | closed-form pinv/ReLU correction | "2× win" was ~95% undertraining; ~1% residual | `hardnet/hardnet_verify.{py,sh}`, `hardnet/hardnet_control.sh` |
| 6 | Disconnected-ball (synthetic, multimodal) | radial bisection | ~94% headline → ~1% at convergence; high-d coverage collapse | `bp/bp_disconnected_sweep.py`, `bp/bp_disc_controls.py`, `bp/bp_disc_convergence.py`, `bp/bp_disc_highdim.py`, `bp/bp_disc_d4_converge.py` |

## The five sources of apparent gain (see FINDINGS §Synthesis)

1. **Under-converged repair** (1,2,3) — artifact; fix the repair budget.
2. **Undertrained network** (5) — artifact; train longer with LR decay.
3. **Direction-dependent repair** (4, and HardNet's correction) — real, ~0.5%.
4. **Nonconvex objective** (5) — real, ~0.6%.
5. **Multimodal/disconnected feasible set** (6) — real ~5.6% routing floor but
   only ~1% objective gain at convergence; headline 94% was undertraining + 2D
   random-search.

## Process note (recorded honestly)

Every *negative* finding held up under controls. Every *positive* "ensembling
genuinely helps" claim drafted during the study (HardNet nonconvex win;
disconnected-ball "survives convergence"; a 13% "structural floor") was
**undercut by deeper convergence controls** and corrected in-place (the FINDINGS
doc keeps the correction notes rather than silently overwriting). Lesson: always
run capacity × training-convergence controls before claiming a positive ensemble
result.

## Open (not done — larger undertakings, decisions for the user)

- A **real multimodal problem** (AC-OPF / nonconvex QCQP) — the §6 disconnected
  set is a synthetic toy.
- A **per-dimension-converged** multimodal study — the §7 high-d sweep is
  confounded by fixed-budget undertraining; `bp_disc_d4_converge.py` was
  inconclusive (training pathology at d=4). Needs a better-posed, stably-trained
  problem.

## Reproducing

`logs/` and `third-party/` are gitignored; the tracked scripts regenerate the
logs. Most BP/disconnected-ball scripts are 2D/CPU and run in minutes; Πnet and
HardNet need a GPU (see their `.sh` SLURM headers — `pi_donti_gpu`,
`rtx_pro_6000`). Env: `conda activate ml4opt` (+ `LD_LIBRARY_PATH` export per
the repo CLAUDE.md); Πnet uses its own `third-party/pinet/.venv`.
