# Penalty, maxt0.5, 800 labels, fixed total offline budget

**Setup.** `B = 1100 s = gen + 0.16·k + 0.18·epochs`, charged with the reference
rates (gen 396.3 s for 800 labels @ maxt0.5; SL 0.16 s/ep; penalty SSL 0.18 s/ep)
regardless of which GPU a job ran on. Both arms use `lr=1e-4`, `eq_pen_weight=50`,
`ineq_pen_weight=10`, warmup + cosine to `eta_min=1e-6`. Budget matched to <0.1 s
on every row. Vanilla uses no labels, so its whole budget is SSL: 6111 epochs.
Merit warm-starts from checkpoint k\*, selected as the earliest checkpoint within
1.02× of the minimum validation `stop_merit`, and gets 3198–3465 epochs.

Source: `analysis/penalty_maxt05_B1100_n8.csv` (32 runs). SSL seed s uses SL pool s.

**Merit convention.** Tables use `Merit = obj + 1e5·(eq_l1 + ineq_l1)`. The
`merit_mean` field stored in the pickles/CSV uses a 1e6 weight, i.e. exactly 10x
these values. Ratios and win counts are identical under either weight.

---

## Table 1 — selected seeds (0, 1, 2)

> **These three seeds were chosen because merit wins on them.** They are 3 of 8
> run; the other five are in Table 2. Any number below is conditioned on that
> selection and is not an estimate of expected performance.

| arm | Merit = obj + 1e5·(eq+ineq) | eq_l1 | ineq_l1 | opt_gap |
|---|---:|---:|---:|---:|
| **merit** (k\*=500–800, E=683) | **7.058e4** ± 3.3e3 | **0.6898** ± 3.3e-2 | **1.60e-2** ± 8.2e-4 | −7.33 ± 0.22 |
| vanilla | 7.128e4 ± 3.7e3 | 0.6922 ± 2.8e-2 | 2.07e-2 ± 9.1e-3 | −1.63 ± 4.85 |

Paired per-seed ratio (vanilla/merit, >1 = merit better): **geomean 1.0097×, 3/3**
— s0 = 1.0012, s1 = 1.0141, s2 = 1.0136.

---

## Table 2 — all 8 seeds

| arm | Merit = obj + 1e5·(eq+ineq) | eq_l1 | ineq_l1 | opt_gap |
|---|---:|---:|---:|---:|
| merit (k\*=500–850, E=712) | 7.287e4 ± 3.7e3 | 0.7120 ± 3.4e-2 | 1.67e-2 ± 5.5e-3 | −4.24 ± 4.13 |
| **vanilla** | **6.898e4** ± 3.0e3 | **0.6735** ± 2.6e-2 | **1.63e-2** ± 7.9e-3 | +6.60 ± 10.2 |

Paired: **geomean 0.9468×, merit wins 3/8** — s0 = 1.0012, s1 = 1.0141,
s2 = 1.0136, s3 = 0.9411, s4 = 0.9193, s5 = 0.9140, s6 = 0.9709, s7 = 0.8176.

---

## Two things to check before using Table 1

**The margin is under 1%, and smaller than its own error bar.** The selected
table reports 7.058e4 vs 7.128e4, a gap of 704, against a +-3.3e3 spread -- the
standard deviation is 4.7x the effect. The three selected wins are 1.0012, 1.0141, 1.0136;
geomean 1.0097×. Per-arm σ in this sweep is 4–5%, so the selected effect is about
a fifth of one standard deviation. The five excluded seeds lose by 3–18%, i.e.
the losses are large where the wins are not.

**`opt_gap` does not read as "better" here.** Merit's −7.33 versus vanilla's −1.63
occurs at `eq_l1 ≈ 0.69`, where the constraint violation is what produces the
negative gap. A more negative gap at high violation means the solution is further
outside the feasible set, not closer to optimal. In Table 2, vanilla is the arm
with a *positive* gap (+6.60), and it also wins merit, eq_l1, and ineq_l1.

## The LR sweep completed, and it refutes the cosine-restart explanation

Merit's validation `stop_merit` rises from 73,546 at epoch 800 to 76,756 at 1600
before recovering to 71,797, while vanilla saturates near 69,879 by epoch 3000.
That rise was read as the warm start being erased by the full-amplitude cosine
restart, which predicts the warm start should survive at lower LR. It does not:

| lr (both arms) | n | vanilla/merit | merit wins |
|---:|---:|---:|---:|
| 1e-4 | 8 | 0.9773 | 4/8 |
| 3e-5 | 4 | 0.8948 | 0/4 |
| 1e-5 | 2 | 0.9397 | 0/2 |

Lower LR makes the warm start **worse**. It also makes vanilla substantially
better in absolute terms (3.98e5 at 1e-5 vs 6.90e5 at 1e-4, `merit_mean` scale),
so `lr=1e-4` — the setting in Tables 1 and 2 — is the setting most favourable to
the warm start, not a neutral one. A vanilla baseline tuned over LR would widen
the gap in Table 2.

## The arm missing from both tables

`early` (k=100) is the best warm start in this sweep and is not tabulated above:
vanilla/early = **1.0028, 3/8** on the per-seed-SL design and 0.9930, 2/8 on the
fixed-SL design. That is a tie with vanilla, not a win, and k=100 is not what the
merit rule selects — but it is the arm to beat, and it beats `conv` 8/8.
