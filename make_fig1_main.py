"""Fig. 1 (main + supplementary) from the cached grids (fig1_bundle.py), no GPU.

Main: held-out training loss of each method on the plane through three permutation-aligned
trained models, seeds (3, 4, 5); rows: 3D surface, top view, and the loss along the straight
line through s0 and s1 (uncapped). The seed triple is chosen by a fixed rule: the triple whose
median edge barrier is closest to each method's median over the 18 edges of its 10-model sheet
(summed percentile distance from 50; ties to the lower-index triple). The rule was fixed after the
three planes had been computed and viewed, before the final layout. Supplementary: the same
layout for triples (0, 1, 2) and (6, 7, 8); the typicality table.
Also writes fig1_main_all*.png: the same layout with hard FS at rho = 1 and 0.8 added (collapsed
seeds marked with an X). Writes paper_figs/fig1/fig1_main*.png and fig1_main_numbers.md.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fig1_plot import figure, surface
from sheet_stats import stats

OUT = "paper_figs/fig1"
M = ["M1", "M4", "M2", "M3f"]  # columns: SL rho 10, SSL rho 10, SL rho 1e5, SL + hard FS
TRIPLES = {"plane": "(0, 1, 2)", "plane_t345": "(3, 4, 5)", "plane_t678": "(6, 7, 8)"}
# Same style for every method: colour relative to the method's best of its 10 trained networks
# (seed_losses.py), cap 4 decades (so no method's plateau saturates except the
# erratic hard-FS region), no mesh lines, the slice line marked in the top view.
MAIN = dict(crop=(-1.0, 2.0, -0.8, 1.7), minima=False, stride=2, mesh=False, mark_slice=True, xyticks=False,
            cap=4.0, elev=42, norm="method")
ROWS = lambda v: [(v, "3d", "own"), (v, "2d", "own"), (v, "slice", "own")]
# Extended figure: the same layout plus hard FS at rho = 1 and 0.8 (no gate, no distance term), whose
# collapsed seeds (merit after FS > 10, m3_candidates.py) are marked with an X.
M_ALL = M + ["M3r1", "M3r08"]
SEEDS = {"plane": (0, 1, 2), "plane_t345": (3, 4, 5), "plane_t678": (6, 7, 8)}
COLLAPSED = {"M3r1": {1, 3, 8}, "M3r08": {2, 5, 6, 7, 8}}


def all_style(v):
    """MAIN style plus the collapsed networks of this seed triple (indices within the triple)."""
    col = {m: {n for n, s in enumerate(SEEDS[v]) if s in bad} for m, bad in COLLAPSED.items()}
    return dict(MAIN, collapsed=col)


def edge_barriers(v, m):
    """Straight-line barrier (decades) on the 3 edges of the seed triangle of plane view v."""
    s = surface(v, m, "own", 99)
    f = RegularGridInterpolator((s["ys"], s["xs"]), s["T"])
    out = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        P = s["pts"][a] + np.linspace(0, 1, 201)[:, None] * (s["pts"][b] - s["pts"][a])
        t = f(P[:, ::-1])
        out.append(t.max() - max(t[0], t[-1]))
    return np.array(out)


def main():
    lines = ["# Fig. 1 numbers (held-out, own training loss, decades = log10 ratio)", ""]
    ref = {m: np.array(stats(f"figures/landscape/fig1/sheet10_wm_test_{m}.npz", m)[0]) for m in M}
    lines += ["## Seed-triple typicality", "",
              "Median straight-line edge barrier of each triple; in brackets its percentile among the "
              "18 edges of the method's 10-model sheet.", "",
              "| triple | " + " | ".join(M) + " | summed abs(pct - 50) |", "|---|" + "---:|" * (len(M) + 1)]
    score = {}
    for v, name in TRIPLES.items():
        cells, dev = [], 0.0
        for m in M:
            med = np.median(edge_barriers(v, m))
            pct = 100 * (ref[m] < med).mean()
            dev += abs(pct - 50)
            cells.append(f"{med:.2f} ({pct:.0f}th)")
        score[v] = dev
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {dev:.0f} |")
    main_view = min(TRIPLES, key=lambda v: (score[v], list(TRIPLES).index(v)))
    lines += ["", f"Chosen for the main figure: seeds {TRIPLES[main_view]}.", ""]
    lines += ["## Along the line s0 - s1 (barrier relative to the higher of the two networks)", "",
              "| triple | method | barrier between s0 and s1 |", "|---|---|---:|"]
    for v, name in TRIPLES.items():
        for m in M:
            s = surface(v, m, "own", 99, MAIN["crop"])
            j0 = int(np.argmin(np.abs(s["ys"])))
            t, seg = s["T"][j0], (s["xs"] >= 0) & (s["xs"] <= 1)
            climb = t[seg].max() - max(np.interp(0, s["xs"], t), np.interp(1, s["xs"], t))
            lines.append(f"| {name} | {m} | {10 ** climb:,.0f}x |")
    figure(ROWS(main_view), M, style=MAIN, row_h=5, label_row=0, path=f"{OUT}/fig1_main.png")
    for v in TRIPLES:
        if v != main_view:
            figure(ROWS(v), M, style=MAIN, row_h=5, label_row=0, path=f"{OUT}/fig1_main_{v}.png")
    for v in TRIPLES:
        suffix = "" if v == main_view else f"_{v}"
        figure(ROWS(v), M_ALL, style=all_style(v), row_h=5, label_row=0, path=f"{OUT}/fig1_main_all{suffix}.png")
    lines += ["", "## Hard FS at rho = 1 and 0.8 on the same triples (barrier s0 to s1)", "",
              "| triple | method | s0, s1 status | barrier between s0 and s1 |", "|---|---|---|---:|"]
    for v, name in TRIPLES.items():
        for m in ["M3r1", "M3r08"]:
            s = surface(v, m, "own", 99, MAIN["crop"])
            j0 = int(np.argmin(np.abs(s["ys"])))
            t, seg = s["T"][j0], (s["xs"] >= 0) & (s["xs"] <= 1)
            climb = t[seg].max() - max(np.interp(0, s["xs"], t), np.interp(1, s["xs"], t))
            status = ", ".join("collapsed" if sd in COLLAPSED[m] else "good" for sd in SEEDS[v][:2])
            lines.append(f"| {name} | {m} | {status} | {10 ** climb:,.0f}x |")
    open(f"{OUT}/fig1_main_numbers.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
