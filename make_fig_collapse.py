"""Hard FS at rho = 1 and 0.8: where do the collapsed seeds sit? (fig1_plot, cached grids)

Collapsed = merit after FS (500 iterations) > 10 (m3_candidates.py): rho 1 seeds 1, 3, 8;
rho 0.8 seeds 2, 5, 6, 7, 8. Planes through aligned seeds with s0 = lowest-index good seed,
s1 = lowest-index collapsed seed, s2 = next good (triple A) or next collapsed (triple B).
Rows: own training loss (3D, top view), merit after FS (top view), own loss along s0 -> s1.
Writes paper_figs/fig1/fig1_collapse.png.
"""
from fig1_plot import figure

COLS = ["M3r1@A", "M3r1@B", "M3r08@A", "M3r08@B"]
LABELS = {"M3r1@A": r"$\rho = 1$, seeds 0, 1, 2", "M3r1@B": r"$\rho = 1$, seeds 0, 1, 3",
          "M3r08@A": r"$\rho = 0.8$, seeds 0, 2, 1", "M3r08@B": r"$\rho = 0.8$, seeds 0, 2, 5"}
COLLAPSED = {"M3r1@A": {1}, "M3r1@B": {1, 2}, "M3r08@A": {1}, "M3r08@B": {1, 2}}  # indices within the triple
STYLE = dict(crop=(-1.0, 2.0, -0.8, 1.7), minima=False, stride=2, mesh=False, mark_slice=True, xyticks=False,
             cap=4.0, elev=42, labels=LABELS, collapsed=COLLAPSED, norm="method",
             cbar_label=r"$\log_{10}$(value / value at the best network), held-out")


if __name__ == "__main__":
    figure([("collapse", "3d", "own"), ("collapse", "2d", "own"), ("collapse", "2d", "merit"),
            ("collapse", "slice", "own")], COLS, style=STYLE, row_h=5, label_row=0,
           path="paper_figs/fig1/fig1_collapse.png")
