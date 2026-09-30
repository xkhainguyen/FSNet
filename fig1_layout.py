# %% [markdown]
# # Fig. 1 layout lab
# Re-renders from the cached grids in `paper_figs/fig1/data/` (refresh with `python fig1_bundle.py`).
# No torch, no GPU. Plotting code lives in `fig1_plot.py`; edit `figure(...)` calls here.
#
# `figure(rows, methods, style, path, label_row)`: rows = list of (view, panel, loss) with
# view in `straight`, `curved`, `plane`, `plane_t345`, `plane_t678`, `hex19`; panel in `3d`, `2d`,
# `slice`, `basins`; loss in `own`, `merit`. Style keys: cap, elev, azim, stride, mesh, minima,
# minima_below, crop (x0, x1, y0, y1), mark_slice, xyticks, cmap, label_size, slice_ymax.

# %%
import importlib
import fig1_plot
importlib.reload(fig1_plot)
from fig1_plot import *

METHODS = ["M1", "M4", "M2", "M3f"]  # SL rho 10, SSL rho 10, SL rho 1e5, SL + hard FS

# %% [markdown]
# ## Main figure (as in `make_fig1_main.py`)

# %%
MAIN = dict(crop=(-1.0, 2.0, -0.8, 1.7), minima=False, stride=2, mesh=False, mark_slice=True, xyticks=False,
            cap=4.0, elev=42, norm="method")
figure([("plane_t345", "3d", "own"), ("plane_t345", "2d", "own"), ("plane_t345", "slice", "own")],
       METHODS, style=MAIN, row_h=5, label_row=0)

# %% [markdown]
# ## Other views (edit freely)

# %%
# Basins of attraction on the same plane (caution: for hard FS the region outside the cliff is
# solver divergence, so its many tiny "basins" are numerical, not traps)
figure([("plane_t345", "2d", "own"), ("plane_t345", "basins", "own")], METHODS, style=MAIN, row_h=5)

# %%
# 10-seed flat sheet vs curved sheet
figure([("straight", "2d", "own"), ("curved", "2d", "own")], METHODS, style=dict(minima=False), row_h=5)

# %%
# Objective mismatch: training loss vs merit on the same plane
figure([("plane_t345", "2d", "own"), ("plane_t345", "2d", "merit")], METHODS, style=MAIN, row_h=5)
