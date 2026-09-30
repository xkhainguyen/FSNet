"""Fill the landscape_lab disk cache for the notebook's default views (run on a GPU node).

Uses the same settings as landscape_lab.ipynb (test split, N_EVAL=500, 31x31), so the
notebook loads these grids instead of recomputing them.
"""
import os

from landscape_lab import Lab

lab = Lab(split="test", n_eval=500).add_from_json("figures/landscape/fig1/ckpts.json")
for m in lab.models:
    for radius in (0.1, 0.5):
        for fs in (False, True):
            lab.grid(m, ("random", 5), ("random", 6), radius=radius, n=31, fs=fs)
        vec = f"figures/landscape/fig1/topvec_test_{m}.pt"
        if os.path.exists(vec):
            lab.grid(m, ("eig", vec), ("random", 6), radius=radius, n=31, fs=False)
lab.plane(["M1", "M3", "M4"], n=31, margin=0.5, fs=True)
print("cache filled:", len(os.listdir(lab.cache_dir)), "grids")
