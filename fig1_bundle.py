"""Collect every Fig. 1 landscape grid into paper_figs/fig1/data/ (float32, all loss components),
so layouts can be iterated in fig1_layout.ipynb without torch or a GPU.

One file per view, keys "<M>/<field>":
  straight  10-seed flat sheets used in fig1_sheet10_M3f.png (weight-matched seeds)
  curved    10-seed curved (Bezier) sheets (connect_sheet.py)
  plane     wide plane through aligned seeds 0, 1, 2 (fig1_manyminima_aligned.png style)
  plane_t345, plane_t678   the same through seeds 3, 4, 5 and 6, 7, 8
  hex19     19-seed flat sheets, ordering 0 (M1, M2, M4)
Fields: xs, ys, points (seed coordinates), and every loss component
(huber, pen2, obj, viol_l1, huber_fs, obj_fs, viol_l1_fs, dist_fs); NaN = off the sheet.
Re-run after new grids land; missing files are skipped.
"""
import os

import numpy as np

F = "figures/landscape/fig1"
OUT = "paper_figs/fig1/data"
MODELS = ["M1", "M2", "M3f", "M4"]
VIEWS = {
    "straight": {"M1": "sheet_test_M1.npz", "M2": "sheet_test_M2.npz",
                 "M3f": "sheet_tri10_test_M3f.npz", "M4": "sheet_test_M4.npz"},
    "curved": {m: f"sheet10_curved_test_{m}.npz" for m in MODELS},
    "plane": {m: f"seedplane_aligned_test_{m}.npz" for m in MODELS},
    "plane_t345": {m: f"seedplane_aligned_t345_test_{m}.npz" for m in MODELS},
    "plane_t678": {m: f"seedplane_aligned_t678_test_{m}.npz" for m in MODELS},
    "hex19": {m: f"sheet_hex19_o0_test_{m}.npz" for m in ["M1", "M2", "M4"]},
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for view, files in VIEWS.items():
        out = {}
        for m, fname in files.items():
            f = f"{F}/{fname}"
            if not os.path.exists(f):
                print(f"skip {view}/{m}: missing {f}")
                continue
            d = dict(np.load(f, allow_pickle=True))
            out[f"{m}/xs"], out[f"{m}/ys"], out[f"{m}/points"] = d["xs"], d["ys"], d["points"]
            for k in d["components"]:
                out[f"{m}/{k}"] = d[f"plane/{k}"].astype(np.float32)
        np.savez_compressed(f"{OUT}/{view}.npz", **out)
        print(f"{OUT}/{view}.npz: {sorted({k.split('/')[0] for k in out})}")


if __name__ == "__main__":
    main()
