"""Shared definitions for the Fig. 1 landscape study (L1 penalty, seed-indexed runs).

  M1   SL, rho 10,  lr 3e-4
  M2   SL, rho 1e5, lr 1e-4 (best lr in its screen)
  M2m  SL, rho 1e5, lr 3e-4 (M2 at M1's lr, to rule out the lr confound)
  M4   SSL, rho 10, lr 1e-3
"""
import glob
import os

D = "results/nonsmooth_nonconvex/socp/SOCPProblem-100-50-50-10000"
FIG = "figures/landscape/fig1"
SPEC = {
    "M1": ("sup_pen", "obj0.1_eq10.0_ineq10.0", "0.0003", 10.0),
    "M2": ("sup_pen", "obj0.1_eq100000.0_ineq100000.0", "0.0001", 1e5),
    "M2m": ("sup_pen", "obj0.1_eq100000.0_ineq100000.0", "0.0003", 1e5),
    "M4": ("penalty", "obj1.0_eq10.0_ineq10.0", "0.001", 10.0),
    "M3": ("sup_pen_fs", "obj0.1_eq10.0_ineq10.0", "0.0003", 10.0),
    "M3r1": ("sup_pen_fs", "obj0.1_eq1.0_ineq1.0", "0.0003", 1.0, 1000),
    "M3r08": ("sup_pen_fs", "obj0.1_eq0.8_ineq0.8", "0.0003", 0.8, 1000),
    # FSNet-style SL + FS: distance term, squared penalty gated on violation >= 1e3, lr 1e-4, 300 epochs
    "M3f": ("sup_pen_fs", "obj0.1_eq10.0_ineq10.0_dist5.0_gate1000.0", "0.0001", 10.0, 300, ""),
}
TITLE = {"M1": r"M1: SL, small $\rho$", "M2": r"M2: SL, large $\rho$",
         "M2m": r"M2 (lr matched to M1): SL, large $\rho$", "M4": r"M4: SSL, small $\rho$",
         "M3": r"M3: SL + FS, small $\rho$",
         "M3r1": r"M3: SL + FS, $\rho = 1$", "M3r08": r"M3: SL + FS, $\rho = 0.8$",
         "M3f": r"M3: SL + FS, FSNet-style"}


def ckpt(m, s):
    meth, w, lr = SPEC[m][:3]
    ep = SPEC[m][4] if len(SPEC[m]) > 4 else 3000
    pen = SPEC[m][5] if len(SPEC[m]) > 5 else "_penl1"
    runs = [r for r in sorted(glob.glob(f"{D}/*_MLP_{meth}_seed{s}_nepochs{ep}_lr{lr}_trainsize7000_{w}"
                                        f"{pen}_dropout0.0_lrschedcosine_etamin1e-06")) if os.path.exists(r + "/model.pt")]
    if not runs:
        raise FileNotFoundError(f"{m} seed {s}")
    return runs[-1] + "/model.pt"


def own_loss(m, c):
    """Own training loss from stored per-sample-mean components c (compute_landscape_compare)."""
    meth, rho = SPEC[m][0], SPEC[m][3]
    if m == "M3f":  # FSNet-style: squared penalty only while pen2 >= 1e3 (sum of eq and ineq here)
        return 100 * c["huber_fs"] + 5.0 * c["dist_fs"] + rho * c["pen2"] * (c["pen2"] >= 1e3)
    first = {"sup_pen": 100 * c["huber"], "sup_pen_fs": 100 * c["huber_fs"], "penalty": c["obj"]}[meth]
    return first + rho * c["viol_l1"]
