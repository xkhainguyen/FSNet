"""Verify our perturbation finding on Bisection-Projection (BP).

BP's repair is geometrically distinct: it's radial bisection from a feasible
interior point (IP) to an infeasible candidate. Unlike L-BFGS (Euclidean
projection minimising ||eq||² + ||ineq||²), bisection follows a 1D LINE and
finds the boundary intersection on that line. Bisection converges by
construction (geometric rate ~bis_step=0.9 → 30 iters gives 1e-30 alpha gap),
so there's no under-convergence failure mode.

Hypothesis: under-convergence story DOESN'T apply here. If perturbation gives
gain, it's the radial-projection geometry — different perturbations define
different rays from IP, hitting different boundary points with different
objective values.

Pre-trained QP model used: third-party/Bisection-Projection/models/qp/
QP_Problem-100-50-10000-10000/NN_Eq/solver_net.pth
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

import torch

ROOT = pathlib.Path("/orcd/scratch/orcd/008/khain/FSNet/third-party/Bisection-Projection")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.optimization_utils import QP_Problem  # noqa: E402


def merit(obj, eq, ineq, rho=1e6):
    return obj + rho * (eq + ineq)


def ip_bisect_partial(
    data: QP_Problem,
    X: torch.Tensor,               # (B, xdim)
    z_infeas_partial: torch.Tensor,  # (B, partial_dim), candidate in partial space
    ip_partial: torch.Tensor,        # (B, partial_dim), IP in partial space
    n_steps: int = 30,
    bis_step: float = 0.9,
    feas_eps: float = 1e-5,
) -> tuple[torch.Tensor, int]:
    """Bisect in partial space; complete_partial enforces equality at every step.
    Returns the boundary FULL Y (B, ydim).
    """
    B = X.shape[0]
    alpha_lower = torch.zeros(B, 1, device=X.device, dtype=X.dtype)
    alpha_upper = torch.ones(B, 1, device=X.device, dtype=X.dtype)
    for k in range(n_steps):
        alpha = (1 - bis_step) * alpha_lower + bis_step * alpha_upper
        zt_partial = alpha * (z_infeas_partial - ip_partial) + ip_partial
        yt_full = data.complete_partial(X, zt_partial)
        # only ineq matters; eq is satisfied by complete_partial up to numerical precision
        ineq = data.ineq_resid(X, yt_full).abs().max(dim=1)[0].unsqueeze(1)
        sub_feas = ineq < feas_eps
        sub_inf = ~sub_feas
        alpha_lower = torch.where(sub_feas, alpha, alpha_lower)
        alpha_upper = torch.where(sub_inf, alpha, alpha_upper)
        if (alpha_upper - alpha_lower).max() < 1e-9:
            break
    zt_partial = alpha_lower * (z_infeas_partial - ip_partial) + ip_partial
    yt_full = data.complete_partial(X, zt_partial)
    return yt_full, k


def metrics(data: QP_Problem, X, Y):
    obj = data.obj_fn(Y)
    eq = data.eq_resid(X, Y).abs().sum(dim=1)
    ineq = data.ineq_resid(X, Y).abs().sum(dim=1)
    return obj, eq, ineq, merit(obj, eq, ineq)


def main():
    print("=== BP verification: random QP100-50-50, pretrained NN_Eq solver ===", flush=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}", flush=True)

    # Dataset
    ds_path = ROOT / "datasets/qp/random_2023_qp_dataset_var100_ineq50_eq50_ex10000"
    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)
    test_size = 1024
    data = QP_Problem(dataset, test_size)
    # move tensors to device
    for attr in dir(data):
        v = getattr(data, attr)
        if torch.is_tensor(v):
            setattr(data, attr, v.to(device=DEVICE, dtype=torch.float64))

    X_test = data.testX.to(DEVICE)
    Y_star = data.testY.to(DEVICE)  # ground-truth optimal
    print(f"X_test.shape={X_test.shape}  Y_star.shape={Y_star.shape}", flush=True)

    # Load pretrained NN solver
    model_path = ROOT / "models/qp/QP_Problem-100-50-10000-10000/NN_Eq/solver_net.pth"
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(model, dict):
        # state dict; we need the class — not available, so skip
        print("ERROR: solver_net.pth is a state dict; need class", flush=True)
        return
    model.eval().to(DEVICE).double()
    print(f"loaded solver_net: {type(model).__name__}", flush=True)

    # NN raw partial prediction
    with torch.no_grad():
        Z_raw = model(X_test)  # (B, partial_dim) probably in [0,1] sigmoid output
        Z_scaled = data.scale(X_test, Z_raw) if hasattr(data, "scale") else Z_raw
        Y_raw_full = data.complete_partial(X_test, Z_scaled)
    print(f"Y_raw_full.shape={Y_raw_full.shape}", flush=True)

    # Baseline metrics
    obj0, eq0, ineq0, m0 = metrics(data, X_test, Y_raw_full)
    feasibility_max = (eq0 + ineq0).max()
    print("\n--- raw NN output (no repair) ---", flush=True)
    print(f"  Obj={float(obj0.mean()):.4f}  EqL1={float(eq0.mean()):.3e}  IneqL1={float(ineq0.mean()):.3e}  Merit={float(m0.mean()):.4f}", flush=True)
    print(f"  Opt ref: Obj_star={float(data.obj_fn(Y_star).mean()):.4f}", flush=True)

    # ---- Interior point: Chebyshev center per sample via cvxpy LP ----
    import cvxpy as cp
    feas_eps_id = 1e-5
    Y_pen0 = data.check_feasibility(X_test, Y_raw_full).abs().max(dim=1)[0]
    infeas_idx0 = Y_pen0 > feas_eps_id
    n_infs = int(infeas_idx0.sum())
    print(f"  Computing Chebyshev-center IP for {n_infs} infeasible samples ...", flush=True)
    Q_np = data.Q_np
    p_np = data.p_np
    A_np = data.A_np
    G_np = data.G_np
    h_np = data.h_np
    L_np = data.L_np
    U_np = data.U_np
    X_inf_np = X_test[infeas_idx0].cpu().numpy()
    ydim = data.ydim
    g_norm = (G_np ** 2).sum(axis=1) ** 0.5  # (nineq,)
    ip_inf_np = []
    t0 = time.time()
    n_failed = 0
    for i in range(n_infs):
        y = cp.Variable(ydim)
        # Minimise 0 (pure feasibility) → returns "a" feasible point (likely at LP vertex).
        # Using SCS solver because CLARABEL fails on this formulation in our env.
        constraints = [G_np @ y <= h_np, y <= U_np, y >= L_np, A_np @ y == X_inf_np[i]]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            ip_inf_np.append(y.value if y.value is not None else None)
        except Exception:
            ip_inf_np.append(None)
        if ip_inf_np[-1] is None:
            n_failed += 1
            ip_inf_np[-1] = Y_star[infeas_idx0][i].cpu().numpy()  # fallback
    print(f"  cvxpy feas-solve wall = {time.time()-t0:.1f}s, n_failed_fallback_to_y_star={n_failed}", flush=True)
    import numpy as _np
    ip_inf_np = torch.tensor(_np.array(ip_inf_np), device=DEVICE, dtype=torch.float64)
    ip_full = Y_star.clone()
    ip_full[infeas_idx0] = ip_inf_np
    o_ip, eq_ip, ineq_ip, m_ip = metrics(data, X_test, ip_full)
    print(f"  IP (cvxpy feas-solve): max EqL1={float(eq_ip.max()):.3e}  max IneqL1={float(ineq_ip.max()):.3e}", flush=True)
    diff = (ip_inf_np - Y_star[infeas_idx0]).norm(dim=1)
    print(f"  ||IP - Y_star||  mean={float(diff.mean()):.4f}  max={float(diff.max()):.4f}", flush=True)
    print(f"  IP obj = {float(data.obj_fn(ip_inf_np).mean()):.4f}  (Y_star_inf obj = {float(data.obj_fn(Y_star[infeas_idx0]).mean()):.4f})", flush=True)

    # ---- Extract partial coords ----
    feas_eps = 1e-5
    Y_pen = data.check_feasibility(X_test, Y_raw_full).abs().max(dim=1)[0]
    infeas_idx = Y_pen > feas_eps
    n_infeas = int(infeas_idx.sum())
    print(f"  Infeasible samples (max violation > {feas_eps}): {n_infeas}/{X_test.shape[0]}", flush=True)
    if n_infeas == 0:
        return

    X_inf = X_test[infeas_idx]
    # Z_scaled is (B, partial_dim) — NN output already in partial space (sigmoid * (U-L) + L).
    z_raw_partial = Z_scaled[infeas_idx]                   # (B_inf, partial_dim)
    ip_partial = ip_inf_np[:, data.partial_vars_idx]       # (B_inf, partial_dim) — IP in partial coords

    # ---- Baseline K=1 ----
    y_bp1, _ = ip_bisect_partial(data, X_inf, z_raw_partial, ip_partial, n_steps=30, bis_step=0.9, feas_eps=feas_eps)
    Y_final = Y_raw_full.clone()
    Y_final[infeas_idx] = y_bp1
    o_a, e_a, i_a, m_a = metrics(data, X_test, Y_final)
    print(f"\n--- BP K=1 baseline (partial-space bisect, n_steps=30) ---", flush=True)
    print(f"  Merit_full={float(m_a.mean()):.6f}  Obj_full={float(o_a.mean()):.6f}  IneqL1_full={float(i_a.mean()):.3e}", flush=True)

    # ---- K perturbation sweep (partial-space perturbations) ----
    print("\n--- K-perturbation sweep (best_merit on infeas subset) ---", flush=True)
    print(f"{'K':>4} {'eps':>6} | {'Obj_inf':>11} {'IneqL1_inf':>12} {'Merit_inf':>14} | {'Merit_full':>14}", flush=True)
    torch.manual_seed(7)
    out_scale = z_raw_partial.std(dim=0).mean()
    print(f"  (partial-space output scale = {float(out_scale):.4f}, n_infeas={n_infeas})", flush=True)
    for K in [1, 5, 20, 100]:
        for eps in [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]:
            if K == 1 and eps != 0.0:
                continue
            B = n_infeas
            noise = eps * out_scale * torch.randn(K, B, z_raw_partial.shape[1], device=DEVICE, dtype=torch.float64)
            if K == 1:
                noise = torch.zeros_like(noise)
            z_pert = z_raw_partial.unsqueeze(0) + noise  # (K, B, p)
            ip_k = ip_partial.unsqueeze(0).expand(K, B, -1).contiguous()
            X_rep = X_inf.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            z_flat = z_pert.reshape(K * B, -1)
            ip_flat = ip_k.reshape(K * B, -1)
            with torch.no_grad():
                y_full_flat, _ = ip_bisect_partial(data, X_rep, z_flat, ip_flat, n_steps=30, bis_step=0.9, feas_eps=feas_eps)
            obj_k = data.obj_fn(y_full_flat).reshape(K, B)
            eq_k = data.eq_resid(X_rep, y_full_flat).abs().sum(dim=1).reshape(K, B)
            ineq_k = data.ineq_resid(X_rep, y_full_flat).abs().sum(dim=1).reshape(K, B)
            m_k = merit(obj_k, eq_k, ineq_k)
            best_idx = m_k.argmin(dim=0)
            arange_b = torch.arange(B, device=DEVICE)
            picked_y = y_full_flat.reshape(K, B, -1)[best_idx, arange_b]
            picked_obj = obj_k[best_idx, arange_b]
            picked_ineq = ineq_k[best_idx, arange_b]
            picked_m = m_k[best_idx, arange_b]
            Y_final = Y_raw_full.clone()
            Y_final[infeas_idx] = picked_y
            o_full, e_full, i_full, mm_full = metrics(data, X_test, Y_final)
            print(
                f"{K:>4} {eps:>6.2f} | "
                f"{float(picked_obj.mean()):11.6f} {float(picked_ineq.mean()):12.3e} "
                f"{float(picked_m.mean()):14.6f} | {float(mm_full.mean()):14.6f}",
                flush=True,
            )

    # ---- Budget sweep at K=100 ε=0.10 ----
    print("\n--- Bisection budget sweep (K=100 ε=0.10) ---", flush=True)
    print(f"{'n_steps':>8} | {'Obj_inf':>11} {'IneqL1_inf':>12} {'Merit_inf':>14}", flush=True)
    torch.manual_seed(7)
    K, eps = 100, 0.10
    B = n_infeas
    noise = eps * out_scale * torch.randn(K, B, z_raw_partial.shape[1], device=DEVICE, dtype=torch.float64)
    z_pert = z_raw_partial.unsqueeze(0) + noise
    ip_k = ip_partial.unsqueeze(0).expand(K, B, -1).contiguous()
    X_rep = X_inf.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
    z_flat = z_pert.reshape(K * B, -1)
    ip_flat = ip_k.reshape(K * B, -1)
    for n_steps in [3, 5, 10, 30, 100, 300]:
        with torch.no_grad():
            y_full_flat, _ = ip_bisect_partial(data, X_rep, z_flat, ip_flat, n_steps=n_steps, bis_step=0.9, feas_eps=feas_eps)
        obj_k = data.obj_fn(y_full_flat).reshape(K, B)
        eq_k = data.eq_resid(X_rep, y_full_flat).abs().sum(dim=1).reshape(K, B)
        ineq_k = data.ineq_resid(X_rep, y_full_flat).abs().sum(dim=1).reshape(K, B)
        m_k = merit(obj_k, eq_k, ineq_k)
        best_idx = m_k.argmin(dim=0)
        arange_b = torch.arange(B, device=DEVICE)
        picked_obj = obj_k[best_idx, arange_b]
        picked_ineq = ineq_k[best_idx, arange_b]
        picked_m = m_k[best_idx, arange_b]
        print(f"{n_steps:>8} | {float(picked_obj.mean()):11.6f} {float(picked_ineq.mean()):12.3e} {float(picked_m.mean()):14.6f}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
