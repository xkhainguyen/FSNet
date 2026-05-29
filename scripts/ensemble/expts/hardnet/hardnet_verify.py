"""Verify our perturbation finding on HardNet (Min & Azizan, NeurIPS 2024).

HardNet-Aff appends a CLOSED-FORM (one-shot) projection layer to a vanilla MLP:

    proj(f, x) = f + pinv(A) @ (ReLU(bl - Af) - ReLU(Af - bu))

This is not iterative — no convergence loop, no tolerance, no max_iter. Theory
predicts: zero "under-convergence" perturbation gain. The projection itself is
direction-dependent (different f → different correction), so some gain is
geometrically possible — similar to BP's radial story.

Problem: HardNet's `opt` dataset = DC3 nonconvex QP100 (same family as our other
verification problems).
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path("/orcd/scratch/orcd/008/khain/FSNet/third-party/hardnet")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets/opt"))
os.chdir(ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

from hardnet_aff import HardNetAff  # noqa: E402


def merit(obj, eq, ineq, rho=1e6):
    return obj + rho * (eq + ineq)


def metrics(data, X, Y):
    obj = data.evaluate(X, Y)
    ineq = torch.clamp(Y @ data.A.T - data.b, 0).sum(dim=1)
    eq = torch.abs(Y @ data.C.T - X).sum(dim=1)
    return obj, eq, ineq, merit(obj, eq, ineq)


def main(epochs: int, seed: int, convex_obj: bool = False):
    tag = "CONVEX-obj (control)" if convex_obj else "nonconvex-obj"
    print(f"=== HardNet verification [{tag}]: DC3 QP100, seed={seed}, epochs={epochs} ===", flush=True)
    print(f"device: {DEVICE}", flush=True)

    # ----- Load dataset -----
    nEx = 10000
    fpath = ROOT / f"datasets/opt/opt_dataset_ex{nEx}"
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    for attr in dir(data):
        v = getattr(data, attr)
        if not callable(v) and not attr.startswith("__") and torch.is_tensor(v):
            try:
                setattr(data, attr, v.to(DEVICE))
            except AttributeError:
                pass
    data._device = DEVICE
    print(f"data: {data}, ydim={data.ydim}, neq={data.neq}, A.shape={data.A.shape}, C.shape={data.C.shape}", flush=True)
    print(f"trainX={data.trainX.shape}, validX={data.validX.shape}, testX={data.testX.shape}", flush=True)

    # ----- Control (1)/(2): optionally drop the nonconvex sin term -----
    # Original: evaluate(X,Y) = (0.5*(Y@Q)*Y + p*sin(Y)).sum(1)  -> nonconvex via sin
    # Convex:   evaluate(X,Y) = (0.5*(Y@Q)*Y + p*Y).sum(1)       -> convex (Q is PSD)
    if convex_obj:
        _Q, _p = data.Q, data.p
        def _convex_evaluate(X, Y):
            return (0.5 * (Y @ _Q) * Y + _p * Y).sum(dim=1)
        data.evaluate = _convex_evaluate  # used by get_train_loss AND our metrics()
        # sanity: confirm patch is live
        _yt = data.trainY[:2]
        print(f"  [convex control] patched data.evaluate; sample obj={float(_convex_evaluate(data.trainX[:2], _yt).mean()):.4f}", flush=True)

    # ----- Build args (mirroring default_args for `opt` + hardnet_aff) -----
    args = {
        "probType": "opt",
        "nEx": nEx,
        "epochs": epochs,
        "batchSize": 200,
        "lr": 1e-4,
        "hiddenSize": 200,
        "softWeight": 10,
        "softEpochs": max(1, epochs // 4),  # 25% warmup with no projection
        "seed": seed,
        "evalFreq": max(1, epochs // 5),
        "saveAllStats": False,
        "resultsSaveFreq": 100,
        "testProj": "none",
        "suffix": "",
    }

    # ----- Train -----
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = HardNetAff(data, args).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args["lr"])
    train_X = data.trainX
    train_Y = data.trainY
    n_train = train_X.shape[0]
    bs = args["batchSize"]
    print(f"Training {epochs} epochs (softEpochs={args['softEpochs']}) ...", flush=True)
    t0 = time.time()
    for epoch in range(epochs):
        if epoch < args["softEpochs"]:
            net.set_projection(False)
        else:
            net.set_projection(True)
        # Shuffle
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        n_seen = 0
        net.train()
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            xb = train_X[idx]
            yb = train_Y[idx]
            opt.zero_grad()
            loss = data.get_train_loss(net, xb, yb, args).sum()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
            n_seen += xb.shape[0]
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            net.eval()
            with torch.no_grad():
                Y_val = net(data.validX, isTest=True)
                obj_val, eq_val, ineq_val, m_val = metrics(data, data.validX, Y_val)
            print(f"  epoch {epoch+1:3d}/{epochs} loss={epoch_loss/n_seen:.4f} | valid Obj={float(obj_val.mean()):.4f} EqL1={float(eq_val.mean()):.3e} IneqL1={float(ineq_val.mean()):.3e}", flush=True)
    print(f"Training done in {time.time()-t0:.1f}s", flush=True)

    # ----- Eval baseline (with projection ON) -----
    net.eval()
    net.set_projection(True)
    Xt = data.testX
    with torch.no_grad():
        Y_proj = net(Xt, isTest=True)
    obj0, eq0, ineq0, m0 = metrics(data, Xt, Y_proj)
    print(f"\n--- Baseline (projection ON, K=1) ---", flush=True)
    print(f"  Obj={float(obj0.mean()):.4f}  EqL1={float(eq0.mean()):.3e}  IneqL1={float(ineq0.mean()):.3e}  Merit={float(m0.mean()):.4f}", flush=True)

    # Optimum reference (if available)
    if hasattr(data, "testY") and data.testY.shape[1] == data.ydim and data.testY.abs().sum() > 0:
        obj_ref = data.evaluate(Xt, data.testY)
        print(f"  Opt ref Obj = {float(obj_ref.mean()):.4f}", flush=True)

    # ----- Raw NN output (no projection) for perturbation -----
    net.set_projection(False)
    with torch.no_grad():
        F_raw = net(Xt, isTest=True)
    net.set_projection(True)
    print(f"  Raw NN F.shape={F_raw.shape}, raw IneqL1 max = {float(torch.clamp(F_raw @ data.A.T - data.b, 0).sum(dim=1).max()):.3e}", flush=True)

    # ----- K-perturbation sweep -----
    print(f"\n--- K-perturbation sweep (projection applied to each perturbed candidate) ---", flush=True)
    print(f"{'K':>4} {'eps':>6} | {'Obj':>10} {'EqL1':>10} {'IneqL1':>12} {'Merit':>14}", flush=True)
    torch.manual_seed(7)
    out_scale = F_raw.std(dim=0).mean()
    print(f"  output scale = {float(out_scale):.4f}", flush=True)
    B = Xt.shape[0]
    for K in [1, 5, 20, 100]:
        for eps in [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]:
            if K == 1 and eps != 0.0:
                continue
            noise = eps * out_scale * torch.randn(K, B, F_raw.shape[1], device=DEVICE, dtype=torch.float64)
            if K == 1:
                noise = torch.zeros_like(noise)
            f_k = F_raw.unsqueeze(0) + noise  # (K, B, ydim)
            f_flat = f_k.reshape(K * B, -1)
            X_rep = Xt.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1)
            # Apply HardNet projection inline
            with torch.no_grad():
                A_eff, bl, bu = data.get_coefficients(X_rep)
                Af = A_eff @ f_flat[:, :, None]
                # opt problem branch uses pinv
                y_proj_flat = f_flat + (torch.linalg.pinv(A_eff) @ (torch.relu(bl[:, :, None] - Af) - torch.relu(Af - bu[:, :, None])))[:, :, 0]
            obj_k = data.evaluate(X_rep, y_proj_flat).reshape(K, B)
            ineq_k = torch.clamp(y_proj_flat @ data.A.T - data.b, 0).sum(dim=1).reshape(K, B)
            eq_k = (y_proj_flat @ data.C.T - X_rep).abs().sum(dim=1).reshape(K, B)
            m_k = merit(obj_k, eq_k, ineq_k)
            best = m_k.argmin(dim=0)
            arange_b = torch.arange(B, device=DEVICE)
            print(
                f"{K:>4} {eps:>6.2f} | "
                f"{float(obj_k[best, arange_b].mean()):10.4f} "
                f"{float(eq_k[best, arange_b].mean()):10.3e} "
                f"{float(ineq_k[best, arange_b].mean()):12.3e} "
                f"{float(m_k[best, arange_b].mean()):14.4f}",
                flush=True,
            )

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convex_obj", action="store_true",
                   help="drop the nonconvex sin term -> convex objective (control 2)")
    args = p.parse_args()
    main(args.epochs, args.seed, convex_obj=args.convex_obj)
