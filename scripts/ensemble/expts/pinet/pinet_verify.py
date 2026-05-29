"""Verify our FSNet/DC3 findings on PI-Net.

We test two claims:
(1) Pre-repair multi-start (K perturbations of the trunk output) reduces merit
    when the repair operator is under-converged.
(2) Increasing the repair budget collapses the perturbation gain.

PI-Net's repair = ADMM orthogonal projection onto the constraint set.
Pre-repair point = trunk MLP output before `self.project_test`.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import train_state

ROOT = pathlib.Path(__file__).resolve().parents[3].parent / "third-party" / "pinet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.QP.load_qp import load_data  # noqa: E402
from benchmarks.QP.run_qp import setup_model  # noqa: E402
from benchmarks.model import (  # noqa: E402
    HardConstrainedMLP,
    setup_pinet,
)
from pinet import (  # noqa: E402
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
)

ID_PATH = ROOT / "src/benchmarks/QP/ids/dc3_nonconvex_1.yaml"
CFG_PATH = ROOT / "src/benchmarks/configs/benchmark_config_default.yaml"


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_raw_model(hyperparameters, eq_constraint, ineq_constraint, box_constraint, dim):
    """Construct an HCNN identical to the trained one but with `raw_test=True`.

    Same params tree (only Dense layers carry weights), so we can apply the
    trained state.params and get the pre-projection trunk output.
    """
    project, project_test, _ = setup_pinet(
        hyperparameters=hyperparameters,
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=box_constraint,
        setup_reps=-1,
    )
    return HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=dim,
        features_list=hyperparameters["features_list"],
        activation=jax.nn.relu,
        raw_train=False,
        raw_test=True,
    ), project_test


def merit(obj, eq_v, ineq_v, rho=1e6):
    return obj + rho * (eq_v + ineq_v)


def eval_perturbed(
    project_test,
    raw_pred,
    b,
    a_mat,
    g_mat,
    h,
    batched_objective,
    K,
    eps,
    rng,
):
    """Apply K perturbations to raw_pred, project each, return per-sample best-merit metrics.

    raw_pred: (B, dim)
    b:        (B, dim_b, 1)  (here dim_b == dim_eq input)
    """
    B, dim = raw_pred.shape
    if K == 1 and eps == 0.0:
        x_in = raw_pred
        b_in = b
    else:
        noise = eps * jax.random.normal(rng, shape=(K, B, dim))
        if K == 1:
            noise = jnp.zeros_like(noise)
        x_in = (raw_pred[None] + noise).reshape(K * B, dim)  # (K*B, dim)
        b_in = jnp.broadcast_to(b[None], (K, B, b.shape[1], 1)).reshape(K * B, b.shape[1], 1)
    proj = project_test(x_in, b_in)  # (K*B, dim)
    obj = batched_objective(proj).reshape(K if K > 1 else 1, -1) if K > 1 else batched_objective(proj)[None]
    # constraints
    eq_v = jnp.max(jnp.abs(a_mat[0] @ proj[..., None] - b_in), axis=1)[..., 0]
    ineq_v = jnp.max(jnp.maximum(g_mat[0] @ proj[..., None] - h, 0), axis=1)[..., 0]
    if K > 1:
        obj = obj.reshape(K, B)
        eq_v = eq_v.reshape(K, B)
        ineq_v = ineq_v.reshape(K, B)
    else:
        obj = obj.reshape(1, B)
        eq_v = eq_v.reshape(1, B)
        ineq_v = ineq_v.reshape(1, B)
    m = merit(obj, eq_v, ineq_v)  # (K, B)
    best_idx = jnp.argmin(m, axis=0)  # (B,)
    pick = lambda t: t[best_idx, jnp.arange(B)]
    return pick(obj), pick(eq_v), pick(ineq_v), pick(m)


def main(n_epochs: int, seed: int = 42):
    print(f"=== PI-Net verification: DC3 nonconvex QP100, seed={seed} ===", flush=True)
    print(f"jax devices: {jax.devices()}", flush=True)
    dataset = load_cfg(ID_PATH)
    hp = load_cfg(CFG_PATH)
    hp["n_epochs"] = n_epochs

    import torch
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    loader_key, mkey, key = jax.random.split(key, 3)

    print("Loading data ...", flush=True)
    (
        a_mat, g_mat, h, x_data, batched_objective,
        train_loader, valid_loader, test_loader, batched_loss,
    ) = load_data(
        use_dc3_dataset=dataset["use_DC3_dataset"],
        use_convex=dataset["use_convex"],
        problem_seed=dataset["problem_seed"],
        problem_var=dataset["problem_var"],
        problem_nineq=dataset["problem_nineq"],
        problem_neq=dataset["problem_neq"],
        problem_examples=dataset["problem_examples"],
        rng_key=loader_key,
        batch_size=hp.get("batch_size", 2048),
        use_jax_loader=True,
        penalty=hp.get("penalty", 0.0),
    )
    print(f"  a_mat {a_mat.shape}  g_mat {g_mat.shape}  h {h.shape}  x_data {x_data.shape}", flush=True)

    print("Setting up model ...", flush=True)
    model, params, setup_time, train_step = setup_model(
        rng_key=mkey,
        hyperparameters=hp,
        proj_method="pinet",
        a_mat=a_mat,
        x_data=x_data,
        g_mat=g_mat,
        h=h,
        batched_loss=batched_loss,
    )
    tx = optax.adam(hp["learning_rate"])
    state = train_state.TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)

    print(f"Training for {n_epochs} epochs ...", flush=True)
    t0 = time.time()
    for ep in range(n_epochs):
        ep_loss = 0.0
        ep_n = 0
        for batch in train_loader:
            x_batch, _ = batch
            loss, state = train_step(state, x_batch[:, :, 0], x_batch)
            ep_loss += float(loss) * x_batch.shape[0]
            ep_n += x_batch.shape[0]
        if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}/{n_epochs}  loss={ep_loss/ep_n:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"Training done in {time.time()-t0:.1f}s", flush=True)

    # ----- Eval baseline (default n_iter_test=100, no perturb) -----
    print("\n--- baseline (K=0, no perturb) ---", flush=True)
    x_test, obj_ref = next(iter(test_loader))
    B = x_test.shape[0]
    pred = state.apply_fn({"params": state.params}, x=x_test[:, :, 0], b=x_test, test=True)
    obj0 = batched_objective(pred)
    eq0 = jnp.max(jnp.abs(a_mat[0] @ pred[..., None] - x_test), axis=1)[..., 0]
    ineq0 = jnp.max(jnp.maximum(g_mat[0] @ pred[..., None] - h, 0), axis=1)[..., 0]
    m0 = merit(obj0, eq0, ineq0)
    print(f"  Obj={float(obj0.mean()):.4f}  EqMax={float(eq0.mean()):.3e}  IneqMax={float(ineq0.mean()):.3e}  Merit={float(m0.mean()):.4f}", flush=True)
    print(f"  Opt ref Obj={float(obj_ref.mean()):.4f}", flush=True)

    # ----- Build raw-only model to get pre-projection output -----
    print("\nBuilding raw_test=True overlay for perturbation eval ...", flush=True)
    # Get constraint objects matching what setup_model wired (look at QP/load_qp.py)
    # The HCNN was wired with eq=Ax=b, ineq=Gx<=h.
    eq_c = EqualityConstraint(a_mat=a_mat, b=x_test, method=None, var_b=True)
    ineq_c = AffineInequalityConstraint(c_mat=g_mat, ub=h, lb=-jnp.inf * jnp.ones_like(h))
    dim = a_mat.shape[2]
    raw_model, _ = build_raw_model(hp, eq_c, ineq_c, None, dim)
    raw_pred = raw_model.apply({"params": state.params}, x=x_test[:, :, 0], b=x_test, test=True)
    print(f"  raw_pred.shape={raw_pred.shape}", flush=True)

    # JIT a project_test variant per n_iter_test
    def make_project(n_iter):
        hp_local = dict(hp)
        hp_local["n_iter_test"] = n_iter
        _, project_test, _ = setup_pinet(
            hyperparameters=hp_local,
            eq_constraint=eq_c,
            ineq_constraint=ineq_c,
            box_constraint=None,
            setup_reps=-1,
        )
        return jax.jit(project_test)

    print("\n=== Perturbation × budget sweep ===", flush=True)
    print(f"{'n_iter':>6} {'K':>3} {'eps':>6} | {'Obj':>10} {'EqMax':>10} {'IneqMax':>10} {'Merit':>14}", flush=True)
    rng = jax.random.PRNGKey(7)
    for n_iter in [10, 50, 100, 500, 1000, 3000]:
        proj_fn = make_project(n_iter)
        for K in [1, 5, 20, 100]:
            for eps in [0.0, 0.01, 0.05, 0.1]:
                if K == 1 and eps != 0.0:
                    continue
                rng, sub = jax.random.split(rng)
                obj, eq, ineq, mer = eval_perturbed(
                    proj_fn, raw_pred, x_test, a_mat, g_mat, h, batched_objective,
                    K=K, eps=eps, rng=sub,
                )
                jax.block_until_ready(mer)
                print(f"{n_iter:>6} {K:>3} {eps:>6.2f} | {float(obj.mean()):10.4f} {float(eq.mean()):10.3e} {float(ineq.mean()):10.3e} {float(mer.mean()):14.4f}", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.n_epochs, args.seed)
