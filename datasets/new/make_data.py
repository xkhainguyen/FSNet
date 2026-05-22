import os

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

import json
import time
import pickle
import numpy as np
import casadi as ca

# ============================================================
# CONFIG
# ============================================================

num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 10
seed = 2025

# structured difficulty knobs
num_modes = 4
conditioning_alpha = 6.0
multistart_k = 3

MAX_CPU_TIME = 11.0

print("=" * 80)
print("Structured Multimodal Nonsmooth Nonconvex SOCP Benchmark")
print("=" * 80)

np.random.seed(seed)

# ============================================================
# CONDITIONED EQUALITY MATRIX
# ============================================================

U_rand, _ = np.linalg.qr(np.random.randn(num_eq, num_eq))
V_rand, _ = np.linalg.qr(np.random.randn(num_var, num_var))

singular_vals = np.exp(
    -conditioning_alpha * np.linspace(0, 1, num_eq)
)

Sigma = np.zeros((num_eq, num_var))
Sigma[:num_eq, :num_eq] = np.diag(singular_vals)

A = U_rand @ Sigma @ V_rand

A_cond = singular_vals.max() / singular_vals.min()

print(f"Condition number of equality manifold: {A_cond:.3e}")

# ============================================================
# MODES
# ============================================================

mode_centers = np.random.uniform(-1, 1, size=(num_modes, num_eq))

X = []
mode_ids = []

for n in range(num_examples):
    mode = np.random.randint(num_modes)

    Xi = (
        mode_centers[mode]
        + 0.20 * np.random.randn(num_eq)
    )

    X.append(Xi)
    mode_ids.append(mode)

X = np.array(X)
mode_ids = np.array(mode_ids)

XL = X.min(axis=0)
XU = X.max(axis=0)

# ============================================================
# VARIABLES / OBJECTIVE
# ============================================================

Q = np.diag(0.1 + np.random.rand(num_var))

p = np.random.uniform(-1, 1, num_var)

L = np.ones(num_var) * -5
U = np.ones(num_var) * 5

# multimodal frequencies
freqs = np.random.randint(2, 8, size=num_var)

# ============================================================
# CONSTRAINTS
# ============================================================

G, h, C, d = [], [], [], []

anchor_points = [
    np.random.uniform(-2, 2, size=num_var)
    for _ in range(num_modes)
]

for i in range(num_ineq):

    Gi = np.random.uniform(
        -1,
        1,
        size=(num_ineq, num_var),
    )

    hi = np.random.uniform(
        -1,
        1,
        size=(num_ineq),
    )

    Ci = np.random.uniform(
        -1,
        1,
        size=(num_var),
    )

    anchor = anchor_points[i % num_modes]

    di = (
        np.linalg.norm(
            Gi @ np.cos(anchor) + hi,
            2,
        )
        - Ci.T @ anchor
        + np.random.uniform(0.0, 0.5)
    )

    G.append(Gi)
    h.append(hi)
    C.append(Ci)
    d.append(di)

G = np.array(G)
h = np.array(h)
C = np.array(C)
d = np.array(d)

# ============================================================
# CASADI MODEL
# ============================================================

y = ca.MX.sym("y", num_var)
t = ca.MX.sym("t")
Xi = ca.MX.sym("Xi", num_eq)

# highly multimodal objective
obj_func = (
    0.5 * ca.mtimes(y.T, ca.mtimes(Q, y))
    + ca.dot(p, ca.sin(y))
    + 0.03 * ca.sum1(ca.cos(freqs * y))
    + 0.01 * ca.sum1(ca.sin(5 * y))
    + 0.1 * t
)

eq_constraints = A @ y - Xi

soc_constraint = ca.dot(y, y) - t**2

ineq_constraints = []

for i in range(num_ineq):

    val = (
        ca.norm_2(
            G[i] @ ca.cos(y) + h[i]
        )
        - (
            ca.dot(C[i], y) + d[i]
        )
    )

    ineq_constraints.append(val)

ineq_constraints.append(soc_constraint)

g_all = ca.vertcat(
    eq_constraints,
    *ineq_constraints,
)

nlp = {
    "x": ca.vertcat(y, t),
    "f": obj_func,
    "g": g_all,
    "p": Xi,
}

# ============================================================
# SOLVER
# ============================================================

opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
    # "ipopt.max_cpu_time": MAX_CPU_TIME,
}

solver = ca.nlpsol(
    "solver",
    "ipopt",
    nlp,
    opts,
)

lbg = np.concatenate([
    np.zeros(num_eq),
    -np.inf * np.ones(num_ineq + 1),
])

ubg = np.concatenate([
    np.zeros(num_eq),
    np.zeros(num_ineq + 1),
])

lbx = np.concatenate([L, [0.0]])
ubx = np.concatenate([U, [np.inf]])

# ============================================================
# DATA LOGGING
# ============================================================

data = {

    # benchmark metadata
    "benchmark_name":
        "structured_multimodal_socp",

    "version":
        "v2",

    "seed":
        seed,

    "num_modes":
        num_modes,

    "conditioning_alpha":
        conditioning_alpha,

    "A_condition_number":
        float(A_cond),

    # problem
    "Q": Q,
    "p": p,
    "A": A,
    "G": G,
    "h": h,
    "C": C,
    "d": d,

    # dataset
    "X": X,
    "mode_ids": mode_ids,

    # bounds
    "YL": L,
    "YU": U,
    "XL": XL,
    "XU": XU,

    # solutions
    "Y": [],

    # solver stats
    "solve_time_sec": [],
    "iter_count": [],
    "return_status": [],
    "success": [],

    # quality
    "obj_value": [],
    "eq_l2": [],
    "ineq_l2": [],
    "ineq_max": [],

    # geometry
    "active_constraints": [],
    "num_active_constraints": [],

    # multistart
    "multistart_obj_std": [],
    "multistart_solution_std": [],
    "best_restart_idx": [],

    # stationarity proxy
    "grad_norm": [],
}

# ============================================================
# MULTISTART SOLVE LOOP
# ============================================================

for n in range(num_examples):

    Xi_n = X[n]

    best_obj = np.inf
    best_res = None
    best_idx = -1

    all_objs = []
    all_solutions = []

    for restart in range(multistart_k):

        init_y = np.random.uniform(
            -2,
            2,
            size=num_var,
        )

        init_t = np.linalg.norm(init_y)

        x0 = np.concatenate([
            init_y,
            [init_t],
        ])

        start = time.time()

        try:

            res = solver(
                x0=x0,
                p=Xi_n,
                lbg=lbg,
                ubg=ubg,
                lbx=lbx,
                ubx=ubx,
            )

            wall = time.time() - start

            objv = float(
                res["f"].full().item()
            )

            sol = res["x"].full().flatten()

            all_objs.append(objv)
            all_solutions.append(sol[:-1])

            if objv < best_obj:
                best_obj = objv
                best_res = res
                best_idx = restart
                best_wall = wall

        except Exception:

            all_objs.append(np.nan)
            all_solutions.append(
                np.full(num_var, np.nan)
            )

    # --------------------------------------------------------
    # failed all restarts
    # --------------------------------------------------------

    if best_res is None:

        print(f"[{n}] ALL RESTARTS FAILED")

        data["Y"].append(
            np.full(num_var, np.nan)
        )

        data["solve_time_sec"].append(np.nan)
        data["iter_count"].append(-1)
        data["return_status"].append(
            "FAILED"
        )
        data["success"].append(False)

        data["obj_value"].append(np.nan)
        data["eq_l2"].append(np.nan)
        data["ineq_l2"].append(np.nan)
        data["ineq_max"].append(np.nan)

        data["active_constraints"].append(
            None
        )
        data["num_active_constraints"].append(
            np.nan
        )

        data["multistart_obj_std"].append(
            np.nan
        )
        data["multistart_solution_std"].append(
            np.nan
        )

        data["best_restart_idx"].append(-1)

        data["grad_norm"].append(np.nan)

        continue

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    st = solver.stats()

    success = bool(
        st.get("success", False)
    )

    status = st.get(
        "return_status",
        "",
    )

    iters = int(
        st.get("iter_count", -1)
    )

    sol_x = best_res["x"].full().flatten()

    y_sol = sol_x[:-1]

    data["Y"].append(y_sol)

    data["solve_time_sec"].append(
        float(best_wall)
    )

    data["iter_count"].append(iters)

    data["return_status"].append(status)

    data["success"].append(success)

    data["obj_value"].append(best_obj)

    # --------------------------------------------------------
    # feasibility
    # --------------------------------------------------------

    g_val = best_res["g"].full().flatten()

    eq_val = g_val[:num_eq]

    ineq_val = g_val[num_eq:]

    eq_l2 = float(
        np.linalg.norm(eq_val, 2)
    )

    ineq_pos = np.maximum(
        ineq_val,
        0.0,
    )

    ineq_l2 = float(
        np.linalg.norm(ineq_pos, 2)
    )

    ineq_max = float(
        np.max(ineq_pos)
    )

    data["eq_l2"].append(eq_l2)

    data["ineq_l2"].append(ineq_l2)

    data["ineq_max"].append(ineq_max)

    # --------------------------------------------------------
    # active set logging
    # --------------------------------------------------------

    active_mask = (
        np.abs(ineq_val) < 1e-4
    )

    data["active_constraints"].append(
        active_mask.astype(np.int8)
    )

    data["num_active_constraints"].append(
        int(active_mask.sum())
    )

    # --------------------------------------------------------
    # multistart geometry
    # --------------------------------------------------------

    finite_objs = np.array([
        x for x in all_objs
        if np.isfinite(x)
    ])

    if len(finite_objs) > 0:

        obj_std = float(
            np.std(finite_objs)
        )

    else:
        obj_std = np.nan

    sols = np.array(all_solutions)

    if np.any(np.isfinite(sols)):

        sol_std = float(
            np.nanstd(sols)
        )

    else:
        sol_std = np.nan

    data["multistart_obj_std"].append(
        obj_std
    )

    data["multistart_solution_std"].append(
        sol_std
    )

    data["best_restart_idx"].append(
        best_idx
    )

    # --------------------------------------------------------
    # stationarity proxy
    # --------------------------------------------------------

    grad_func = ca.Function(
        "grad_f",
        [ca.vertcat(y, t)],
        [
            ca.gradient(
                obj_func,
                ca.vertcat(y, t),
            )
        ],
    )

    grad_val = grad_func(
        sol_x
    ).full().flatten()

    grad_norm = float(
        np.linalg.norm(
            grad_val,
            2,
        )
    )

    data["grad_norm"].append(
        grad_norm
    )

    # --------------------------------------------------------
    # logging
    # --------------------------------------------------------

    print(
        f"[{n:05d}] "
        f"mode={mode_ids[n]} "
        f"obj={best_obj:.4f} "
        f"eq={eq_l2:.2e} "
        f"ineq={ineq_max:.2e} "
        f"active={active_mask.sum()} "
        f"obj_std={obj_std:.3e}"
    )

# ============================================================
# STRUCTURED SPLITS
# ============================================================

train_mask = mode_ids < (num_modes - 1)

ood_mask = mode_ids == (num_modes - 1)

data["split_train_idx"] = np.where(
    train_mask
)[0]

data["split_ood_mode_idx"] = np.where(
    ood_mask
)[0]

# ============================================================
# BEST PARTIAL
# ============================================================

det_best = -np.inf
best_partial = None

for i in range(1000):

    np.random.seed(i)

    partial_vars = np.random.choice(
        num_var,
        num_var - num_eq,
        replace=False,
    )

    other_vars = np.setdiff1d(
        np.arange(num_var),
        partial_vars,
    )

    _, det = np.linalg.slogdet(
        A[:, other_vars]
    )

    if det > det_best:

        det_best = det
        best_partial = partial_vars

print("best_det", det_best)

data["best_partial"] = best_partial

# ============================================================
# SAVE
# ============================================================

out_dir = (
    "datasets/structured_multimodal_socp"
)

os.makedirs(out_dir, exist_ok=True)

out_name = (
    f"seed{seed}_"
    f"var{num_var}_"
    f"ineq{num_ineq}_"
    f"eq{num_eq}_"
    f"modes{num_modes}_"
    f"ex{num_examples}"
)

pickle_path = os.path.join(
    out_dir,
    out_name + ".pkl",
)

json_path = os.path.join(
    out_dir,
    out_name + "_meta.json",
)

with open(pickle_path, "wb") as f:
    pickle.dump(data, f)

meta = {
    "benchmark_name":
        data["benchmark_name"],

    "version":
        data["version"],

    "seed":
        seed,

    "num_examples":
        num_examples,

    "num_var":
        num_var,

    "num_eq":
        num_eq,

    "num_ineq":
        num_ineq,

    "num_modes":
        num_modes,

    "conditioning_alpha":
        conditioning_alpha,

    "A_condition_number":
        float(A_cond),
}

with open(json_path, "w") as f:
    json.dump(
        meta,
        f,
        indent=2,
    )

print("=" * 80)
print("Finished generating benchmark!")
print("pickle:", pickle_path)
print("meta:", json_path)
print("=" * 80)