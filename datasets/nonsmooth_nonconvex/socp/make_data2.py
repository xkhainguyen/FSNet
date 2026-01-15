import os

NUM_THREADS = "1" #64
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

# Optional but often helpful
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import pickle
import casadi as ca
import time

num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 8000
seed = 2025

print(f"Nonsmooth nonconvex SOCP problem with {num_var} variables, {num_ineq} inequalities, {num_eq} equalities and {num_examples} examples")
np.random.seed(seed)

Q = np.diag(np.random.rand(num_var) * 0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
XL = X.min(axis=0)
XU = X.max(axis=0)

L = np.ones(num_var) * -5
U = np.ones(num_var) * 5
x0 = np.random.uniform(-1, 1, size=(num_var))

G, h, C, d = [], [], [], []
for i in range(num_ineq):
    G.append(np.random.uniform(-1, 1, size=(num_ineq, num_var)))
    h.append(np.random.uniform(-1, 1, size=(num_ineq)))
    C.append(np.random.uniform(-1, 1, size=(num_var)))
    d.append(np.linalg.norm(G[i] @ x0 + h[i], 2) - C[i].T @ x0)

G = np.array(G)
h = np.array(h)
C = np.array(C)
d = np.array(d)

# -------------------------
# Build NLP ONCE with Xi as parameter
# -------------------------
y = ca.MX.sym("y_var", num_var)
t = ca.MX.sym("t_var")
Xi = ca.MX.sym("Xi", num_eq)  # parameter

obj_func = 0.5 * ca.mtimes(y.T, ca.mtimes(Q, y)) + ca.dot(p, ca.sin(y)) + 0.1 * t

eq_constraints = A @ y - Xi
soc = ca.dot(y, y) - t**2

ineq_constraints = []
for i in range(num_ineq):
    ineq_constraints.append(
        ca.norm_2(G[i] @ ca.cos(y) + h[i]) - (ca.dot(C[i], y) + d[i])
    )
ineq_constraints.append(soc)
ineq_constraints = ca.vertcat(*ineq_constraints)

g_all = ca.vertcat(eq_constraints, ineq_constraints)
nlp = {"x": ca.vertcat(y, t), "f": obj_func, "g": g_all, "p": Xi}

# -------------------------
# IPOPT options: time budget
# -------------------------
MAX_CPU_TIME = 11.0
opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
    # "ipopt.max_cpu_time": MAX_CPU_TIME,
}

solver = ca.nlpsol("solver", "ipopt", nlp, opts)
print("Max CPU time per example:", MAX_CPU_TIME)

# Bounds (constant across examples)
lbg = np.concatenate([np.zeros(num_eq), -np.inf * np.ones(num_ineq + 1)])
ubg = np.concatenate([np.zeros(num_eq), np.zeros(num_ineq + 1)])
lbx = np.concatenate([L, [0.0]])
ubx = np.concatenate([U, [np.inf]])

# -------------------------
# Data dictionary (expanded for cost+quality)
# -------------------------
data = {
    "Q": Q,
    "p": p,
    "A": A,
    "X": X,
    "G": G,
    "h": h,
    "C": C,
    "d": d,
    "YL": L,
    "YU": U,
    "XL": XL,
    "XU": XU,

    "solver_name": "ipopt",
    "solver_opts": opts,
    "budget_type": "max_cpu_time",
    "budget_value": float(MAX_CPU_TIME),

    "Y_subopt": [],

    # cost
    "solve_time_sec": [],
    "iter_count": [],
    "return_status": [],
    "success": [],

    # quality
    "obj_value": [],
    "eq_l2": [],
    "ineq_max": [],
    "ineq_l2": [],
}

Y = []

# -------------------------
# Solve loop (fast)
# -------------------------
for n in range(num_examples):
    Xi_n = X[n]

    start = time.time()
    res = solver(p=Xi_n, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    wall = time.time() - start

    st = solver.stats()
    success = bool(st.get("success", False))
    status = st.get("return_status", "")
    iters = int(st.get("iter_count", -1))

    data["solve_time_sec"].append(float(wall))
    data["success"].append(success)
    data["return_status"].append(status)
    data["iter_count"].append(iters)

    sol_x = res["x"].full().flatten()
    y_sol = sol_x[:-1]
    Y.append(y_sol)

    objv = float(res["f"].full().item())
    data["obj_value"].append(objv)

    # Post-hoc feasibility from g(x); ineq are defined as <= 0
    g_val = res["g"].full().flatten()
    eq_val = g_val[:num_eq]
    ineq_val = g_val[num_eq:]

    eq_l2 = float(np.linalg.norm(eq_val, 2))
    ineq_pos = np.maximum(ineq_val, 0.0)
    ineq_max = float(np.max(ineq_pos)) if ineq_pos.size else 0.0
    ineq_l2 = float(np.linalg.norm(ineq_pos, 2))

    data["eq_l2"].append(eq_l2)
    data["ineq_max"].append(ineq_max)
    data["ineq_l2"].append(ineq_l2)

    print(f"Example {n}: time={wall:.4f}s iters={iters} obj={objv:.4g} eq_l2={eq_l2:.3g} ineq_max={ineq_max:.3g}")
    st = solver.stats()
    print(
        f"status={st.get('return_status','')}, success={st.get('success', False)}, "
        f"iters={st.get('iter_count', -1)}, "
        f"t_wall_nlp_f={st.get('t_wall_nlp_f','NA')}, "
        f"t_wall_total={st.get('t_wall_total','NA')}"
    )

    # if n >= 10:
    #     break

data["Y_subopt"] = np.array(Y)

# -------------------------
# Your best_partial block (unchanged, but fix det init)
# -------------------------
det_best = -np.inf
best_partial = None
for i in range(1000):
    np.random.seed(i)
    partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
    other_vars = np.setdiff1d(np.arange(num_var), partial_vars)
    _, det = np.linalg.slogdet(A[:, other_vars])
    if det > det_best:
        det_best = det
        best_partial = partial_vars

print("best_det", det_best)
data["best_partial"] = best_partial

# Save
out_path = f"datasets/nonsmooth_nonconvex/socp/random{seed}_socp_dataset_var{num_var}_ineq{num_ineq}_eq{num_eq}_ex{num_examples}_maxt{MAX_CPU_TIME:.1f}"
with open(out_path, "wb") as f:
    pickle.dump(data, f)

print("Finished generating data!", out_path)
