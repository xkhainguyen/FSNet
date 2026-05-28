import torch
from typing import Tuple, Optional, Callable

# Differentiable and nondifferentiable L-BFGS solver

@torch.jit.script
def _search_direction(
    g: torch.Tensor,               # (B, n)
    S: torch.Tensor,               # (m, B, n) stacked s‑vectors
    Y: torch.Tensor,               # (m, B, n) stacked y‑vectors
    gamma: torch.Tensor            # (B, 1) or scalar
) -> torch.Tensor:                 # returns d (B, n)
    """
    Compute d = −H_k^{-1} g_k for L‑BFGS in batch mode using two-loop recursion.

    Parameters
    ----------
    g : torch.Tensor
        Current gradient, shape (B, n)
    S : torch.Tensor
        History of s_i vectors, shape (m, B, n)
    Y : torch.Tensor
        History of y_i vectors, shape (m, B, n)
    gamma : torch.Tensor
        Scalar or (B,1) scaling for the initial Hessian approximation

    Returns
    -------
    torch.Tensor
        Search direction, shape (B, n)
    """
    m = S.shape[0]  # history length
    eps = 1e-10
    rho = 1.0 / ((S * Y).sum(dim=2, keepdim=True) + eps)  # (m,B,1)

    # First loop (reverse order)
    q = g.clone()
    alphas = []
    for i in range(m - 1, -1, -1):
        alpha_i = rho[i] * (S[i] * q).sum(dim=1, keepdim=True)  # (B,1)
        alphas.append(alpha_i)
        q = q - alpha_i * Y[i]

    # Apply initial Hessian approximation: gamma * I
    r = gamma * q

    # Second loop (forward order)
    alphas = alphas[::-1]
    for i in range(m):
        beta = rho[i] * (Y[i] * r).sum(dim=1, keepdim=True)
        r = r + S[i] * (alphas[i] - beta)

    return -r


@torch.jit.script
def compute_gamma(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the initial Hessian scaling factor γ = s^T y / y^T y.
    
    Parameters
    ----------
    S : torch.Tensor
        History of s vectors, shape (m, B, n)
    Y : torch.Tensor
        History of y vectors, shape (m, B, n)
        
    Returns
    -------
    torch.Tensor
        Scaling factor, shape (B, 1)
    """
    eps = 1e-10
    s_dot_y = (S[-1] * Y[-1]).sum(dim=1, keepdim=True)
    y_dot_y = (Y[-1] * Y[-1]).sum(dim=1, keepdim=True) + eps
    return s_dot_y / y_dot_y


class LBFGSConfig:
    """Configuration class for L-BFGS parameters."""
    def __init__(
        self,
        max_iter: int = 20,
        memory: int = 20,
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        scale: float = 1.0,
        c: float = 1e-4,
        rho_ls: float = 0.5,
        max_ls_iter: int = 10,
        verbose: bool = False
    ):
        self.max_iter = max_iter
        self.memory = memory
        self.val_tol = val_tol
        self.grad_tol = grad_tol
        self.scale = scale
        self.c = c
        self.rho_ls = rho_ls
        self.max_ls_iter = max_ls_iter
        self.verbose = verbose


def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure.

    Calling with ``reduce=True`` (default) returns the batch-mean scalar — kept
    for backward compatibility with the autograd-graph callers. Calling with
    ``reduce=False`` returns the per-sample tensor of shape ``(B,)``, used by
    the per-sample convergence check.
    """
    def _obj(y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        eq_residual = (data.eq_resid(x, y) ** 2).sum(dim=1)     # (B,)
        ineq_residual = (data.ineq_resid(x, y) ** 2).sum(dim=1) # (B,)
        out = scale * (eq_residual + ineq_residual)              # (B,)
        return out.mean(0) if reduce else out
    return _obj


def _check_convergence(f_val_per_sample: torch.Tensor, g: torch.Tensor, config: LBFGSConfig) -> torch.Tensor:
    """Per-sample convergence check.

    Returns a (B,) mask. Sample i is considered converged when **both** the
    per-sample objective is below ``val_tol`` AND the per-sample gradient norm
    is below ``grad_tol`` — i.e. we're at a low-residual *and* near-stationary
    point. Callers ``.all()`` the result to decide loop exit.

    The previous semantics was ``val | grad`` (either triggers exit). That
    was the cause of the batch-1 quality regression: a single sample can hit
    the val-tol threshold (squared residual small) while its L1 residual is
    still substantial, or hit grad-tol at a near-stationary but non-feasible
    point. At larger batch the OR threshold was effectively forgiving because
    different samples failed different criteria and the loop kept running for
    the worst. The ``&`` semantics removes that fragile coupling and gives
    consistent per-sample convergence regardless of batch composition.
    """
    val_converged = f_val_per_sample / config.scale < config.val_tol  # (B,) bool
    grad_converged = g.norm(dim=1) < config.grad_tol                   # (B,) bool
    return val_converged & grad_converged                               # (B,) bool


def _backtracking_line_search(
    y: torch.Tensor,
    d: torch.Tensor,
    g: torch.Tensor,
    f_val_per_sample: torch.Tensor,
    obj_func: Callable,
    config: LBFGSConfig,
) -> torch.Tensor:
    """Per-sample backtracking line search.

    Returns a (B,) step tensor — each sample's step is set independently when
    its own Armijo condition is satisfied (and frozen thereafter). This makes
    the line search batch-size-invariant: the converged ``y`` no longer
    depends on whether the K perturbed candidates are processed sequentially
    or as one big batch.

    Requires ``g`` to be the **per-sample** gradient (``autograd.grad`` of the
    sum-reduced objective, NOT mean-reduced) and ``f_val_per_sample`` to be
    the (B,) un-reduced objective. The caller is responsible for using sum-
    reduction in the autograd path.
    """
    B = y.shape[0]
    step = torch.ones(B, device=y.device, dtype=y.dtype)
    accepted = torch.zeros(B, dtype=torch.bool, device=y.device)

    with torch.no_grad():
        dir_deriv = (g * d).sum(dim=1)  # (B,)
        for _ in range(config.max_ls_iter):
            y_trial = y + step.unsqueeze(-1) * d
            f_trial = obj_func(y_trial, reduce=False)  # (B,)
            armijo = f_trial <= f_val_per_sample + config.c * step * dir_deriv  # (B,)
            accepted = accepted | armijo
            if accepted.all():
                break
            step = torch.where(accepted, step, step * config.rho_ls)

    return step


def lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Differentiable L‑BFGS solver with vectorized two‑loop recursion.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    config : LBFGSConfig, optional
        Configuration object. If None, uses default parameters from kwargs.
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution, shape (B, n)
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Initialize
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    # History buffers
    S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    # Create objective function
    obj_func = _create_objective_function(x, data, config.scale)

    # Per-sample objective + per-sample gradient (sum-reduce for autograd).
    f_val_vec = obj_func(y, reduce=False)
    g = torch.autograd.grad(f_val_vec.sum(), y, create_graph=True)[0]

    for k in range(config.max_iter):
        # Per-sample convergence check
        if _check_convergence(f_val_vec, g, config).all():
            if config.verbose:
                print(f"Converged at iteration {k}")
            break

        # Compute search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g  # Steepest descent for first iteration

        # Per-sample line search → (B,) step
        step = _backtracking_line_search(y, d, g, f_val_vec, obj_func, config)

        # Update solution
        y_next = y + step.unsqueeze(-1) * d
        f_next_vec = obj_func(y_next, reduce=False)
        g_next = torch.autograd.grad(f_next_vec.sum(), y_next, create_graph=True)[0]

        # Update history
        S_hist[hist_ptr] = y_next - y
        Y_hist[hist_ptr] = g_next - g
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)

        # Prepare for next iteration
        y = y_next
        f_val_vec = f_next_vec.clone()
        g = g_next.clone()

        if config.verbose and k % 5 == 0:
            print(f"Iter {k:3d}: f_mean = {f_next_vec.mean().item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step_mean = {step.mean().item():.3e}")

    return y


def nondiff_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    S_hist: Optional[torch.Tensor] = None,
    Y_hist: Optional[torch.Tensor] = None,
    hist_len: int = 0,
    hist_ptr: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable L‑BFGS solver that doesn't build backward graph.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    config : LBFGSConfig, optional
        Configuration object
    S_hist, Y_hist : torch.Tensor, optional
        Pre-existing history buffers
    hist_len, hist_ptr : int
        History tracking variables
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution, shape (B, n)
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Initialize without gradient tracking
    y = y_init.detach().clone().requires_grad_(True)
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    # Initialize history buffers if not provided
    if S_hist is None:
        S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
        Y_hist = torch.zeros_like(S_hist)
        hist_len = 0
        hist_ptr = 0
    
    obj_func = _create_objective_function(x, data, config.scale)

    f_val_vec = obj_func(y, reduce=False)
    g = torch.autograd.grad(f_val_vec.sum(), y, create_graph=False)[0]

    for k in range(config.max_iter):
        y.requires_grad_(False)
        g = g.detach()

        # Per-sample convergence check
        if _check_convergence(f_val_vec, g, config).all():
            if config.verbose:
                print(f"Converged at iteration {k}")
            break

        # Compute search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g

        # Per-sample line search → (B,) step
        step = _backtracking_line_search(y, d, g, f_val_vec, obj_func, config)

        y_next = y + step.unsqueeze(-1) * d

        # Update history with detached tensors
        y_next.requires_grad_(True)
        f_next_vec = obj_func(y_next, reduce=False)
        g_next, = torch.autograd.grad(f_next_vec.sum(), y_next, create_graph=False)

        S_hist[hist_ptr] = (y_next - y).detach()
        Y_hist[hist_ptr] = (g_next - g).detach()
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)

        y = y_next.detach()
        f_val_vec = f_next_vec.detach().clone()
        g = g_next.clone()

        if config.verbose and k % 5 == 0:
            print(f"Iter {k:3d}: f_mean = {f_next_vec.mean().item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step_mean = {step.mean().item():.3e}")

    return y


def hybrid_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 20,
    config: Optional[LBFGSConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Hybrid L‑BFGS solver with truncated backpropagation.
    
    Starts with differentiable L‑BFGS and switches to non-differentiable
    after max_diff_iter iterations for memory efficiency.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    max_diff_iter : int
        Number of differentiable iterations before switching
    config : LBFGSConfig, optional
        Configuration object
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution with gradient connection to first max_diff_iter steps
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Create a config for the differentiable phase
    diff_config = LBFGSConfig(
        max_iter=max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose
    )
    
    # Run differentiable phase (shortened version of lbfgs_solve)
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    obj_func = _create_objective_function(x, data, config.scale)
    f_val_vec = obj_func(y, reduce=False)
    g = torch.autograd.grad(f_val_vec.sum(), y, create_graph=True)[0]

    for k in range(max_diff_iter):
        if _check_convergence(f_val_vec, g, diff_config).all():
            if config.verbose:
                print(f"Converged in differentiable phase at iteration {k}")
            return y

        # Search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g

        # Per-sample line search → (B,) step
        step = _backtracking_line_search(y, d, g, f_val_vec, obj_func, diff_config)

        # Update
        y_next = y + step.unsqueeze(-1) * d
        f_next_vec = obj_func(y_next, reduce=False)
        g_next = torch.autograd.grad(f_next_vec.sum(), y_next, create_graph=True)[0]

        # Update history
        S_hist[hist_ptr] = y_next - y
        Y_hist[hist_ptr] = g_next - g
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)

        y = y_next
        f_val_vec = f_next_vec.clone()
        g = g_next.clone()

        if config.verbose and k % 5 == 0:
            print(f"Diff iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step = {step:.3e}")
    
    # Switch to non-differentiable phase
    remaining_config = LBFGSConfig(
        max_iter=config.max_iter - max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose
    )
    
    y_nondiff = nondiff_lbfgs_solve(
        x, y, data, remaining_config,
        S_hist=S_hist,
        Y_hist=Y_hist,
        hist_len=hist_len,
        hist_ptr=hist_ptr
    )
    
    # Return with gradient connection only to differentiable phase
    return y + (y_nondiff - y).detach()