import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from models.neural_networks import EnsembleMLP, MixtureOfExperts, MultiHeadMLP
from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


class Evaluator:
    """Separate evaluator class for model evaluation."""
    
    def __init__(self, opt_problem, method, config):
        """Initialize evaluator."""
        self.opt_problem = opt_problem
        self.method = method
        self.config = config
        self.config_method = config[method]

    @torch.no_grad()
    def evaluate_batch(self, model, input_batch):
        model.eval()

        time_start = time.time()
        X_batch = input_batch.to(DEVICE)

        Y_final = self._get_final_prediction(model, X_batch)
        time_end = time.time()

        # Per-sample objective / violations
        obj_pred = self.opt_problem.obj_fn(Y_final)          # [B]

        eq_resid = self.opt_problem.eq_resid(X_batch, Y_final)     # [B, meq]
        ineq_resid = self.opt_problem.ineq_resid(X_batch, Y_final) # [B, mineq]

        eq_l1 = eq_resid.abs().sum(dim=1)          # [B]
        ineq_l1 = ineq_resid.abs().sum(dim=1)      # [B]

        # Return CPU tensors so caller can aggregate/scatter_add efficiently
        return {
            "objective": obj_pred.detach().to("cpu", dtype=torch.float32),
            "eq_violation_l1": eq_l1.detach().to("cpu", dtype=torch.float32),
            "ineq_violation_l1": ineq_l1.detach().to("cpu", dtype=torch.float32),
            "sol_time": time_end - time_start,
        }

    @torch.no_grad()
    def evaluate_loss(self, model, data_loader):
        model.eval()

        total_loss = 0.0
        total_samples = 0

        for X_batch, _Y_true in data_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)

            # Pre-post-process predictions for penalty terms
            Y_pred = model(X_batch)
            Y_pred_scaled = self.opt_problem.scale(Y_pred)

            eq_l2 = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
            ineq_l2 = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

            Y_final = self._get_final_prediction(model, X_batch)

            obj_pred = self.opt_problem.obj_fn(Y_final)
            distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square().mean()

            loss = (
                self.config_method["obj_weight"] * obj_pred
                + self.config_method["dist_weight"] * distance
                + self.config_method["eq_pen_weight"] * eq_l2
                + self.config_method["ineq_pen_weight"] * ineq_l2
            )

            total_loss += loss.sum().item()
            total_samples += X_batch.size(0)

        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate_merit(self, model, data_loader):
        model.eval()

        total_merit = 0.0
        total_samples = 0

        for X_batch, _Y_true in data_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)

            Y_final = self._get_final_prediction(model, X_batch)

            eq_resid = self.opt_problem.eq_resid(X_batch, Y_final)
            ineq_resid = self.opt_problem.ineq_resid(X_batch, Y_final)

            eq_l1 = eq_resid.abs().sum(dim=1)
            ineq_l1 = ineq_resid.abs().sum(dim=1)

            obj_pred = self.opt_problem.obj_fn(Y_final)

            merit = 1.0 * obj_pred + 1e6 * eq_l1 + 1e6 * ineq_l1

            total_merit += merit.sum().item()
            total_samples += X_batch.size(0)

        return total_merit / max(1, total_samples)
    
    @torch.no_grad()
    def evaluate(self, model, data_loader, split_name="eval", return_detailed=False):
        """
        Comprehensive evaluation of the model.
        
        Args:
            model: The neural network model
            data_loader: DataLoader for evaluation data
            split_name: Name of the split (train/val/test)
            return_detailed: Whether to return detailed predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = []
        detailed_results = [] if return_detailed else None
        
        total_time = 0
        
        for batch_idx, (X_batch, Y_true) in enumerate(data_loader):
            X_batch = X_batch.to(DEVICE)
            Y_true = Y_true.to(DEVICE)
            
            start_time = time.time()
            
            Y_final = self._get_final_prediction(model, X_batch)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Compute comprehensive metrics
            batch_metrics = self._compute_batch_metrics(X_batch, Y_final, Y_true)
            batch_metrics['inference_time'] = batch_time
            all_metrics.append(batch_metrics)
            
            # Store detailed results if requested
            if return_detailed:
                detailed_results.append({
                    'X': X_batch.cpu(),
                    'Y_final': Y_final.cpu(),
                    'Y_true': Y_true.cpu(),
                    'metrics': batch_metrics
                })
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics['total_time'] = total_time
        aggregated_metrics['avg_inference_time'] = total_time / len(data_loader)
        
        # Print summary
        self._print_evaluation_summary(split_name, aggregated_metrics)
        
        if return_detailed:
            return aggregated_metrics, detailed_results
        return aggregated_metrics
    
    @torch.enable_grad()
    def _post_process_predictions(self, X_batch, Y_pred_scaled):
        """Apply method-specific post-processing.

        When ``config['skip_repair']`` is True the repair step is bypassed and
        ``Y_pred_scaled`` is returned as-is, regardless of method. Use to ablate
        the contribution of the repair layer at eval time.
        """
        if self.config.get('skip_repair', False):
            return Y_pred_scaled

        max_iter_override = self.config.get('repair_max_iter_override', None)

        if self.method == "FSNet" or self.method == "S3Net" or self.method == 'semi':
            return nondiff_lbfgs_solve(
                X_batch, Y_pred_scaled, self.opt_problem,
                val_tol=self.config_method.get('test_val_tol', 1e-6),
                memory=self.config_method.get('memory_size', 20),
                max_iter=max_iter_override if max_iter_override is not None
                          else self.config_method.get('max_iter', 20),
                scale=self.config_method.get('scale', 1),
                per_sample=self.config.get('per_sample_lbfgs',
                                            self.config_method.get('per_sample_lbfgs', False)),
            )
        elif self.method == "DC3" or self.method == "sup_partial":
            Y_completion = self.opt_problem.complete_partial(X_batch, Y_pred_scaled)
            return grad_steps(self.opt_problem, X_batch, Y_completion, self.config)
        elif self.method == "projection":
            return self.opt_problem.qpth_projection(X_batch, Y_pred_scaled)
        else:
            return Y_pred_scaled

    def _get_final_prediction(self, model, X_batch):
        """
        Get final prediction, handling ensemble post-processing modes.

        "pre"  = agg(NNs) + Opt : aggregate raw NN outputs, post-process once.
        "post" = agg(NNs + Opts): post-process each member, then aggregate.

        Falls back to the standard single-model path for non-ensemble models.
        """
        if isinstance(model, MixtureOfExperts):
            moe_strategy = self.config.get('moe_strategy', 'vanilla')
            if moe_strategy == 'top2_best_merit':
                candidate_top_k = int(self.config.get('moe_candidate_top_k', 2))
                _out, candidates, _topk_idx, topk_weights = model.forward_candidates(
                    X_batch, candidate_top_k=candidate_top_k)
                finals = []
                for i in range(candidates.shape[1]):
                    y_scaled = self.opt_problem.scale(candidates[:, i, :])
                    y_final = self._post_process_predictions(X_batch, y_scaled)
                    finals.append(y_final)
                return self._aggregate_moe_predictions(
                    finals, X_batch, topk_weights, agg='best_merit')

            moe_post = self.config.get('moe_post', 'pre')
            if moe_post == 'post':
                _out, candidates, _topk_idx, topk_weights = model.forward_candidates(X_batch)
                finals = []
                for i in range(candidates.shape[1]):
                    y_scaled = self.opt_problem.scale(candidates[:, i, :])
                    y_final = self._post_process_predictions(X_batch, y_scaled)
                    finals.append(y_final)
                return self._aggregate_moe_predictions(finals, X_batch, topk_weights)

            Y_pred = model(X_batch)
            Y_pred_scaled = self.opt_problem.scale(Y_pred)
            return self._post_process_predictions(X_batch, Y_pred_scaled)

        is_ensemble_like = isinstance(model, (EnsembleMLP, MultiHeadMLP))
        if not is_ensemble_like:
            Y_pred = model(X_batch)
            Y_pred_scaled = self.opt_problem.scale(Y_pred)

            perturb_k = int(self.config.get('inference_perturb_k', 0))
            if perturb_k > 1:
                return self._perturb_repair_aggregate(X_batch, Y_pred_scaled, perturb_k)

            return self._post_process_predictions(X_batch, Y_pred_scaled)

        ensemble_post = self.config.get('ensemble_post', 'pre')
        all_preds = model.forward_all(X_batch)  # (M, B, out)

        if ensemble_post == 'post':
            scaled_stack = torch.stack(
                [self.opt_problem.scale(all_preds[i])
                 for i in range(all_preds.shape[0])],
                dim=0,
            )  # (M, B, out)
            finals = self._batched_post_process(X_batch, scaled_stack)
            return self._aggregate_predictions(finals, X_batch)

        scaled = [self.opt_problem.scale(all_preds[i])
                  for i in range(all_preds.shape[0])]
        aggregated = self._aggregate_predictions(scaled, X_batch)
        return self._post_process_predictions(X_batch, aggregated)

    def _aggregate_moe_predictions(self, finals, X_batch, topk_weights, agg=None):
        """Aggregate post-processed MoE candidate predictions.

        Strategies:
            router      - per-sample pick candidate with highest routing weight
            mean        - element-wise mean over candidates
            best_obj    - per-sample pick candidate with lowest objective
            best_merit  - per-sample pick candidate with lowest merit
                         merit = obj + 1e6*(eq_viol + ineq_viol)
        """
        agg = agg or self.config.get('moe_agg', self.config.get('ensemble_agg', 'best_merit'))
        stacked = torch.stack(finals, dim=0)  # (K, B, out)

        if agg == 'router':
            pick = topk_weights.argmax(dim=1)  # (B,)
            B = stacked.shape[1]
            return stacked[pick, torch.arange(B, device=stacked.device)]

        if agg == 'mean':
            return stacked.mean(dim=0)

        if agg in ('best_obj', 'best_merit'):
            scores = []
            for f in finals:
                obj = self.opt_problem.obj_fn(f)  # (B,)
                if agg == 'best_merit':
                    eq_l1 = self.opt_problem.eq_resid(X_batch, f).abs().sum(dim=1)
                    ineq_l1 = self.opt_problem.ineq_resid(X_batch, f).abs().sum(dim=1)
                    obj = obj + 1e6 * (eq_l1 + ineq_l1)
                scores.append(obj)
            scores = torch.stack(scores, dim=0)  # (K, B)
            best_idx = scores.argmin(dim=0)      # (B,)
            B = stacked.shape[1]
            return stacked[best_idx, torch.arange(B, device=stacked.device)]

        raise ValueError(f"Unknown MoE aggregation strategy: {agg}")

    def _batched_post_process(self, X_batch, Y_candidates):
        """Run the per-sample repair on K candidates.

        Args:
            X_batch: (B, xdim) input batch.
            Y_candidates: (K, B, ydim) K candidate scaled predictions per sample.

        Returns:
            list of K tensors of shape (B, ydim) — the repaired candidates.

        Defaults to a Python loop over K (each call has homogeneous samples).
        Set ``vectorize_repair=True`` to stack into a single (K·B)-batch call
        — but **expect different (typically worse) numbers**: the L-BFGS
        implementation in this repo uses a single global line-search step
        size and a batch-mean objective for convergence checking
        (``utils/lbfgs.py:120-140``, lines 107-117). With heterogeneous
        K-candidate batches the global step size can't fit all samples and
        the mean-based convergence check exits too early. Sequential calls
        avoid this because each call has homogeneous samples (the k-th
        perturbation of B inputs, all close to each other).

        Proper fix would require per-sample line search in L-BFGS.
        """
        K, _, _ = Y_candidates.shape
        if not self.config.get('vectorize_repair', False) or K == 1:
            return [self._post_process_predictions(X_batch, Y_candidates[k])
                    for k in range(K)]

        # Experimental vectorised path — keep available for L2O variants where
        # the L-BFGS line-search heterogeneity isn't a problem.
        B, D = Y_candidates.shape[1], Y_candidates.shape[2]
        Y_flat = Y_candidates.reshape(K * B, D)
        X_flat = X_batch.repeat(K, 1)
        Y_repaired_flat = self._post_process_predictions(X_flat, Y_flat)
        return list(Y_repaired_flat.reshape(K, B, D).unbind(0))

    def _perturb_repair_aggregate(self, X_batch, Y_pred_scaled, K):
        """Single-model "free ensemble" via perturbed repair starts.

        Take one NN's scaled prediction, replicate K times with additive noise,
        run the repair step on each perturbed start, then aggregate with the
        configured ``ensemble_agg`` (typically ``best_merit``). This turns any
        single-model checkpoint into a K-member post-style ensemble at
        inference time with zero retraining cost.

        Config knobs:
            inference_perturb_k    : K (number of perturbed restarts)
            inference_perturb_eps  : Gaussian std (scalar or relative to output range)
            inference_perturb_dist : 'gauss' (default), 'antithetic', 'sphere'
            inference_perturb_keep_original : include unperturbed pred as restart 0 (default True)
            vectorize_repair : run all K repairs in one (K·B)-batch L-BFGS call (default True)
        """
        eps = float(self.config.get('inference_perturb_eps', 0.05))
        dist = self.config.get('inference_perturb_dist', 'gauss')
        keep_original = self.config.get('inference_perturb_keep_original', True)

        B, D = Y_pred_scaled.shape
        device = Y_pred_scaled.device

        n_remaining = K - 1 if keep_original else K

        if dist == 'antithetic':
            n_pairs = (n_remaining + 1) // 2
            zs = torch.randn(n_pairs, B, D, device=device, dtype=Y_pred_scaled.dtype)
            perts = torch.cat([zs, -zs], dim=0)[:n_remaining] * eps
        elif dist == 'sphere':
            zs = torch.randn(n_remaining, B, D, device=device, dtype=Y_pred_scaled.dtype)
            zs = zs / zs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            perts = zs * eps
        else:
            perts = torch.randn(n_remaining, B, D,
                                 device=device, dtype=Y_pred_scaled.dtype) * eps

        candidates = [Y_pred_scaled] if keep_original else []
        for k in range(n_remaining):
            candidates.append(Y_pred_scaled + perts[k])

        Y_stacked = torch.stack(candidates, dim=0)  # (K, B, D)
        finals = self._batched_post_process(X_batch, Y_stacked)

        return self._aggregate_predictions(finals, X_batch)

    def _aggregate_predictions(self, finals, X_batch):
        """
        Aggregate post-processed predictions from ensemble members.

        Strategies:
            mean         – element-wise mean
            median       – element-wise median
            best_obj   – per-sample pick the member with the lowest objective
            best_merit – per-sample pick the member with the lowest
                           merit = obj + 1e6*(eq_viol + ineq_viol)
        """
        agg = self.config.get('ensemble_agg', 'mean')
        stacked = torch.stack(finals, dim=0)  # (M, B, out)

        if agg == 'mean':
            return stacked.mean(dim=0)

        if agg == 'median':
            return stacked.median(dim=0).values

        if agg in ('best_obj', 'best_merit'):
            scores = []
            for f in finals:
                obj = self.opt_problem.obj_fn(f)  # (B,)
                if agg == 'best_merit':
                    eq_l1 = self.opt_problem.eq_resid(X_batch, f).abs().sum(dim=1)
                    ineq_l1 = self.opt_problem.ineq_resid(X_batch, f).abs().sum(dim=1)
                    obj = obj + 1e6 * (eq_l1 + ineq_l1)
                scores.append(obj)
            scores = torch.stack(scores, dim=0)  # (M, B)
            best_idx = scores.argmin(dim=0)       # (B,)
            B = stacked.shape[1]
            return stacked[best_idx, torch.arange(B, device=stacked.device)]

        raise ValueError(f"Unknown ensemble aggregation strategy: {agg}")

    def _compute_merit(self, obj, eq_vio, ineq_vio):
        """Compute merit function value."""
        obj_weight = 1
        eq_weight = 1e6
        ineq_weight = 1e6
        return obj_weight * obj + eq_weight * eq_vio + ineq_weight * ineq_vio
    
    def _compute_batch_metrics(self, X_batch, Y_final, Y_true):
        """Compute comprehensive metrics for a batch."""
        # Objective values
        obj_pred = self.opt_problem.obj_fn(Y_final)
        obj_true = self.opt_problem.obj_fn(Y_true)
        
        # Constraint violations
        eq_resid = self.opt_problem.eq_resid(X_batch, Y_final)
        ineq_resid = self.opt_problem.ineq_resid(X_batch, Y_final)
        
        eq_violation_l2 = eq_resid.square().sum(dim=1)
        ineq_violation_l2 = ineq_resid.square().sum(dim=1)
        eq_violation_l1 = eq_resid.abs().sum(dim=1)
        ineq_violation_l1 = ineq_resid.abs().sum(dim=1)
        eq_violation_max = eq_resid.abs().max(dim=1)[0]
        ineq_violation_max = ineq_resid.abs().max(dim=1)[0]
        
        # Optimality gap in percent for consistent reporting across files.
        opt_gap = 100.0 * (obj_pred - obj_true) / obj_true.abs()
        # Solution distance
        solution_distance = torch.norm(Y_final - Y_true, dim=1).square()

        merit = self._compute_merit(obj_pred, eq_violation_l1, ineq_violation_l1)

        # Merit metrics
        return {
            # Objective metrics
            'objective': obj_pred.mean().item(),
            'objective_max': obj_pred.max().item(),
            'true_objective': obj_true.mean().item(),
            'true_objective_max': obj_true.max().item(),
            'opt_gap_mean': opt_gap.mean().item(),
            'opt_gap_std': opt_gap.std().item(),
            'opt_gap_max': opt_gap.max().item(),
            'opt_gap_min': opt_gap.min().item(),

            'merit_mean': merit.mean().item(),
            'merit_std': merit.std().item(),
            'merit_max': merit.max().item(),
            'merit_min': merit.min().item(),
            
            # Constraint violations (L2)
            'eq_violation_l2_mean': eq_violation_l2.mean().item(),
            'eq_violation_l2_max': eq_violation_l2.max().item(),
            'ineq_violation_l2_mean': ineq_violation_l2.mean().item(),
            'ineq_violation_l2_max': ineq_violation_l2.max().item(),
            
            # Constraint violations (l1)
            'eq_violation_l1_mean': eq_violation_l1.mean().item(),
            'eq_violation_l1_max': eq_violation_l1.max().item(),
            'ineq_violation_l1_mean': ineq_violation_l1.mean().item(),
            'ineq_violation_l1_max': ineq_violation_l1.max().item(),
            
            # Constraint violations (L∞)
            'eq_violation_max_mean': eq_violation_max.mean().item(),
            'eq_violation_max_max': eq_violation_max.max().item(),
            'ineq_violation_max_mean': ineq_violation_max.mean().item(),
            'ineq_violation_max_max': ineq_violation_max.max().item(),
            
            # Solution quality
            'solution_distance_mean': solution_distance.mean().item(),
            'solution_distance_std': solution_distance.std().item(),
            'solution_distance_max': solution_distance.max().item(),
        }
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across batches."""
        if not all_metrics:
            return {}
        
        keys = all_metrics[0].keys() - {'inference_time'}
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in all_metrics]
            if key.endswith('_std'):
                # For std metrics, compute overall std
                aggregated[key] = np.std([m[key.replace('_std', '_mean')] for m in all_metrics])
            else:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _print_evaluation_summary(self, split_name, metrics):
        """Log evaluation summary."""
        log.info("%s EVALUATION RESULTS:", split_name.upper())
        log.info("=" * 50)
        log.info("Obj:         %.6e", metrics.get('objective', 0))
        log.info("Opt Gap (%%): %.6e +/- %.6e", metrics.get('opt_gap_mean', 0), metrics.get('opt_gap_std', 0))
        log.info("Eq Vio l1:   %.6e (max: %.6e)", metrics.get('eq_violation_l1_mean', 0), metrics.get('eq_violation_l1_max', 0))
        log.info("Ineq Vio l1: %.6e (max: %.6e)", metrics.get('ineq_violation_l1_mean', 0), metrics.get('ineq_violation_l1_max', 0))
        log.info("Sol Dist:    %.6e +/- %.6e", metrics.get('solution_distance_mean', 0), metrics.get('solution_distance_std', 0))
        log.info("Merit:       %.6e +/- %.6e", metrics.get('merit_mean', 0), metrics.get('merit_std', 0))
        log.info("Avg Inf T:   %.4fs", metrics.get('avg_inference_time', 0))
        log.info("=" * 50)
    
    def evaluate_multiple_batch_sizes(self, model, dataset, batch_sizes, split_name="test"):
        """
        Evaluate model with multiple batch sizes and collect detailed results for all successful ones.
        
        Args:
            model: The neural network model
            dataset: Dataset to evaluate on
            batch_sizes: List of batch sizes to test
            split_name: Name of the evaluation split
            
        Returns:
            Tuple of (results_dict, detailed_results_dict)
        """
        results = {}
        all_detailed_results = {}
        
        for batch_size in batch_sizes:
            log.info("Evaluating with batch_size=%d", batch_size)
            
            try:
                # Create data loader with specific batch size
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                )
                
                # Evaluate with detailed results
                metrics, detailed_results = self.evaluate(
                    model, data_loader, f"{split_name}_bs{batch_size}", 
                    return_detailed=True
                )
                
                results[batch_size] = {
                    'metrics': metrics,
                    'batch_size': batch_size,
                }
                
                all_detailed_results[batch_size] = detailed_results
                
                # Clear cache after each evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log.warning("Batch size %d failed (OOM)", batch_size)
                    results[batch_size] = {
                        'error': 'OOM',
                        'batch_size': batch_size
                    }
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # Print comparison summary
        self._print_batch_size_comparison(results, split_name)
        
        return results, all_detailed_results
    
    def _print_batch_size_comparison(self, results, split_name):
        """Log comparison of results across batch sizes."""
        header = f"{'BS':<8} {'Obj':<12} {'OptGap':<12} {'EqViol':<12} {'IneqViol':<12} {'Time':<8}"
        log.info("%s BATCH SIZE COMPARISON:", split_name.upper())
        log.info("=" * 70)
        log.info(header)
        log.info("-" * 70)

        for batch_size, result in results.items():
            if 'error' in result:
                log.info("%-8d OOM", batch_size)
            else:
                m = result['metrics']
                log.info("%-8d %-12.4e %-12.4e %-12.4e %-12.4e %.2fs",
                         batch_size,
                         m.get('objective', 0), m.get('opt_gap_mean', 0),
                         m.get('eq_violation_l1_mean', 0),
                         m.get('ineq_violation_l1_mean', 0),
                         m.get('total_time', 0))

        log.info("=" * 70)
