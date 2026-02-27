import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader

from models.neural_networks import EnsembleMLP
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

            merit = 1.0 * obj_pred + 1e5 * eq_l1 + 1e5 * ineq_l1

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
        """Apply method-specific post-processing."""
        if self.method == "FSNet" or self.method == "S3Net" or self.method == 'semi':
            return nondiff_lbfgs_solve(
                X_batch, Y_pred_scaled, self.opt_problem,
                val_tol=self.config_method.get('test_val_tol', 1e-6),
                memory=self.config_method.get('memory_size', 20),
                max_iter=self.config_method.get('max_iter', 20),
                scale=self.config_method.get('scale', 1)
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
        if not isinstance(model, EnsembleMLP):
            Y_pred = model(X_batch)
            Y_pred_scaled = self.opt_problem.scale(Y_pred)
            return self._post_process_predictions(X_batch, Y_pred_scaled)

        ensemble_post = self.config.get('ensemble_post', 'pre')
        all_preds = model.forward_all(X_batch)  # (M, B, out)

        if ensemble_post == 'post':
            finals = []
            for i in range(all_preds.shape[0]):
                y_scaled = self.opt_problem.scale(all_preds[i])
                y_final = self._post_process_predictions(X_batch, y_scaled)
                finals.append(y_final)
            return self._aggregate_predictions(finals, X_batch)

        scaled = [self.opt_problem.scale(all_preds[i])
                  for i in range(all_preds.shape[0])]
        aggregated = self._aggregate_predictions(scaled, X_batch)
        return self._post_process_predictions(X_batch, aggregated)

    def _aggregate_predictions(self, finals, X_batch):
        """
        Aggregate post-processed predictions from ensemble members.

        Strategies:
            mean         – element-wise mean
            median       – element-wise median
            greedy_obj   – per-sample pick the member with the lowest objective
            greedy_merit – per-sample pick the member with the lowest
                           merit = obj + 1e5*(eq_viol + ineq_viol)
        """
        agg = self.config.get('ensemble_agg', 'mean')
        stacked = torch.stack(finals, dim=0)  # (M, B, out)

        if agg == 'mean':
            return stacked.mean(dim=0)

        if agg == 'median':
            return stacked.median(dim=0).values

        if agg in ('greedy_obj', 'greedy_merit'):
            scores = []
            for f in finals:
                obj = self.opt_problem.obj_fn(f)  # (B,)
                if agg == 'greedy_merit':
                    eq_l1 = self.opt_problem.eq_resid(X_batch, f).abs().sum(dim=1)
                    ineq_l1 = self.opt_problem.ineq_resid(X_batch, f).abs().sum(dim=1)
                    obj = obj + 1e5 * (eq_l1 + ineq_l1)
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
        
        # Optimality gap
        opt_gap = (obj_pred - obj_true) / obj_true.abs()         
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
        log.info("Opt Gap:     %.6e +/- %.6e", metrics.get('opt_gap_mean', 0), metrics.get('opt_gap_std', 0))
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

