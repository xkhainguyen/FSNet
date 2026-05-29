import copy
import logging
import numpy as np
import os
import pickle
import random
import time
import yaml as _yaml
from typing import Dict, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve, hybrid_lbfgs_solve
from models.neural_networks import MLP, SampledContextMLPv1, SampledContextMLPv2, LocalContextMLPv1, LocalContextMLPv2, EnsembleMLP, MixtureOfExperts, MultiHeadMLP
from utils.evaluator import Evaluator

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


def get_model_size_stats(model):
    """Return parameter-count and memory-size stats for a PyTorch model."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_bytes = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'param_bytes': int(param_bytes),
        'buffer_bytes': int(buffer_bytes),
        'total_bytes': int(total_bytes),
        'total_mb': float(total_bytes / (1024 ** 2)),
    }


def append_seed_suffix(name, seed):
    """Append the run seed as the final suffix of a directory name."""
    return f"{name}_seed{seed}"


def load_instance(config):
    """Loads problem instance, data, and sets up save directory.

    The run directory is structured as::

        results/{prob_type}/{prob_name}/{prob_str}/{method}_seed{seed}_{timestamp}/
            config.yaml   # full config snapshot (human-readable)
            model.pt      # model weights + config (binary)
            results.pkl   # training / test metrics (binary)
            train.log     # log file (added by logging setup)
    """

    seed = config['seed']
    method = config['method']
    train_size = config['train_size']
    val_size = config['val_size']
    test_size = config['test_size']
    en_subopt = config['en_subopt']
    prob_type = config['prob_type']
    prob_name = config['prob_name']
    prob_size = config['prob_size']

    problem_registry = {
        'convex':              {'qp': QPProblem, 'qcqp': QCQPProblem, 'socp': SOCPProblem},
        'nonconvex':           {'qp': nonconvexQPProblem, 'qcqp': nonconvexQCQPProblem, 'socp': nonconvexSOCPProblem},
        'nonsmooth_nonconvex': {'qp': nonsmooth_nonconvexQPProblem, 'qcqp': nonsmooth_nonconvexQCQPProblem, 'socp': nonsmooth_nonconvexSOCPProblem},
    }

    if prob_type not in problem_registry or prob_name not in problem_registry[prob_type]:
        raise NotImplementedError(f"Problem '{prob_type}/{prob_name}' not implemented")

    seed_data = 2025
    dataset_filepath = os.path.join(
        'datasets', prob_type, prob_name,
        f"random{seed_data}_{prob_name}_dataset_var{prob_size[0]}_ineq{prob_size[1]}_eq{prob_size[2]}_ex{prob_size[3]}"
    )

    if method in ("sup", "sup_partial", "sup_pen", "S3Net", "semi"):
        if en_subopt == 1:
            dataset_filepath += f'_subopt_noise{config["subopt_ratio"]}_bias{config["subopt_ratio"]}'
        elif en_subopt == 2:
            if config['subopt_ratio'] == 1:
                dataset_filepath += '_tol1e0_ready'
            elif config['subopt_ratio'] == -10:
                dataset_filepath += '_tol1em1_ready'
        elif en_subopt == 3:
            dataset_filepath += f'_maxt{config["subopt_ratio"]}_ready'

    log.info("Loading dataset: %s", dataset_filepath)
    with open(dataset_filepath, 'rb') as f:
        dataset = pickle.load(f)

    opt_problem = problem_registry[prob_type][prob_name](
        dataset, train_size, val_size, test_size, seed, en_subopt)

    opt_problem.device = DEVICE
    log.info("Device: %s", DEVICE)
    for attr in dir(opt_problem):
        var = getattr(opt_problem, attr)
        if torch.is_tensor(var):
            try:
                setattr(opt_problem, attr, var.to(DEVICE))
            except AttributeError:
                pass

    # ---- build save directory (skip for eval-only runs) ----
    if config.get('_eval_only'):
        return opt_problem, None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if config['ablation']:
        ablation_name = append_seed_suffix(
            f"dist_{config['FSNet']['dist_weight']}_diff_{config['FSNet']['max_diff_iter']}",
            seed,
        )
        result_save_dir = os.path.join(
            'ablation_results', prob_type, prob_name, str(opt_problem),
            f"{config['network']}_{method}",
            ablation_name)
    else:
        lr_str = f"{config[method]['lr']:.0e}".replace("+", "")
        run_name = (f"{timestamp}_{method}"
                    f"_e{config[method]['num_epochs']}"
                    f"_lr{lr_str}"
                    f"_n{train_size}")
        run_name += f"_hdim{config['hidden_dim']}"
        if config.get('ensemble_size', 1) > 1:
            run_name += (f"_ens{config['ensemble_size']}"
                         f"_{config.get('ensemble_mode', 'vanilla')}"
                         f"_{config.get('ensemble_post', 'pre')}")
        if config.get('network') == 'SampledContextMLPv1':
            ctx_cfg = config.get('SampledContextMLPv1', {})
            run_name += f"_ctx{int(ctx_cfg.get('num_context_points', config.get('context_num_points', 16)))}"
        if config.get('network') == 'SampledContextMLPv2':
            ctx_cfg = config.get('SampledContextMLPv2', {})
            run_name += (
                f"_ctxv2k{int(ctx_cfg.get('num_context_points', 4))}"
                f"e{int(ctx_cfg.get('context_encoder_dim', 128))}"
            )
        if config.get('network') == 'LocalContextMLPv1':
            run_name += "_localctxv1"
        if config.get('network') == 'LocalContextMLPv2':
            local_cfg = config.get('LocalContextMLPv2', {})
            run_name += f"_localctxv2d{local_cfg.get('local_delta_scale', 0.2)}"
        if config.get('network') == 'MoE':
            run_name += f"_moe{config['MoE']['num_experts']}k{config['MoE']['top_k']}_temp{config['MoE']['gate_temperature']}_noise{config['MoE']['gate_noise_std']}"
            if config.get('moe_strategy', 'vanilla') != 'vanilla':
                run_name += f"_{config['moe_strategy']}"
        if config.get('network') == 'MultiHeadMLP':
            mhe_cfg = config.get('MultiHeadMLP', {}) or {}
            run_name += f"_mhe{int(mhe_cfg.get('num_heads', config.get('mhe_num_heads', 5)))}"
            if mhe_cfg.get('head_hidden_dim') is not None:
                run_name += f"h{int(mhe_cfg['head_hidden_dim'])}"
        if en_subopt != 0:
            run_name += f"_subopt{en_subopt}_{config['subopt_ratio']}"
        if config['checkpoint']:
            ckpt_tag = os.path.basename(os.path.dirname(config['checkpoint']))
            ckpt_model = os.path.splitext(os.path.basename(config['checkpoint']))[0]
            run_name += f"_finetune_{ckpt_tag}_{ckpt_model}"
        run_name = append_seed_suffix(run_name, seed)
        result_save_dir = os.path.join(
            'results', prob_type, prob_name, str(opt_problem), run_name)

    os.makedirs(result_save_dir, exist_ok=True)

    _save_config_yaml(config, result_save_dir)

    return opt_problem, result_save_dir


def _save_config_yaml(config, save_dir):
    """Persist a human-readable copy of the full config."""
    serialisable = {k: v for k, v in config.items()
                    if not k.startswith('_')}
    path = os.path.join(save_dir, 'config.yaml')
    with open(path, 'w') as f:
        _yaml.dump(serialisable, f, default_flow_style=False, sort_keys=False)
    log.info("Config saved: %s", path)


def _get_mlp_output_layers(model, init_name):
    """Return final Linear layers from every MLP-like predictor in a model."""
    output_layers = []
    for module in model.modules():
        if isinstance(module, MLP):
            final_linear = module.mlp[-2]
            if not isinstance(final_linear, nn.Linear):
                raise ValueError(f"{init_name} init expected MLP to end with Linear + Sigmoid")
            output_layers.append(final_linear)

    if not output_layers:
        raise ValueError(f"{init_name} init requires at least one MLP output layer")

    return output_layers


def _initialize_output_center_layer(model, method, gain):
    """Initialize only output layers without using solution labels."""
    output_layers = _get_mlp_output_layers(model, 'output_center')
    for layer in output_layers:
        if layer.bias is None:
            raise ValueError("output_center init requires output-layer bias")
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    log.info(
        "Applied %s output-center init to %d output layer(s): "
        "final weight=xavier_uniform(gain=%.2f), final bias=0",
        method,
        len(output_layers),
        gain,
    )


def _initialize_mean_bias_output_layer(model, opt_problem, method, gain):
    """Bias the final sigmoid output toward the train-set mean solution.

    Hidden layers keep their default PyTorch initialization.
    The final linear layer gets a small Xavier initialization and a bias set to
    the logit of the normalized train-set mean solution.
    """
    output_layers = _get_mlp_output_layers(model, 'mean_bias')
    ref_layer = output_layers[0]
    output_dim = ref_layer.bias.numel()

    train_Y = opt_problem.train_dataset.tensors[1].to(
        device=ref_layer.weight.device,
        dtype=ref_layer.weight.dtype,
    )

    if output_dim == opt_problem.ydim:
        lower = opt_problem.L.view(1, -1).to(device=train_Y.device, dtype=train_Y.dtype)
        upper = opt_problem.U.view(1, -1).to(device=train_Y.device, dtype=train_Y.dtype)
        init_Y = train_Y
    elif output_dim == len(opt_problem.partial_vars):
        partial_vars = torch.as_tensor(opt_problem.partial_vars, device=train_Y.device)
        lower = opt_problem.L[partial_vars].view(1, -1).to(device=train_Y.device, dtype=train_Y.dtype)
        upper = opt_problem.U[partial_vars].view(1, -1).to(device=train_Y.device, dtype=train_Y.dtype)
        init_Y = train_Y[:, partial_vars]
    else:
        raise ValueError(
            f"mean_bias init output dim {output_dim} does not match full "
            f"dim {opt_problem.ydim} or partial dim {len(opt_problem.partial_vars)}"
        )

    denom = torch.clamp(upper - lower, min=1.0e-12)
    y_mean_norm = ((init_Y - lower) / denom).mean(dim=0)
    y_mean_norm = torch.clamp(y_mean_norm, min=1.0e-4, max=1.0 - 1.0e-4)

    for layer in output_layers:
        if layer.bias is None or layer.bias.numel() != y_mean_norm.numel():
            raise ValueError("mean_bias init target dimension does not match train-set mean")
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        with torch.no_grad():
            layer.bias.copy_(torch.logit(y_mean_norm))

    log.info(
        "Applied %s mean-bias init to %d output layer(s): final weight=xavier_uniform(gain=%.2f), "
        "final bias=logit(train_mean_norm), mean(train_mean_norm)=%.4f",
        method,
        len(output_layers),
        gain,
        y_mean_norm.mean().item(),
    )


def create_model(opt_problem, method, config):
    """Creates and returns a neural network model."""
    
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    network = config['network']
    dropout = config["dropout"]
    context_cfg_v1 = config.get("SampledContextMLPv1", {})
    context_cfg_v2 = config.get("SampledContextMLPv2", {})
    local_ctx_cfg = config.get("LocalContextMLPv2", {})

    if method == "DC3" or method == "sup_partial":
        out_dim = opt_problem.partial_vars.shape[0]
    else:
        out_dim = opt_problem.ydim

    if network == 'MLP':
        model = MLP(opt_problem.xdim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
    elif network == 'SampledContextMLPv1':
        context_num_points = int(context_cfg_v1.get('num_context_points', config.get('context_num_points', 16)))
        context_normalize = bool(context_cfg_v1.get('normalize', config.get('context_normalize', True)))
        context_eps = float(context_cfg_v1.get('eps', config.get('context_eps', 1e-8)))

        model = SampledContextMLPv1(
            opt_problem.xdim,
            hidden_dim,
            out_dim,
            problem_type=config['prob_type'],
            problem_name=config['prob_name'],
            L=opt_problem.L,
            U=opt_problem.U,
            A=opt_problem.A,
            G=opt_problem.G,
            h=opt_problem.h,
            Q=opt_problem.Q,
            p=opt_problem.p,
            c=opt_problem.c,
            H=getattr(opt_problem, 'H', None),
            C=getattr(opt_problem, 'C', None),
            d=getattr(opt_problem, 'd', None),
            num_context_points=context_num_points,
            seed=config.get('seed', 2025),
            context_normalize=context_normalize,
            context_eps=context_eps,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif network == 'SampledContextMLPv2':
        context_num_points = int(context_cfg_v2.get('num_context_points', 4))
        context_normalize = bool(context_cfg_v2.get('normalize', True))
        context_eps = float(context_cfg_v2.get('eps', 1e-8))
        context_encoder_dim = int(context_cfg_v2.get('context_encoder_dim', 128))

        model = SampledContextMLPv2(
            opt_problem.xdim,
            hidden_dim,
            out_dim,
            problem_type=config['prob_type'],
            problem_name=config['prob_name'],
            L=opt_problem.L,
            U=opt_problem.U,
            A=opt_problem.A,
            G=opt_problem.G,
            h=opt_problem.h,
            Q=opt_problem.Q,
            p=opt_problem.p,
            c=opt_problem.c,
            H=getattr(opt_problem, 'H', None),
            C=getattr(opt_problem, 'C', None),
            d=getattr(opt_problem, 'd', None),
            num_context_points=context_num_points,
            seed=config.get('seed', 2025),
            context_normalize=context_normalize,
            context_eps=context_eps,
            context_encoder_dim=context_encoder_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif network == 'LocalContextMLPv1':
        model = LocalContextMLPv1(
            opt_problem.xdim,
            hidden_dim,
            out_dim,
            problem_type=config['prob_type'],
            problem_name=config['prob_name'],
            L=opt_problem.L,
            U=opt_problem.U,
            A=opt_problem.A,
            G=opt_problem.G,
            h=opt_problem.h,
            Q=opt_problem.Q,
            p=opt_problem.p,
            c=opt_problem.c,
            H=getattr(opt_problem, 'H', None),
            C=getattr(opt_problem, 'C', None),
            d=getattr(opt_problem, 'd', None),
            num_layers=num_layers,
            dropout=dropout,
        )
    elif network == 'LocalContextMLPv2':
        model = LocalContextMLPv2(
            opt_problem.xdim,
            hidden_dim,
            out_dim,
            problem_type=config['prob_type'],
            problem_name=config['prob_name'],
            L=opt_problem.L,
            U=opt_problem.U,
            A=opt_problem.A,
            G=opt_problem.G,
            h=opt_problem.h,
            Q=opt_problem.Q,
            p=opt_problem.p,
            c=opt_problem.c,
            H=getattr(opt_problem, 'H', None),
            C=getattr(opt_problem, 'C', None),
            d=getattr(opt_problem, 'd', None),
            local_delta_scale=float(local_ctx_cfg.get('local_delta_scale', 0.2)),
            num_layers=num_layers,
            dropout=dropout,
        )
    elif network == 'MultiHeadMLP':
        mhe_cfg = config.get('MultiHeadMLP', {})
        num_heads = int(mhe_cfg.get('num_heads', config.get('mhe_num_heads', 5)))
        head_hidden_dim = mhe_cfg.get('head_hidden_dim',
                                       config.get('mhe_head_hidden_dim', None))
        head_hidden_dim = int(head_hidden_dim) if head_hidden_dim else None
        model = MultiHeadMLP(
            opt_problem.xdim, hidden_dim, out_dim,
            num_heads=num_heads,
            head_hidden_dim=head_hidden_dim,
            num_layers=num_layers, dropout=dropout,
        )
    elif network == 'MoE':
        moe_cfg = config.get('MoE', {})
        num_experts = moe_cfg.get('num_experts', config.get('num_experts', 4))
        top_k = moe_cfg.get('top_k', config.get('top_k', 2))
        gate_temperature = moe_cfg.get('gate_temperature', config.get('moe_gate_temperature', 1.0))
        gate_noise_std = moe_cfg.get('gate_noise_std', config.get('moe_gate_noise_std', 0.0))
        model = MixtureOfExperts(
            opt_problem.xdim, hidden_dim, out_dim,
            num_experts=num_experts, top_k=top_k,
            gate_temperature=gate_temperature,
            gate_noise_std=gate_noise_std,
            num_layers=num_layers, dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown network type: {network}")
    model = model.to(DEVICE)

    init_scheme = config.get(method, {}).get('init', 'default')
    init_gain = float(config.get(method, {}).get('init_gain', 0.1))
    if init_scheme == 'output_center':
        _initialize_output_center_layer(model, method, init_gain)
    elif init_scheme == 'mean_bias':
        _initialize_mean_bias_output_layer(model, opt_problem, method, init_gain)
    elif init_scheme != 'default':
        raise ValueError(f"Unknown init scheme for {method}: {init_scheme}")

    if isinstance(model, (SampledContextMLPv1, SampledContextMLPv2)):
        if isinstance(model, SampledContextMLPv2):
            fit_batch_size = int(context_cfg_v2.get('fit_batch_size', 256))
        else:
            fit_batch_size = int(context_cfg_v1.get('fit_batch_size', config.get('context_fit_batch_size', 256)))
        train_X = opt_problem.train_dataset.tensors[0]
        model.fit_context_stats(train_X, batch_size=fit_batch_size)

    return model


class Trainer:
    def __init__(self, opt_problem, config, save_dir=None):
        """Initializes the Trainer with opt_problem, method, and configuration."""
        self.opt_problem = opt_problem
        self.method = config['method']
        self.config = config
        self.save_dir = save_dir
        
        self.config_method = config[self.method]
        self.evaluator = Evaluator(opt_problem, self.method, config)
        self.en_feasibility = False
        self.en_penalty = False
        
        self._initialize_params()

    def _log_model_size(self, model=None, prefix="Model size"):
        """Log parameter-count and memory-size stats for a model."""
        stats = get_model_size_stats(model or self.model)
        log.info("%s: params=%d  trainable=%d  size=%.4f MB",
                 prefix, stats['total_params'], stats['trainable_params'], stats['total_mb'])
        return stats

    def compute_batch_loss(self, X_batch: torch.Tensor, Y_pred: torch.Tensor, Y_label: torch.Tensor, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and additional metrics."""
        Y_pred_scaled = self.opt_problem.scale(Y_pred)
        metrics = {}
        if self.method == "penalty":
            return self._penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "adaptive_penalty":
            return self._adaptive_penalty_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "FSNet":
            return self._fsnet_loss(X_batch, Y_pred_scaled, metrics)
        elif self.method == "S3Net":            
            return self._s3net_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "semi":            
            return self._semi_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup":
            return self._sup_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup_partial":
            return self._sup_partial_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "sup_pen":
            return self._sup_pen_loss(X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics)
        elif self.method == "DC3": 
            return self._dc3_loss(X_batch, Y_pred_scaled, metrics)            
        elif self.method == "projection":
            return self._projection_loss(X_batch, Y_pred_scaled, metrics)
        

    def _penalty_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the penalty loss."""
        obj = self.opt_problem.obj_fn(Y_pred_scaled)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
    
        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['eq_pen_weight'] * eq_violation + \
               self.config_method['ineq_pen_weight'] * ineq_violation

        if isinstance(self.model, LocalContextMLPv2) and self.model.last_coarse_prediction is not None:
            coarse_scaled = self.opt_problem.scale(self.model.last_coarse_prediction)
            coarse_obj = self.opt_problem.obj_fn(coarse_scaled)
            coarse_eq_violation = self.opt_problem.eq_resid(X_batch, coarse_scaled).square().sum(dim=1)
            coarse_ineq_violation = self.opt_problem.ineq_resid(X_batch, coarse_scaled).square().sum(dim=1)
            coarse_weight = float(self.config.get('LocalContextMLPv2', {}).get('coarse_loss_weight', 0.5))
            coarse_loss = self.config_method['obj_weight'] * coarse_obj + \
                          self.config_method['eq_pen_weight'] * coarse_eq_violation + \
                          self.config_method['ineq_pen_weight'] * coarse_ineq_violation
            loss = loss + coarse_weight * coarse_loss
            metrics['coarse_eq_violation'] = coarse_eq_violation.mean().item()
            metrics['coarse_ineq_violation'] = coarse_ineq_violation.mean().item()

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
        })
        return loss, metrics

    def _adaptive_penalty_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the adaptive penalty loss."""
        obj = self.opt_problem.obj_fn(Y_pred_scaled)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)

        loss = self.config_method['obj_weight'] * obj + \
               self.adaptive_eq_weight * eq_violation + \
               self.adaptive_ineq_weight * ineq_violation

        with torch.no_grad():
            self.adaptive_eq_weight = torch.clamp(self.adaptive_eq_weight + self.config_method['increasing_rate'] * eq_violation.mean(), min=0.0, max=self.config_method['eq_pen_weight_max'])
            self.adaptive_ineq_weight = torch.clamp(self.adaptive_ineq_weight + self.config_method['increasing_rate'] * ineq_violation.mean(), min=0.0, max=self.config_method['ineq_pen_weight_max'])
            if self.adaptive_eq_weight >= self.config_method['eq_pen_weight_max']:
                self.adaptive_eq_weight = self.config_method['eq_pen_weight_max']/2
            if self.adaptive_ineq_weight >= self.config_method['ineq_pen_weight_max']:
                self.adaptive_ineq_weight = self.config_method['ineq_pen_weight_max']/2

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
        })
        return loss, metrics
    
    def _fsnet_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the FSNet loss."""
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        # Feasibility refinement using hybrid L-BFGS
        Y_final = hybrid_lbfgs_solve(
            X_batch,
            Y_pred_scaled,
            self.opt_problem,
            val_tol=self.config_method['val_tol'],
            memory=self.config_method['memory_size'],
            max_iter=self.config_method['max_iter'],
            max_diff_iter=self.config_method['max_diff_iter'],
            scale=self.config_method['scale'],
            per_sample=self.config_method.get('per_sample_lbfgs', False),
        )
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square().mean()

        if pre_eq_violation.mean() >= 1e3 or pre_ineq_violation.mean() >= 1e3:
            loss = self.config_method['obj_weight'] * obj + \
                   self.config_method['dist_weight'] * distance +\
                   self.config_method['eq_pen_weight'] * pre_eq_violation + \
                   self.config_method['ineq_pen_weight'] * pre_ineq_violation
        else:
            loss = self.config_method['obj_weight'] * obj + \
                   self.config_method['dist_weight'] * distance
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics

    def _sup_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = huber(Y_final - Y_label).mean(dim=1)  # [B]

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics

    def _sup_partial_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        Y_final = Y_pred_scaled

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()
        obj = distance
        eq_violation = distance
        ineq_violation = distance
        eq_violation_l1 = distance
        ineq_violation_l1 = distance
        
        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = huber(Y_final - Y_label[:, self.opt_problem.partial_vars]).mean(dim=1)  # [B]

        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _sup_pen_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        sup_weight = 100.0 # prev 1.0

        # per-sample robust supervised loss
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = sup_weight * huber(Y_final - Y_label).mean(dim=1)  # [B]

        loss_eq_term = self.config_method['eq_pen_weight'] * pre_eq_violation 
        loss_ineq_term = self.config_method['ineq_pen_weight'] * pre_ineq_violation

        self.en_penalty = True
        if self.en_penalty:           
            loss += loss_eq_term + loss_ineq_term
            
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _s3net_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, Y_label: torch.Tensor, metrics: Dict, epoch_metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        pre_eq_violation = self.opt_problem.eq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)
        pre_ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_pred_scaled).square().sum(dim=1)

        if epoch_metrics['epoch'] > -1:
            if not self.en_penalty:
                log.debug("Enabling penalty terms")
            self.en_penalty = True

        if epoch_metrics['epoch'] > -1:
            if not self.en_feasibility:
                log.debug("Enabling feasibility seeking")
            self.en_feasibility = True

        if self.en_feasibility:
            Y_final = hybrid_lbfgs_solve(
                X_batch,
                Y_pred_scaled,
                self.opt_problem,
                val_tol=self.config_method['val_tol'],
                memory=self.config_method['memory_size'],
                max_iter=self.config_method['max_iter'],
                max_diff_iter=self.config_method['max_diff_iter'],
                scale=self.config_method['scale'],
                per_sample=self.config_method.get('per_sample_lbfgs', False),
            )
        else:
            Y_final = Y_pred_scaled
            
        obj = self.opt_problem.obj_fn(Y_final)
        
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)

        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        sup_weight = 2.0

        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        loss = sup_weight * huber(Y_final - Y_label).mean(dim=1)

        loss_obj_term = self.config_method['obj_weight'] * obj
        loss_dist_term = self.config_method['dist_weight'] * distance
        loss_eq_term = self.config_method['eq_pen_weight'] * pre_eq_violation
        loss_ineq_term = self.config_method['ineq_pen_weight'] * pre_ineq_violation

        if self.en_penalty:
            if pre_eq_violation.mean() >= 1e3 or pre_ineq_violation.mean() >= 1e3:              
                loss += loss_obj_term + loss_dist_term + loss_eq_term + loss_ineq_term
            else:
                loss += loss_obj_term + loss_dist_term
            
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
    
    def _semi_loss(self, X_batch, Y_pred_scaled, Y_label, metrics, epoch_metrics):
        if epoch_metrics['epoch'] > -1:
            if not self.en_penalty:
                log.debug("Enabling penalty terms")
            self.en_penalty = True

        if epoch_metrics['epoch'] > -1:
            if not self.en_feasibility:
                log.debug("Enabling feasibility seeking")
            self.en_feasibility = True

        B = X_batch.shape[0]
        idx_sup = torch.randperm(B, device=X_batch.device)[:B // 2]
        idx_unsup = torch.tensor([i for i in range(B) if i not in idx_sup], device=X_batch.device)

        # --- forward feasibility refinement (DC3 / FSNet-like) ---
        if self.en_feasibility:
            Y_final = hybrid_lbfgs_solve(
                X_batch, Y_pred_scaled, self.opt_problem,
                val_tol=self.config_method['val_tol'],
                memory=self.config_method['memory_size'],
                max_iter=self.config_method['max_iter'],
                max_diff_iter=self.config_method['max_diff_iter'],
                scale=self.config_method['scale'],
                per_sample=self.config_method.get('per_sample_lbfgs', False),
            )
        else:
            Y_final = Y_pred_scaled

        # --- compute objectives and constraints ---
        obj = self.opt_problem.obj_fn(Y_final)
        eq_viol = self.opt_problem.eq_resid(X_batch, Y_final)
        ineq_viol = self.opt_problem.ineq_resid(X_batch, Y_final)
        eq_violation = eq_viol.square().sum(dim=1)
        ineq_violation = ineq_viol.square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square()

        # --- supervised subset ---
        def huber(x, delta=1e-1):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5*x.pow(2)/delta, ax - 0.5*delta)

        # sup_weight = np.exp(-epoch_metrics['epoch'] / (0.3 * self.config['num_epochs']))
        sup_weight = 1.0
        loss_sup = torch.zeros(B, device=X_batch.device)
        loss_sup[idx_sup] = sup_weight * huber(Y_final[idx_sup] - Y_label[idx_sup]).mean(dim=1)

        # --- self-supervised subset ---
        loss_unsup = (
            self.config_method['obj_weight'] * obj +
            self.config_method['dist_weight'] * distance +
            self.config_method['eq_pen_weight'] * eq_violation +
            self.config_method['ineq_pen_weight'] * ineq_violation
        )
        loss_unsup[idx_sup] = 0.0  # only apply unsupervised terms to unlabeled subset

        # --- total loss ---
        loss = loss_sup + loss_unsup
        loss = loss.mean()

        # --- metric logging ---
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.mean().item(),
        })
        return loss, metrics
            
    def _dc3_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the DC3 loss."""
        Y_completion = self.opt_problem.complete_partial(X_batch, Y_pred_scaled)
        Y_final = grad_steps(self.opt_problem, X_batch, Y_completion, self.config)
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        
        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['eq_pen_weight'] * eq_violation + \
               self.config_method['ineq_pen_weight'] * ineq_violation
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
        })

        return loss, metrics
    
    def _projection_loss(self, X_batch: torch.Tensor, Y_pred_scaled: torch.Tensor, metrics: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the projection loss."""
        Y_final = self.opt_problem.qpth_projection(X_batch, Y_pred_scaled)
        obj = self.opt_problem.obj_fn(Y_final)
        eq_violation = self.opt_problem.eq_resid(X_batch, Y_final).square().sum(dim=1)
        ineq_violation = self.opt_problem.ineq_resid(X_batch, Y_final).square().sum(dim=1)
        eq_violation_l1 = self.opt_problem.eq_resid(X_batch, Y_final).abs().sum(dim=1)
        ineq_violation_l1 = self.opt_problem.ineq_resid(X_batch, Y_final).abs().sum(dim=1)
        distance = torch.norm(Y_final - Y_pred_scaled, dim=1).square().mean()

        loss = self.config_method['obj_weight'] * obj + \
               self.config_method['dist_weight'] * distance
        
        metrics.update({
            'obj': obj.mean().item(),
            'eq_violation': eq_violation.mean().item(),
            'ineq_violation': ineq_violation.mean().item(),
            'eq_violation_l1': eq_violation_l1.mean().item(),
            'ineq_violation_l1': ineq_violation_l1.mean().item(),
            'distance': distance.item(),
        })

        return loss, metrics

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'obj': 0.0, 'loss': 0.0, 'eq_violation': 0.0, 'ineq_violation': 0.0,
                         'eq_violation_l1': 0.0, 'ineq_violation_l1': 0.0, 'distance': 0.0,
                         'moe_aux_loss': 0.0, 'moe_gate_entropy': 0.0,
                         'moe_gate_max_prob': 0.0, 'epoch': epoch}

        for batch_idx, (X_batch, Y_label) in enumerate(train_loader):
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_label = Y_label.to(DEVICE, non_blocking=True)

            if getattr(self.model, 'is_multihead', False):
                all_preds = self.model.forward_all(X_batch)  # (M, B, out)
                M = all_preds.shape[0]
                total_loss = 0.0
                batch_metrics = {}
                for m in range(M):
                    loss_m, metrics_m = self.compute_batch_loss(
                        X_batch, all_preds[m], Y_label, epoch_metrics)
                    total_loss = total_loss + loss_m.mean()
                    if m == 0:
                        batch_metrics = metrics_m
                scalar_loss = total_loss / M
            else:
                Y_pred = self.model(X_batch)
                loss, batch_metrics = self.compute_batch_loss(X_batch, Y_pred, Y_label, epoch_metrics)
                scalar_loss = loss.mean()
            if isinstance(self.model, MixtureOfExperts):
                moe_cfg = self.config.get('MoE', {})
                moe_aux_weight = moe_cfg.get('aux_loss_weight', self.config.get('moe_aux_loss_weight', 0.01))
                moe_aux = moe_aux_weight * self.model.aux_loss
                scalar_loss = scalar_loss + moe_aux
                batch_metrics['moe_aux_loss'] = moe_aux.item()
                batch_metrics['moe_gate_entropy'] = self.model.gate_entropy.item()
                batch_metrics['moe_gate_max_prob'] = self.model.gate_max_prob.item()

            self.optimizer.zero_grad()
            scalar_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            epoch_metrics['loss'] += scalar_loss.item()
        
            self.scheduler.step()
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        epoch_metrics['epoch'] *= num_batches
            
        return epoch_metrics
    
    @property
    def use_wandb(self):
        return self.config.get('wandb', False) and WANDB_AVAILABLE

    def _initialize_params(self) -> None:
        if self.method == 'adaptive_penalty' or self.method == 'sup_pen':
            self.adaptive_eq_weight = self.config_method['eq_pen_weight']
            self.adaptive_ineq_weight = self.config_method['ineq_pen_weight']

        moe_cfg = self.config.get('MoE', {})
        self._moe_routing_state = {
            'warmup_epochs': int(moe_cfg.get('warmup_epochs', self.config.get('moe_warmup_epochs', 30))),
            'start_temp': float(moe_cfg.get('start_temp', self.config.get('moe_start_temp', moe_cfg.get('gate_temperature', self.config.get('moe_gate_temperature', 1.0))))),
            'final_temp': float(moe_cfg.get('final_temp', self.config.get('moe_final_temp', 1.0))),
            'noise_start': float(moe_cfg.get('gate_noise_std', self.config.get('moe_gate_noise_std', 0.0))),
            'noise_final': float(moe_cfg.get('gate_noise_final', self.config.get('moe_gate_noise_final', 0.0))),
            'decay_epochs': int(moe_cfg.get('temp_decay_epochs', self.config.get('moe_temp_decay_epochs', 200))),
        }

    def _update_moe_routing(self, epoch: int) -> None:
        """Stabilize MoE with dense warmup and annealed gate settings."""
        if not isinstance(self.model, MixtureOfExperts):
            return

        st = self._moe_routing_state
        warmup = max(st['warmup_epochs'], 0)
        after = max(epoch - warmup, 0)
        decay = max(st['decay_epochs'], 1)
        frac = min(after / decay, 1.0)

        gate_temp = st['start_temp'] + (st['final_temp'] - st['start_temp']) * frac
        gate_noise = st['noise_start'] + (st['noise_final'] - st['noise_start']) * frac
        force_dense = epoch < warmup

        self.model.set_routing(
            force_dense=force_dense,
            gate_temperature=gate_temp,
            gate_noise_std=gate_noise,
        )

           
    def _update_epoch_params(self, epoch: int) -> None:
        """Update parameters based on epoch."""
        # FSNet tolerance decay
        if ((self.method == 'FSNet' or self.method == 'S3Net' or self.method == 'semi') and (epoch + 1) % self.config_method['decay_tol_step'] == 0) and self.config['checkpoint'] is None:
            self.config_method['val_tol'] = np.clip(
                self.config_method['val_tol'] / 10, 
                a_min=1e-9, 
                a_max=1e-6
            )
        self._update_moe_routing(epoch)
        

    def train(self):
        """Main training loop with detailed results collection."""

        train_loader, val_loader = self._prepare_data_loaders()

        if self.config['checkpoint']:
            log.info("Loading checkpoint: %s", self.config['checkpoint'])
            ckpt = torch.load(self.config['checkpoint'], map_location=DEVICE, weights_only=False)
            ckpt['config']['dropout'] = self.config['dropout']
            self.model = create_model(self.opt_problem, self.method, ckpt['config'])
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model = create_model(self.opt_problem, self.method, self.config)

        self._log_model_size()
        self._init_optimizer_and_scheduler(train_loader)

        log.info("lr=%.2e  weight_decay=1e-3  epochs=%d",
                 self.config_method['lr'], self.config_method['num_epochs'])

        # Training history
        train_history = []
        val_history = []

        train_start = time.time()
        for epoch in range(self.config_method['num_epochs']):
            self._update_epoch_params(epoch)
            epoch_start = time.time()
            
            # Train for one epoch
            self.model.train()
            epoch_metrics = self.train_epoch(train_loader, epoch)
            train_history.append({'epoch': epoch, **epoch_metrics})
            epoch_end = time.time()
       
            log.info("Ep %d/%d  Loss=%.2f  Obj=%.2f  EqV=%.6f  IneqV=%.6f  T=%.2fs",
                     epoch + 1, self.config_method['num_epochs'],
                     epoch_metrics['loss'], epoch_metrics.get('obj', 0),
                     epoch_metrics.get('eq_violation_l1', 0),
                     epoch_metrics.get('ineq_violation_l1', 0),
                     epoch_end - epoch_start)
            if isinstance(self.model, MixtureOfExperts):
                log.info("MoE  aux=%.4e  H(gate)=%.4f  maxP=%.4f  temp=%.3f  noise=%.3f",
                         epoch_metrics.get('moe_aux_loss', 0.0),
                         epoch_metrics.get('moe_gate_entropy', 0.0),
                         epoch_metrics.get('moe_gate_max_prob', 0.0),
                         float(getattr(self.model, 'gate_temperature', 1.0)),
                         float(getattr(self.model, 'gate_noise_std', 0.0)))

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': epoch_metrics['loss'],
                    'train/objective': epoch_metrics.get('obj', 0),
                    'train/eq_violation_l1': epoch_metrics.get('eq_violation_l1', 0),
                    'train/ineq_violation_l1': epoch_metrics.get('ineq_violation_l1', 0),
                    'train/eq_violation_l2': epoch_metrics.get('eq_violation', 0),
                    'train/ineq_violation_l2': epoch_metrics.get('ineq_violation', 0),
                    'train/distance': epoch_metrics.get('distance', 0),
                    'train/moe_aux_loss': epoch_metrics.get('moe_aux_loss', 0),
                    'train/moe_gate_entropy': epoch_metrics.get('moe_gate_entropy', 0),
                    'train/moe_gate_max_prob': epoch_metrics.get('moe_gate_max_prob', 0),
                    'train/epoch_time': epoch_end - epoch_start,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })

            if epoch % self.config['eval_step'] == 0:
                log.info("Validation at epoch %d", epoch)
                val_metrics = self.evaluator.evaluate(self.model, val_loader, f"validation_epoch_{epoch}")
                val_history.append({**val_metrics, 'epoch': epoch})

                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'val/objective': val_metrics.get('objective', 0),
                        'val/opt_gap_mean': val_metrics.get('opt_gap_mean', 0),
                        'val/opt_gap_max': val_metrics.get('opt_gap_max', 0),
                        'val/eq_violation_l1': val_metrics.get('eq_violation_l1_mean', 0),
                        'val/ineq_violation_l1': val_metrics.get('ineq_violation_l1_mean', 0),
                        'val/eq_violation_l2': val_metrics.get('eq_violation_l2_mean', 0),
                        'val/ineq_violation_l2': val_metrics.get('ineq_violation_l2_mean', 0),
                        'val/merit_mean': val_metrics.get('merit_mean', 0),
                        'val/solution_distance': val_metrics.get('solution_distance_mean', 0),
                        'val/inference_time': val_metrics.get('avg_inference_time', 0),
                    })

                if self.save_dir and self.config['save_intermediate']:
                    self._save_model(epoch)

        train_end = time.time()
        training_time = train_end - train_start
        log.info("Training completed in %.2fs", training_time)

        if hasattr(self.opt_problem, 'test_dataset'):
            log.info("=" * 60)
            log.info("TEST EVALUATION")
            log.info("=" * 60)

            test_batch_sizes = self.config.get('test_batch_sizes', [256, 512])
            log.info("Test batch sizes: %s", test_batch_sizes)
            
            # Run evaluation with all batch sizes and collect detailed results for all
            batch_size_results, all_detailed_results = self.evaluator.evaluate_multiple_batch_sizes(
                self.model, 
                self.opt_problem.test_dataset, 
                test_batch_sizes, 
                "test"
            )
            
            # Combine all test results
            final_test_results = {
                'batch_size_comparison': batch_size_results,
                'detailed_results_all_batch_sizes': all_detailed_results
            }

            if self.use_wandb:
                for bs, result in batch_size_results.items():
                    if 'error' not in result:
                        metrics = result['metrics']
                        wandb.log({
                            f'test/bs{bs}/objective': metrics.get('objective', 0),
                            f'test/bs{bs}/opt_gap_mean': metrics.get('opt_gap_mean', 0),
                            f'test/bs{bs}/opt_gap_max': metrics.get('opt_gap_max', 0),
                            f'test/bs{bs}/eq_violation_l1': metrics.get('eq_violation_l1_mean', 0),
                            f'test/bs{bs}/ineq_violation_l1': metrics.get('ineq_violation_l1_mean', 0),
                            f'test/bs{bs}/merit_mean': metrics.get('merit_mean', 0),
                            f'test/bs{bs}/solution_distance': metrics.get('solution_distance_mean', 0),
                            f'test/bs{bs}/total_time': metrics.get('total_time', 0),
                        })
                first_valid = next((r['metrics'] for r in batch_size_results.values() if 'error' not in r), None)
                if first_valid:
                    wandb.summary.update({
                        'test/objective': first_valid.get('objective', 0),
                        'test/opt_gap_mean': first_valid.get('opt_gap_mean', 0),
                        'test/eq_violation_l1': first_valid.get('eq_violation_l1_mean', 0),
                        'test/ineq_violation_l1': first_valid.get('ineq_violation_l1_mean', 0),
                        'test/merit_mean': first_valid.get('merit_mean', 0),
                        'test/solution_distance': first_valid.get('solution_distance_mean', 0),
                    })
        else:
            log.warning("No test dataset available")
            final_test_results = {}

        if self.save_dir:
            self._save_model_and_results(
                train_history, 
                val_history, 
                final_test_results, 
                training_time
            )
        
        return self.model

    # ------------------------------------------------------------------
    # Ensemble training
    # ------------------------------------------------------------------
    def train_ensemble(self):
        """Train a deep ensemble with either vanilla or FGE mode."""
        mode = self.config.get('ensemble_mode', 'vanilla')
        if mode == 'vanilla':
            return self._train_vanilla_ensemble()
        elif mode == 'fge':
            return self._train_fge_ensemble()
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")

    def _prepare_data_loaders(self):
        """Shared data loader setup used by both single and ensemble training.

        Automatically adjusts batch size for small datasets and attaches a
        seeded Generator to the train loader for reproducible shuffling.
        """
        batch_size = self.config['batch_size']
        train_size = len(self.opt_problem.train_dataset)
        thresholds = [(50, 16), (100, 32), (500, 64), (1000, 128), (5000, 256)]
        for limit, bs in thresholds:
            if train_size <= limit:
                batch_size = bs
                break
        self.config['batch_size'] = batch_size

        generator = self.config.get('_generator', None)

        train_loader = DataLoader(
            self.opt_problem.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            self.opt_problem.val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        log.info("Batch size: %d (train_size=%d)", batch_size, train_size)
        return train_loader, val_loader

    def _init_optimizer_and_scheduler(self, train_loader):
        """Create optimizer and LR scheduler for the current self.model."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config_method['lr'],
            weight_decay=0.001,
            fused=True,
        )
        warmup_steps = len(train_loader)
        total_steps = len(train_loader) * self.config_method['num_epochs']
        s1 = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[s1, s2], milestones=[warmup_steps])

    def _init_model_and_optimizer(self, train_loader, seed=None):
        """Create a fresh model, optimizer and scheduler."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.model = create_model(self.opt_problem, self.method, self.config)
        self._log_model_size(prefix="Member model size")
        self._init_optimizer_and_scheduler(train_loader)

    def _run_training_loop(self, train_loader, val_loader, num_epochs,
                           start_epoch=0, member_tag="", save_tag=None,
                           wandb_suffix=""):
        """Run the core training loop for *num_epochs* epochs.

        Args:
            save_tag: When set and ``save_intermediate`` is True, intermediate
                checkpoints are written as ``{save_tag}_epoch_{epoch}.pt``.
            wandb_suffix: Suffix appended to wandb metric names (e.g.
                ``"_m0"`` for vanilla ensemble members).  Each metric
                becomes its own section (e.g. ``train_loss/m0``) so that
                members are easy to compare within one chart.
        """
        train_history, val_history = [], []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            self._update_epoch_params(epoch)
            epoch_start = time.time()

            self.model.train()
            epoch_metrics = self.train_epoch(train_loader, epoch)
            train_history.append({'epoch': epoch, **epoch_metrics})
            epoch_end = time.time()

            log.info("%sEp %d/%d  Loss=%.2f  Obj=%.2f  EqV=%.6f  IneqV=%.6f  T=%.2fs",
                     member_tag, epoch + 1, start_epoch + num_epochs,
                     epoch_metrics['loss'], epoch_metrics.get('obj', 0),
                     epoch_metrics.get('eq_violation_l1', 0),
                     epoch_metrics.get('ineq_violation_l1', 0),
                     epoch_end - epoch_start)

            if self.use_wandb:
                ws = wandb_suffix
                def _k(section, name):
                    """Build wandb key: 'section/name' or 'section_name/mX'."""
                    if ws:
                        return f'{section}_{name}/{ws.lstrip("_")}'
                    return f'{section}/{name}'
                wandb.log({
                    'epoch': epoch,
                    _k('train', 'loss'): epoch_metrics['loss'],
                    _k('train', 'objective'): epoch_metrics.get('obj', 0),
                    _k('train', 'eq_violation_l1'): epoch_metrics.get('eq_violation_l1', 0),
                    _k('train', 'ineq_violation_l1'): epoch_metrics.get('ineq_violation_l1', 0),
                    _k('train', 'distance'): epoch_metrics.get('distance', 0),
                    (f'lr/{ws.lstrip("_")}' if ws else 'lr'): self.optimizer.param_groups[0]['lr'],
                })

            if epoch % self.config['eval_step'] == 0:
                log.info("%sValidation at epoch %d", member_tag, epoch)
                val_metrics = self.evaluator.evaluate(self.model, val_loader, f"{member_tag}val_epoch_{epoch}")
                val_history.append({**val_metrics, 'epoch': epoch})

                if self.use_wandb:
                    ws = wandb_suffix
                    def _kv(section, name):
                        if ws:
                            return f'{section}_{name}/{ws.lstrip("_")}'
                        return f'{section}/{name}'
                    wandb.log({
                        'epoch': epoch,
                        _kv('val', 'objective'): val_metrics.get('objective', 0),
                        _kv('val', 'opt_gap_mean'): val_metrics.get('opt_gap_mean', 0),
                        _kv('val', 'eq_violation_l1'): val_metrics.get('eq_violation_l1_mean', 0),
                        _kv('val', 'ineq_violation_l1'): val_metrics.get('ineq_violation_l1_mean', 0),
                        _kv('val', 'merit_mean'): val_metrics.get('merit_mean', 0),
                    })

                if self.save_dir and self.config['save_intermediate'] and save_tag:
                    self._save_member_checkpoint(save_tag, epoch)
            
                # if self.config['method'] == 'FSNet' and epoch > 100: # debug FSNet convergence
                #     break
        return train_history, val_history

    def _evaluate_and_save_ensemble(self, ensemble_model, train_history, val_history, training_time):
        """Run test evaluation on the ensemble and save results."""
        self.model = ensemble_model
        final_test_results = {}

        if hasattr(self.opt_problem, 'test_dataset'):
            log.info("=" * 60)
            log.info("ENSEMBLE TEST EVALUATION")
            log.info("=" * 60)

            test_batch_sizes = self.config.get('test_batch_sizes', [256, 512])
            batch_size_results, all_detailed_results = self.evaluator.evaluate_multiple_batch_sizes(
                ensemble_model,
                self.opt_problem.test_dataset,
                test_batch_sizes,
                "ensemble_test",
            )

            final_test_results = {
                'batch_size_comparison': batch_size_results,
                'detailed_results_all_batch_sizes': all_detailed_results,
            }

            if self.use_wandb:
                first_valid = next((r['metrics'] for r in batch_size_results.values() if 'error' not in r), None)
                if first_valid:
                    wandb.summary.update({
                        'ensemble_test/objective': first_valid.get('objective', 0),
                        'ensemble_test/opt_gap_mean': first_valid.get('opt_gap_mean', 0),
                        'ensemble_test/eq_violation_l1': first_valid.get('eq_violation_l1_mean', 0),
                        'ensemble_test/ineq_violation_l1': first_valid.get('ineq_violation_l1_mean', 0),
                        'ensemble_test/merit_mean': first_valid.get('merit_mean', 0),
                        'ensemble_test/solution_distance': first_valid.get('solution_distance_mean', 0),
                    })

        if self.save_dir:
            self._save_model_and_results(train_history, val_history, final_test_results, training_time)

        return ensemble_model

    # ---- Vanilla deep ensemble ----
    def _train_vanilla_ensemble(self):
        """Train M models from independent random initializations."""
        M = self.config['ensemble_size']
        base_seed = self.config['seed']
        train_loader, val_loader = self._prepare_data_loaders()

        all_train_history, all_val_history = [], []
        member_models = []

        train_start = time.time()
        for i in range(M):
            member_seed = base_seed + i
            log.info("=" * 60)
            log.info("VANILLA ENSEMBLE: member %d/%d  seed=%d", i + 1, M, member_seed)
            log.info("=" * 60)

            self._init_model_and_optimizer(train_loader, seed=member_seed)
            tag = f"[m{i}] "
            hist_t, hist_v = self._run_training_loop(
                train_loader, val_loader,
                num_epochs=self.config_method['num_epochs'],
                member_tag=tag,
                save_tag=f"member_{i}",
                wandb_suffix=f"_m{i}",
            )
            all_train_history.extend(hist_t)
            all_val_history.extend(hist_v)
            member_models.append(copy.deepcopy(self.model))

        training_time = time.time() - train_start
        log.info("Vanilla ensemble completed in %.2fs (%d members)", training_time, M)

        ensemble_model = EnsembleMLP(member_models).to(DEVICE)
        return self._evaluate_and_save_ensemble(ensemble_model, all_train_history, all_val_history, training_time)

    # ---- Fast Geometric Ensembling ----
    def _train_fge_ensemble(self):
        """Pre-train one model, then collect snapshots with cyclical LR."""
        M = self.config['ensemble_size']
        pretrain_ratio = self.config.get('fge_pretrain_ratio', 0.8)
        total_epochs = self.config_method['num_epochs']
        pretrain_epochs = int(total_epochs * pretrain_ratio)
        fge_epochs = total_epochs - pretrain_epochs
        cycle_length = max(1, fge_epochs // M)

        train_loader, val_loader = self._prepare_data_loaders()

        log.info("FGE plan: %d pre-train + %d snapshot epochs (%d snapshots, cycle=%d)",
                 pretrain_epochs, fge_epochs, M, cycle_length)

        log.info("=" * 60)
        log.info("FGE PHASE 1: Pre-training")
        log.info("=" * 60)

        self._init_model_and_optimizer(train_loader, seed=self.config['seed'])

        train_start = time.time()
        hist_t, hist_v = self._run_training_loop(
            train_loader, val_loader,
            num_epochs=pretrain_epochs,
            member_tag="[pretrain] ",
            save_tag="pretrain",
        )

        log.info("=" * 60)
        log.info("FGE PHASE 2: Cyclical LR snapshot collection")
        log.info("=" * 60)

        lr_max = self.config.get('fge_lr_max') or self.config_method['lr']
        lr_min = 1e-6
        snapshots = []
        steps_per_epoch = len(train_loader)

        for snap_idx in range(M):
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cycle_length * steps_per_epoch,
                T_mult=1,
                eta_min=lr_min,
            )
            for g in self.optimizer.param_groups:
                g['lr'] = lr_max

            tag = f"[fge-snap{snap_idx}] "
            h_t, h_v = self._run_training_loop(
                train_loader, val_loader,
                num_epochs=cycle_length,
                start_epoch=pretrain_epochs + snap_idx * cycle_length,
                member_tag=tag,
                save_tag=f"member_{snap_idx}",
            )
            hist_t.extend(h_t)
            hist_v.extend(h_v)

            snapshots.append(copy.deepcopy(self.model))
            snap_epoch = pretrain_epochs + (snap_idx + 1) * cycle_length
            log.info("Snapshot %d/%d collected at epoch %d", snap_idx + 1, M, snap_epoch)

        training_time = time.time() - train_start
        log.info("FGE completed in %.2fs (%d snapshots)", training_time, M)

        ensemble_model = EnsembleMLP(snapshots).to(DEVICE)
        return self._evaluate_and_save_ensemble(ensemble_model, hist_t, hist_v, training_time)

    def _serialisable_config(self):
        return {k: v for k, v in self.config.items() if not k.startswith('_')}

    def _build_model_payload(self, model=None):
        """Build the dict persisted as a .pt checkpoint."""
        model = model or self.model
        return {
            'model_state_dict': model.state_dict(),
            'model_architecture_str': str(model),
            'config': self._serialisable_config(),
        }

    def _save_model(self, epoch: int = None):
        """Save an intermediate checkpoint (model weights only)."""
        if not self.save_dir:
            log.warning("No save_dir set, skipping checkpoint.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        filename = f"model_{epoch}.pt" if epoch is not None else "model.pt"
        path = os.path.join(self.save_dir, filename)
        try:
            torch.save(self._build_model_payload(), path)
            log.info("Model saved: %s", path)
        except Exception as e:
            log.error("Error saving model: %s", e)

    def _save_member_checkpoint(self, tag: str, epoch):
        """Save an intermediate checkpoint for a single ensemble member.

        Written to ``{save_dir}/members/{tag}_epoch_{epoch}.pt``.
        """
        if not self.save_dir:
            return
        members_dir = os.path.join(self.save_dir, "members")
        os.makedirs(members_dir, exist_ok=True)
        path = os.path.join(members_dir, f"{tag}_epoch_{epoch}.pt")
        try:
            torch.save(self._build_model_payload(), path)
            log.info("Member checkpoint saved: %s", path)
        except Exception as e:
            log.error("Error saving member checkpoint: %s", e)

    def _save_ensemble_members(self, ensemble_model):
        """Save each member of an EnsembleMLP as a separate .pt file.

        Layout::

            {save_dir}/members/
                member_0.pt
                member_1.pt
                ...
        """
        if not self.save_dir:
            return
        members_dir = os.path.join(self.save_dir, "members")
        os.makedirs(members_dir, exist_ok=True)
        for i, member in enumerate(ensemble_model.members):
            path = os.path.join(members_dir, f"member_{i}.pt")
            try:
                torch.save(self._build_model_payload(model=member), path)
                log.info("Member %d saved: %s", i, path)
            except Exception as e:
                log.error("Error saving member %d: %s", i, e)

    def _save_model_and_results(self, train_history, val_history,
                                test_results_data, training_time):
        """Save model.pt, test_summary.yaml, and results.pkl.

        Layout::

            model.pt            – model weights
            test_summary.yaml   – human-readable aggregated test metrics
            results.pkl         – detailed per-sample test tensors only
            members/            – (ensembles) individual member checkpoints
        """
        if not self.save_dir:
            log.warning("No save_dir set, skipping save.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        log.info("Saving to: %s", self.save_dir)
        model_size = get_model_size_stats(self.model)

        # ---- model.pt ----
        model_path = os.path.join(self.save_dir, "model.pt")
        try:
            torch.save(self._build_model_payload(), model_path)
            log.info("Model saved: %s", model_path)
        except Exception as e:
            log.error("Error saving model: %s", e)

        if isinstance(self.model, EnsembleMLP):
            self._save_ensemble_members(self.model)

        # ---- test_summary.yaml (human-readable) ----
        summary = {
            'seed': self.config.get('seed', 'N/A'),
            'method': self.method,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_time_seconds': round(training_time, 2),
            'pytorch_version': torch.__version__,
            'device': str(DEVICE),
            'model_size': {
                'total_params': model_size['total_params'],
                'trainable_params': model_size['trainable_params'],
                'param_bytes': model_size['param_bytes'],
                'buffer_bytes': model_size['buffer_bytes'],
                'total_bytes': model_size['total_bytes'],
                'total_mb': round(model_size['total_mb'], 8),
            },
            'opt_gap_unit': 'percent',
        }
        if test_results_data and 'batch_size_comparison' in test_results_data:
            bs_metrics = {}
            for bs in sorted(test_results_data['batch_size_comparison'], key=int):
                result = test_results_data['batch_size_comparison'][bs]
                if 'error' in result:
                    bs_metrics[int(bs)] = {'error': result['error']}
                else:
                    metrics = result['metrics']
                    bs_metrics[int(bs)] = {
                        k: round(float(metrics[k]), 8)
                        for k in sorted(metrics)
                    }
            summary['test'] = bs_metrics
        summary_path = os.path.join(self.save_dir, "test_summary.yaml")
        try:
            with open(summary_path, 'w') as f:
                _yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
            log.info("Test summary saved: %s", summary_path)
        except Exception as e:
            log.error("Error saving test summary: %s", e)

        # ---- results.pkl (detailed per-sample tensors) ----
        detailed = {
            'train_history': train_history,
            'val_history': val_history,
            'model_size': model_size,
        }
        if test_results_data and 'detailed_results_all_batch_sizes' in test_results_data:
            detailed['test_detailed'] = test_results_data['detailed_results_all_batch_sizes']
        results_path = os.path.join(self.save_dir, "results.pkl")
        try:
            with open(results_path, 'wb') as f:
                pickle.dump(detailed, f)
            log.info("Detailed results saved: %s", results_path)
        except Exception as e:
            log.error("Error saving detailed results: %s", e)

        # ---- W&B artifact ----
        if self.use_wandb:
            try:
                artifact = wandb.Artifact(
                    f"model-{self.config.get('prob_type','')}-"
                    f"{self.config.get('prob_name','')}-{self.method}",
                    type='model',
                    metadata={'training_time': training_time},
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                wandb.summary['training_time_seconds'] = training_time
                log.info("W&B artifact logged")
            except Exception as e:
                log.warning("W&B artifact logging failed: %s", e)
