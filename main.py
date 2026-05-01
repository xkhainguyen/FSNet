import logging
import os
import random
import yaml
import numpy as np
import torch
import time
import argparse
from utils.trainer import load_instance, Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

log = logging.getLogger(__name__)

PROBLEM_TYPES = ['convex', 'nonconvex', 'nonsmooth_nonconvex']
PROBLEM_NAMES = ['qp', 'qcqp', 'socp']


def set_seed(seed: int, strict: bool = False) -> torch.Generator | None:
    """Set seeds and optionally enable strict deterministic mode.

    strict=False keeps fast/non-strict CuDNN behavior, which can lead to
    better optimisation trajectories for this task.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return None


def setup_logging(save_dir: str = None, level: int = logging.INFO):
    """Configure root logger with console + optional file handler.

    Console output is kept minimal (message only).
    The log file gets the full format with timestamps for later analysis.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
        fh.setFormatter(file_fmt)
        root.addHandler(fh)

METHODS = [
    'penalty', 'adaptive_penalty', 'FSNet', 'DC3', 'projection',
    'sup', 'sup_partial', 'sup_pen', 'semi', 'S3Net',
]


def _merge_defaults(config):
    """Merge ``defaults`` section into every method dict so that
    ``config[method]`` is always a complete flat dict."""
    defaults = config.pop('defaults', {})
    for method_name in METHODS:
        method_cfg = config.get(method_name)
        if method_cfg is None:
            method_cfg = {}
        config[method_name] = {**defaults, **method_cfg}


def _normalize_moe_config(config):
    """Ensure canonical nested MoE config while supporting legacy flat keys."""
    moe = dict(config.get('MoE', {}) or {})

    legacy_map = {
        'num_experts': 'num_experts',
        'top_k': 'top_k',
        'moe_aux_loss_weight': 'aux_loss_weight',
        'moe_gate_temperature': 'gate_temperature',
        'moe_gate_noise_std': 'gate_noise_std',
        'moe_warmup_epochs': 'warmup_epochs',
        'moe_start_temp': 'start_temp',
        'moe_final_temp': 'final_temp',
        'moe_gate_noise_final': 'gate_noise_final',
        'moe_temp_decay_epochs': 'temp_decay_epochs',
    }
    for legacy_key, nested_key in legacy_map.items():
        if nested_key not in moe and legacy_key in config:
            moe[nested_key] = config[legacy_key]

    defaults = {
        'num_experts': 4,
        'top_k': 2,
        'aux_loss_weight': 0.01,
        'gate_temperature': 1.0,
        'gate_noise_std': 0.0,
        'warmup_epochs': 30,
        'start_temp': 2.0,
        'final_temp': 1.0,
        'gate_noise_final': 0.0,
        'temp_decay_epochs': 200,
    }
    for key, val in defaults.items():
        moe.setdefault(key, val)

    config['MoE'] = moe


def _normalize_context_configs(config):
    """Ensure canonical nested configs for SampledContextMLPv1 and SampledContextMLPv2."""
    legacy_map = {
        'context_num_points': 'num_context_points',
        'context_normalize': 'normalize',
        'context_fit_batch_size': 'fit_batch_size',
        'context_eps': 'eps',
        'context_encoder_dim': 'context_encoder_dim',
    }

    ctx_v1 = dict(config.get('SampledContextMLPv1', {}) or {})
    for legacy_key, nested_key in legacy_map.items():
        if nested_key not in ctx_v1 and legacy_key in config:
            ctx_v1[nested_key] = config[legacy_key]
    for key, val in {
        'num_context_points': 16,
        'normalize': True,
        'fit_batch_size': 256,
        'eps': 1.0e-8,
    }.items():
        ctx_v1.setdefault(key, val)
    config['SampledContextMLPv1'] = ctx_v1

    ctx_v2 = dict(config.get('SampledContextMLPv2', {}) or {})
    for legacy_key, nested_key in legacy_map.items():
        if nested_key not in ctx_v2 and legacy_key in config:
            ctx_v2[nested_key] = config[legacy_key]
    for key, val in {
        'num_context_points': 4,
        'normalize': True,
        'fit_batch_size': 256,
        'eps': 1.0e-8,
        'context_encoder_dim': 128,
    }.items():
        ctx_v2.setdefault(key, val)
    config['SampledContextMLPv2'] = ctx_v2

    local_ctx = dict(config.get('LocalContextMLPv2', {}) or {})
    for key, val in {
        'local_delta_scale': 0.2,
        'coarse_loss_weight': 0.5,
    }.items():
        local_ctx.setdefault(key, val)
    config['LocalContextMLPv2'] = local_ctx


def create_parser():
    """Create and configure the argument parser, then load and process the configuration."""
    parser = argparse.ArgumentParser(description='Neural Network Optimization')

    # General
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--method', type=str, choices=METHODS,
                        help='Training method')
    parser.add_argument('--prob_type', type=str, choices=PROBLEM_TYPES)
    parser.add_argument('--prob_name', type=str, choices=PROBLEM_NAMES)
    parser.add_argument('--prob_size', type=int, nargs='+', default=[100, 50, 50, 10000])
    parser.add_argument('--network', type=str, default='MLP')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--strict_repro', action='store_true',
                        help='Enable strict deterministic behavior (CuDNN deterministic + seeded DataLoader)')
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--en_subopt', type=int, default=0)
    parser.add_argument('--subopt_ratio', type=float, default=0)
    parser.add_argument('--save_intermediate', type=bool, default=False)

    # Data
    parser.add_argument('--train_size', type=int, default=7000)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_size', type=int)
    parser.add_argument('--test_size', type=int)

    # Model / training (override method defaults)
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--init', type=str, choices=['default', 'mean_bias'],
                        help='Network initialization scheme for the selected method')
    parser.add_argument('--num_experts', type=int)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--moe_num_experts', type=int)
    parser.add_argument('--moe_top_k', type=int)
    parser.add_argument('--moe_aux_loss_weight', type=float)
    parser.add_argument('--moe_gate_temperature', type=float)
    parser.add_argument('--moe_gate_noise_std', type=float)
    parser.add_argument('--moe_warmup_epochs', type=int)
    parser.add_argument('--moe_start_temp', type=float)
    parser.add_argument('--moe_final_temp', type=float)
    parser.add_argument('--moe_gate_noise_final', type=float)
    parser.add_argument('--moe_temp_decay_epochs', type=int)
    parser.add_argument('--moe_post', type=str, choices=['pre', 'post'])
    parser.add_argument('--moe_strategy', type=str,
                        choices=['vanilla', 'top2_best_merit'])
    parser.add_argument('--moe_agg', type=str,
                        choices=['router', 'mean', 'best_obj', 'best_merit'])
    parser.add_argument('--moe_candidate_top_k', type=int)
    parser.add_argument('--context_num_points', type=int)
    parser.add_argument('--context_fit_batch_size', type=int)
    parser.add_argument('--context_eps', type=float)
    parser.add_argument('--context_encoder_dim', type=int)
    parser.add_argument('--local_delta_scale', type=float)
    parser.add_argument('--local_coarse_loss_weight', type=float)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--dist_weight', type=float)
    parser.add_argument('--max_diff_iter', type=int)
    parser.add_argument('--val_tol', type=float)
    parser.add_argument('--decay_tol_step', type=int)
    parser.add_argument('--memory_size', type=int)

    # Ensemble
    parser.add_argument('--ensemble_size', type=int, default=1)
    parser.add_argument('--ensemble_mode', type=str, default='vanilla',
                        choices=['vanilla', 'fge'])
    parser.add_argument('--fge_pretrain_ratio', type=float, default=0.8)
    parser.add_argument('--fge_lr_max', type=float, default=None)
    parser.add_argument('--ensemble_post', type=str, default='pre',
                        choices=['pre', 'post'])
    parser.add_argument('--ensemble_agg', type=str, default='mean',
                        choices=['mean', 'median', 'best_obj', 'best_merit'])

    # W&B
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='FSNet')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    _merge_defaults(config)
    _normalize_moe_config(config)
    _normalize_context_configs(config)

    # ── Top-level overrides ──
    if args.method:
        config['method'] = args.method
    config['seed'] = args.seed
    if args.prob_type:
        config['prob_type'] = args.prob_type
    if args.prob_name:
        config['prob_name'] = args.prob_name
    if args.prob_size:
        config['prob_size'] = args.prob_size
    if args.network:
        config['network'] = args.network

    config['checkpoint'] = args.checkpoint
    config['en_subopt'] = args.en_subopt
    config['subopt_ratio'] = args.subopt_ratio
    config['save_intermediate'] = args.save_intermediate
    config['ablation'] = args.ablation

    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.train_size:
        config['train_size'] = args.train_size
    if args.val_size:
        config['val_size'] = args.val_size
    if args.test_size:
        config['test_size'] = args.test_size
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['num_layers'] = args.num_layers
    if args.dropout:
        config['dropout'] = args.dropout

    # MoE overrides
    if args.moe_num_experts is not None:
        config['MoE']['num_experts'] = args.moe_num_experts
    elif args.num_experts is not None:
        config['MoE']['num_experts'] = args.num_experts
    if args.moe_top_k is not None:
        config['MoE']['top_k'] = args.moe_top_k
    elif args.top_k is not None:
        config['MoE']['top_k'] = args.top_k
    if args.moe_aux_loss_weight is not None:
        config['MoE']['aux_loss_weight'] = args.moe_aux_loss_weight
    if args.moe_gate_temperature is not None:
        config['MoE']['gate_temperature'] = args.moe_gate_temperature
    if args.moe_gate_noise_std is not None:
        config['MoE']['gate_noise_std'] = args.moe_gate_noise_std
    if args.moe_warmup_epochs is not None:
        config['MoE']['warmup_epochs'] = args.moe_warmup_epochs
    if args.moe_start_temp is not None:
        config['MoE']['start_temp'] = args.moe_start_temp
    if args.moe_final_temp is not None:
        config['MoE']['final_temp'] = args.moe_final_temp
    if args.moe_gate_noise_final is not None:
        config['MoE']['gate_noise_final'] = args.moe_gate_noise_final
    if args.moe_temp_decay_epochs is not None:
        config['MoE']['temp_decay_epochs'] = args.moe_temp_decay_epochs
    if args.moe_post is not None:
        config['moe_post'] = args.moe_post
    if args.moe_strategy is not None:
        config['moe_strategy'] = args.moe_strategy
    if args.moe_agg is not None:
        config['moe_agg'] = args.moe_agg
    if args.moe_candidate_top_k is not None:
        config['moe_candidate_top_k'] = args.moe_candidate_top_k
    if args.context_num_points is not None:
        config['SampledContextMLPv1']['num_context_points'] = args.context_num_points
        config['SampledContextMLPv2']['num_context_points'] = args.context_num_points
    if args.context_fit_batch_size is not None:
        config['SampledContextMLPv1']['fit_batch_size'] = args.context_fit_batch_size
        config['SampledContextMLPv2']['fit_batch_size'] = args.context_fit_batch_size
    if args.context_eps is not None:
        config['SampledContextMLPv1']['eps'] = args.context_eps
        config['SampledContextMLPv2']['eps'] = args.context_eps
    if args.context_encoder_dim is not None:
        config['SampledContextMLPv2']['context_encoder_dim'] = args.context_encoder_dim
    if args.local_delta_scale is not None:
        config['LocalContextMLPv2']['local_delta_scale'] = args.local_delta_scale
    if args.local_coarse_loss_weight is not None:
        config['LocalContextMLPv2']['coarse_loss_weight'] = args.local_coarse_loss_weight

    # ── Method-specific overrides (applied to config[method] only) ──
    method = config['method']
    method_overrides = {
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'scale': args.scale,
        'dist_weight': args.dist_weight,
        'max_diff_iter': args.max_diff_iter,
        'val_tol': args.val_tol,
        'decay_tol_step': args.decay_tol_step,
        'memory_size': args.memory_size,
        'init': args.init,
    }
    for key, val in method_overrides.items():
        if val is not None:
            config[method][key] = val

    # Ensemble
    config['ensemble_size'] = args.ensemble_size
    config['ensemble_mode'] = args.ensemble_mode
    config['fge_pretrain_ratio'] = args.fge_pretrain_ratio
    config['fge_lr_max'] = args.fge_lr_max
    config['ensemble_post'] = args.ensemble_post
    config['ensemble_agg'] = args.ensemble_agg

    # W&B
    config['wandb'] = args.wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    config['wandb_run_name'] = args.wandb_run_name
    config['wandb_tags'] = args.wandb_tags
    config['strict_repro'] = args.strict_repro

    return args, config

def main():
    args, config = create_parser()

    prob_type = config.get('prob_type', 'Error')
    prob_name = config.get('prob_name', 'Error')

    generator = set_seed(config['seed'], strict=config.get('strict_repro', False))
    if generator is not None:
        config['_generator'] = generator

    opt_problem, result_save_dir = load_instance(config)

    setup_logging(save_dir=result_save_dir)

    log.info("Problem: %s/%s  size=%s  seed=%d", prob_type, prob_name,
             config['prob_size'], config['seed'])

    if config['wandb']:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        run_name = config['wandb_run_name'] or os.path.basename(result_save_dir)
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=run_name,
            tags=config['wandb_tags'],
            config={k: v for k, v in config.items() if k != '_generator'},
        )

    log.info("Method: %s  epochs: %d  save_dir: %s",
             config['method'], config[config['method']]['num_epochs'], result_save_dir)

    trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)

    if config['ensemble_size'] > 1:
        log.info("Ensemble training: %d members, mode=%s",
                 config['ensemble_size'], config['ensemble_mode'])
        trainer.train_ensemble()
    else:
        trainer.train()

    if config['wandb']:
        wandb.finish()

    log.info("Done")

if __name__ == "__main__":
    main()
