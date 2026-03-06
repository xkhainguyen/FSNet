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


def set_seed(seed: int) -> torch.Generator:
    """Set all random seeds for full reproducibility. Returns a torch Generator
    that should be passed to DataLoaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
    parser.add_argument('--seed', type=int, default=2025)
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
    parser.add_argument('--scale', type=float)
    parser.add_argument('--dist_weight', type=float)
    parser.add_argument('--max_diff_iter', type=int)

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

    # ── Method-specific overrides (applied to config[method] only) ──
    method = config['method']
    method_overrides = {
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'scale': args.scale,
        'dist_weight': args.dist_weight,
        'max_diff_iter': args.max_diff_iter,
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

    return args, config

def main():
    args, config = create_parser()

    prob_type = config.get('prob_type', 'Error')
    prob_name = config.get('prob_name', 'Error')

    generator = set_seed(config['seed'])
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