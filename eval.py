"""
Evaluate saved model checkpoints on the test set.

Supports:
  - Single model:      python eval.py --checkpoints path/to/model.pt
  - Saved ensemble:    python eval.py --checkpoints path/to/ensemble_model.pt
  - Ad-hoc ensemble from multiple single models:
      python eval.py --checkpoints model_a.pt model_b.pt model_c.pt
  - Override ensemble eval strategy:
      python eval.py --checkpoints m1.pt m2.pt --ensemble_post post --ensemble_agg best_merit
"""

import argparse
import glob
import logging
import os
import yaml
import yaml as _yaml
import torch
import time

from main import setup_logging
from utils.trainer import load_instance, create_model, DEVICE
from utils.evaluator import Evaluator
from models.neural_networks import MLP, EnsembleMLP, MixtureOfExperts

log = logging.getLogger(__name__)


def load_single_model(ckpt_path, opt_problem):
    """Load a single MLP from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    method = config['method']
    model = create_model(opt_problem, method, config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def load_model_from_checkpoint(ckpt_path, opt_problem):
    """Load a model from a checkpoint, auto-detecting single vs ensemble."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    state_dict = ckpt['model_state_dict']

    is_ensemble = any(k.startswith('members.') for k in state_dict.keys())

    if is_ensemble:
        member_indices = {int(k.split('.')[1]) for k in state_dict if k.startswith('members.')}
        n_members = len(member_indices)
        members = [create_model(opt_problem, config['method'], config) for _ in range(n_members)]
        model = EnsembleMLP(members).to(DEVICE)
        model.load_state_dict(state_dict)
        log.info("Loaded EnsembleMLP (%d members) from %s", n_members, ckpt_path)
    else:
        model = create_model(opt_problem, config['method'], config)
        model.load_state_dict(state_dict)
        log.info("Loaded single MLP from %s", ckpt_path)

    model.eval()
    return model, config


def resolve_checkpoints(run_dir):
    """Given a run directory, return a list of checkpoint paths.

    - If ``members/member_*.pt`` exist, return them sorted (ensemble from members).
    - Otherwise return ``model.pt`` (single model or saved EnsembleMLP).
    """
    members_pattern = os.path.join(run_dir, "members", "member_[0-9]*.pt")
    # Filter out intermediate epoch checkpoints (keep only member_N.pt)
    member_files = sorted(
        [f for f in glob.glob(members_pattern)
         if os.path.basename(f).startswith("member_") and "_epoch_" not in os.path.basename(f)]
    )
    if member_files:
        log.info("Found %d member checkpoints in %s", len(member_files), run_dir)
        return member_files

    model_pt = os.path.join(run_dir, "model.pt")
    if os.path.isfile(model_pt):
        return [model_pt]

    raise FileNotFoundError(f"No model.pt or members/member_*.pt found in {run_dir}")


def create_eval_parser():
    parser = argparse.ArgumentParser(description='Evaluate saved model checkpoints')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run_dir', type=str,
                       help='Path to a training run directory. Auto-discovers '
                            'members/member_*.pt (ensemble) or model.pt (single).')
    group.add_argument('--checkpoints', type=str, nargs='+',
                       help='One or more .pt checkpoint paths. '
                            'Multiple single-model checkpoints are combined into an ensemble.')

    parser.add_argument('--config', type=str, default=None,
                        help='Override YAML config (default: use config saved in checkpoint)')

    parser.add_argument('--ensemble_size', type=int, default=None,
                        help='Use only the first N members (default: all)')
    parser.add_argument('--ensemble_post', type=str, default='pre', choices=['pre', 'post'],
                        help='"pre" = ens(NNs)+Opt, "post" = ens(NNs+Opts)')
    parser.add_argument('--ensemble_agg', type=str, default='mean',
                        choices=['mean', 'median', 'best_obj', 'best_merit'],
                        help='Aggregation strategy for ensemble members')

    parser.add_argument('--test_batch_sizes', type=int, nargs='+', default=None,
                        help='Test batch sizes (default: from saved config)')

    parser.add_argument('--wandb', action='store_true', help='Log results to W&B')
    parser.add_argument('--wandb_project', type=str, default='FSNet-eval')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None)

    return parser


def _build_eval_save_dir(config):
    """Build save directory for eval results.

    Format: ``results/.../TIMESTAMP_eval_METHOD_seed_SEED[_ensN_POST_AGG]``
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prob_type = config['prob_type']
    prob_name = config['prob_name']
    method = config['method']
    seed = config.get('seed', 0)

    tag = f"{timestamp}_eval_{method}_seed{seed}"
    if config.get('ensemble_size', 1) > 1:
        tag += (f"_ens{config['ensemble_size']}"
                f"_{config.get('ensemble_post', 'pre')}"
                f"_{config.get('ensemble_agg', 'mean')}")

    prob_str = f"{prob_name.upper()}Problem"
    first_ckpt = config.get('_first_checkpoint_path', '')
    for p in first_ckpt.split(os.sep):
        if 'Problem' in p:
            prob_str = p
            break

    return os.path.join('results', prob_type, prob_name, prob_str, tag)


def _save_eval_results(save_dir, config, batch_size_results, eval_time):
    """Save test_summary.yaml and config.yaml for an eval run."""
    os.makedirs(save_dir, exist_ok=True)

    serialisable = {k: v for k, v in config.items() if not k.startswith('_')}
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        _yaml.dump(serialisable, f, default_flow_style=False, sort_keys=False)

    summary = {
        'method': config['method'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'eval_time_seconds': round(eval_time, 2),
        'ensemble_size': config.get('ensemble_size', 1),
        'ensemble_post': config.get('ensemble_post', 'pre'),
        'ensemble_agg': config.get('ensemble_agg', 'mean'),
        'checkpoints': config.get('_checkpoint_paths', []),
        'opt_gap_unit': 'percent',
    }
    bs_metrics = {}
    for bs, result in batch_size_results.items():
        if 'error' in result:
            bs_metrics[int(bs)] = {'error': result['error']}
        else:
            bs_metrics[int(bs)] = {
                k: round(float(v), 8) for k, v in result['metrics'].items()
            }
    summary['test'] = bs_metrics

    with open(os.path.join(save_dir, 'test_summary.yaml'), 'w') as f:
        _yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    log.info("Eval results saved to: %s", save_dir)


def main():
    parser = create_eval_parser()
    args = parser.parse_args()

    setup_logging()

    checkpoints = args.checkpoints or resolve_checkpoints(args.run_dir)

    if args.ensemble_size is not None:
        if args.ensemble_size > len(checkpoints):
            log.warning("--ensemble_size %d > available checkpoints (%d), using all",
                        args.ensemble_size, len(checkpoints))
        else:
            checkpoints = checkpoints[:args.ensemble_size]
            log.info("Using first %d of %d checkpoints", args.ensemble_size, len(checkpoints))

    first_ckpt = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
    config = first_ckpt['config']

    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config.update(yaml_config)

    config['ensemble_post'] = args.ensemble_post
    config['ensemble_agg'] = args.ensemble_agg
    config['wandb'] = args.wandb
    config['_eval_only'] = True
    config['_checkpoint_paths'] = checkpoints
    config['_first_checkpoint_path'] = checkpoints[0]
    if args.test_batch_sizes:
        config['test_batch_sizes'] = args.test_batch_sizes

    method = config['method']
    prob_type = config['prob_type']
    prob_name = config['prob_name']

    log.info("Evaluation: %s/%s  method=%s", prob_type, prob_name, method)

    opt_problem, _ = load_instance(config)

    if len(checkpoints) == 1:
        model, _ = load_model_from_checkpoint(checkpoints[0], opt_problem)
    else:
        log.info("Loading %d checkpoints as ensemble", len(checkpoints))
        members = []
        for ckpt_path in checkpoints:
            m, ckpt_cfg = load_single_model(ckpt_path, opt_problem)
            if ckpt_cfg['prob_type'] != prob_type or ckpt_cfg['prob_name'] != prob_name:
                raise ValueError(
                    f"Checkpoint {ckpt_path} is for {ckpt_cfg['prob_type']}/{ckpt_cfg['prob_name']}, "
                    f"but first checkpoint is for {prob_type}/{prob_name}")
            members.append(m)
            log.info("  [%d] %s  method=%s seed=%s",
                     len(members), ckpt_path, ckpt_cfg['method'], ckpt_cfg.get('seed', '?'))
        model = EnsembleMLP(members).to(DEVICE)
        config['ensemble_size'] = len(members)
        log.info("Created EnsembleMLP with %d members", len(members))

    model.eval()

    if args.wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"eval_{prob_type}_{prob_name}_{method}"
            wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=run_name, tags=args.wandb_tags,
                config={k: v for k, v in config.items() if not k.startswith('_')})
        except ImportError:
            log.warning("wandb not installed, skipping")
            config['wandb'] = False

    evaluator = Evaluator(opt_problem, method, config)
    test_batch_sizes = config.get('test_batch_sizes', [256, 512])
    is_ensemble = isinstance(model, EnsembleMLP)

    log.info("Model: %s  test_batch_sizes=%s",
             f"EnsembleMLP({len(model.members)} members)" if is_ensemble else "MLP",
             test_batch_sizes)
    if is_ensemble:
        log.info("ensemble_post=%s  ensemble_agg=%s",
                 config['ensemble_post'], config['ensemble_agg'])

    log.info("=" * 60)
    log.info("TEST EVALUATION")
    log.info("=" * 60)

    start_time = time.time()
    batch_size_results, all_detailed_results = evaluator.evaluate_multiple_batch_sizes(
        model, opt_problem.test_dataset, test_batch_sizes, "test")
    eval_time = time.time() - start_time
    log.info("Evaluation completed in %.2fs", eval_time)

    eval_save_dir = _build_eval_save_dir(config)
    _save_eval_results(eval_save_dir, config, batch_size_results, eval_time)

    if config.get('wandb', False):
        import wandb
        first_valid = next(
            (r['metrics'] for r in batch_size_results.values() if 'error' not in r), None)
        if first_valid:
            wandb.summary.update({
                'test/objective': first_valid.get('objective', 0),
                'test/opt_gap_mean': first_valid.get('opt_gap_mean', 0),
                'test/eq_violation_l1': first_valid.get('eq_violation_l1_mean', 0),
                'test/ineq_violation_l1': first_valid.get('ineq_violation_l1_mean', 0),
                'test/merit_mean': first_valid.get('merit_mean', 0),
                'test/solution_distance': first_valid.get('solution_distance_mean', 0),
                'eval_time_seconds': eval_time,
            })
        wandb.finish()

    log.info("Done")


if __name__ == "__main__":
    main()
