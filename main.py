import yaml
import torch
import time
import argparse
from utils.trainer import load_instance, Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Define available problem types and problems
PROBLEM_TYPES = ['convex', 'nonconvex', 'nonsmooth_nonconvex']
PROBLEM_NAMES = ['qp', 'qcqp', 'socp']

def create_parser():
    """Create and configure the argument parser, then load and process the configuration."""
    parser = argparse.ArgumentParser(description='Neural Network Optimization')
    
    # General parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--method', type=str, 
                        help='Training method (penalty, adaptive_penalty, FSNet, DC3, projection, sup, sup_partial, sup_pen, semi, S3Net)')
    parser.add_argument('--prob_type', type=str, choices=PROBLEM_TYPES,
                        help='Problem type (convex, nonconvex, nonsmooth_nonconvex)')
    parser.add_argument('--prob_name', type=str, choices=PROBLEM_NAMES,
                        help='Problem name (qp, qcqp, socp)')
    parser.add_argument('--prob_size', type=int, nargs='+', default=[100, 50, 50, 10000],
                        help='Problem size parameters [n, m, p, N] (default: [100, 50, 50, 10000])')
    parser.add_argument('--network', type=str, default='MLP',
                        help='Type of neural network to use')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--en_subopt', type=int, default=0, help='Enable suboptimality in training')
    parser.add_argument('--subopt_ratio', type=float, default=0, help='Suboptimality ratio if en_subopt is True')
    parser.add_argument('--save_intermediate', type=bool, default=False, help='Save intermediate models during training')

    # Dataset parameters
    parser.add_argument('--train_size', type=int, help='Size of training dataset', default=7000)
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--val_size', type=int, help='Size of validation dataset')
    parser.add_argument('--test_size', type=int, help='Size of test dataset')
    parser.add_argument('--dropout', type=float, help='Dropout rate for the model')

    # Neural network parameters
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, help='Learning rate decay step size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, help='Number of hidden layers')
    
    # Feasibility seeking parameters
    parser.add_argument('--scale', type=float, help='Scale')
    parser.add_argument('--dist_weight', type=float, help='Distance weight')
    parser.add_argument('--max_diff_iter', type=int, help='Maximum number of iterations for keeping the track of gradient')

    # Ensemble parameters
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of ensemble members (1 = single model, >1 = ensemble)')
    parser.add_argument('--ensemble_mode', type=str, default='vanilla', choices=['vanilla', 'fge'],
                        help='Ensemble training mode: vanilla (independent inits) or fge (Fast Geometric Ensembling)')
    parser.add_argument('--fge_pretrain_ratio', type=float, default=0.8,
                        help='FGE: fraction of total epochs for pre-training before snapshot collection')
    parser.add_argument('--fge_lr_max', type=float, default=None,
                        help='FGE: max LR during cyclical phase (defaults to base lr)')
    parser.add_argument('--ensemble_post', type=str, default='pre', choices=['pre', 'post'],
                        help='Ensemble eval: "pre" = ens(NNs)+Opt (avg then post-process), '
                             '"post" = ens(NNs+Opts) (post-process each then aggregate)')
    parser.add_argument('--ensemble_agg', type=str, default='mean',
                        choices=['mean', 'median', 'greedy_obj', 'greedy_merit'],
                        help='Ensemble aggregation: mean, median, greedy_obj (best objective), '
                             'greedy_merit (best merit = obj + penalty*violations)')

    # Weights & Biases parameters
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='FSNet', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team or user)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (auto-generated if not set)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='W&B tags for the run')

    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    if args.method:
        config['seed'] = args.seed
    if args.method:
        config['method'] = args.method
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

    # Override dataset parameters
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.train_size:
        config['train_size'] = args.train_size
    if args.val_size:
        config['val_size'] = args.val_size
    if args.test_size:
        config['test_size'] = args.test_size
    
    # Override neural network parameters
    if args.lr:
        config[args.method]['lr'] = args.lr
    if args.lr_decay:
        config[args.method]['lr_decay'] = args.lr_decay
    if args.lr_decay_step:
        config[args.method]['lr_decay_step'] = args.lr_decay_step
    if args.num_epochs:
        config[args.method]['num_epochs'] = args.num_epochs
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['num_layers'] = args.num_layers
    if args.dropout:
        config['dropout'] = args.dropout
    
    # Feasibility seeking parameters
    if args.scale:
        config['FSNet']['scale'] = args.scale
        config['S3Net']['scale'] = args.scale
        config['semi']['scale'] = args.scale
    if args.dist_weight is not None:
        config['FSNet']['dist_weight'] = args.dist_weight
        config['S3Net']['dist_weight'] = args.dist_weight
        config['semi']['dist_weight'] = args.dist_weight
    if args.max_diff_iter is not None:
        config['FSNet']['max_diff_iter'] = args.max_diff_iter
        config['S3Net']['max_diff_iter'] = args.max_diff_iter
        config['semi']['max_diff_iter'] = args.max_diff_iter

    # Ablation study flag
    config['ablation'] = args.ablation

    # Ensemble
    config['ensemble_size'] = args.ensemble_size
    config['ensemble_mode'] = args.ensemble_mode
    config['fge_pretrain_ratio'] = args.fge_pretrain_ratio
    config['fge_lr_max'] = args.fge_lr_max
    config['ensemble_post'] = args.ensemble_post
    config['ensemble_agg'] = args.ensemble_agg

    # Weights & Biases
    config['wandb'] = args.wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    config['wandb_run_name'] = args.wandb_run_name
    config['wandb_tags'] = args.wandb_tags

    return args, config

def main():
    # Parse command-line arguments and get processed config
    args, config = create_parser()
    
    # Get the problem type and name from config (with defaults)
    prob_type = config.get('prob_type', 'Error')
    prob_name = config.get('prob_name', 'Error')
    
    print(f"\n======= Running for problem: {prob_type}/{prob_name} =======\n")

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Initialize Weights & Biases
    if config['wandb']:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        # date time string + method
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = config['wandb_run_name'] or f"{timestamp}_{config['method']}"
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            name=run_name,
            tags=config['wandb_tags'],
            config=config,
        )

    # Load data 
    print(f"Loading problem instance: {prob_type}/{prob_name} with size {config['prob_size']}")
    opt_problem, result_save_dir = load_instance(config)
    
    # Train and test the model
    print(f"Training model using {config['method']} method with seed {config['seed']} for {config[config['method']]['num_epochs']} epochs")
    start_time = time.time()
    
    # Instantiate and use the Trainer
    trainer = Trainer(opt_problem=opt_problem, config=config, save_dir=result_save_dir)
    if config['ensemble_size'] > 1:
        print(f"\nEnsemble training: {config['ensemble_size']} members, mode={config['ensemble_mode']}")
        trainer.train_ensemble()
    else:
        trainer.train()
    
    training_time = time.time() - start_time
    print(f"Training and testing completed in {training_time:.2f} seconds")

    if config['wandb']:
        wandb.finish()

    print("Done!!!")

if __name__ == "__main__":
    main()