"""Portable single-run entry point. Run from the repository root."""
import argparse
import json
from pathlib import Path
from utils.config import load_config

def parse_args():
    p = argparse.ArgumentParser(description='Train FMGAD')
    p.add_argument('--config', default='configs/books.yaml')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--data-root')
    p.add_argument('--output-dir')
    p.add_argument('--epochs', type=int)
    p.add_argument('--episodes-per-epoch', type=int)
    p.add_argument('--eval-episodes', type=int)
    p.add_argument('--dry-run', action='store_true', help='Print resolved config without training')
    return p.parse_args()

def resolve_config(args):
    c = load_config(args.config)
    ds = c['dataset']['name'].lower()
    seed = args.seed
    c['seed'] = seed
    c.setdefault('runtime', {}).update(seed=seed, device=args.device)
    c['device'] = args.device
    c.setdefault('episode', {})['seed'] = seed
    split_dir = Path(c.setdefault('split', {}).get('save_path', 'splits/default.json')).parent
    c['split'].update(seed=seed, save_path=str(split_dir/f'{ds}_seed{seed}.json'))
    if args.data_root: c['dataset']['root'] = args.data_root
    output = Path(args.output_dir or f'results/{ds}/seed_{seed}')
    c.setdefault('experiment', {}).update(output_root=str(output.parent), output_dir=str(output),
                                         run_name=f'{ds}_seed{seed}')
    t = c.setdefault('training', {})
    t['checkpoint_dir'] = str(output/'checkpoints')
    for attr, key in [('epochs', 'num_epochs'), ('episodes_per_epoch', 'episodes_per_epoch'),
                      ('eval_episodes', 'eval_episodes')]:
        value = getattr(args, attr)
        if value is not None:
            if value < 1: raise ValueError(f'{attr} must be positive')
            t[key] = value
    return c

def main():
    args = parse_args()
    config = resolve_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from train import train
    train(config)

if __name__ == '__main__': main()
