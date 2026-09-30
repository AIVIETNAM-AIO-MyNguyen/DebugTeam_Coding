"""
Standalone Evaluation Script.

Usage:
    python scripts/run_eval.py --config configs/cifar100_mlp_mixer.yaml --checkpoint runs/<exp_name>/checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.config.parser import load_config
from src.data import build_dataloaders
from src.engine.evaluator import Evaluator
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MLP checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (defaults to config.yaml in checkpoint run dir)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(name="eval")

    checkpoint_path = Path(args.checkpoint)
    config_path = args.config
    if config_path is None:
        candidate = checkpoint_path.parent.parent / "config.yaml"
        if candidate.is_file():
            config_path = str(candidate)
            logger.info(f"Auto-detected configuration from: {config_path}")
        else:
            raise ValueError("No --config specified and config.yaml not found in checkpoint experiment directory.")

    # Load configuration
    cfg = load_config(config_path=config_path)
    set_seed(seed=cfg.training.get("seed", 42), deterministic=True)

    # Build data
    logger.info(f"Loading data for {cfg.data.dataset}...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    target_loader = test_loader if args.split == "test" else val_loader

    # Build model
    logger.info(f"Building model {cfg.model.name}...")
    model = build_model(cfg)

    # Load checkpoint
    logger.info(f"Loading weights from {args.checkpoint}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    # Run evaluation
    evaluator = Evaluator(device=device)
    metrics = evaluator.evaluate(model, target_loader)

    logger.info("=" * 40)
    logger.info(f"Evaluation Results on [{args.split.upper()}] split:")
    logger.info(f"Loss:     {metrics['loss']:.4f}")
    logger.info(f"Top-1 Acc: {metrics['top1']:.2f}%")
    logger.info(f"Top-5 Acc: {metrics['top5']:.2f}%")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
