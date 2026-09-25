"""
Main Training Script.

Usage:
    python scripts/run_train.py --config configs/cifar100_mlp_mixer.yaml
    python scripts/run_train.py --config configs/tiny_imagenet_mlp_mixer.yaml --training.epochs 50 --training.lr 5e-4
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to enable imports regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.parser import load_config, save_config
from src.data import build_dataloaders
from src.engine.trainer import Trainer
from src.models import build_model
from src.utils.logger import setup_logger
from src.utils.naming import generate_experiment_name, setup_experiment_dir
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP-variants on vision datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar100_mlp_mixer.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default.yaml",
        help="Path to base default YAML configuration file",
    )
    # Remaining args are parsed as dot-notation overrides (e.g. --training.lr 1e-4)
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown_args = parse_args()

    # 1. Load config with CLI overrides
    cfg = load_config(
        config_path=args.config,
        default_config_path=args.default_config,
        cli_overrides=unknown_args,
    )

    # 2. Control Seed & Determinism
    seed = cfg.training.get("seed", 42)
    deterministic = cfg.training.get("deterministic", True)
    set_seed(seed=seed, deterministic=deterministic)

    # 3. Setup Experiment Name & Directory Structure
    exp_name = cfg.experiment.get("name")
    if not exp_name:
        exp_name = generate_experiment_name(
            model_name=cfg.model.name,
            dataset_name=cfg.data.dataset,
            tag=cfg.experiment.get("tag", "run"),
            seed=seed,
        )
        cfg.experiment.name = exp_name

    base_dir = cfg.experiment.get("base_dir", "runs")
    exp_dirs = setup_experiment_dir(exp_name=exp_name, base_dir=base_dir)

    # 4. Save Config Snapshot for Perfect Reproducibility
    save_config(cfg, exp_dirs["root"] / "config.yaml")

    # 5. Initialize Logger
    logger = setup_logger(name="main", log_file=exp_dirs["logs"] / "train.log")
    logger.info(f"Experiment initialized: {exp_name}")
    logger.info(f"Artifacts will be stored at: {exp_dirs['root']}")
    logger.info(f"Seed set to {seed} (deterministic={deterministic})")

    # 6. Build DataLoaders
    logger.info(f"Building data pipeline for dataset: {cfg.data.dataset}")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 7. Build Model
    logger.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg)
    total_params = model.num_parameters(trainable_only=False)
    trainable_params = model.num_parameters(trainable_only=True)
    logger.info(f"Model {cfg.model.name} constructed. Parameters: {trainable_params:,} trainable ({total_params:,} total)")

    # 8. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        exp_dirs=exp_dirs,
        logger=logger,
    )

    results = trainer.fit()
    logger.info(f"Experiment finished successfully! Best Acc@1: {results['best_top1']:.2f}%")


if __name__ == "__main__":
    main()
