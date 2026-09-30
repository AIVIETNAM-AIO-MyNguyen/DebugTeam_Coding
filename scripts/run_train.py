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
from src.utils.distributed import cleanup_distributed, init_distributed_mode, is_main_process
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

    # 0. Initialize DDP Multi-GPU (e.g. 2x T4 on Kaggle via torchrun)
    is_distributed, rank, local_rank, world_size = init_distributed_mode()

    # 1. Load config with CLI overrides
    cfg = load_config(
        config_path=args.config,
        default_config_path=args.default_config,
        cli_overrides=unknown_args,
    )

    # 2. Control Seed & Determinism
    base_seed = cfg.training.get("seed", 42)
    seed = base_seed + rank
    deterministic = cfg.training.get("deterministic", True)
    set_seed(seed=seed, deterministic=deterministic)

    # 3. Setup Experiment Name & Directory Structure (rank 0 only)
    exp_name = cfg.experiment.get("name")
    if not exp_name:
        exp_name = generate_experiment_name(
            model_name=cfg.model.name,
            dataset_name=cfg.data.dataset,
            tag=cfg.experiment.get("tag", "run"),
            seed=base_seed,
        )
        cfg.experiment.name = exp_name

    base_dir = cfg.experiment.get("base_dir", "runs")
    if is_main_process():
        exp_dirs = setup_experiment_dir(exp_name=exp_name, base_dir=base_dir)
        save_config(cfg, exp_dirs["root"] / "config.yaml")
    else:
        root = Path(base_dir) / exp_name
        exp_dirs = {
            "root": root,
            "checkpoints": root / "checkpoints",
            "logs": root / "logs",
            "metrics": root / "metrics",
        }

    if is_distributed:
        import torch.distributed as dist
        dist.barrier()

    # 4. Initialize Logger
    log_file = (exp_dirs["logs"] / "train.log") if is_main_process() else None
    logger = setup_logger(name=f"main_r{rank}", log_file=log_file)
    if is_main_process():
        logger.info(f"Experiment initialized: {exp_name}")
        logger.info(f"Distributed training: {is_distributed} (world_size={world_size})")
        logger.info(f"Artifacts will be stored at: {exp_dirs['root']}")
        logger.info(f"Seed set to {base_seed} (deterministic={deterministic})")

    # 5. Build DataLoaders (automatically uses DistributedSampler when in DDP)
    if is_main_process():
        logger.info(f"Building data pipeline for dataset: {cfg.data.dataset}")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    if is_main_process():
        logger.info(f"Train batches per GPU: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 6. Build Model
    if is_main_process():
        logger.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg)
    total_params = model.num_parameters(trainable_only=False)
    trainable_params = model.num_parameters(trainable_only=True)
    if is_main_process():
        logger.info(f"Model {cfg.model.name} constructed. Parameters: {trainable_params:,} trainable ({total_params:,} total)")

    # 7. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        exp_dirs=exp_dirs,
        logger=logger,
    )

    results = trainer.fit()
    if is_main_process():
        logger.info(f"Experiment finished successfully! Best Acc@1: {results['best_top1']:.2f}%")

    cleanup_distributed()


if __name__ == "__main__":
    main()
