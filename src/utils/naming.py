"""
Experiment Naming Convention and Directory Management.

Enforces a standardized naming format across team members:
    <YYYYMMDD_HHMMSS>_<model>_<dataset>_<tag>_s<seed>

And manages artifact directories:
    runs/<experiment_name>/
        ├── checkpoints/    # best.pt, last.pt
        ├── logs/           # console and training log files
        ├── metrics/        # training metrics CSV/JSON
        ├── config.yaml     # snapshot of runtime config
        └── git_info.json   # git commit, branch, dirty flag
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def generate_experiment_name(
    model_name: str,
    dataset_name: str,
    tag: Optional[str] = "baseline",
    seed: int = 42,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generates a standardized experiment name.

    Args:
        model_name: Name of the model architecture (e.g., 'mlp_mixer', 'resmlp').
        dataset_name: Name of the dataset (e.g., 'cifar100', 'tiny_imagenet').
        tag: Custom descriptive tag (e.g., 'baseline', 'lr1e-3', 'patch4').
        seed: Random seed used in the run.
        timestamp: Optional timestamp string. Defaults to current datetime 'YYYYMMDD_HHMMSS'.

    Returns:
        Formatted experiment name string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean strings to avoid invalid path characters
    clean_model = model_name.lower().replace("-", "_").strip()
    clean_dataset = dataset_name.lower().replace("-", "_").strip()
    clean_tag = (tag or "run").lower().replace(" ", "_").replace("/", "_").strip()

    return f"{timestamp}_{clean_model}_{clean_dataset}_{clean_tag}_s{seed}"


def get_git_info() -> Dict[str, Any]:
    """
    Captures current git repository state for experiment provenance.
    Safe against non-git directories or environments without git.

    Returns:
        Dictionary containing commit hash, branch name, dirty status, etc.
    """
    git_info: Dict[str, Any] = {
        "is_git_repo": False,
        "commit_hash": None,
        "branch": None,
        "is_dirty": None,
        "error": None,
    }

    try:
        # Check if inside git work tree
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        git_info["is_git_repo"] = True

        # Commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_info["commit_hash"] = commit

        # Branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_info["branch"] = branch

        # Check for uncommitted changes (dirty)
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_info["is_dirty"] = len(status) > 0

    except Exception as e:
        git_info["error"] = str(e)

    return git_info


def setup_experiment_dir(
    exp_name: str,
    base_dir: str = "runs",
) -> Dict[str, Path]:
    """
    Creates structured directory for a new experiment.

    Directory layout:
        <base_dir>/<exp_name>/
            ├── checkpoints/
            ├── logs/
            └── metrics/

    Args:
        exp_name: Experiment name.
        base_dir: Base directory for all runs (defaults to 'runs').

    Returns:
        Dictionary mapping directory keys ('root', 'checkpoints', 'logs', 'metrics') to Path objects.
    """
    root = Path(base_dir) / exp_name
    checkpoints_dir = root / "checkpoints"
    logs_dir = root / "logs"
    metrics_dir = root / "metrics"

    for d in (root, checkpoints_dir, logs_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save git info
    git_info = get_git_info()
    with open(root / "git_info.json", "w", encoding="utf-8") as f:
        json.dump(git_info, f, indent=2)

    return {
        "root": root,
        "checkpoints": checkpoints_dir,
        "logs": logs_dir,
        "metrics": metrics_dir,
    }
