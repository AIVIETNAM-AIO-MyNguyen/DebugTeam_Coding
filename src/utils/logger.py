"""
Logging and Metric Tracking Utilities.

Provides structured console and file logging along with JSON/CSV metric logging.
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def setup_logger(
    name: str = "mlp_exp",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Sets up a logger with unified formatting for console and file output.

    Args:
        name: Name of the logger.
        log_file: Optional path to a file where logs should be appended.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricTracker:
    """
    Tracks and records training/validation metrics epoch by epoch.
    Exports history to JSON and CSV files.
    """

    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        self.history: List[Dict[str, Any]] = []

    def record(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Records metrics for a given epoch and updates files on disk."""
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)

        if self.save_dir:
            self._save_json()
            self._save_csv()

    def _save_json(self) -> None:
        assert self.save_dir is not None
        out_path = self.save_dir / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _save_csv(self) -> None:
        assert self.save_dir is not None
        if not self.history:
            return
        out_path = self.save_dir / "metrics.csv"
        keys = list(self.history[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def get_best(self, metric_name: str, mode: str = "max") -> Optional[Dict[str, Any]]:
        """Returns the entry with the best score for metric_name ('max' or 'min')."""
        if not self.history or metric_name not in self.history[0]:
            return None

        if mode == "max":
            return max(self.history, key=lambda x: x.get(metric_name, float("-inf")))
        elif mode == "min":
            return min(self.history, key=lambda x: x.get(metric_name, float("inf")))
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'max' or 'min'.")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
