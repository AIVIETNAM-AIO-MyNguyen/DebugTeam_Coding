"""
Training Engine for MLP Vision Experiments.

Features:
- Automatic Mixed Precision (AMP) for fast GPU execution
- Soft target CrossEntropy support for Mixup/CutMix
- Gradient clipping and learning rate scheduling
- Top-1 and Top-5 accuracy tracking
- Metric persistence (JSON/CSV) and best checkpoint tracking
- Usable via CLI scripts AND inside Jupyter / Google Colab / Kaggle notebooks
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.checkpoint import save_checkpoint
from src.utils.logger import AverageMeter, MetricTracker, setup_logger


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> list:
    """Computes precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        # Handle one-hot / smoothed targets from Mixup
        if target.dim() > 1:
            target = target.argmax(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k <= output.size(1):
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            else:
                res.append(0.0)
        return res


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss with support for probability distribution targets (Mixup/CutMix)."""

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class Trainer:
    """
    Modular trainer designed for both script execution and interactive notebook use.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        cfg: Any,
        exp_dirs: Optional[Dict[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        logger: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_dirs = exp_dirs

        self.is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.is_distributed:
            import os
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.model = model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        else:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.model = model.to(self.device)

        # Logger and metric tracker (rank 0 only records metrics to disk)
        from src.utils.distributed import is_main_process
        log_file = (exp_dirs["logs"] / "train.log") if (exp_dirs and "logs" in exp_dirs and is_main_process()) else None
        self.logger = logger or setup_logger(name="trainer", log_file=log_file)
        metrics_dir = exp_dirs["metrics"] if (exp_dirs and "metrics" in exp_dirs and is_main_process()) else None
        self.tracker = MetricTracker(save_dir=metrics_dir)

        # Criterion setup (handle Mixup soft targets or standard labels)
        self.use_mixup = self.cfg.get("augmentation", {}).get("use_mixup", False)
        label_smoothing = self.cfg.get("training", {}).get("label_smoothing", 0.0)
        if self.use_mixup:
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.eval_criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Mixed Precision AMP
        self.use_amp = self.cfg.get("training", {}).get("use_amp", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 1
        self.best_top1 = 0.0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.cfg.get("optimizer", {}).get("name", "adamw").lower()
        lr = float(self.cfg.get("training", {}).get("lr", 1e-3))
        weight_decay = float(self.cfg.get("optimizer", {}).get("weight_decay", 1e-4))

        if opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            momentum = float(self.cfg.get("optimizer", {}).get("momentum", 0.9))
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _build_scheduler(self):
        sched_name = self.cfg.get("scheduler", {}).get("name", "cosine").lower()
        epochs = int(self.cfg.get("training", {}).get("epochs", 100))
        min_lr = float(self.cfg.get("scheduler", {}).get("min_lr", 1e-6))

        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        elif sched_name == "step":
            step_size = int(self.cfg.get("scheduler", {}).get("step_size", 30))
            gamma = float(self.cfg.get("scheduler", {}).get("gamma", 0.1))
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_name in ("none", "null"):
            return None
        else:
            return None

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs training for a single epoch."""
        self.model.train()
        loss_meter = AverageMeter("Loss", ":.4f")
        top1_meter = AverageMeter("Acc@1", ":.2f")
        top5_meter = AverageMeter("Acc@5", ":.2f")

        # Set epoch for DistributedSampler to guarantee deterministic shuffling
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        from src.utils.distributed import is_main_process
        grad_clip = self.cfg.get("training", {}).get("grad_clip", 1.0)
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}] Train", leave=False, disable=not is_main_process())

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Record batch metrics
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(prec1, batch_size)
            top5_meter.update(prec5, batch_size)

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
            })

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": loss_meter.avg,
            "train_top1": top1_meter.avg,
            "train_top5": top5_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Runs validation evaluation for a single epoch."""
        self.model.eval()
        loss_meter = AverageMeter("Val Loss", ":.4f")
        top1_meter = AverageMeter("Val Acc@1", ":.2f")
        top5_meter = AverageMeter("Val Acc@5", ":.2f")

        from src.utils.distributed import is_main_process
        pbar = tqdm(self.val_loader, desc=f"Epoch [{epoch}] Val", leave=False, disable=not is_main_process())

        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.eval_criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(prec1, batch_size)
            top5_meter.update(prec5, batch_size)

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "top1": f"{top1_meter.avg:.2f}%",
            })

        return {
            "val_loss": loss_meter.avg,
            "val_top1": top1_meter.avg,
            "val_top5": top5_meter.avg,
        }

    def fit(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the full training loop across all epochs.
        """
        total_epochs = epochs or self.cfg.get("training", {}).get("epochs", 100)
        from src.utils.distributed import is_main_process
        if is_main_process():
            self.logger.info(f"Starting training on device: {self.device}")
            self.logger.info(f"Total epochs: {total_epochs}, Initial LR: {self.optimizer.param_groups[0]['lr']}")

        start_time = time.time()

        for epoch in range(self.start_epoch, total_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Combine epoch metrics (rank 0 records & saves checkpoints)
            if is_main_process():
                epoch_metrics = {**train_metrics, **val_metrics}
                self.tracker.record(epoch, epoch_metrics)

                is_best = val_metrics["val_top1"] > self.best_top1
                if is_best:
                    self.best_top1 = val_metrics["val_top1"]

                # Save checkpoints
                if self.exp_dirs and "checkpoints" in self.exp_dirs:
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    save_checkpoint(
                        state={
                            "epoch": epoch,
                            "model_state": raw_model.state_dict(),
                            "optimizer_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                            "scaler_state": self.scaler.state_dict() if self.scaler else None,
                            "best_top1": self.best_top1,
                            "metrics": epoch_metrics,
                        },
                        is_best=is_best,
                        checkpoint_dir=self.exp_dirs["checkpoints"],
                    )

                # Log summary
                self.logger.info(
                    f"Epoch [{epoch:03d}/{total_epochs:03d}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Acc: {train_metrics['train_top1']:.2f}% | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_top1']:.2f}% | "
                    f"Best Val: {self.best_top1:.2f}%"
                )

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed / 60:.2f} minutes. Best Val Acc@1: {self.best_top1:.2f}%")

        return {
            "best_top1": self.best_top1,
            "history": self.tracker.history,
        }
