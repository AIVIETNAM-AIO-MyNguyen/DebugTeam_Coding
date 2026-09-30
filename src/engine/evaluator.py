"""
Evaluation Engine for Standalone Inference and Testing.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm

from src.engine.trainer import accuracy
from src.utils.logger import AverageMeter


class Evaluator:
    """
    Evaluator for standalone testing and notebook analysis.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs evaluation on a dataset.

        Args:
            model: Neural network model.
            dataloader: DataLoader to evaluate on.
            return_preds: If True, returns arrays of all predicted and true labels.

        Returns:
            Dictionary with metrics ('loss', 'top1', 'top5', and optionally 'all_preds', 'all_targets').
        """
        model.eval()
        model = model.to(self.device)

        loss_meter = AverageMeter("Loss", ":.4f")
        top1_meter = AverageMeter("Acc@1", ":.2f")
        top5_meter = AverageMeter("Acc@5", ":.2f")

        all_preds = []
        all_targets = []

        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Handle one-hot targets if passed
            label_targets = targets.argmax(dim=1) if targets.dim() > 1 else targets

            outputs = model(images)
            loss = self.criterion(outputs, label_targets)

            prec1, prec5 = accuracy(outputs, label_targets, topk=(1, 5))
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_meter.update(prec1, batch_size)
            top5_meter.update(prec5, batch_size)

            if return_preds:
                preds = outputs.argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_targets.extend(label_targets.cpu().tolist())

        results = {
            "loss": loss_meter.avg,
            "top1": top1_meter.avg,
            "top5": top5_meter.avg,
        }

        if return_preds:
            results["all_preds"] = all_preds
            results["all_targets"] = all_targets

        return results
