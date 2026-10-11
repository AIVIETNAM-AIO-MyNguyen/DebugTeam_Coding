"""
Automated Ablation Suite Runner & Summary Table Generator.

Runs the Golden Ablation Experiments sequentially and compiles
an academic comparison table ready for technical reports.

Features:
- DDP Multi-GPU support via `--nproc_per_node` (e.g. 2x T4 on Kaggle)
- Auto-detection and skipping of already completed runs to preserve GPU time
- Robust JSON metrics parser (handles both list and dict schemas)
- Automated Markdown table output
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.parser import load_config


ABLATION_RUNS = [
    {
        "id": "Ablation 1 (Pure Linear)",
        "config": "configs/ablation/01_pyramid_resmlp_pure_linear.yaml",
        "description": "Pyramid-ResMLP without RepTokenMix (pure Linear spatial mixing)",
    },
    {
        "id": "Ablation 2 (DW 3x3 Only)",
        "config": "configs/ablation/02_pyramid_resmlp_rep_3x3.yaml",
        "description": "RepTokenMix with single local 3x3 DW-Conv branch",
    },
    {
        "id": "Ablation 3 (Full RepTokenMix)",
        "config": "configs/ablation/03_pyramid_resmlp_full_reptoken.yaml",
        "description": "Full RepTokenMix (Linear + DW 1x1 + DW 3x3) + Overlapping Stem",
    },
    {
        "id": "Ablation 4 (Non-Overlapping Stem)",
        "config": "configs/ablation/04_pyramid_resmlp_non_overlapping_stem.yaml",
        "description": "RepTokenMix with standard patchify stem (no boundary overlap)",
    },
]


def run_single_ablation(
    config_path: str,
    epochs: int,
    extra_args: List[str],
    nproc_per_node: int = 1,
) -> None:
    if nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            str(REPO_ROOT / "scripts" / "run_train.py"),
            "--config", config_path,
            "--training.epochs", str(epochs),
        ] + extra_args
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_train.py"),
            "--config", config_path,
            "--training.epochs", str(epochs),
        ] + extra_args

    print(f"\n{'='*80}")
    print(f"Launching Ablation: {config_path} ({epochs} epochs, {nproc_per_node} GPUs)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd, check=True)


def parse_latest_run_metrics(tag: str) -> Dict[str, Any]:
    """Finds the most recent run matching the tag and reads best_top1."""
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return {"val_top1": 0.0, "val_top5": 0.0, "run_dir": "N/A", "epochs": 0}

    matching = sorted(runs_dir.glob(f"*_{tag}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matching:
        # Fallback search
        matching = sorted(runs_dir.glob(f"*{tag}*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if matching:
        latest = matching[0]
        metrics_file = latest / "metrics" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    best_top1 = max((epoch_data.get("val_top1", 0.0) for epoch_data in data), default=0.0)
                    best_top5 = max((epoch_data.get("val_top5", 0.0) for epoch_data in data), default=0.0)
                    completed_epochs = len(data)
                elif isinstance(data, dict):
                    best_top1 = max((epoch_data.get("val_top1", 0.0) for epoch_data in data.values()), default=0.0)
                    best_top5 = max((epoch_data.get("val_top5", 0.0) for epoch_data in data.values()), default=0.0)
                    completed_epochs = len(data)
                else:
                    best_top1, best_top5, completed_epochs = 0.0, 0.0, 0

                return {
                    "val_top1": best_top1,
                    "val_top5": best_top5,
                    "run_dir": latest.name,
                    "epochs": completed_epochs,
                }

    return {"val_top1": 0.0, "val_top5": 0.0, "run_dir": "N/A", "epochs": 0}


def main():
    parser = argparse.ArgumentParser(description="Run Golden Ablation Suite")
    parser.add_argument("--epochs", type=int, default=40, help="Epoch budget per ablation run (default: 40)")
    parser.add_argument("--dry_run", action="store_true", help="Print configs without training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size override")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of GPUs per node (default: 1)")
    parser.add_argument("--skip_completed", action="store_true", default=True, help="Skip runs that have already completed target epochs")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain even if run already completed")
    args, unknown = parser.parse_known_args()

    print(f"\n=======================================================")
    print(f"    Pyramid-ResMLP Golden Ablation Suite Runner")
    print(f"    Total Runs: {len(ABLATION_RUNS)} | Epochs per run: {args.epochs}")
    print(f"    GPUs per run: {args.nproc_per_node}")
    print(f"=======================================================\n")

    for item in ABLATION_RUNS:
        print(f"[{item['id']}] -> {item['config']}")
        print(f"  Description: {item['description']}")

    if args.dry_run:
        print("\nDry-run mode completed. Remove --dry_run to execute.")
        return

    summary = []
    extra = ["--data.batch_size", str(args.batch_size)] + unknown

    for item in ABLATION_RUNS:
        cfg = load_config(item["config"], default_config_path="configs/default.yaml")
        tag = cfg.experiment.tag

        # Check if already completed
        existing = parse_latest_run_metrics(tag)
        if not args.force_retrain and existing["epochs"] >= args.epochs:
            print(f"\n[SKIP] {item['id']} already completed with {existing['epochs']}/{args.epochs} epochs.")
            print(f"       Best Val Acc@1: {existing['val_top1']:.2f}% (Run: {existing['run_dir']})\n")
            summary.append({
                "id": item["id"],
                "description": item["description"],
                "top1": existing["val_top1"],
                "top5": existing["val_top5"],
                "run_dir": existing["run_dir"],
            })
            continue

        run_single_ablation(
            config_path=item["config"],
            epochs=args.epochs,
            extra_args=extra,
            nproc_per_node=args.nproc_per_node,
        )

        metrics = parse_latest_run_metrics(tag)
        summary.append({
            "id": item["id"],
            "description": item["description"],
            "top1": metrics["val_top1"],
            "top5": metrics["val_top5"],
            "run_dir": metrics["run_dir"],
        })

    # Print Final Markdown Summary
    print(f"\n\n{'='*80}")
    print(f"           FINAL ABLATION SUITE RESULTS TABLE")
    print(f"{'='*80}\n")
    print("| Experiment | Architecture Setting | Val Top-1 (%) | Val Top-5 (%) | Run Artifact |")
    print("| :--- | :--- | :---: | :---: | :--- |")
    for s in summary:
        print(f"| **{s['id']}** | {s['description']} | **{s['top1']:.2f}%** | {s['top5']:.2f}% | `{s['run_dir']}` |")
    print("\n")


if __name__ == "__main__":
    main()
