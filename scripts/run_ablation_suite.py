"""
Automated Ablation Suite Runner & Summary Table Generator.

Runs the Golden Ablation Experiments sequentially and compiles
an academic comparison table ready for technical reports.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

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


def run_single_ablation(config_path: str, epochs: int, extra_args: List[str]) -> Path:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_train.py"),
        "--config", config_path,
        "--training.epochs", str(epochs),
    ] + extra_args

    print(f"\n{'='*80}")
    print(f"Launching Ablation: {config_path} ({epochs} epochs)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd, check=True)


def parse_latest_run_metrics(tag: str) -> Dict[str, Any]:
    """Finds the most recent run matching the tag and reads best_top1."""
    runs_dir = REPO_ROOT / "runs"
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
                best_top1 = max((epoch_data.get("val_top1", 0.0) for epoch_data in data.values()), default=0.0)
                best_top5 = max((epoch_data.get("val_top5", 0.0) for epoch_data in data.values()), default=0.0)
                return {"val_top1": best_top1, "val_top5": best_top5, "run_dir": latest.name}
    return {"val_top1": 0.0, "val_top5": 0.0, "run_dir": "N/A"}


def main():
    parser = argparse.ArgumentParser(description="Run Golden Ablation Suite")
    parser.add_argument("--epochs", type=int, default=40, help="Epoch budget per ablation run (default: 40)")
    parser.add_argument("--dry_run", action="store_true", help="Print configs without training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size override")
    args, unknown = parser.parse_known_args()

    print(f"\n=======================================================")
    print(f"    Pyramid-ResMLP Golden Ablation Suite Runner")
    print(f"    Total Runs: {len(ABLATION_RUNS)} | Epochs per run: {args.epochs}")
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
        run_single_ablation(item["config"], epochs=args.epochs, extra_args=extra)
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
