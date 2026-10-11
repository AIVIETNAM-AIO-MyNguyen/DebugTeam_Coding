"""
Hardware Inference Profiling & Zero-Overhead Validation Pipeline.

Measures:
- Latency (ms/batch): Mean, Std, P50, P95, P99 using CUDA events with warmup
- Throughput (images/sec)
- GPU Memory Footprint (Peak VRAM in MB)
- Numerical Equivalence (allclose tolerance check before and after fusion)
- Empirical Zero-Overhead Speedup validation (Train mode vs Fused Deploy mode)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.config.parser import load_config
from src.models import build_model, list_models


def measure_gpu_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 50,
    num_iters: int = 200,
) -> Tuple[Dict[str, float], float]:
    """
    Measures GPU inference latency with CUDA Events and peak VRAM allocation.

    Returns:
        metrics_dict: {mean_ms, std_ms, p50_ms, p95_ms, p99_ms, throughput_img_s}
        peak_vram_mb: Peak allocated GPU memory during inference
    """
    device = input_tensor.device
    batch_size = input_tensor.size(0)

    # 1. Warmup Phase (prime caches, cuDNN benchmark kernels, CUDA context)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # 2. Reset peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 3. Timed Iterations
    latencies = []
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(num_iters):
                starter.record()
                _ = model(input_tensor)
                ender.record()
                torch.cuda.synchronize(device)
                curr_latency = starter.elapsed_time(ender)  # milliseconds
                latencies.append(curr_latency)
    else:
        import time
        with torch.no_grad():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))
    std_ms = float(np.std(latencies_arr))
    p50_ms = float(np.percentile(latencies_arr, 50))
    p95_ms = float(np.percentile(latencies_arr, 95))
    p99_ms = float(np.percentile(latencies_arr, 99))
    throughput = float((batch_size * 1000.0) / mean_ms) if mean_ms > 0 else 0.0

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "throughput_img_s": throughput,
    }, peak_vram_mb


def run_profiling(
    cfg: Any,
    batch_sizes: List[int],
    num_warmup: int = 50,
    num_iters: int = 200,
    device_str: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Conducts comparative profiling between:
    1. Train Mode (unfused, parallel branches active)
    2. Deploy Mode (analytically fused via locality_injection())
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    image_size = cfg.get("data", {}).get("image_size", None) or cfg.get("model", {}).get("image_size", 64)

    print(f"\n{'='*80}")
    print(f"Hardware Inference Profiling: {cfg.model.name} (Resolution: {image_size}x{image_size})")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Warmup: {num_warmup} iters | Measurement: {num_iters} iters")
    print(f"{'='*80}\n")

    results = []

    for bs in batch_sizes:
        print(f"--- Profiling Batch Size: {bs} ---")
        x = torch.randn(bs, 3, image_size, image_size, device=device)

        # 1. Build Model in Train Mode
        model_train = build_model(cfg).to(device)
        model_train.eval()

        train_metrics, train_vram = measure_gpu_latency(
            model_train, x, num_warmup=num_warmup, num_iters=num_iters
        )

        # Get sample output from train model
        with torch.no_grad():
            out_train = model_train(x[:2])

        # 2. Fuse to Deploy Mode via Locality Injection
        if hasattr(model_train, "locality_injection") and callable(model_train.locality_injection):
            model_train.locality_injection()
            deploy_active = True
        else:
            deploy_active = False

        deploy_metrics, deploy_vram = measure_gpu_latency(
            model_train, x, num_warmup=num_warmup, num_iters=num_iters
        )

        # Verify numerical equivalence
        with torch.no_grad():
            out_deploy = model_train(x[:2])
        max_diff = float((out_train - out_deploy).abs().max().item())
        is_exact = max_diff < 1e-4

        # Parameter counting
        params_train = sum(p.numel() for p in model_train.parameters())

        speedup = train_metrics["mean_ms"] / deploy_metrics["mean_ms"] if deploy_metrics["mean_ms"] > 0 else 1.0
        vram_reduction = ((train_vram - deploy_vram) / train_vram * 100.0) if train_vram > 0 else 0.0

        res_entry = {
            "batch_size": bs,
            "train_latency_ms": train_metrics["mean_ms"],
            "train_latency_std": train_metrics["std_ms"],
            "train_throughput": train_metrics["throughput_img_s"],
            "train_vram_mb": train_vram,
            "deploy_latency_ms": deploy_metrics["mean_ms"],
            "deploy_latency_std": deploy_metrics["std_ms"],
            "deploy_throughput": deploy_metrics["throughput_img_s"],
            "deploy_vram_mb": deploy_vram,
            "speedup": speedup,
            "vram_reduction_pct": vram_reduction,
            "max_abs_diff": max_diff,
            "is_exact": is_exact,
        }
        results.append(res_entry)

        print(f"  Train Mode  : {train_metrics['mean_ms']:.2f} +/- {train_metrics['std_ms']:.2f} ms | {train_metrics['throughput_img_s']:.1f} img/s | {train_vram:.1f} MB")
        print(f"  Deploy Mode : {deploy_metrics['mean_ms']:.2f} +/- {deploy_metrics['std_ms']:.2f} ms | {deploy_metrics['throughput_img_s']:.1f} img/s | {deploy_vram:.1f} MB")
        print(f"  Speedup     : {speedup:.2f}x | VRAM Saved: {vram_reduction:.1f}% | Numerical Diff: {max_diff:.2e} (Exact: {is_exact})\n")

    return results


def print_markdown_table(results: List[Dict[str, Any]], model_name: str):
    """Prints a GitHub-Flavored Markdown summary table."""
    print(f"\n### Zero-Overhead Inference Profiling Results: `{model_name}`\n")
    print("| Batch | Train Latency (ms) | Deploy Latency (ms) | Speedup | Train Throughput (img/s) | Deploy Throughput (img/s) | VRAM Train (MB) | VRAM Deploy (MB) | Equivalence |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        equiv_str = "Exact" if r["is_exact"] else f"{r['max_abs_diff']:.1e}"
        print(
            f"| {r['batch_size']} "
            f"| {r['train_latency_ms']:.2f} +/- {r['train_latency_std']:.2f} "
            f"| **{r['deploy_latency_ms']:.2f} +/- {r['deploy_latency_std']:.2f}** "
            f"| **{r['speedup']:.2f}x** "
            f"| {r['train_throughput']:.1f} "
            f"| **{r['deploy_throughput']:.1f}** "
            f"| {r['train_vram_mb']:.1f} "
            f"| {r['deploy_vram_mb']:.1f} "
            f"| {equiv_str} |"
        )
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Hardware Inference Profiling")
    parser.add_argument("--config", type=str, default="configs/tiny_imagenet_pyramid_res_mlp.yaml", help="Path to YAML config")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 16, 32, 64, 128], help="Batch sizes to test")
    parser.add_argument("--num_warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--num_iters", type=int, default=200, help="Timed iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="runs/profiling", help="Directory to save artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config, default_config_path="configs/default.yaml")
    results = run_profiling(
        cfg=cfg,
        batch_sizes=args.batch_sizes,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        device_str=args.device,
    )

    print_markdown_table(results, cfg.model.name)

    # Save to disk
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{cfg.model.name}_profiling_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved profiling metrics to: {out_json}")


if __name__ == "__main__":
    main()
