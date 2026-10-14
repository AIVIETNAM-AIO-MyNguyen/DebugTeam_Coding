# Runs Directory Structure

This directory houses all experimental artifacts, hardware profiling benchmarks, ablation studies, and training logs.

## Directory Layout

```text
runs/
├── ablation_study/               # Golden Ablation Suite (Tiny-ImageNet-200, 60 epochs)
│   ├── ablation_study.txt        # Full console log of all 4 ablation runs on dual Tesla T4
│   └── ablation_learning_curves.png # High-resolution (300 DPI) comparison curves
│
├── profiling/                    # Hardware Inference Profiling (Section 5.3.a)
│   ├── benchmark.txt             # Hardware inference profiling log on Tesla T4
│   └── pyramid_res_mlp_profiling_results.json # Quantitative latency, throughput, VRAM, and numerical diff metrics
│
├── notebooks/                    # Standalone experiment notebooks
│   ├── HybridFormer_CIFAR100.ipynb
│   └── PoolFormer_CIFAR100.ipynb
│
├── legacy_logs/                  # Historical experiment logs, configs, and JSON metrics
│   ├── pyramid_res_mlp.*         # 100-epoch runs on CIFAR-100
│   ├── pyramid_res_mlp_pure.*
│   ├── pyramid_resmlp_droppath.*
│   ├── resmlp_local_stem.*
│   ├── tiny_image_pyramid_res_mlp.*
│   └── train_res_mlp_local_injection_in_token_mix.*
│
├── 20261006_152836_.../          # Standard timestamped training run folders
├── 20261006_153544_.../          # (Checkpoints, logs, metrics.json, metrics.csv)
│
└── parse_runs.py                 # Utility script to parse legacy experiment convergence
```

## Experiment Log Index

| Category | Location | Content |
| :--- | :--- | :--- |
| **Ablation Suite** | `runs/ablation_study/` | 4 Golden Ablations on Tiny-ImageNet (Pure Linear vs DW 3x3 vs Full RepTokenMix vs Non-overlap stem) |
| **Hardware Profiling** | `runs/profiling/` | CUDA Event latency (ms), throughput (img/s), VRAM (MB) across batch sizes 1-128 |
| **Notebooks** | `runs/notebooks/` | Interactive benchmarking notebooks |
| **Legacy Runs** | `runs/legacy_logs/` | Earlier CIFAR-100 baseline experiments |
