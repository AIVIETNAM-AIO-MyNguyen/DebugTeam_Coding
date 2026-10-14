# Experimental Results Directory

This directory aggregates empirical benchmark results, architectural diagrams, and evaluation outputs across project milestones.

---

## Directory Structure

```text
results/
├── ca_1p9M_100epochs/               # CA-Mixer (1.9M params, 100 epochs on CIFAR-100)
│   ├── env.json                     # System hardware and library environment metadata
│   ├── fig_pareto.png               # Pareto efficiency plot (Accuracy vs. GFLOPs / Parameters)
│   ├── fig_training_curves.png      # Training & validation loss and accuracy curves
│   ├── summary.csv                  # Main benchmark metrics summary
│   ├── summary_numeric.csv          # Unformatted numerical metrics for automated parsing
│   ├── table_configs.csv            # Detailed layer and hyperparameter configuration
│   ├── table_main.tex               # Formatted LaTeX table for reports/papers
│   ├── table_matched.csv            # Parameter-matched comparison table
│   └── table_pyramid_vs_iso.csv     # Multi-scale vs. isotropic comparison table
│
├── Channel MLP (Details).png        # Shift-ResMLP inverted bottleneck channel MLP diagram
├── Shift-ResMLP Architecture.png    # Shift-ResMLP complete macro architecture diagram
├── Shift-ResMLP Block.png           # Shift-ResMLP residual block dataflow diagram
├── Shift-ResMLP_training_log.txt    # Shift-ResMLP training execution log
└── Spatial-Shift (Details).png      # Parameter-free 4-direction spatial shift operator diagram
```

---

## Experiment Summaries

### 1. CA-Mixer (1.9M, 100 Epochs) (`ca_1p9M_100epochs/`)
- **Model:** CA-Mixer with patch size 4, hidden dimension 160, depth 6, $\kappa = 0.5$, $T$-jitter = 0.25, update probability = 0.8.
- **Dataset:** CIFAR-100 (100 epochs, batch size 256, AdamW, cosine schedule).
- **Key Metrics:**
  - **Top-1 Accuracy:** 64.92%
  - **Top-5 Accuracy:** 85.72%
  - **Parameters:** 1.901M
  - **Complexity:** 0.489 GFLOPs / image
  - **Throughput:** ~3,942 images / second

### 2. Shift-ResMLP Baseline (Root)
- **Model:** ResMLP with zero-parameter 4-direction coordinate shift token mixing.
- **Artifacts:** High-resolution architectural diagrams and full training logs referenced in milestone reports.
