# MLP-Variants Vision Experimentation Platform

A modular, reproducible, and extensible research codebase for benchmarking **MLP-based vision architectures** (MLP-Mixer, ResMLP, gMLP, CycleMLP, hire-MLP, etc.) across standardized vision datasets (**CIFAR-100**, **Tiny-ImageNet-200**, and **ImageNet-100 / Subsets**).

Built for collaborative teams with:
- **Strict Reproducibility**: Unified seed control across Python, NumPy, PyTorch, cuDNN, and multi-process DataLoader workers.
- **Reproducible Package Management**: Powered by **UV** for ultra-fast, cross-platform dependency locking.
- **Zero-Friction Dual Execution**: Designed to run seamlessly in both CLI terminal scripts (`.py`) and interactive notebooks (`.ipynb`) locally, on **Google Colab**, or **Kaggle**.
- **Model Registry Pattern**: Plug-and-play architecture where teammates can branch out, define new MLP variants in a single file with `@register_model`, and immediately run experiments without touching core engine code.
- **Standardized Naming & Provenance**: Automated experiment run directories tracking `git_info.json` (commit hash, branch, dirty flag), configs, metrics (JSON/CSV), and checkpoints.

---

## Table of Contents
1. [Repository Structure](#1-repository-structure)
2. [Quickstart with UV (Local Environment)](#2-quickstart-with-uv-local-environment)
3. [Running on Google Colab & Kaggle](#3-running-on-google-colab--kaggle)
4. [Teammate Branching & Development Workflow](#4-teammate-branching--development-workflow)
5. [Seed Control & Determinism](#5-seed-control--determinism)
6. [Experiment Naming Convention & Directory Layout](#6-experiment-naming-convention--directory-layout)
7. [Data Pipelines](#7-data-pipelines)
8. [Configuration System & CLI Overrides](#8-configuration-system--cli-overrides)
9. [Training & Evaluation Usage](#9-training--evaluation-usage)
10. [Adding a New MLP Variant](#10-adding-a-new-mlp-variant)

---

## 1. Repository Structure

```text
DebugTeam_Coding/
├── pyproject.toml              # UV / pip package configuration (setuptools build backend)
├── .python-version             # Python version pin (3.11)
├── .gitignore                  # Clean ignores for checkpoints, datasets, caches
├── README.md                   # Team documentation & onboarding guide
├── configs/                    # Hierarchical YAML configuration files
│   ├── default.yaml            # Base default parameters
│   ├── cifar100_mlp_mixer.yaml # CIFAR-100 baseline config
│   ├── tiny_imagenet_mlp_mixer.yaml # Tiny-ImageNet baseline config
│   └── imagenet_subset_mlp_mixer.yaml # ImageNet subset baseline config
├── src/                        # Core Python package
│   ├── __init__.py
│   ├── config/                 # YAML parser, deep-merging & CLI overrides
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── utils/                  # Seed control, experiment naming, logging, checkpoints
│   │   ├── __init__.py
│   │   ├── seed.py             # Deterministic random/numpy/torch/worker seeding
│   │   ├── naming.py           # Experiment naming and artifact folder creation
│   │   ├── logger.py           # Structured logging and JSON/CSV metric tracker
│   │   └── checkpoint.py       # Checkpoint save/load with best metric tracking
│   ├── data/                   # Data pipelines and augmentations
│   │   ├── __init__.py         # Unified `build_dataloaders(cfg)` factory
│   │   ├── transforms.py       # Augmentation pipelines, Mixup/CutMix collator
│   │   ├── cifar100.py         # CIFAR-100 pipeline (auto-download)
│   │   ├── tiny_imagenet.py    # Tiny-ImageNet pipeline (auto-download & parser)
│   │   └── imagenet_subset.py  # ImageNet subset pipeline & mock data generator
│   ├── models/                 # Model registry & architectures
│   │   ├── __init__.py         # `build_model(cfg)` factory
│   │   ├── base.py             # BaseClassifier and @register_model registry
│   │   └── mlp_mixer.py        # Reference MLP-Mixer implementation
│   └── engine/                 # Training and evaluation loops
│       ├── __init__.py
│       ├── trainer.py          # Modular Trainer (AMP, scheduler, grad clip, metrics)
│       └── evaluator.py        # Standalone evaluation & inference
├── scripts/                    # Command-line entry points
│   ├── run_train.py            # CLI training script
│   ├── run_eval.py             # CLI evaluation script
│   └── download_data.py        # Dataset downloader helper
├── notebooks/                  # Interactive Jupyter notebooks
│   ├── 00_quickstart_colab_kaggle.ipynb # Interactive training walkthrough
│   └── 01_data_exploration.ipynb        # Dataset & augmentation visualizer
└── tests/                      # Fast unit tests
    ├── test_seed.py            # Test seed reproducibility
    ├── test_config.py          # Test config parsing and CLI overrides
    ├── test_models.py          # Test MLP forward pass across image sizes
    └── test_data.py            # Test dataset loading and transforms
```

---

## 2. Quickstart with UV (Local Environment)

We use [uv](https://github.com/astral-sh/uv) for blazing-fast, deterministic virtual environments.

### 2.1 Installation
If you don't have `uv` installed:
```powershell
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 Create Environment and Sync Dependencies
```bash
# 1. Create a Python 3.11 virtual environment
uv venv --python 3.11

# 2. Sync all project dependencies (installs torch, torchvision, timm, etc.)
uv sync

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

Alternatively, you can run any command directly with `uv run`:
```bash
uv run python scripts/run_train.py --config configs/cifar100_mlp_mixer.yaml
```

---

## 3. Running on Google Colab & Kaggle

The repository is built to work immediately inside Google Colab or Kaggle with zero setup friction.

### Setup Cell in Colab / Kaggle:
```python
# 1. Clone repository (if running in a fresh Colab instance)
!git clone https://github.com/<your-org>/DebugTeam_Coding.git
%cd DebugTeam_Coding

# 2. Install editable package
!pip install -e .

# 3. Verify GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Interactive Usage inside Notebook:
```python
from src.config import load_config
from src.utils.seed import set_seed
from src.data import build_dataloaders
from src.models import build_model
from src.engine import Trainer

# 1. Set seed
set_seed(42, deterministic=True)

# 2. Load config
cfg = load_config("configs/cifar100_mlp_mixer.yaml")
cfg.training.epochs = 5  # Quick test

# 3. Build data and model
train_loader, val_loader, test_loader = build_dataloaders(cfg)
model = build_model(cfg)

# 4. Train
trainer = Trainer(model, train_loader, val_loader, cfg)
results = trainer.fit()
```

### 3.2 Dual GPU Training on Kaggle (2x NVIDIA T4 with DDP)

Kaggle allows you to turn on **Accelerator: GPU T4 x2** (in the notebook right-hand sidebar under *Session Options*).

Our codebase has native **DistributedDataParallel (DDP)** support. To utilize both T4 GPUs in parallel:

#### Check GPU count in notebook:
```python
import torch
print("GPU Count:", torch.cuda.device_count())  # Returns: 2
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

#### Launch training across both GPUs:
In a Kaggle notebook cell, run:
```bash
!torchrun --nproc_per_node=2 scripts/run_train.py \
    --config configs/cifar100_mlp_mixer.yaml \
    --data.batch_size 128 \
    --training.epochs 100
```

#### How it works:
- `torchrun` launches 2 independent worker processes (one on GPU 0, one on GPU 1).
- **`DistributedSampler`** automatically partitions each dataset evenly without overlap.
- Gradients are averaged across both GPUs via NCCL all-reduce backend.
- Total effective batch size = `data.batch_size * 2` (e.g. $128 \times 2 = 256$).
- Only **Rank 0** prints logs, updates progress bars, and writes checkpoint files to disk to prevent file contention.

Open and run `notebooks/00_quickstart_colab_kaggle.ipynb` for a complete step-by-step interactive workflow.

---

## 4. Teammate Branching & Development Workflow

To prevent conflicts and ensure clean collaboration, teammates should follow this branching pattern:

1. **Checkout `main` and pull latest changes:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create your feature or experiment branch:**
   ```bash
   # Convention: experiment/<your_name>/<model_or_idea>
   git checkout -b experiment/tan/resmlp_cifar100
   ```

3. **Implement your model variant:**
   Create a new file in `src/models/<your_model>.py` and decorate your class with `@register_model("<model_name>")`.

4. **Create or adjust a config:**
   Add `configs/cifar100_<your_model>.yaml`.

5. **Run your experiment:**
   ```bash
   python scripts/run_train.py --config configs/cifar100_<your_model>.yaml
   ```

6. **Track your artifacts:**
   Results will automatically be stored in `runs/<timestamp>_<model>_<dataset>_<tag>_s<seed>/` with `git_info.json` capturing your branch and commit.

7. **Commit and push:**
   ```bash
   git add src/models/<your_model>.py configs/cifar100_<your_model>.yaml
   git commit -m "feat: implement ResMLP variant on CIFAR-100"
   git push -u origin experiment/tan/resmlp_cifar100
   ```

---

## 5. Seed Control & Determinism

MLP architectures are sensitive to weight initialization and data augmentation ordering. `src/utils/seed.py` implements complete deterministic seeding across:
- Python built-in `random` & `PYTHONHASHSEED`
- `numpy.random`
- `torch.manual_seed` & `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- Multi-process DataLoader worker initialization (`seed_worker`)
- DataLoader sampling via seeded `torch.Generator`

### Usage:
```python
from src.utils.seed import set_seed

# Enforces complete determinism
set_seed(seed=42, deterministic=True)
```

In YAML configs:
```yaml
training:
  seed: 42
  deterministic: true
```

---

## 6. Experiment Naming Convention & Directory Layout

To avoid naming collisions and lost artifacts, every run generates a standardized name:
```
<YYYYMMDD_HHMMSS>_<model>_<dataset>_<tag>_s<seed>
```
*Example:* `20260925_113000_mlp_mixer_cifar100_baseline_s42`

### Run Directory Layout:
```text
runs/20260925_113000_mlp_mixer_cifar100_baseline_s42/
├── config.yaml          # Full snapshot of runtime configuration
├── git_info.json        # Branch name, commit hash, dirty flag
├── logs/
│   └── train.log        # Detailed console and file training log
├── metrics/
│   ├── metrics.json     # Epoch-by-epoch loss & accuracy log
│   └── metrics.csv      # CSV formatted metrics for easy plotting
└── checkpoints/
    ├── last_model.pt    # Most recent epoch weights & optimizer state
    └── best_model.pt    # Checkpoint with highest Validation Top-1 Accuracy
```

---

## 7. Data Pipelines

We provide optimized pipelines for three standard vision benchmarks:

| Dataset | Native Resolution | Classes | Train Samples | Val / Test Samples | Auto-Download |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-100** | $32 \times 32$ | 100 | 45,000 | 5,000 / 10,000 | Yes (via torchvision) |
| **Tiny-ImageNet-200** | $64 \times 64$ | 200 | 100,000 | 10,000 | Yes (Stanford CS231n) |
| **ImageNet Subset (ImageNet-100)** | $224 \times 224$ | 100 | ~130,000 | 5,000 | Yes (Mock generator fallback) |

### Features:
- **Automatic Downloading**: Run `python scripts/download_data.py --dataset all` or enable `download: true` in config.
- **Tiny-ImageNet Annotation Parser**: Correctly parses `val_annotations.txt` directly without rearranging files on disk.
- **ImageNet-100 Subsetting**: Dynamically extracts $N$ classes from standard ImageNet directory, remapping class IDs to $0 \dots N-1$. Includes a lightweight synthetic mock generator for fast pipeline testing without downloading 150GB.
- **Augmentation & Regularization**: RandAugment, AutoAugment, RandomErasing (Cutout), ColorJitter, and batch-level **Mixup / CutMix** collation.

---

## 8. Configuration System & CLI Overrides

Configurations are organized in `configs/` as clean YAML files. The parser supports:
1. **Inheritance & Merging**: Merge `configs/default.yaml` with any experiment-specific YAML.
2. **Attribute Access**: Access keys cleanly as `cfg.model.hidden_dim` or `cfg["model"]["hidden_dim"]`.
3. **CLI Dot-Notation Overrides**: Override any nested key dynamically without modifying the YAML file:

```bash
python scripts/run_train.py \
    --config configs/cifar100_mlp_mixer.yaml \
    --training.lr 5e-4 \
    --training.seed 100 \
    --data.batch_size 64 \
    --model.dropout 0.2
```

---

## 9. Training & Evaluation Usage

### Train a Model (CLI)
```bash
# CIFAR-100
python scripts/run_train.py --config configs/cifar100_mlp_mixer.yaml

# Tiny-ImageNet
python scripts/run_train.py --config configs/tiny_imagenet_mlp_mixer.yaml

# ImageNet-100 Subset
python scripts/run_train.py --config configs/imagenet_subset_mlp_mixer.yaml
```

### Evaluate a Checkpoint (CLI)
```bash
python scripts/run_eval.py \
    --config configs/cifar100_mlp_mixer.yaml \
    --checkpoint runs/<exp_name>/checkpoints/best_model.pt \
    --split test
```

### Pre-download Datasets
```bash
python scripts/download_data.py --dataset cifar100
python scripts/download_data.py --dataset tiny_imagenet
python scripts/download_data.py --dataset mock_imagenet
```

---

## 10. Adding a New MLP Variant

To add a new architecture (e.g. `ResMLP`, `gMLP`, `CycleMLP`), follow these 3 simple steps:

### Step 1: Create Model Class in `src/models/`
Create `src/models/my_mlp.py`:
```python
import torch
import torch.nn as nn
from src.models.base import BaseClassifier, register_model

@register_model("my_mlp")
class MyMLP(BaseClassifier):
    def __init__(self, image_size=32, num_classes=100, hidden_dim=256, **kwargs):
        super().__init__()
        # Define your architecture here
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).mean(dim=2)
        return self.head(x)
```

### Step 2: Import in `src/models/__init__.py`
Add:
```python
import src.models.my_mlp  # Registers "my_mlp"
```

### Step 3: Create YAML Config
Create `configs/cifar100_my_mlp.yaml`:
```yaml
experiment:
  tag: "my_mlp_baseline"

model:
  name: "my_mlp"
  hidden_dim: 256
```

### Step 4: Run!
```bash
python scripts/run_train.py --config configs/cifar100_my_mlp.yaml
```
Your model is immediately wired up with AMP, LR scheduling, data pipelines, checkpointing, and seed control!
