# Week 3 Experimental Progress Report: Architecture Exploration & Benchmark Summary

**Project:** DebugTeam_Coding — MLP-Based Vision Architectures on CIFAR-100 & Scaled Benchmarks  
**Authors:** Tan, Nhat Minh, Tien Anh, and Collaborative Peers  
**Date:** October 2026  
**Scope:** Comprehensive synthesis of Week 3 experiments across branches:
1. **Current Branch (`experiment/tan/resmlp_cifar100`):** `pyramid_res_mlp`, `pyramid_res_mlp_pure` (with completed training runs), and peer notebooks in `run/` (`PoolFormer-S12`, `HybridFormer`).
2. **CA-Mixer Branch (`ca-mixer-idea`):** Cellular Automata Token Mixing (`CA-Mixer`) across complexity and resolution scaling.
3. **Spatial Shift Branch (`experiment/tienanh/resmlp_cifar100`):** Parameter-Free Spatial Communication (`Shift-ResMLP`) and Isotropic `ResMLPv2` with complete empirical training logs.

---

## 1. Executive Summary & Master Benchmark Comparison

In Week 2, our team identified a critical limitation in columnar/isotropic MLP architectures (e.g., vanilla ResMLP): they required massive parameter budgets (~2.19M to 14M+) to achieve modest Top-1 accuracy (62.40% on CIFAR-100) because they lacked hierarchical multi-scale feature downsampling and local inductive bias. Week 3 investigated architectural solutions across three independent research axes:
1. **Hierarchical Multi-Stage Pyramid Architectures (`Pyramid-ResMLP` & `Pyramid-ResMLP Pure`):** Combines overlapping patch downsampling with structural re-parameterization in token mixing (RepTokenMix block), with `Pyramid-ResMLP Pure` serving as a strict ablation control.
2. **Resolution-Agnostic Cellular Automata Mixing (`CA-Mixer`):** Employs Neural Cellular Automata (NCA) with biologically inspired local perception operators, enabling variable-resolution inference with constant parameter count.
3. **Parameter-Free Spatial Shift (`Shift-ResMLP`):** Replaces matrix multiplication spatial mixing with zero-parameter 4-direction channel shifts, retaining only channel MLP weights.
4. **MetaFormer & Hybrid Architectures (`PoolFormer-S12` & `HybridFormer`):** Explores non-parametric pooling versus multi-head self-attention hybridization.

### Master Benchmark: CIFAR-100 Benchmark Comparison

| Architecture | Paradigm | Spatial Token Mixer | Deploy Params | Epochs | Top-1 Val Acc (%) | Training Time | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ResMLP Baseline** | Isotropic | Pure Linear $(N \times N)$ | 2.19M | 100 | 62.40% | 54.6 min | Week 02 baseline ($N=64$) |
| **RepMLP Baseline** | Hierarchical | RepConv 3-Branch | 2.16M | 100 | 71.46% | 107.1 min | Week 02 baseline (3 stages) |
| **Pyramid-ResMLP Pure** | Hierarchical | Linear $(N_i \times N_i)$ | **1.93M** | 100 | **68.22%** | **44.9 min** | Pure linear token mixing ablation |
| **Pyramid-ResMLP** | Hierarchical | RepToken (Fused) | **1.79M** | 100 | **72.18%** | **44.5 min** | 1.94M train params fuse to 1.79M |
| **Shift-ResMLP** | Isotropic | 4-Direction Shift | 14.26M | 100 | **72.92%** | 110.2 min | Parameter-free spatial communication |
| **CA-Mixer** | Isotropic NCA | Cellular Automata | **1.90M** | 100 | **64.92%** | **58.4 min** | 0.489 GFLOPs, 2.20% ECE (also 62.87% @ 1.21M, 30ep) |
| **PoolFormer-S12** | Hierarchical | Non-param. AvgPool | 11.45M | 100 | 59.50% | ~53 min | Canonical MetaFormer ($224\times224$) |
| **HybridFormer** | Hierarchical | PoolFormer + MHSA | 10.59M | 300 | **75.70%** | ~160 min | Pooling (S1-2) + MHSA (S3-4), test: 75.27% |

### Master Benchmark: Tiny-ImageNet ($64\times64$, 200 Classes) High-Resolution Scaling

| Model Configuration | Stage Channels ($C$) | Stage Depths ($B$) | Regularization Scheme | Deploy Params | Top-1 Val Acc (%) | Top-5 Val Acc (%) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Pyramid-ResMLP (Base)** | $[96, 192, 384]$ | $[2, 4, 6]$ | Standard weight decay ($\lambda=0.05$) | 9,370,760 | 57.85% | 79.12% |
| **Pyramid-ResMLP (DropPath)** | $[112, 224, 448]$ | $[2, 4, 6]$ | DropPath 0.05, $\lambda=0.10$, Erase 0.25 | **12,728,104** | **62.93%** | **82.83%** |

---

## 2. Experiments on Current Branch (`experiment/tan/resmlp_cifar100`)

### 2.1. Pyramid-ResMLP (`src/models/pyramid_res_mlp.py`)

#### A. Architectural Design
Pyramid-ResMLP transitions the isotropic ResMLP into a hierarchical multi-stage backbone inspired by feature pyramid networks and MetaFormer principles.
1. **Overlapping Patch Embedding Stem:**
   - Instead of large non-overlapping patch slicing (which damages local boundary continuity), it uses an overlapping convolution stem:
     $$\text{Stem: } \text{Conv2d}(3, C_1, k=3, s=2, p=1) \rightarrow \text{BatchNorm2d}(C_1)$$
   - For CIFAR-100 ($32\times32$), it reduces the resolution to $16\times16$ tokens with $C_1 = 64$.
2. **Three-Stage Hierarchical Backbone:**
   - **Stage 1:** Resolution $16\times16$ ($N_1 = 256$ tokens), $C_1 = 64$, Depth $B_1 = 2$.
   - **ConvDownsample 1 $\rightarrow$ 2:** $\text{Conv2d}(64, 128, k=3, s=2, p=1) + \text{BN}$, reducing tokens to $8\times8$ ($N_2 = 64$).
   - **Stage 2:** Resolution $8\times8$ ($N_2 = 64$ tokens), $C_2 = 128$, Depth $B_2 = 2$.
   - **ConvDownsample 2 $\rightarrow$ 3:** $\text{Conv2d}(128, 256, k=3, s=2, p=1) + \text{BN}$, reducing tokens to $4\times4$ ($N_3 = 16$).
   - **Stage 3:** Resolution $4\times4$ ($N_3 = 16$ tokens), $C_3 = 256$, Depth $B_3 = 2$.
3. **Structural Re-parameterization in Token Mixing (RepTokenMix block):**
   - In standard ResMLP, spatial communication relies entirely on a dense matrix $\mathbf{W} \in \mathbb{R}^{N_i \times N_i}$, which completely ignores 2D neighbor spatial proximity.
   - During **training**, RepTokenMix block runs three parallel paths:
     1. Dense token projection: $\text{Linear}(N_i, N_i)$.
     2. Local $1\times1$ depthwise convolution: $\text{Conv2d}(C_i, C_i, k=1, s=1, \text{groups}=C_i) + \text{BN}$.
     3. Local $3\times3$ depthwise convolution: $\text{Conv2d}(C_i, C_i, k=3, s=1, p=1, \text{groups}=C_i) + \text{BN}$.
   - During **deployment**, the depthwise convolutions and batch norms are mapped into equivalent 2D matrices and fused directly into the $\text{Linear}(N_i, N_i)$ weights and biases via `locality_injection()`.
   - **Inference cost:** Zero additional parameters, zero extra FLOPs, and zero kernel-launch overhead compared to pure linear mixing.
4. **Channel Mixing (`ChannelMixLayer`):**
   - Affine normalization: $y = \alpha \odot x + \beta$.
   - Two-layer feedforward network with expansion factor 4: $\text{Linear}(C_i, 4C_i) \rightarrow \text{GELU} \rightarrow \text{Linear}(4C_i, C_i)$.
   - LayerScale: Residual scaled by learnable $\gamma_1, \gamma_2 \in \mathbb{R}^{C_i}$.
   - Stochastic depth (`DropPath`) for regularization.

#### B. Model Size & Configuration Breakdown
- **CIFAR-100 Configuration (`configs/cifar100_pyramid_res_mlp.yaml`):**
  - `channels`: `[64, 128, 256]`, `num_blocks`: `[2, 2, 2]`, `patch_size`: `2`, `expansion_factor`: `4`.
  - **Trainable Parameters (Training):** `1,938,820` (~1.94M).
  - **Deployable Parameters (Fused):** `1,785,828` (~1.79M).
  - *Parameter reduction upon fusion:* `152,992` parameters saved by eliminating parallel DWConv + BN branches.
- **Tiny-ImageNet Base Configuration (`runs/tiny_image_pyramid_res_mlp.yaml`):**
  - `image_size`: `64`, `num_classes`: `200`, `channels`: `[96, 192, 384]`, `num_blocks`: `[2, 4, 6]`.
  - **Trainable Parameters:** `9,566,312` (~9.57M).
  - **Deployable Parameters:** `9,370,760` (~9.37M).
- **Tiny-ImageNet Scaled Configuration (`runs/pyramid_resmlp_droppath.yaml`):**
  - `image_size`: `64`, `num_classes`: `200`, `channels`: `[112, 224, 448]`, `num_blocks`: `[2, 4, 6]`, `drop_path_rate`: `0.05`.
  - **Trainable Parameters:** `12,931,272` (~12.93M).
  - **Deployable Parameters:** `12,728,104` (~12.73M).

#### C. Training Results & Metrics
- **CIFAR-100 (100 Epochs Benchmark, 2 GPUs):**
  - Optimization: AdamW ($\text{lr}=10^{-3}$, weight decay $=0.05$, Cosine Annealing to $10^{-6}$).
  - Augmentations: RandAugment ($N=2, M=9$), Random Erasing (0.1), Color Jitter (0.1), Mixup ($\alpha=0.8$), CutMix ($\alpha=1.0$).
  - **Best Val Top-1 Accuracy:** **72.18%** (achieved at Epoch 89).
  - **Best Val Top-5 Accuracy:** **92.04%** (Epoch 89).
  - **Final Epoch 100:** Val Top-1: `72.14%`, Train Top-1: `64.85%`, Train Loss: `2.5382`, Val Loss: `1.2169`.
  - **Training Time:** **44.53 min** (over 2 GPUs).

---

### 2.2. Pyramid-ResMLP Pure (`src/models/pyramid_res_mlp_pure.py`)

#### A. Architectural Design
- Designed specifically as an **ablation control** to isolate the benefits of the pyramid multi-stage downsampling backbone from the benefits of structural re-parameterization.
- Retains the exact same overlapping patch embedding stem, stage channels `[64, 128, 256]`, and `ConvDownsample` modules as `Pyramid-ResMLP`.
- Replaces RepTokenMix block with PureTokenMix layer:
  $$y = \mathbf{W}_{\text{spatial}} x, \quad \mathbf{W}_{\text{spatial}} \in \mathbb{R}^{N_i \times N_i}$$
  completely omitting the parallel $1\times1$ and $3\times3$ depthwise convolution branches.

#### B. Model Size Breakdown
- `channels`: `[64, 128, 256]`, `num_blocks`: `[2, 2, 2]`.
- **Total Parameters:** **1,926,276** (~1.93M).
- *Difference:* Exactly 12,544 parameters fewer than training-time `Pyramid-ResMLP`, corresponding to the eliminated conv kernels and batch normalization scales/biases across all 6 blocks.

#### C. Full Empirical Training Results (`runs/pyramid_res_mlp_pure.json` & `.log`)
- **Training Setup:**
  - Dataset: CIFAR-100 ($32\times32$, 100 classes).
  - Hardware: Distributed training over 2 GPUs (DDP).
  - Optimization: AdamW ($\text{lr}=10^{-3}$, weight decay $=0.05$, Cosine Annealing to $10^{-6}$), batch size 256.
  - Augmentation: RandAugment, Random Erasing (0.1), Color Jitter (0.1), Mixup ($\alpha=0.8$), CutMix ($\alpha=1.0$), Label Smoothing (0.1), AMP FP16.
- **Empirical Results:**
  - **Total Training Duration:** **44.91 minutes**.
  - **Best Validation Top-1 Accuracy:** **68.22%** (achieved at **Epoch 90**, Val Loss: `1.3402`, Train Top-1: `59.36%`, Train Loss: `2.6839`).
  - **Best Validation Top-5 Accuracy:** **89.98%**.
  - **Final Epoch 100:** Val Top-1: `67.98%`, Val Top-5: `89.90%`, Val Loss: `1.3546`, Train Top-1: `57.18%`, Train Loss: `2.7778`.
- **Convergence Trajectory Across Epochs:**

| Epoch | Train Loss | Train Top-1 (%) | Val Loss | Val Top-1 (%) | Best Val Top-1 (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.3971 | 5.67% | 3.7385 | 13.42% | 13.42% |
| **5** | 3.9461 | 16.74% | 2.8799 | 29.44% | 29.44% |
| **10** | 3.6858 | 25.13% | 2.2554 | 43.04% | 43.04% |
| **20** | 3.3793 | 35.48% | 1.8562 | 53.36% | 53.36% |
| **30** | 3.1968 | 41.73% | 1.7218 | 57.46% | 57.46% |
| **40** | 3.0374 | 45.97% | 1.5107 | 61.86% | 61.86% |
| **50** | 2.9806 | 49.09% | 1.4519 | 64.48% | 64.48% |
| **60** | 2.8471 | 51.80% | 1.4081 | 66.46% | 66.46% |
| **70** | 2.8004 | 56.26% | 1.3831 | 66.56% | 66.56% |
| **80** | 2.8281 | 55.68% | 1.3638 | 67.54% | 67.54% |
| **90 (Best)** | **2.6839** | **59.36%** | **1.3402** | **68.22%** | **68.22%** |
| **100** | 2.7778 | 57.18% | 1.3546 | 67.98% | 68.22% |

#### D. The Two-Step Ablation Breakthrough
Comparing the three models reveals the precise origin of our performance gains:
1. **Gain from Hierarchical Pyramid Design:**
   $$\text{ResMLP Isotropic (62.40\%)} \longrightarrow \text{Pyramid-ResMLP Pure (68.22\%)} \quad (\mathbf{+5.82\%})$$
   Switching from a single-stage isotropic column to a 3-stage pyramid provides $+5.82\%$ accuracy while reducing parameters from $2.19\text{M}$ to $1.93\text{M}$ and training time from $54.65\text{ min}$ to $44.91\text{ min}$.
2. **Gain from Structural Re-parameterization (RepTokenMix block):**
   $$\text{Pyramid-ResMLP Pure (68.22\%)} \longrightarrow \text{Pyramid-ResMLP (72.18\%)} \quad (\mathbf{+3.96\%})$$
   Injecting 2D spatial locality via parallel depthwise convolutions during training adds another $+3.96\%$ accuracy, and fusing them upon deployment achieves this with **zero added latency or parameters** (only $1.79\text{M}$ deployable params).
3. **Cumulative Total Gain:**
   $$\text{ResMLP Baseline (62.40\%)} \longrightarrow \text{Pyramid-ResMLP (72.18\%)} \quad (\mathbf{+9.78\%})$$

---

### 2.3. Peer Notebook 1: PoolFormer-S12 (`run/PoolFormer_CIFAR100.ipynb`)

#### A. Architectural Design
- Implements the foundational MetaFormer architecture proposed by Yu et al. (CVPR 2022).
- Demonstrates that the token mixer can be completely non-parametric:
  $$\text{TokenMixer}(x) = \text{AvgPool2D}(x, k=3, s=1, p=1) - x$$
- Structure:
  - 4 stages with patch embedding and downsampling:
    - Stage 1 stem: $\text{Conv2d}(3, 64, k=7, s=4, p=2)$ (stride 4 reduces $224\times224 \rightarrow 56\times56$).
    - Stages 2–4 downsamplers: $\text{Conv2d}(C_{\text{in}}, C_{\text{out}}, k=3, s=2, p=1)$.
  - Stage Depths: `(2, 2, 6, 2)`, Stage Widths: `(64, 128, 320, 512)`.
  - Normalization: $\text{GroupNorm}(1, C)$ (equivalent to LayerNorm over channels).
  - Channel MLP: $1\times1$ convs with expansion ratio 4.0 and GELU.
  - Residual stabilization: $\text{LayerScale} = 10^{-5}$ and $\text{StochasticDepth} = 0.1$.

#### B. Model Size & Setup
- **Input Resolution:** **$224\times224$** (CIFAR-100 images upsampled using bicubic `RandomResizedCrop`).
- **Number of Classes:** 100.
- **Total Parameters:** **11,453,476** (~11.45M).

#### C. Training Results
- **Training Strategy:** 100 epochs, AdamW with 5 warmup epochs, batch size 64 across 2 GPUs (DDP).
- **Best Validation Accuracy:** **59.50%** (achieved at Epoch 91).
- **Final CIFAR-100 Test Accuracy:** **59.02%**.
- **Analysis:** Non-parametric average pooling achieves 59.50% on CIFAR-100, but is limited by the very low learning rate ($6.25\times10^{-5}$) and the blur induced by upscaling $32\times32$ to $224\times224$.

---

### 2.4. Peer Notebook 2: HybridFormer (`run/HybridFormer_CIFAR100.ipynb`)

#### A. Architectural Design
- A novel hybrid architecture exploring the synergy between MetaFormer pooling and Multi-Head Self-Attention (MHSA).
- **Macro Design:**
  - **Early Stages (Stages 0 & 1):** Use parameter-free PoolFormer blocks ($3\times3$ AvgPool) with $\text{GroupNorm}$ to extract fine-grained local textures efficiently at high spatial resolutions.
  - **Late Stages (Stages 2 & 3):** Switch to Multi-Head Self-Attention (`F.scaled_dot_product_attention`, head dimension 64) with `ChannelNorm` and MetaFormer skip scaling ($r_1, r_2$).
  - **Channel MLP:** Equipped with `StarReLU` activation:
    $$\text{StarReLU}(x) = s \cdot (\text{ReLU}(x))^2 + b$$
  - **Patch Embedding:** Non-strided $\text{Conv2d}(3, 64, k=3, s=1, p=1)$ in Stage 0 preserves native $32\times32$ resolution, followed by $s=2$ downsamplers for subsequent stages ($32\times32 \rightarrow 16\times16 \rightarrow 8\times8 \rightarrow 4\times4$).
  - **Classifier Head:** 2-layer MLP with StarReLU: $\text{Linear}(384, 1536) \rightarrow \text{StarReLU} \rightarrow \text{Linear}(1536, 100)$.

#### B. Model Size & Setup
- **Input Resolution:** **$32\times32$** (native CIFAR-100).
- **Depths:** `(2, 2, 6, 2)`, **Dims:** `(64, 128, 256, 384)`.
- **Total Parameters:** **10,594,622** (~10.59M).

#### C. Training Results
- **Training Strategy:** 300 epochs, AdamW ($\text{lr}=5\times10^{-4}$), batch size 256 over 2 GPUs (DDP).
- **Best Validation Accuracy:** **75.70%** (achieved at Epoch 275).
- **Final CIFAR-100 Test Accuracy:** **75.27%** (Test Loss: `1.1412`).
- **Analysis:** Demonstrates the ceiling of MetaFormer hybridization on CIFAR-100: reserving self-attention for lower spatial resolutions ($8\times8$ and $4\times4$) captures global semantic context while keeping computational complexity low.

---

## 3. Experiments on Branch `ca-mixer-idea` (Cellular Automata Mixer)

### 3.1. CA-Mixer Architectural Concept & Mechanism
- Standard MLP token mixers (e.g., ResMLP) flatten spatial features into $N$ tokens and multiply by a fixed matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$, which creates strict resolution locking: testing on different resolutions is mathematically impossible without retraining or resizing.
- **CA-Mixer Solution:** Replace the matrix multiplication with **Neural Cellular Automata (NCA)** updates!
  - **Perception:** Uses fixed, biologically inspired local convolution operators:
    $$\mathbf{K} = [\text{Sobel}_X, \text{Sobel}_Y, \text{Laplacian}] \in \mathbb{R}^{3 \times 3 \times 3}$$
    convolved channel-wise with input state $h$.
  - **Update MLP $f_\theta$:** Shared $1\times1$ convolutions: $\text{Conv2d}(3C, C_{\text{hidden}}, 1) \rightarrow \text{GELU} \rightarrow \text{Conv2d}(C_{\text{hidden}}, C, 1) \rightarrow \tanh$.
  - **Stochastic Firing:** Updates occur with a cell fire rate ($p=0.8$) to simulate asynchronous cellular behavior and prevent co-adaptation.
  - **Iterative Dynamics:** State updates are iterated for $T$ steps:
    $$h_{t+1} = h_t + m \odot f_\theta(\mathbf{K} \ast h_t), \quad T = \max(H, W)$$
  - **Resolution Agnostic:** Since $\mathbf{K}$ and $f_\theta$ are 2D convolutions, the weights are **100% independent of image resolution $H \times W$**.

### 3.2. Complexity & Parameter Invariance across Resolutions
- **Configuration:** `patch_size`: `4`, `dim`: `128`, `depth`: `6`, `ca_hidden`: `128`, `ca_channel_hidden`: `512`, `fire_rate`: `0.8`.
- **Parameter Invariance across Resolutions:**
  - $32\times32$ ($N=64$ tokens): **1.2075M params**, 0.525 GFLOPs, 2,072 img/s.
  - $64\times64$ ($N=256$ tokens): **1.2075M params (Constant!)**, 3.797 GFLOPs, 506 img/s.
  - $96\times96$ ($N=576$ tokens): **1.2075M params (Constant!)**, 12.358 GFLOPs, 142 img/s.
  - $128\times128$ ($N=1024$ tokens): **1.2075M params (Constant!)**, 28.752 GFLOPs, 64 img/s.

### 3.3. Empirical Performance on CIFAR-100 (3 Seeds Benchmark)
- **Top-1 Accuracy:** **$62.87 \pm 0.24\%$**
- **Top-5 Accuracy:** **$88.09 \pm 0.30\%$**
- **Test Loss:** **$1.388 \pm 0.009$**
- **Expected Calibration Error (ECE):** **$3.82 \pm 0.22\%$**
- **Train-Test Generalization Gap:** **$20.74 \pm 0.56\%$** *(slashed overfitting by over 17% compared to dense unconstrained token mixing).*
- **Zero-Shot Native Resolution Robustness:**
  - Native $24\times24$: **53.65%** Top-1.
  - Native $32\times32$: **62.87%** Top-1.
  - Native $40\times40$: **47.83%** Top-1.
  - Native $48\times48$: **32.15%** Top-1.

### 3.4. Parameter-Matched 100-Epoch Scaling (Commit `6c70e3e`)
- **Configuration:** `patch=4, dim=160, depth=6, nca_hidden=160, channel_hidden=652`
- **Dynamic Step Budget:** $T = \lceil 0.5 \times \max(h,w) \rceil = 4$ steps at $32\times32$, $T_{\text{jitter}}=0.25$, `update_prob=0.8`, `frozen_filters=True`, `identity_kernel=False`.
- **Training Recipe:** 100 epochs, AdamW ($\eta = 10^{-3}$, weight decay $= 0.05$), batch size 256, warmup 5 epochs, label smoothing 0.1, AMP FP16. Single Tesla T4 GPU.
- **Empirical Results:**
  - **Top-1 Validation Accuracy:** **64.92%** (+2.05% gain over the 30-epoch 1.21M baseline).
  - **Top-5 Validation Accuracy:** **85.72%**.
  - **Test Loss:** **1.503**.
  - **Expected Calibration Error (ECE):** **2.20%** (best calibration across all Week 03 evaluated models!).
  - **Parameters:** **1.901M** (`1,901,100`), directly matching the parameter budget of Pyramid-ResMLP Pure (1.93M) and Pyramid-ResMLP (1.94M / 1.79M).
  - **FLOPs:** **0.489 GFLOPs/image**.
  - **Training Speed:** 35.0 s/epoch, 3,942 img/s, **58.4 minutes total**, 2.80 GB VRAM.
  - **Train-Test Gap:** 35.01%.

---

## 4. Experiments on Branch `experiment/tienanh/resmlp_cifar100`

### 4.1. Shift-ResMLP (`src/models/res_mlp_spatial_shift.py`)

#### A. Architectural Design
- Investigates **parameter-free spatial communication** by replacing the linear token-mixing matrix with a 4-way coordinate spatial shift operation (`SpatialShift`).
- **Operational Flow:**
  1. Input tokens $(B, N, C)$ are reshaped back to the 2D feature grid $(B, H, W, C)$ where $H = W = \text{image\_size} / \text{patch\_size} = 8$.
  2. The channel dimension $C$ is partitioned into 4 equal slices of size $C/4$:
     - **Slice 1 (Left Shift):** `shifted[:, :, :-1, 0*c:1*c] = x[:, :, 1:, 0*c:1*c]` (padded with 0 on the right).
     - **Slice 2 (Right Shift):** `shifted[:, :, 1:, 1*c:2*c] = x[:, :, :-1, 1*c:2*c]` (padded with 0 on the left).
     - **Slice 3 (Up Shift):** `shifted[:, :-1, :, 2*c:3*c] = x[:, 1:, :, 2*c:3*c]` (padded with 0 on the bottom).
     - **Slice 4 (Down Shift):** `shifted[:, 1:, :, 3*c:4*c] = x[:, :-1, :, 3*c:4*c]` (padded with 0 on the top).
  3. The shifted tensor is flattened back to $(B, N, C)$.
  4. LayerScale $\gamma_1$ and residual addition: $x \leftarrow x + \gamma_1 \odot \text{Shift}(x)$.
- **Key Property:** The spatial mixing operation contains **strictly 0 learnable parameters**! It acts as an inductive bias that routes neighboring information across spatial locations without any matrix multiplication.

#### B. Model Size Breakdown
- **Trained Model Capacity:**
  - Instantiated with default code settings (`hidden_dim=384, num_blocks=12, mlp_ratio=4.0`):
    - Patch embedding: $3 \times 4 \times 4 \times 384 + 384 = 18,816$.
    - 12 Blocks: Each block has Affine layers ($4 \times 384$) and Channel MLP ($2 \times 384 \times 1536 + 1536 + 384 = 1,181,568$).
    - Spatial Shift: **0 parameters**.
    - Head & Norm: $384 + 384 + 384 \times 100 + 100 = 39,268$.
    - **Total Parameters:** **14,264,548** (~14.26M).
- **Target Compact Configuration (`hidden_dim=128, num_blocks=8, mlp_ratio=2.0`):**
  - **Total Parameters:** **552,932** (~0.55M).

#### C. Full Empirical Training Results (`results/Shift-ResMLP_training_log.txt`)
- **Training Setup:**
  - Dataset: CIFAR-100 ($32\times32$, 100 classes).
  - Optimization: AdamW ($\text{lr}=10^{-3}$, weight decay $=0.05$, Cosine Annealing to $10^{-6}$).
  - Augmentation: RandAugment, Random Erasing (0.1), Color Jitter (0.1), Mixup ($\alpha=0.8$), CutMix ($\alpha=1.0$), Label Smoothing (0.1), AMP FP16.
  - Batch size: 128, Total epochs: 100.
  - Device: CUDA single GPU.
- **Training Performance:**
  - **Total Training Duration:** **110.21 minutes**.
  - **Best Validation Top-1 Accuracy:** **72.92%** (achieved at **Epoch 88**, with Val Loss: `1.1330` and Train Loss: `2.6176`).
  - **Final Epoch 100:** Val Acc: **72.56%**, Train Acc: **63.37%**, Val Loss: **1.1197**, Train Loss: **2.6201**.
- **Convergence Trajectory Across Epochs:**

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Best Val Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4.4850 | 3.59% | 4.0979 | 7.56% | 7.56% |
| **5** | 4.1824 | 10.87% | 3.4197 | 20.02% | 20.02% |
| **10** | 3.9309 | 17.49% | 2.8236 | 30.50% | 30.50% |
| **20** | 3.5482 | 29.73% | 2.0848 | 47.24% | 47.24% |
| **30** | 3.3027 | 37.66% | 1.6972 | 56.68% | 56.68% |
| **40** | 3.1142 | 45.07% | 1.4567 | 62.62% | 62.62% |
| **50** | 2.9645 | 49.92% | 1.2926 | 67.28% | 67.28% |
| **60** | 2.8211 | 54.66% | 1.2175 | 69.46% | 69.46% |
| **70** | 2.8154 | 56.01% | 1.1696 | 70.64% | 70.64% |
| **80** | 2.6339 | 61.73% | 1.1366 | 71.86% | 71.86% |
| **88 (Best)** | **2.6176** | **62.84%** | **1.1330** | **72.92%** | **72.92%** |
| **90** | 2.6708 | 61.59% | 1.1255 | 72.58% | 72.92% |
| **100** | 2.6201 | 63.37% | 1.1197 | 72.56% | 72.92% |

- **Key Takeaway:** Shift-ResMLP achieves **72.92%** accuracy on CIFAR-100 without a single learnable parameter in the spatial mixer. This proves empirically that pure coordinate feature translation is sufficient to break permutation invariance and provide effective spatial receptive fields.

---

### 4.2. ResMLPv2 (`src/models/res_mlp_v2.py`)

#### A. Architectural Design
- Clean reference implementation of the standard ResMLP baseline (Touvron et al., 2021) adapted for CIFAR-100.
- Uses non-overlapping linear patch projection `nn.Linear(patch_dim, hidden_dim)`, Affine normalization, cross-patch communication via `nn.Linear(num_patches, num_patches)`, and cross-channel communication via `Linear(hidden_dim, 4*hidden_dim) -> GELU -> Linear(4*hidden_dim, hidden_dim)`.

#### B. Model Size Comparison vs. Shift-ResMLP
- **Full Configuration (`hidden_dim=384, num_blocks=12`):**
  - **ResMLPv2:** `14,314,468` params.
  - **Shift-ResMLP:** `14,264,548` params.
  - *Difference:* Exactly $12 \times (64 \times 64 + 64) = 49,920$ parameters, representing the total weights of the 12 spatial linear mixing layers in ResMLPv2 that are completely eliminated in Shift-ResMLP.
- **Target Compact Configuration (`hidden_dim=128, num_blocks=8`):**
  - **ResMLPv2:** `586,212` params.
  - **Shift-ResMLP:** `552,932` params (saving $8 \times (64 \times 64 + 64) = 33,280$ parameters).

---

## 5. Comparative Insights for Week 3 Report

1. **Parameter Efficiency: Pyramid-ResMLP vs. Shift-ResMLP:**
   - Both `Pyramid-ResMLP` and `Shift-ResMLP` break through the 72% accuracy ceiling on CIFAR-100:
     - `Pyramid-ResMLP`: **72.18%** Val Acc with only **1,785,828 deployable parameters** and **44.53 min** training time.
     - `Shift-ResMLP`: **72.92%** Val Acc with **14,264,548 parameters** and **110.21 min** training time.
   - While Shift-ResMLP eliminates spatial weights, its single-stage isotropic columnar design forces high channel capacity ($D=384, L=12$) to achieve high performance. In contrast, Pyramid-ResMLP's hierarchical downsampling achieves comparable accuracy (+0.72% vs Week 2 RepMLP) with **8× fewer parameters** (1.79M vs 14.26M) and **2.5× faster training**.
2. **The Pure Pyramid Ablation:**
   - The completion of `Pyramid-ResMLP Pure` (68.22%) gives us a definitive 2-step ablation proving that:
     1. Multi-stage pyramid feature downsampling contributes **+5.82%** (62.40% $\rightarrow$ 68.22%).
     2. Structural re-parameterization contributes **+3.96%** (68.22% $\rightarrow$ 72.18%).
3. **Resolution Flexibility & Calibration: CA-Mixer:**
   - CA-Mixer provides a completely orthogonal advantage: resolution invariance. At 1.90M parameters over 100 epochs, it scales to **64.92%** Top-1 validation accuracy (+2.05% gain over the 30-epoch 1.21M baseline) with an outstanding Expected Calibration Error of **2.20%** (best among all models) and 58.4 min training time, while evaluating on arbitrary resolutions with zero parameter changes.
4. **Hybrid Attention Ceiling:**
   - `HybridFormer` demonstrates that combining lightweight pooling with late-stage self-attention reaches **75.70%** Val Acc (75.27% test), setting the performance benchmark for MetaFormer hybrids on CIFAR-100.

---

## 6. Official Architecture Diagrams Index in Week 03 Report

All model architectures are visually documented in `reports/w03/figures/` and embedded directly in the compiled report (`reports/w03/main.pdf`):

| Model Architecture | Source / Location | Figure in Report | Description |
| :--- | :--- | :--- | :--- |
| **Pyramid-ResMLP & Pyramid-ResMLP Pure** | `reports/w03/figures/Pyramid ResMLP and Pyramid ResMLP Pure.png` | **Figure 1** | 3-stage pyramid macro-flow, block design, training RepTokenMix, locality injection fusion, and pure linear ablation mixer. |
| **Shift-ResMLP** | `results/Shift-ResMLP Architecture.png`, `results/Shift-ResMLP Block.png`, `results/Spatial-Shift (Details).png`, `results/Channel MLP (Details).png` | **Figure 2** | Multi-panel subfigures detailing macro dataflow, block residual paths, parameter-free 4-direction coordinate shift, and channel MLP. |
| **CA-Mixer** | `reports/w03/figures/ca_mixer_arch.png`, `fig_complexity.png`, `fig_resolution.png` | **Figures 3 & 4** | Neural Cellular Automata (NCA) iterative dynamics, parameter scaling invariance, and zero-shot resolution robustness. |
| **PoolFormer-S12** | `diagrams/poolformer.pdf` | **Figure 5** | Standalone vector PDF diagram of the 4-stage PoolFormer macro-architecture and non-parametric average pooling block. |
| **HybridFormer** | `diagrams/hybridformer.pdf` | **Figure 6** | Standalone vector PDF diagram of the 4-stage HybridFormer combining pooling in Stages 1–2 with self-attention in Stages 3–4. |
| **Training Trajectories** | `reports/w03/figures/training_curves_comparison.png` | **Figure 7** | 100-epoch validation accuracy and loss convergence comparison on CIFAR-100. |

