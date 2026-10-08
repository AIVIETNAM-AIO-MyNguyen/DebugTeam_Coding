# Bảng Đặc Tả Kỹ Thuật: ConvMixer & Res-ConvMixer trên CIFAR-100
> **Tài liệu bàn giao cho nhóm (Dành cho Tân & Tiến để tổng hợp báo cáo và thuyết trình)**

---

## 1. Phân Tích Dung Lượng Mô Hình (Model Scaling & Capacity Tiers)

Để giải quyết bài toán phân loại **100 lớp phức tạp** trên CIFAR-100 mà không bị thiếu dung lượng học (underfitting), kiến trúc được chia làm 3 cấp độ quy mô chuẩn:

| Cấp độ (Tier) | Cấu hình (`dim`, `depth`) | Tham số ConvMixer | Tham số Res-ConvMixer (Hybrid) | So sánh tương đương | Mục đích sử dụng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Tiny** | $D=256, L=8$ | **0.67M** | **2.25M** | MobileNet-v3 | Chạy thử nghiệm nhanh (smoke test) |
| **Tier 2: Standard (Khuyên dùng)** | $D=512, L=12$ | **3.54M** | **13.02M** | **ResNet-18 / ResMLP-12** | **Cấu hình tối ưu cho CIFAR-100 (Acc cao, train nhanh)** |
| **Tier 3: High-Capacity** | $D=768, L=16$ | **10.20M** | **38.59M** | **ResNet-50 / ResMLP-24** | Đua Top-1 Accuracy cao nhất |

---

## 2. Bảng So Sánh Chi Tiết Kiến Trúc Cấu Hình Standard (Tier 2)

| Thành phần / Tham số | ResMLP-S12 (Baseline) | ConvMixer-512/12 | Res-ConvMixer-512/12 (Hybrid) |
| :--- | :---: | :---: | :---: |
| **Kích thước ảnh đầu vào (`image_size`)** | $32 \times 32 \times 3$ | $32 \times 32 \times 3$ | $32 \times 32 \times 3$ |
| **Kích thước Patch (`patch_size`)** | $4 \times 4$ | **$2 \times 2$** (bắt chi tiết mịn) | **$2 \times 2$** |
| **Số lượng Patch ($S = (H/p) \times (W/p)$)** | $8 \times 8 = 64$ | $16 \times 16 = 256$ | $16 \times 16 = 256$ |
| **Kích thước Vector ẩn (`dim`)** | 384 | **512** | **512** |
| **Số tầng kiến trúc (`depth`)** | 12 | **12** | **12** |
| **Cơ chế trộn không gian (Spatial Mixing)** | Linear $S \times S$ (toàn cục, cứng nhắc) | **Depthwise Conv ($7 \times 7$)** | **Depthwise Conv ($7 \times 7$) + Affine2d** |
| **Cơ chế chuẩn hóa (Normalization)** | Affine ($y = \alpha \cdot x + \beta$) | BatchNorm2d | **Affine2d** (độc lập kích thước Batch) |
| **Cơ chế trộn kênh (Channel Mixing)** | Inverted MLP ($D \to 2D \to D$) | Pointwise Conv ($1 \times 1$) | **Inverted Bottleneck MLP ($D \to 2D \to D$)** |
| **Tổng số tham số (Total Parameters)** | ~15.4M | **~3.54M (Gọn gàng, hiệu quả)** | **~13.02M (Ngang ResNet-18)** |
| **Val-Accuracy dự kiến trên CIFAR-100** | ~63% – 66% | **~78% – 81%** 🏆 | **~79% – 82%** 🏆 |

---

## 3. Thiết Lập Tham Số Huấn Luyện (Training Hyperparameters Setup)

File cấu hình chính thức: `configs/cifar100_conv_mixer.yaml` & `configs/cifar100_res_conv_mixer.yaml`:

```yaml
experiment:
  tag: "cifar100_conv_mixer"
  base_dir: "runs"

training:
  epochs: 100                      # 100 epochs
  seed: 42
  deterministic: true
  lr: 0.001                        # Learning rate khởi đầu
  use_amp: true                    # Bật Mixed Precision FP16 trên T4 GPU
  grad_clip: 1.0
  label_smoothing: 0.1

optimizer:
  name: "adamw"
  weight_decay: 0.05

scheduler:
  name: "cosine"
  min_lr: 1.0e-6

data:
  dataset: "cifar100"
  batch_size: 128                  # Batch size 128
  num_workers: 4
  val_split: 0.1

augmentation:
  rand_augment: true
  random_erasing: 0.1
  color_jitter: 0.1
  use_mixup: true                  # alpha = 0.8
  cutmix_alpha: 1.0                # alpha = 1.0

model:
  name: "conv_mixer"               # hoặc "res_conv_mixer"
  image_size: 32
  patch_size: 2
  dim: 512                         # Cấu hình chuẩn Standard
  depth: 12                        # 12 tầng
  kernel_size: 7
  num_classes: 100
```
