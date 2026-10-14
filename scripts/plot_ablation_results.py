"""
Script plotting the Golden Ablation Suite Learning Curves
from the extracted 60-epoch training logs.
"""

import matplotlib.pyplot as plt
import numpy as np

# Epochs 1 to 60
epochs = list(range(1, 61))

# Extracted Validation Top-1 Accuracies (%) for the 4 ablation runs
val_acc_ablation_1 = [
    9.01, 16.47, 21.35, 23.62, 26.01, 28.06, 29.66, 31.65, 32.61, 34.93,
    35.32, 36.84, 37.78, 38.00, 38.84, 40.65, 41.04, 41.68, 42.01, 42.90,
    43.48, 44.87, 45.41, 45.12, 45.77, 46.48, 47.07, 46.80, 47.34, 47.87,
    48.26, 48.47, 48.87, 48.96, 49.12, 50.22, 50.48, 51.12, 51.21, 51.21,
    51.23, 51.90, 52.28, 52.40, 51.92, 52.14, 52.76, 52.76, 52.74, 53.19,
    53.22, 53.24, 53.75, 52.94, 53.13, 53.67, 53.57, 52.62, 53.39, 53.56
]

val_acc_ablation_2 = [
    9.88, 17.09, 22.75, 25.08, 29.55, 31.00, 32.61, 33.87, 35.30, 37.48,
    38.70, 39.04, 39.55, 41.42, 42.21, 43.84, 44.48, 45.36, 45.91, 46.87,
    47.03, 48.19, 48.60, 48.89, 48.67, 50.03, 50.46, 50.92, 51.50, 52.00,
    52.10, 52.60, 53.63, 53.67, 53.25, 53.64, 54.36, 54.58, 54.62, 55.00,
    55.65, 55.53, 55.44, 55.87, 55.75, 56.63, 56.27, 56.56, 56.68, 56.68,
    56.93, 56.43, 56.99, 56.85, 56.97, 56.66, 56.86, 57.22, 56.72, 56.97
]

val_acc_ablation_3 = [
    9.27, 15.59, 19.67, 25.11, 26.67, 30.70, 32.33, 33.31, 35.92, 37.50,
    38.17, 39.60, 39.22, 40.73, 41.91, 43.08, 43.61, 44.16, 45.46, 45.80,
    46.76, 46.92, 48.59, 49.18, 49.93, 49.45, 49.93, 50.14, 51.05, 51.32,
    52.19, 51.62, 51.70, 52.25, 52.85, 53.41, 54.12, 53.61, 54.17, 54.12,
    54.84, 54.49, 54.69, 55.45, 55.34, 55.85, 56.16, 55.96, 56.28, 55.65,
    56.56, 56.07, 55.92, 55.95, 56.59, 56.70, 56.66, 56.56, 56.96, 56.84
]

val_acc_ablation_4 = [
    10.83, 17.67, 22.51, 25.02, 28.24, 30.73, 32.63, 34.11, 35.37, 36.94,
    38.54, 39.29, 38.92, 40.23, 42.02, 42.92, 44.02, 44.10, 44.71, 45.44,
    46.18, 46.97, 47.72, 47.62, 48.12, 48.85, 49.55, 49.96, 50.30, 50.31,
    51.30, 51.44, 52.50, 52.68, 52.63, 52.88, 53.64, 54.17, 53.89, 53.84,
    54.86, 54.69, 55.27, 55.51, 55.29, 55.62, 55.99, 56.17, 56.18, 56.26,
    56.29, 56.28, 56.70, 56.56, 56.94, 56.40, 56.53, 56.75, 56.55, 56.49
]

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(epochs, val_acc_ablation_1, label="Ablation 1: Pure Linear (Baseline, 53.75%)", color="#d9534f", linestyle="--", linewidth=2.0)
plt.plot(epochs, val_acc_ablation_2, label="Ablation 2: DW 3x3 Only (57.22%)", color="#0275d8", linewidth=2.2)
plt.plot(epochs, val_acc_ablation_3, label="Ablation 3: Full RepTokenMix (1x1+3x3, 56.96%)", color="#5cb85c", linewidth=2.2)
plt.plot(epochs, val_acc_ablation_4, label="Ablation 4: Non-Overlapping Stem (56.94%)", color="#f0ad4e", linestyle=":", linewidth=2.0)

plt.title("Ablation Study on Tiny-ImageNet-200 (60 Epochs)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Training Epoch", fontsize=12, labelpad=8)
plt.ylabel("Validation Top-1 Accuracy (%)", fontsize=12, labelpad=8)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10, loc="lower right", framealpha=0.95)
plt.tight_layout()

output_path = "ablation_learning_curves.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully to {output_path}")
