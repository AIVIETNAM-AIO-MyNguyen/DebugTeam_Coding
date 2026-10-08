import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load ResMLP data
res_df = pd.read_csv('runs/resmlp.csv')

# 2. Load RepMLP data
rep_data = []
with open('runs/rep_mlp.log', 'r') as f:
    for line in f:
        m = re.search(
            r'Epoch \[(\d+)/100\] Train Loss: ([\d\.]+) \| Train Acc: ([\d\.]+)% \| Val Loss: ([\d\.]+) \| Val Acc: ([\d\.]+)%',
            line
        )
        if m:
            rep_data.append({
                'epoch': int(m.group(1)),
                'train_loss': float(m.group(2)),
                'train_acc': float(m.group(3)),
                'val_loss': float(m.group(4)),
                'val_acc': float(m.group(5)),
            })
rep_df = pd.DataFrame(rep_data)

# 3. Create publication-quality figure
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif'
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

# Colors
color_res = '#d62728'      # Crimson Red for ResMLP
color_res_train = '#9467bd'# Muted purple/gray for ResMLP train
color_rep = '#1b7837'      # Forest Green for RepMLP
color_rep_train = '#7fbc41'# Lighter green for RepMLP train

epochs = res_df['epoch']

# --- Left Subplot: Cross-Entropy Loss ---
ax1 = axes[0]
ax1.plot(epochs, res_df['train_loss'], label='ResMLP Train Loss', color='#7f7f7f', linestyle='--', linewidth=1.6, alpha=0.85)
ax1.plot(epochs, res_df['val_loss'], label='ResMLP Val Loss', color=color_res, linestyle='-', linewidth=2.2)
ax1.plot(rep_df['epoch'], rep_df['train_loss'], label='RepMLP Train Loss', color=color_rep_train, linestyle='--', linewidth=1.6, alpha=0.9)
ax1.plot(rep_df['epoch'], rep_df['val_loss'], label='RepMLP Val Loss', color=color_rep, linestyle='-', linewidth=2.2)

ax1.set_title('Cross-Entropy Loss (100 Epochs)', fontweight='bold', pad=10)
ax1.set_xlabel('Epoch', labelpad=6)
ax1.set_ylabel('Loss', labelpad=6)
ax1.set_xlim(-1, 103)
ax1.set_ylim(0.5, 4.8)
ax1.grid(True, linestyle=':', color='gray', alpha=0.4)
ax1.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')

# --- Right Subplot: Top-1 Validation Accuracy ---
ax2 = axes[1]
ax2.plot(epochs, res_df['val_top1'], label='ResMLP Val Top-1 (Best: 62.40%)', color=color_res, linestyle='-', linewidth=2.2)
ax2.plot(rep_df['epoch'], rep_df['val_acc'], label='RepMLP Val Top-1 (Best: 71.46%)', color=color_rep, linestyle='-', linewidth=2.2)

# Benchmark reference dashed lines
ax2.axhline(y=62.40, color=color_res, linestyle=':', linewidth=1.4, alpha=0.75)
ax2.axhline(y=71.46, color=color_rep, linestyle=':', linewidth=1.4, alpha=0.75)

ax2.set_title('Top-1 Validation Accuracy (CIFAR-100)', fontweight='bold', pad=10)
ax2.set_xlabel('Epoch', labelpad=6)
ax2.set_ylabel('Accuracy (%)', labelpad=6)
ax2.set_xlim(-1, 103)
ax2.set_ylim(0, 80)
ax2.grid(True, linestyle=':', color='gray', alpha=0.4)
ax2.legend(loc='lower right', framealpha=0.95, edgecolor='#cccccc')

plt.tight_layout()
output_path = 'reports/w02/figures/learning_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated {output_path}")
