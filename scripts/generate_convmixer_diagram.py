import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, axes = plt.subplots(1, 3, figsize=(18, 11), dpi=200)
fig.patch.set_facecolor('#F8FAFC')

# Color palette
RESMLP_COLOR = '#3B82F6'    # Blue
CONV_COLOR = '#10B981'      # Green
HYBRID_COLOR = '#8B5CF6'    # Purple
BG_CARD = '#FFFFFF'
BORDER_COLOR = '#CBD5E1'
TEXT_MAIN = '#0F172A'
TEXT_MUTED = '#64748B'

def draw_block_card(ax, title, subtitle, color):
    ax.set_facecolor(BG_CARD)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Outer frame
    card = patches.FancyBboxPatch((0.4, 0.4), 9.2, 15.2, boxstyle="round,pad=0.2",
                                  facecolor=BG_CARD, edgecolor=color, linewidth=2.5)
    ax.add_patch(card)
    
    # Header badge
    badge = patches.FancyBboxPatch((0.8, 14.2), 8.4, 1.0, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='none')
    ax.add_patch(badge)
    ax.text(5.0, 14.7, title, color='#FFFFFF', fontsize=13, fontweight='bold', ha='center', va='center')
    ax.text(5.0, 13.9, subtitle, color=TEXT_MUTED, fontsize=9.5, fontstyle='italic', ha='center')

def draw_box(ax, y, text, detail, bg='#F1F5F9', border='#94A3B8', height=0.9, width=7.2):
    x = (10 - width) / 2
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                                 facecolor=bg, edgecolor=border, linewidth=1.2)
    ax.add_patch(box)
    ax.text(5.0, y + height*0.62, text, color=TEXT_MAIN, fontsize=10, fontweight='bold', ha='center')
    if detail:
        ax.text(5.0, y + height*0.25, detail, color=TEXT_MUTED, fontsize=8.2, ha='center')

def draw_arrow(ax, y_start, y_end):
    ax.annotate('', xy=(5.0, y_end), xytext=(5.0, y_start),
                arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.8))

# ==================== 1. ResMLP ====================
ax1 = axes[0]
draw_block_card(ax1, "A. ResMLP Block", "Touvron et al. (NeurIPS 2021)", RESMLP_COLOR)

# Input
draw_box(ax1, 12.5, "Input Patches", "X ∈ ℝ^(B × S × C)   (S=64, C=128)", '#EFF6FF', RESMLP_COLOR)
draw_arrow(ax1, 12.5, 11.6)

# Cross-patch (Spatial)
draw_box(ax1, 10.7, "AffineTransform #1", "y = α ⊙ X + β", '#F8FAFC', '#94A3B8')
draw_arrow(ax1, 10.7, 9.8)

draw_box(ax1, 8.9, "Cross-Patch Linear (Transpose)", "W_spatial ∈ ℝ^(S × S)  [Global Mixing]", '#DBEAFE', RESMLP_COLOR, height=0.95)
draw_arrow(ax1, 8.9, 8.0)

draw_box(ax1, 7.1, "AffineTransform #2 & Residual", "X = X + Affine(Linear(Affine(X)^T)^T)", '#F8FAFC', '#94A3B8')
draw_arrow(ax1, 7.1, 6.2)

# Cross-channel (MLP)
draw_box(ax1, 5.3, "AffineTransform #3", "Pre-norm channel scale & shift", '#F8FAFC', '#94A3B8')
draw_arrow(ax1, 5.3, 4.4)

draw_box(ax1, 3.4, "Channel MLP (Expansion 2x/4x)", "Linear(C, 2C) → GELU → Linear(2C, C)", '#DBEAFE', RESMLP_COLOR, height=1.0)
draw_arrow(ax1, 3.4, 2.5)

draw_box(ax1, 1.6, "Affine #4 & Residual Add", "Output: X ∈ ℝ^(B × S × C)", '#EFF6FF', RESMLP_COLOR)

# Badge Note
ax1.text(5.0, 0.8, "⚠ Limitation: S×S matrix has no 2D inductive bias", color='#DC2626', fontsize=8, ha='center', fontweight='bold')


# ==================== 2. ConvMixer ====================
ax2 = axes[1]
draw_block_card(ax2, "B. ConvMixer Block", "Trockman & Kolter (ICLR/TMLR 2022)", CONV_COLOR)

# Input
draw_box(ax2, 12.5, "Patch Embedding", "Conv2d(p=2, stride=2) + GELU + BN", '#ECFDF5', CONV_COLOR)
draw_arrow(ax2, 12.5, 11.6)

# Spatial Depthwise
draw_box(ax2, 10.6, "Depthwise Conv2d (k×k)", "groups = D (e.g. k=7×7, D=256)", '#D1FAE5', CONV_COLOR, height=1.0)
draw_arrow(ax2, 10.6, 9.7)

draw_box(ax2, 8.8, "GELU Activation + BatchNorm", "Local receptive field per channel", '#F8FAFC', '#94A3B8')
draw_arrow(ax2, 8.8, 7.9)

draw_box(ax2, 7.0, "Residual Addition (Spatial)", "X = X + BN(GELU(Depthwise(X)))", '#ECFDF5', CONV_COLOR)
draw_arrow(ax2, 7.0, 6.1)

# Channel Pointwise
draw_box(ax2, 5.1, "Pointwise Conv2d (1×1)", "Conv2d(D → D, kernel=1×1)", '#D1FAE5', CONV_COLOR)
draw_arrow(ax2, 5.1, 4.2)

draw_box(ax2, 3.3, "GELU + BatchNorm", "Cross-channel feature fusion", '#F8FAFC', '#94A3B8')
draw_arrow(ax2, 3.3, 2.4)

draw_box(ax2, 1.5, "Output Feature Map", "X ∈ ℝ^(B × D × H/p × W/p)", '#ECFDF5', CONV_COLOR)

# Badge Note
ax2.text(5.0, 0.8, "✔ Strength: Translation equivariance + fast depthwise", color='#059669', fontsize=8, ha='center', fontweight='bold')


# ==================== 3. Res-ConvMixer (Hybrid) ====================
ax3 = axes[2]
draw_block_card(ax3, "C. Res-ConvMixer (Hybrid)", "Proposed Principled Combination", HYBRID_COLOR)

# Input
draw_box(ax3, 12.5, "Input Patches / Affine2d", "X ∈ ℝ^(B × D × H/p × W/p)", '#F5F3FF', HYBRID_COLOR)
draw_arrow(ax3, 12.5, 11.6)

# Depthwise Spatial + Affine
draw_box(ax3, 10.5, "Depthwise Conv2d (k=7) + GELU", "groups = D (Spatial Receptive Field)", '#EDE9FE', HYBRID_COLOR, height=1.0)
draw_arrow(ax3, 10.5, 9.6)

draw_box(ax3, 8.7, "ResMLP Affine2d Transform", "Affine(Depthwise(Affine(X)))", '#F8FAFC', '#94A3B8')
draw_arrow(ax3, 8.7, 7.8)

draw_box(ax3, 6.9, "Residual Addition (Spatial)", "X = X + Affine_out(GELU(DW(Affine_in(X))))", '#F5F3FF', HYBRID_COLOR)
draw_arrow(ax3, 6.9, 6.0)

# ResMLP Channel MLP (Pointwise 1x1 with expansion)
draw_box(ax3, 4.9, "Channel MLP (Expansion 2x)", "Conv2d(1×1, D→2D) → GELU → Conv2d(1×1, 2D→D)", '#EDE9FE', HYBRID_COLOR, height=1.1)
draw_arrow(ax3, 4.9, 3.8)

draw_box(ax3, 2.7, "Dual Affine2d + Residual Add", "Channel scale & shift (batch-size invariant)", '#F8FAFC', '#94A3B8')
draw_arrow(ax3, 2.7, 1.8)

draw_box(ax3, 0.9, "Output Feature Map", "X ∈ ℝ^(B × D × H/p × W/p)", '#F5F3FF', HYBRID_COLOR)

plt.suptitle("Architecture Comparison: ResMLP vs ConvMixer vs Proposed Res-ConvMixer Hybrid",
             fontsize=16, fontweight='bold', color=TEXT_MAIN, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

out_path = Path("reports/convmixer_resmlp_architecture.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Diagram saved successfully to: {out_path}")
