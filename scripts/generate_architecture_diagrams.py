"""
Generate publication-quality diagrams:
1. Recreated Figure 1: ResMLP Architecture (from Touvron et al., 2021)
2. Upgraded RepMLP Architecture: Global & Local Perceptron with Structural Re-parameterization (Ding et al., 2022)
3. Side-by-Side Comparison: ResMLP vs RepMLP Evolution

Usage:
    .venv/bin/python scripts/generate_architecture_diagrams.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="black", textcolor="black", fontsize=9, fontweight="bold", boxstyle="round,pad=0.2", lw=1.2):
    """Draws a styled rounded rectangle with centered text."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        mutation_scale=1.0,
        zorder=3
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=textcolor, zorder=4
        )
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=1.4, arrowstyle="-|>", mutation_scale=12):
    """Draws a clean directional connecting arrow."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
            shrinkA=0, shrinkB=0
        ),
        zorder=2
    )


def create_resmlp_figure1():
    """
    Recreates Figure 1 of Touvron et al. (NeurIPS 2021).
    """
    fig, ax = plt.subplots(figsize=(16, 5.8), dpi=300)
    ax.set_xlim(-0.5, 24.0)
    ax.set_ylim(-1.5, 7.2)
    ax.axis("off")

    # Colors
    c_affine = "#D6EAF8"    # Soft blue
    c_linear = "#FCF3CF"    # Soft yellow
    c_gelu = "#FCF3CF"      # Soft yellow
    c_pool = "#E8DAEF"      # Soft purple
    c_matrix = "#D4EFDF"    # Soft green

    # 1. Image patches mock (3x3 grid)
    px, py = 0.2, 2.5
    for row in range(3):
        for col in range(3):
            cell_color = "#E59866" if (row + col) % 2 == 0 else "#EDBB99"
            rect = patches.Rectangle((px + col * 0.55, py + (2 - row) * 0.55), 0.5, 0.5, facecolor=cell_color, edgecolor="white", lw=1.5, zorder=3)
            ax.add_patch(rect)
    ax.text(px + 0.8, py - 0.45, "Image patches", ha="center", va="center", fontsize=9.5, fontweight="bold")

    # Arrow to Linear Projection
    draw_arrow(ax, px + 1.7, py + 0.8, 2.5, py + 0.8)

    # Linear Projection
    draw_box(ax, 2.5, py + 0.25, 0.95, 1.1, "Linear", c_linear, fontsize=9)

    # Arrow to Patches representation
    draw_arrow(ax, 3.45, py + 0.8, 4.15, py + 0.8)

    # Matrix representation 1: channels x patches
    mx1, my1 = 4.15, py + 0.15
    for r in range(4):
        rect = patches.Rectangle((mx1, my1 + r * 0.32), 1.0, 0.26, facecolor=c_matrix, edgecolor="#27AE60", lw=1.0, zorder=3)
        ax.add_patch(rect)
    ax.text(mx1 + 0.5, my1 - 0.35, "channels", ha="center", va="center", fontsize=8, color="#555")
    ax.text(mx1 + 1.25, my1 + 0.65, "patches", ha="center", va="center", fontsize=8, color="#555", rotation=-90)

    # Arrow into ResMLP Layer
    draw_arrow(ax, 5.4, py + 0.8, 6.1, py + 0.8)

    # -------------------------------------------------------------
    # Outer Dotted Box: ResMLP Layer (xB)
    # -------------------------------------------------------------
    layer_box = patches.FancyBboxPatch(
        (6.1, 0.9), 14.1, 5.8,
        boxstyle="round,pad=0.15",
        facecolor="none",
        edgecolor="#2C3E50",
        linestyle="--",
        linewidth=1.8,
        zorder=1
    )
    ax.add_patch(layer_box)
    ax.text(13.15, 6.35, r"$\mathbf{ResMLP\ Layer\ (\times B)}$", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#2C3E50")

    # Sublayer 1: Cross-patch sublayer container
    box_cp = patches.FancyBboxPatch(
        (6.3, 1.1), 6.6, 4.8,
        boxstyle="round,pad=0.1",
        facecolor="#F4F6F6",
        edgecolor="#7F8C8D",
        linewidth=1.2,
        zorder=1
    )
    ax.add_patch(box_cp)
    ax.text(9.6, 5.5, "Cross-patch sublayer", ha="center", va="center", fontsize=10.5, fontweight="bold", fontstyle="italic", color="#34495E")

    # Sublayer 2: Cross-channel sublayer container
    box_cc = patches.FancyBboxPatch(
        (13.3, 1.1), 6.7, 4.8,
        boxstyle="round,pad=0.1",
        facecolor="#F4F6F6",
        edgecolor="#7F8C8D",
        linewidth=1.2,
        zorder=1
    )
    ax.add_patch(box_cc)
    ax.text(16.65, 5.5, "Cross-channel sublayer", ha="center", va="center", fontsize=10.5, fontweight="bold", fontstyle="italic", color="#34495E")

    # ----------------- Inside Cross-patch Sublayer -----------------
    y_flow = 3.3
    # Affine 1
    draw_box(ax, 6.5, y_flow - 0.45, 0.7, 0.9, "Affine", c_affine, fontsize=8)
    draw_arrow(ax, 7.2, y_flow, 7.5, y_flow)

    # Transpose 1 [T]
    draw_box(ax, 7.5, y_flow - 0.35, 0.5, 0.7, r"$\curvearrowleft T$", "#E5E7E9", fontsize=8)
    draw_arrow(ax, 8.0, y_flow, 8.25, y_flow)

    # Transposed Matrix: patches x channels
    mx2, my2 = 8.25, y_flow - 0.55
    for c in range(4):
        rect = patches.Rectangle((mx2 + c * 0.22, my2), 0.18, 1.1, facecolor=c_matrix, edgecolor="#27AE60", lw=0.9, zorder=3)
        ax.add_patch(rect)
    ax.text(mx2 + 0.45, my2 - 0.3, "patches", ha="center", va="center", fontsize=7.5, color="#555")
    ax.text(mx2 + 1.05, my2 + 0.55, "channel", ha="center", va="center", fontsize=7.5, color="#555", rotation=-90)

    draw_arrow(ax, 9.25, y_flow, 9.55, y_flow)

    # Linear
    draw_box(ax, 9.55, y_flow - 0.45, 0.75, 0.9, "Linear", c_linear, fontsize=8)
    draw_arrow(ax, 10.3, y_flow, 10.55, y_flow)

    # Transpose 2 [T]
    draw_box(ax, 10.55, y_flow - 0.35, 0.5, 0.7, r"$\curvearrowleft T$", "#E5E7E9", fontsize=8)
    draw_arrow(ax, 11.05, y_flow, 11.3, y_flow)

    # Restored Matrix: channels x patches
    mx3, my3 = 11.3, y_flow - 0.55
    for r in range(4):
        rect = patches.Rectangle((mx3, my3 + r * 0.28), 0.75, 0.22, facecolor=c_matrix, edgecolor="#27AE60", lw=0.9, zorder=3)
        ax.add_patch(rect)
    ax.text(mx3 + 0.37, my3 - 0.3, "channels", ha="center", va="center", fontsize=7.5, color="#555")
    ax.text(mx3 + 0.95, my3 + 0.55, "patches", ha="center", va="center", fontsize=7.5, color="#555", rotation=-90)

    draw_arrow(ax, 12.2, y_flow, 12.4, y_flow)

    # Affine 2
    draw_box(ax, 12.4, y_flow - 0.45, 0.7, 0.9, "Affine", c_affine, fontsize=8)

    # Skip-connection 1 (bottom bypass)
    ax.plot([6.2, 6.2, 13.0, 13.0], [y_flow, 1.5, 1.5, y_flow], color="black", lw=1.4, zorder=2)
    draw_arrow(ax, 13.0, 1.5, 13.0, y_flow - 0.05)
    ax.text(9.6, 1.7, "Skip-connection", ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#2C3E50")

    # Connection between sublayers
    draw_arrow(ax, 13.1, y_flow, 13.5, y_flow)

    # ----------------- Inside Cross-channel Sublayer -----------------
    # Affine 1
    draw_box(ax, 13.5, y_flow - 0.45, 0.7, 0.9, "Affine", c_affine, fontsize=8)
    draw_arrow(ax, 14.2, y_flow, 14.5, y_flow)

    # Linear 1
    draw_box(ax, 14.5, y_flow - 0.45, 0.75, 0.9, "Linear", c_linear, fontsize=8)
    draw_arrow(ax, 15.25, y_flow, 15.5, y_flow)

    # GeLU
    draw_box(ax, 15.5, y_flow - 0.45, 0.7, 0.9, "GeLU", c_gelu, fontsize=8)
    draw_arrow(ax, 16.2, y_flow, 16.45, y_flow)

    # Linear 2
    draw_box(ax, 16.45, y_flow - 0.45, 0.75, 0.9, "Linear", c_linear, fontsize=8)
    draw_arrow(ax, 17.2, y_flow, 17.45, y_flow)

    # Affine 2
    draw_box(ax, 17.45, y_flow - 0.45, 0.7, 0.9, "Affine", c_affine, fontsize=8)

    # Skip-connection 2 (bottom bypass)
    ax.plot([13.2, 13.2, 18.3, 18.3], [y_flow, 1.5, 1.5, y_flow], color="black", lw=1.4, zorder=2)
    draw_arrow(ax, 18.3, 1.5, 18.3, y_flow - 0.05)
    ax.text(15.75, 1.7, "Skip-connection", ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#2C3E50")

    # Arrow out of ResMLP layer
    draw_arrow(ax, 18.35, y_flow, 20.6, y_flow)

    # ----------------- Head: Pooling & Linear -----------------
    draw_box(ax, 20.6, y_flow - 0.5, 0.9, 1.0, "Pooling", c_pool, fontsize=8.5)
    draw_arrow(ax, 21.5, y_flow, 21.9, y_flow)
    draw_box(ax, 21.9, y_flow - 0.5, 0.9, 1.0, "Linear", c_linear, fontsize=8.5)
    draw_arrow(ax, 22.8, y_flow, 23.6, y_flow)

    # Caption at bottom
    caption = (
        r"$\mathbf{Figure\ 1:\ The\ ResMLP\ architecture.}$ After linearly projecting the image patches into high dimensional"
        "\n"
        r"embeddings, ResMLP sequentially processes them with (1) a cross-patch linear sublayer; (2) a cross-channel"
        "\n"
        r"two-layer MLP. The MLP is the same as the FCN sublayer of a Transformer. Each sublayer has a residual"
        "\n"
        r"connection and two Affine element-wise transformations."
    )
    ax.text(12.0, -0.65, caption, ha="center", va="center", fontsize=9.5, family="serif")

    plt.tight_layout()
    out_path = REPORTS_DIR / "resmlp_figure1_recreated.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[1/3] Successfully recreated Figure 1 at: {out_path}")


def create_repmlp_upgraded_figure():
    """
    Generates the Upgraded Architecture Diagram:
    RepMLP with Global Perceptron, Local Perceptron (Locality Injection),
    and Structural Re-parameterization.
    """
    fig, ax = plt.subplots(figsize=(16, 9.0), dpi=300)
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-1.6, 10.4)
    ax.axis("off")

    # Modern color palette
    c_gp = "#D5F5E3"       # Soft green (Global Perceptron)
    c_fc = "#FCF3CF"       # Soft yellow (Linear / FC)
    c_conv1 = "#FADBD8"    # Soft rose/red (1x1 Conv)
    c_conv3 = "#EDBB99"    # Soft orange (3x3 Conv)
    c_bn = "#D6EAF8"       # Soft blue (BatchNorm)
    c_fuse = "#E8DAEF"     # Soft purple (Structural Fusion)
    c_ffn = "#FEF9E7"      # Very soft yellow (FFN)
    c_pool = "#E8DAEF"     # Soft purple (Pooling)

    # Title Banner
    ax.text(
        12.0, 10.0,
        "RepMLP Architecture: Re-parameterizing Convolutions into Vision MLPs",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1B2631"
    )
    ax.text(
        12.0, 9.5,
        "Key Upgrade: Global Perceptron (SE Attention) + Local Perceptron (Parallel Convs Fused into FC at Inference)",
        ha="center", va="center", fontsize=11, fontstyle="italic", color="#566573"
    )

    # 1. Input Image / Patches
    px, py = 0.5, 4.2
    for r in range(4):
        for c in range(4):
            color = "#5DADE2" if (r + c) % 2 == 0 else "#85C1E9"
            ax.add_patch(patches.Rectangle((px + c * 0.35, py + (3 - r) * 0.35), 0.3, 0.3, facecolor=color, edgecolor="white", lw=1.2, zorder=3))
    ax.text(px + 0.7, py - 0.45, "Input Patches\n$(B, C, H, W)$", ha="center", va="center", fontsize=9, fontweight="bold")

    draw_arrow(ax, px + 1.6, py + 0.7, 2.7, py + 0.7)

    # -------------------------------------------------------------------------
    # CONTAINER: Global Perceptron (Top Branch)
    # -------------------------------------------------------------------------
    gp_box = patches.FancyBboxPatch(
        (2.8, 6.9), 7.6, 2.1,
        boxstyle="round,pad=0.15",
        facecolor="#EAFAF1",
        edgecolor="#27AE60",
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(gp_box)
    ax.text(6.6, 8.7, r"$\mathbf{Global\ Perceptron\ (Channel\ Context\ Modulation)}$", ha="center", va="center", fontsize=10, fontweight="bold", color="#196F3D")

    draw_box(ax, 3.1, 7.2, 1.1, 0.8, "GAP\n$(1\\times 1)$", c_gp, fontsize=8)
    draw_arrow(ax, 4.2, 7.6, 4.5, 7.6)
    draw_box(ax, 4.5, 7.2, 1.2, 0.8, "Conv $1\\times 1$\n(Reduce)", c_fc, fontsize=8)
    draw_arrow(ax, 5.7, 7.6, 6.0, 7.6)
    draw_box(ax, 6.0, 7.2, 0.8, 0.8, "ReLU", c_fc, fontsize=8)
    draw_arrow(ax, 6.8, 7.6, 7.1, 7.6)
    draw_box(ax, 7.1, 7.2, 1.2, 0.8, "Conv $1\\times 1$\n(Expand)", c_fc, fontsize=8)
    draw_arrow(ax, 8.3, 7.6, 8.6, 7.6)
    draw_box(ax, 8.6, 7.2, 1.1, 0.8, "Sigmoid\nVector $g$", c_gp, fontsize=8)

    # -------------------------------------------------------------------------
    # CONTAINER: RepMLP Block with Local Perceptron
    # -------------------------------------------------------------------------
    main_box = patches.FancyBboxPatch(
        (2.8, 0.8), 13.0, 5.7,
        boxstyle="round,pad=0.2",
        facecolor="#FDFEFE",
        edgecolor="#2980B9",
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(main_box)
    ax.text(9.3, 6.15, r"$\mathbf{RepMLP\ Block\ with\ Locality\ Injection\ (Local\ Perceptron)}$", ha="center", va="center", fontsize=12, fontweight="bold", color="#1B4F72")

    # Partition / Reshape
    draw_box(ax, 3.2, 3.4, 1.4, 1.0, "Partition\n$S\\times S$ grid", "#EAECEE", fontsize=8.5)
    draw_arrow(ax, 2.7, py + 0.7, 3.2, py + 0.7)

    # Split to 3 branches during training
    bx, by = 4.6, 3.9
    draw_arrow(ax, bx, by, 5.4, 4.9)   # to Conv 1x1
    draw_arrow(ax, bx, by, 5.4, 3.9)   # to FC3 (Spatial Linear)
    draw_arrow(ax, bx, by, 5.4, 2.9)   # to Conv 3x3

    # Branch 1: Conv 1x1 + BN
    draw_box(ax, 5.4, 4.5, 1.5, 0.8, "Conv $1\\times 1$\n(Pointwise)", c_conv1, fontsize=8)
    draw_arrow(ax, 6.9, 4.9, 7.2, 4.9)
    draw_box(ax, 7.2, 4.5, 1.1, 0.8, "BN", c_bn, fontsize=8)

    # Branch 2: FC3 (Main Spatial Linear)
    draw_box(ax, 5.4, 3.5, 2.9, 0.8, r"$\mathbf{FC3\ (Spatial\ Linear)}$", c_fc, fontsize=9.5, edgecolor="#B7950B", lw=1.8)

    # Branch 3: Conv 3x3 + BN
    draw_box(ax, 5.4, 2.5, 1.5, 0.8, "Conv $3\\times 3$\n(Local Context)", c_conv3, fontsize=8)
    draw_arrow(ax, 6.9, 2.9, 7.2, 2.9)
    draw_box(ax, 7.2, 2.5, 1.1, 0.8, "BN", c_bn, fontsize=8)

    # Label: Training Mode
    ax.text(6.8, 5.65, "Training Mode: 3 Parallel Branches", ha="center", va="center", fontsize=9, fontweight="bold", color="#C0392B")

    # Merge branches (+)
    draw_arrow(ax, 8.3, 4.9, 9.0, 4.0)
    draw_arrow(ax, 8.3, 3.9, 9.0, 3.9)
    draw_arrow(ax, 8.3, 2.9, 9.0, 3.8)

    # Sum circle (+)
    circle_sum = patches.Circle((9.2, 3.9), 0.25, facecolor="#F2F4F4", edgecolor="black", lw=1.5, zorder=3)
    ax.add_patch(circle_sum)
    ax.text(9.2, 3.9, "+", ha="center", va="center", fontsize=13, fontweight="bold")

    # Structural Re-parameterization Callout Box
    fuse_callout = patches.FancyBboxPatch(
        (5.0, 1.05), 5.2, 0.95,
        boxstyle="round,pad=0.1",
        facecolor="#F4ECF7",
        edgecolor="#8E44AD",
        linestyle="--",
        linewidth=1.4,
        zorder=2
    )
    ax.add_patch(fuse_callout)
    ax.text(
        7.6, 1.52,
        r"$\mathbf{Structural\ Re-parameterization\ (Inference):}$" "\n"
        r"$\mathbf{W_{fused} = W_{FC3} + W_{1\times 1} + W_{3\times 3}}\ (\mathbf{0\ Extra\ FLOPs})$",
        ha="center", va="center", fontsize=8.2, color="#5B2C6F"
    )

    # Arrow from Sum to Modulation
    draw_arrow(ax, 9.45, 3.9, 10.2, 3.9)

    # Modulation with Global Vector (Multiply)
    circle_mult = patches.Circle((10.4, 3.9), 0.28, facecolor="#E8F8F5", edgecolor="#16A085", lw=1.6, zorder=3)
    ax.add_patch(circle_mult)
    ax.text(10.4, 3.9, r"$\otimes$", ha="center", va="center", fontsize=14, fontweight="bold", color="#117A65")

    # Connecting arrow from Global Perceptron to Multiplication
    ax.plot([9.7, 10.4, 10.4], [7.6, 7.6, 4.2], color="#16A085", lw=1.6, linestyle=":", zorder=2)
    draw_arrow(ax, 10.4, 4.4, 10.4, 4.2, color="#16A085")

    # Reshape / Unpartition
    draw_arrow(ax, 10.7, 3.9, 11.2, 3.9)
    draw_box(ax, 11.2, 3.4, 1.5, 1.0, "Unpartition\n$(B, C, H, W)$", "#EAECEE", fontsize=8.5)

    # Skip-connection inside RepMLP block
    ax.plot([2.5, 2.5, 13.5, 13.5], [py + 0.7, 0.4, 0.4, 3.9], color="#2C3E50", lw=1.4, zorder=2)
    draw_arrow(ax, 13.5, 0.4, 13.5, 3.65)
    circle_res = patches.Circle((13.5, 3.9), 0.25, facecolor="#F2F4F4", edgecolor="black", lw=1.5, zorder=3)
    ax.add_patch(circle_res)
    ax.text(13.5, 3.9, "+", ha="center", va="center", fontsize=13, fontweight="bold")
    draw_arrow(ax, 12.7, 3.9, 13.25, 3.9)
    ax.text(8.0, 0.55, "Block Skip-connection", ha="center", va="center", fontsize=8.5, fontstyle="italic")

    # Arrow to FFN Block
    draw_arrow(ax, 13.75, 3.9, 16.3, 3.9)

    # -------------------------------------------------------------------------
    # CONTAINER: Feed-Forward Network (FFN)
    # -------------------------------------------------------------------------
    ffn_box = patches.FancyBboxPatch(
        (16.3, 1.6), 5.4, 4.6,
        boxstyle="round,pad=0.15",
        facecolor="#FEFDE8",
        edgecolor="#D4AC0D",
        linewidth=1.6,
        zorder=1
    )
    ax.add_patch(ffn_box)
    ax.text(19.0, 5.8, r"$\mathbf{Feed-Forward\ Network\ (FFN)}$", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#7D6608")

    draw_box(ax, 16.6, 3.4, 1.15, 0.95, "Conv $1\\times 1$\n(Expand)", c_fc, fontsize=8)
    draw_arrow(ax, 17.75, 3.85, 18.05, 3.85)
    draw_box(ax, 18.05, 3.4, 0.85, 0.95, "GELU", "#FCF3CF", fontsize=8)
    draw_arrow(ax, 18.9, 3.85, 19.2, 3.85)
    draw_box(ax, 19.2, 3.4, 1.15, 0.95, "Conv $1\\times 1$\n(Project)", c_fc, fontsize=8)

    # FFN Skip-connection
    ax.plot([15.9, 15.9, 21.0, 21.0], [3.9, 2.2, 2.2, 3.9], color="#2C3E50", lw=1.3, zorder=2)
    draw_arrow(ax, 21.0, 2.2, 21.0, 3.65)
    circle_ffn = patches.Circle((21.0, 3.9), 0.25, facecolor="#F2F4F4", edgecolor="black", lw=1.5, zorder=3)
    ax.add_patch(circle_ffn)
    ax.text(21.0, 3.9, "+", ha="center", va="center", fontsize=13, fontweight="bold")
    draw_arrow(ax, 20.35, 3.85, 20.75, 3.85)

    # Arrow to Head
    draw_arrow(ax, 21.25, 3.9, 22.1, 3.9)

    # -------------------------------------------------------------------------
    # Classifier Head
    # -------------------------------------------------------------------------
    draw_box(ax, 22.1, 3.35, 1.0, 1.0, "Global\nPooling", c_pool, fontsize=8.5)
    draw_arrow(ax, 23.1, 3.85, 23.4, 3.85)
    draw_box(ax, 23.4, 3.35, 0.9, 1.0, "Linear\nHead", c_fc, fontsize=8.5)

    # Caption
    caption = (
        r"$\mathbf{Upgraded\ RepMLP\ Architecture:}$ Integrates (1) $\mathbf{Global\ Perceptron}$ for global channel squeeze-and-excitation;"
        "\n"
        r"(2) $\mathbf{Local\ Perceptron}$ with parallel $1\times 1$ and $3\times 3$ convolutions during training to inject local inductive bias;"
        "\n"
        r"(3) $\mathbf{Structural\ Re-parameterization}$ fusing all conv branches into FC3 at test time for 0 extra inference latency."
    )
    ax.text(12.0, -0.9, caption, ha="center", va="center", fontsize=10, family="serif")

    plt.tight_layout()
    out_path = REPORTS_DIR / "repmlp_architecture_upgraded.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[2/3] Successfully generated Upgraded RepMLP Diagram at: {out_path}")


def create_comparison_figure():
    """
    Creates a visual side-by-side comparison between:
    - Traditional ResMLP (Figure 1)
    - Upgraded Rep-ResMLP / RepMLP with Local Perceptron
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), dpi=300)

    for ax in axes:
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(-0.5, 5.5)
        ax.axis("off")

    # Colors
    c_blue = "#D4E6F1"
    c_yellow = "#FCF3CF"
    c_red = "#F5B7B1"
    c_green = "#D5F5E3"

    # ========================== TOP: RESMLP ==========================
    ax1 = axes[0]
    ax1.text(0.2, 5.0, "A. Standard ResMLP (Touvron et al., 2021) — Pure Matrix Mixing", fontsize=12, fontweight="bold", color="#1B4F72")

    # Input
    draw_box(ax1, 0.2, 2.0, 1.8, 1.0, "Image\nPatches", "#EDBB99", fontsize=9)
    draw_arrow(ax1, 2.0, 2.5, 2.6, 2.5)
    draw_box(ax1, 2.6, 2.0, 1.2, 1.0, "Linear\nProj", c_yellow, fontsize=8.5)
    draw_arrow(ax1, 3.8, 2.5, 4.4, 2.5)

    # Box ResMLP Layer
    box1 = patches.FancyBboxPatch((4.4, 0.4), 14.5, 4.3, boxstyle="round,pad=0.15", facecolor="#F8F9F9", edgecolor="#2C3E50", linestyle="--", lw=1.5)
    ax1.add_patch(box1)
    ax1.text(11.65, 4.4, "Standard ResMLP Layer (xB)", ha="center", va="center", fontsize=10.5, fontweight="bold")

    # Cross-patch
    draw_box(ax1, 4.8, 2.0, 1.0, 1.0, "Affine", c_blue, fontsize=8.5)
    draw_arrow(ax1, 5.8, 2.5, 6.2, 2.5)
    draw_box(ax1, 6.2, 1.8, 3.2, 1.4, "Cross-Patch Linear\n(No 2D local context)\n$S\\times S$ Flat Matrix", c_yellow, fontsize=9, edgecolor="#B7950B")
    draw_arrow(ax1, 9.4, 2.5, 9.8, 2.5)
    draw_box(ax1, 9.8, 2.0, 1.0, 1.0, "Affine", c_blue, fontsize=8.5)

    # Cross-channel
    draw_arrow(ax1, 10.8, 2.5, 11.4, 2.5)
    draw_box(ax1, 11.4, 2.0, 1.0, 1.0, "Affine", c_blue, fontsize=8.5)
    draw_arrow(ax1, 12.4, 2.5, 12.8, 2.5)
    draw_box(ax1, 12.8, 1.9, 1.8, 1.2, "Linear\n(Expansion)", c_yellow, fontsize=8.5)
    draw_arrow(ax1, 14.6, 2.5, 14.9, 2.5)
    draw_box(ax1, 14.9, 2.0, 0.9, 1.0, "GeLU", c_yellow, fontsize=8.5)
    draw_arrow(ax1, 15.8, 2.5, 16.1, 2.5)
    draw_box(ax1, 16.1, 1.9, 1.6, 1.2, "Linear\n(Project)", c_yellow, fontsize=8.5)
    draw_arrow(ax1, 17.7, 2.5, 18.0, 2.5)
    draw_box(ax1, 18.0, 2.0, 0.8, 1.0, "Affine", c_blue, fontsize=8)

    # Head
    draw_arrow(ax1, 18.9, 2.5, 19.8, 2.5)
    draw_box(ax1, 19.8, 2.0, 1.4, 1.0, "Average\nPooling", "#E8DAEF", fontsize=8.5)
    draw_arrow(ax1, 21.2, 2.5, 21.6, 2.5)
    draw_box(ax1, 21.6, 2.0, 1.4, 1.0, "Linear\nClassifier", c_yellow, fontsize=8.5)

    # ========================== BOTTOM: UPGRADED REPMLP ==========================
    ax2 = axes[1]
    ax2.text(0.2, 5.0, "B. Upgraded with Local Perceptron & Re-parameterization (Ding et al., 2022)", fontsize=12, fontweight="bold", color="#78281F")

    # Input
    draw_box(ax2, 0.2, 2.0, 1.8, 1.0, "Image\nPatches", "#EDBB99", fontsize=9)
    draw_arrow(ax2, 2.0, 2.5, 2.6, 2.5)
    draw_box(ax2, 2.6, 2.0, 1.2, 1.0, "Linear\nProj", c_yellow, fontsize=8.5)
    draw_arrow(ax2, 3.8, 2.5, 4.4, 2.5)

    # Box Upgraded Layer
    box2 = patches.FancyBboxPatch((4.4, 0.3), 14.5, 4.4, boxstyle="round,pad=0.15", facecolor="#FEF9E7", edgecolor="#B03A2E", linestyle="-", lw=1.8)
    ax2.add_patch(box2)
    ax2.text(11.65, 4.45, "Upgraded Block with Local & Global Perceptrons", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#78281F")

    # Global perceptron tag
    gp_tag = patches.FancyBboxPatch((4.8, 3.4), 4.2, 0.7, boxstyle="round,pad=0.1", facecolor=c_green, edgecolor="#27AE60", lw=1.2)
    ax2.add_patch(gp_tag)
    ax2.text(6.9, 3.75, r"$\mathbf{+ Global\ Perceptron\ (SE\ Attention)}$", ha="center", va="center", fontsize=8.5, color="#196F3D")

    # Local perceptron parallel branches
    draw_box(ax2, 4.8, 2.2, 1.7, 0.8, r"$\mathbf{Conv\ 1\times 1 + BN}$", c_red, fontsize=8)
    draw_box(ax2, 4.8, 1.2, 1.7, 0.8, r"$\mathbf{Conv\ 3\times 3 + BN}$", "#FAD7A0", fontsize=8)
    draw_box(ax2, 6.8, 1.6, 2.5, 1.5, r"$\mathbf{FC3\ (Spatial\ Linear)}$" "\n(Main Branch)", c_yellow, fontsize=9, edgecolor="#B7950B", lw=1.5)

    # Fusion label
    ax2.annotate(
        "FUSED AT INFERENCE\n(0 Extra FLOPs)",
        xy=(8.0, 1.6), xytext=(8.0, 0.6),
        ha="center", fontsize=7.5, fontweight="bold", color="#8E44AD",
        arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.2)
    )

    draw_arrow(ax2, 9.3, 2.35, 9.9, 2.35)
    draw_box(ax2, 9.9, 1.9, 0.9, 0.9, r"$\otimes\ g$", c_green, fontsize=10, fontweight="bold")

    # FFN
    draw_arrow(ax2, 10.8, 2.35, 11.4, 2.35)
    draw_box(ax2, 11.4, 1.7, 1.8, 1.3, "FFN\n(Conv 1x1)", c_yellow, fontsize=8.5)
    draw_arrow(ax2, 13.2, 2.35, 13.6, 2.35)
    draw_box(ax2, 13.6, 1.9, 0.9, 0.9, "GELU", c_yellow, fontsize=8.5)
    draw_arrow(ax2, 14.5, 2.35, 14.9, 2.35)
    draw_box(ax2, 14.9, 1.7, 1.8, 1.3, "FFN\n(Conv 1x1)", c_yellow, fontsize=8.5)

    # Output head
    draw_arrow(ax2, 18.9, 2.35, 19.8, 2.35)
    draw_box(ax2, 19.8, 1.85, 1.4, 1.0, "Average\nPooling", "#E8DAEF", fontsize=8.5)
    draw_arrow(ax2, 21.2, 2.35, 21.6, 2.35)
    draw_box(ax2, 21.6, 1.85, 1.4, 1.0, "Linear\nClassifier", c_yellow, fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "resmlp_to_repmlp_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3/3] Successfully generated Comparison Diagram at: {out_path}")


if __name__ == "__main__":
    create_resmlp_figure1()
    create_repmlp_upgraded_figure()
    create_comparison_figure()
    print("All architecture diagrams successfully generated!")
