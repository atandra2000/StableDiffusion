"""
Architecture & Pipeline Overview Chart — Stable Diffusion
Generates a 6-panel dark-themed figure since training has not yet begun.
Run: python assets/generate_plots.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ── Style ────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
ORANGE = "#f0883e"
RED    = "#f85149"
PURPLE = "#a371f7"
CYAN   = "#39d353"
GRAY   = "#8b949e"
WHITE  = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   WHITE,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "text.color":        WHITE,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    "Stable Diffusion from Scratch  —  Architecture & Pipeline Overview",
    fontsize=16, fontweight="bold", color=WHITE, y=0.98
)

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.45, wspace=0.35,
    left=0.06, right=0.97, top=0.93, bottom=0.06
)

# ════════════════════════════════════════════════════════════════════════════
# Panel 1 — Model Component Breakdown (Frozen vs Trainable)
# ════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])

components = ["VAE\nEncoder", "VAE\nDecoder", "CLIP\nText Enc.", "UNet\n(Trainable)"]
params_m   = [83, 49, 123, 860]
colors     = [RED, RED, ORANGE, GREEN]
bar_labels = ["Frozen (83M)", "Frozen (49M)", "Frozen (123M)", "Trainable (860M)"]

bars = ax1.barh(components, params_m, color=colors, height=0.55,
                edgecolor=BORDER, linewidth=0.8)

for bar, p in zip(bars, params_m):
    ax1.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
             f"{p}M", va="center", color=WHITE, fontsize=9, fontweight="bold")

ax1.set_xlim(0, 1050)
ax1.set_xlabel("Parameters (millions)", fontsize=9)
ax1.set_title("Model Components", fontsize=11, fontweight="bold", color=BLUE, pad=8)
ax1.grid(axis="x", alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)

frozen_patch    = mpatches.Patch(color=RED,    label="Frozen (VAE)")
frozen2_patch   = mpatches.Patch(color=ORANGE, label="Frozen (CLIP)")
trainable_patch = mpatches.Patch(color=GREEN,  label="Trainable (UNet)")
ax1.legend(handles=[frozen_patch, frozen2_patch, trainable_patch],
           loc="lower right", fontsize=8, framealpha=0.3,
           facecolor=PANEL, edgecolor=BORDER)

total = sum(params_m)
ax1.text(0.02, -0.18,
         f"Total: {total}M params  |  Trainable: 860M ({860/total*100:.0f}%)",
         transform=ax1.transAxes, color=GRAY, fontsize=8)


# ════════════════════════════════════════════════════════════════════════════
# Panel 2 — DDPM Beta Schedule
# ════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])

T = 1000
beta_start, beta_end = 0.00085, 0.012
betas = np.linspace(beta_start**0.5, beta_end**0.5, T) ** 2
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
t = np.arange(T)

ax2_twin = ax2.twinx()
ax2_twin.tick_params(colors=GRAY)
ax2_twin.spines[["top", "left", "bottom"]].set_visible(False)
ax2_twin.spines["right"].set_edgecolor(BORDER)

l1, = ax2.plot(t, betas,           color=ORANGE, lw=1.8, label="β(t)  — noise added")
l2, = ax2.plot(t, alphas_cumprod,  color=BLUE,   lw=2.0, label="ᾱ(t)  — signal retained")
l3, = ax2_twin.plot(t, np.sqrt(alphas_cumprod),         color=GREEN,  lw=1.6, linestyle="--", label="√ᾱ(t)")
l4, = ax2_twin.plot(t, np.sqrt(1 - alphas_cumprod),     color=PURPLE, lw=1.6, linestyle="--", label="√(1−ᾱ(t))")

ax2.set_xlabel("Timestep t", fontsize=9)
ax2.set_ylabel("β(t) / ᾱ(t)", fontsize=9, color=GRAY)
ax2_twin.set_ylabel("Scaling factors", fontsize=9, color=GRAY)
ax2.set_title("DDPM Scaled-Linear Noise Schedule", fontsize=11,
              fontweight="bold", color=BLUE, pad=8)
ax2.grid(alpha=0.3)
ax2.spines[["top"]].set_visible(False)
ax2.spines[["left", "bottom", "right"]].set_edgecolor(BORDER)

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc="center right", fontsize=8,
           framealpha=0.3, facecolor=PANEL, edgecolor=BORDER)


# ════════════════════════════════════════════════════════════════════════════
# Panel 3 — UNet Channel Progression
# ════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])

stages   = ["Input\n(4,64,64)", "Stage 0\n64×64", "Stage 1\n32×32", "Stage 2\n16×16",
            "Stage 3\n8×8", "Bottleneck\n8×8", "Stage 3\n8×8",  "Stage 2\n16×16",
            "Stage 1\n32×32", "Stage 0\n64×64", "Output\n(4,64,64)"]
channels = [4, 320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320, 4]
attn     = [False, False, True, True, True, True, True, True, True, False, False]

colors_unet = [BLUE if a else ORANGE for a in attn]
colors_unet[0]  = GRAY
colors_unet[-1] = GRAY
colors_unet[5]  = RED     # bottleneck

xs = np.arange(len(stages))
bars3 = ax3.bar(xs, channels, color=colors_unet, width=0.7,
                edgecolor=BORDER, linewidth=0.6)

for i, (bar, ch) in enumerate(zip(bars3, channels)):
    if ch > 0:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f"{ch}", ha="center", va="bottom", fontsize=7,
                 color=WHITE, fontweight="bold")

ax3.set_xticks(xs)
ax3.set_xticklabels(stages, fontsize=6.5, rotation=30, ha="right")
ax3.set_ylabel("Channels", fontsize=9)
ax3.set_title("UNet Channel Progression (Encoder → Decoder)", fontsize=11,
              fontweight="bold", color=BLUE, pad=8)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)

enc_patch  = mpatches.Patch(color=ORANGE, label="No attention")
attn_patch = mpatches.Patch(color=BLUE,   label="+ Cross-attention")
btn_patch  = mpatches.Patch(color=RED,    label="Bottleneck")
ax3.legend(handles=[enc_patch, attn_patch, btn_patch],
           loc="upper center", fontsize=7.5, framealpha=0.3,
           facecolor=PANEL, edgecolor=BORDER, ncol=3)


# ════════════════════════════════════════════════════════════════════════════
# Panel 4 — Data Pipeline Flow
# ════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis("off")
ax4.set_title("Data Pipeline  (5 Steps)", fontsize=11,
              fontweight="bold", color=BLUE, pad=8)

steps = [
    ("01", "Download Metadata",   "LAION-2B parquets via HF Hub",    GREEN),
    ("02", "Filter Metadata",     "aesthetic≥6.5  |  CLIP sim≥0.28", ORANGE),
    ("03", "Download Images",     "img2dataset  |  512px WebDataset", BLUE),
    ("04", "Preprocess → Cache",  "Tokenize captions  |  Parquet",   PURPLE),
    ("05", "Build HF Dataset",    "Train/val split  |  load_from_disk",CYAN),
]

for i, (num, title, desc, color) in enumerate(steps):
    y = 9.0 - i * 1.75
    rect = mpatches.FancyBboxPatch(
        (0.3, y - 0.55), 9.2, 1.1,
        boxstyle="round,pad=0.12",
        facecolor=PANEL, edgecolor=color, linewidth=1.5
    )
    ax4.add_patch(rect)
    ax4.text(0.75, y + 0.08, num, fontsize=11, fontweight="bold",
             color=color, va="center")
    ax4.text(1.7, y + 0.08, title, fontsize=9, fontweight="bold",
             color=WHITE, va="center")
    ax4.text(1.7, y - 0.28, desc, fontsize=7.5, color=GRAY, va="center")

    if i < len(steps) - 1:
        ax4.annotate("", xy=(5.0, y - 0.65), xytext=(5.0, y - 0.55),
                     arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))

ax4.text(5, 0.25, "→ encode_pipeline.py  (Dual-GPU VAE latent encoding)",
         ha="center", fontsize=8, color=ORANGE)


# ════════════════════════════════════════════════════════════════════════════
# Panel 5 — Training Hyperparameter Summary
# ════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis("off")
ax5.set_title("Training Configuration", fontsize=11,
              fontweight="bold", color=BLUE, pad=8)

params = [
    ("Hardware",          "2× RTX PRO 4500  (32 GB each)"),
    ("Dataset",           "LAION-2B-en-aesthetic  ~12M imgs"),
    ("Effective Batch",   "128 × 2 GPUs × 2 accum = 512"),
    ("Optimizer",         "AdamW  lr=1e-4  wd=1e-2  fused"),
    ("LR Schedule",       "500-step warmup + CosineAnnealing"),
    ("Precision",         "bfloat16  +  torch.compile"),
    ("UNet Loss",         "ε-prediction MSE  (T=1000 steps)"),
    ("EMA",               "decay=0.9999  warmup-corrected"),
    ("Inference",         "DDIM  30 steps  CFG scale=7.5"),
    ("Grad Checkpointing","Optional  (saves ~30% VRAM)"),
    ("Status",            "▶  Encoding latents to RAM (~42 GB)"),
]

col_w = [3.5, 6.0]
header_y = 9.5
for j, header in enumerate(["Parameter", "Value"]):
    ax5.text(0.3 + sum(col_w[:j]) * 0.35, header_y, header,
             fontsize=9, fontweight="bold", color=BLUE)

ax5.axhline(y=header_y - 0.35, xmin=0.02, xmax=0.98,
            color=BORDER, linewidth=0.8)

for i, (key, val) in enumerate(params):
    y = header_y - 0.9 - i * 0.78
    color = ORANGE if "Status" in key else WHITE
    bg_color = "#1f2937" if i % 2 == 0 else PANEL
    ax5.text(0.3, y, key,  fontsize=8.5, color=GRAY,  va="center")
    ax5.text(4.0, y, val,  fontsize=8.5, color=color, va="center",
             fontweight="bold" if "Status" in key else "normal")

ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)


# ════════════════════════════════════════════════════════════════════════════
# Panel 6 — Training Status / Progress
# ════════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis("off")
ax6.set_title("Project Status", fontsize=11,
              fontweight="bold", color=BLUE, pad=8)

pipeline_stages = [
    ("Download Metadata",      True,  "Complete"),
    ("Filter Metadata",        True,  "Complete"),
    ("Download Images",        True,  "Complete"),
    ("Preprocess → Cache",     True,  "Complete"),
    ("Build HF Dataset",       True,  "Complete"),
    ("Encode Latents (VAE)",   False, "In Progress"),
    ("Train UNet",             False, "Pending"),
    ("Validate (DDIM)",        False, "Pending"),
    ("Generate Samples",       False, "Pending"),
]

for i, (stage, done, label) in enumerate(pipeline_stages):
    y = 9.1 - i * 0.95
    icon  = "●" if done else ("◐" if label == "In Progress" else "○")
    color = GREEN if done else (ORANGE if label == "In Progress" else GRAY)

    ax6.text(0.4, y, icon,  fontsize=13, color=color, va="center")
    ax6.text(1.4, y + 0.05, stage, fontsize=9, color=WHITE, va="center",
             fontweight="bold" if label == "In Progress" else "normal")
    ax6.text(1.4, y - 0.28, label, fontsize=7.5, color=color, va="center")

# Progress bar
done_count = sum(1 for _, d, _ in pipeline_stages if d)
in_progress = sum(1 for _, _, l in pipeline_stages if l == "In Progress")
progress = (done_count + in_progress * 0.5) / len(pipeline_stages)

bar_x, bar_y, bar_w, bar_h = 0.5, 0.4, 9.0, 0.45
bg_rect = mpatches.FancyBboxPatch((bar_x, bar_y), bar_w, bar_h,
                                   boxstyle="round,pad=0.05",
                                   facecolor="#1f2937", edgecolor=BORDER, linewidth=1.0)
ax6.add_patch(bg_rect)

fill_rect = mpatches.FancyBboxPatch((bar_x, bar_y), bar_w * progress, bar_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=ORANGE, edgecolor="none", linewidth=0)
ax6.add_patch(fill_rect)
ax6.text(5.0, bar_y + bar_h / 2, f"{progress*100:.0f}% Pipeline Complete",
         ha="center", va="center", fontsize=9, color=BG, fontweight="bold")


# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "assets/architecture_overview.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved: {out_path}")
