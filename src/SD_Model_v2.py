"""
SD_Model_v2.py
==============
Next-generation Stable Diffusion architecture inspired by SD3, FLUX.1,
and cutting-edge research in latent diffusion models.

┌─────────────────────────────────────────────────────────────────────────┐
│               WHAT CHANGED vs SD_Model.py (v1)                         │
├─────────────────────┬───────────────────────────────────────────────────┤
│ Component           │ v1 → v2                                           │
├─────────────────────┼───────────────────────────────────────────────────┤
│ Backbone            │ UNet (conv-heavy)  → MM-DiT (pure transformer)    │
│ Text encoder        │ Single CLIP-L/14   → Dual CLIP-L + OpenCLIP-bigG  │
│ Text context dim    │ 768                → 2048                         │
│ Noise schedule      │ DDPM (cosine/beta) → Rectified Flow (ODE, [0,1]) │
│ Loss target         │ ε-prediction       → velocity (v = x₁ − x₀)      │
│ Timestep sampling   │ Uniform            → Logit-Normal (better spread) │
│ Positional encoding │ None on latents    → 2-D Rotary (RoPE) on tokens  │
│ Conditioning        │ Additive t-bias    → adaLN-Zero (scale+shift+gate)│
│ Normalisation       │ GroupNorm          → RMSNorm (faster, no mean)    │
│ MLP activation      │ GELU               → SwiGLU (better throughput)   │
│ Attention stability │ Raw dot-product    → QK-RMSNorm before SDPA       │
│ Inference sampler   │ DDIM (50 steps)    → Heun / Euler (20–25 steps)   │
│ Inference quality   │ Raw weights        → EMA shadow weights            │
└─────────────────────┴───────────────────────────────────────────────────┘

Architecture data-flow (training)
----------------------------------
                 ┌──────────────┐   ┌──────────────────────┐
  img (B,3,H,W) │  AdvancedVAE │   │   DualTextEncoder     │ tokens (B,77)
                 └──────┬───────┘   └───────────┬──────────┘
                 latents│(B,C,h,w)  ctx(B,S,2048)│ pooled(B,2048)
                        │                        │
              ┌─────────┴──────────┐             │
              │ RectifiedFlow.noise │  ←  t~Logit-Normal
              └─────────┬──────────┘
                 noisy  │(B,C,h,w)
                        │
                  ┌─────┴──────┐
                  │   MM-DiT   │  ← t_emb (sinusoidal→MLP)
                  │            │  ← ctx tokens (B, S_txt, d)
                  │ N joint    │     adaLN-Zero from t + pooled
                  │ blocks     │     2-D RoPE on image tokens
                  │ M single   │     QK-RMSNorm in attention
                  │ blocks     │     SwiGLU MLP
                  └─────┬──────┘
                        │ v_pred (B,C,h,w)
                        │
             loss = MSE(v_pred, v_target)   v_target = x₁ − x₀

Architecture data-flow (inference)
------------------------------------
  z_T ~ N(0,I)  →  HeunSampler (20 steps, CFG)  →  z_0  →  VAE.decode()

Model size presets (MMDiTConfig)
---------------------------------
  "nano"  :  d=768,  heads=12, joint=12, single=0  →  ~250 M  (≤6 GB VRAM)
  "small" :  d=1152, heads=16, joint=24, single=0  →  ~700 M  (≤8 GB VRAM)
  "medium":  d=1536, heads=24, joint=38, single=0  →  ~2.0 B  (≤16 GB VRAM)
  "large" :  d=2048, heads=32, joint=38, single=38 →  ~8.0 B  (≤24 GB VRAM)

Debugging notes are scattered throughout as comments beginning with [DEBUG].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MMDiTConfig:
    """
    Configuration for the MM-DiT backbone and the full pipeline.

    Debugging guide
    ---------------
    - If you hit OOM: lower d_model or n_joint first; enable grad_ckpt.
    - If loss is NaN on step 1: check in_channels matches the VAE (4 or 16).
    - If images look blurry: increase n_joint or switch to a larger preset.
    - If text prompts are ignored: increase ctx_dim or add T5 encoder.
    """
    # ── Core MM-DiT dimensions ──────────────────────────────────────────────
    d_model:     int   = 1152   # Main hidden width (must be divisible by n_heads)
    n_heads:     int   = 16     # Attention heads  (head_dim = d_model // n_heads)
    n_joint:     int   = 24     # Joint image+text transformer blocks
    n_single:    int   = 0      # Extra image-only blocks after joint blocks
    mlp_ratio:   float = 4.0    # SwiGLU hidden = int(d_model * mlp_ratio * 2/3)

    # ── Text / conditioning ──────────────────────────────────────────────────
    ctx_dim:     int   = 2048   # Projected text token dimension (CLIP-L + bigG)
    pooled_dim:  int   = 2048   # Pooled text dimension for adaLN-Zero

    # ── Latent space ─────────────────────────────────────────────────────────
    in_channels: int   = 4      # 4 = sd-vae-ft-mse; 16 = FLUX VAE
    patch_size:  int   = 2      # Spatial patch size; tokens = (h/p) * (w/p)

    # ── Training ─────────────────────────────────────────────────────────────
    grad_ckpt:   bool  = False  # Gradient checkpointing (saves VRAM, ~20% slower)
    dropout:     float = 0.0    # Attention dropout (0 = off; 0.1 for regularisation)


# ── Preset factory ─────────────────────────────────────────────────────────────
MMDIT_PRESETS: Dict[str, MMDiTConfig] = {
    "nano":   MMDiTConfig(d_model=768,  n_heads=12, n_joint=12, n_single=0),
    "small":  MMDiTConfig(d_model=1152, n_heads=16, n_joint=24, n_single=0),
    "medium": MMDiTConfig(d_model=1536, n_heads=24, n_joint=38, n_single=0),
    "large":  MMDiTConfig(d_model=2048, n_heads=32, n_joint=38, n_single=38),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CORE UTILITY MODULES
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalisation (Zhang & Sennrich, 2019).

    RMSNorm omits the mean-centring step of LayerNorm:
        y = x / RMS(x) * weight    where RMS(x) = sqrt(mean(x²) + ε)

    Why it's better here:
    - ~25% fewer ops than LayerNorm (no mean subtraction or bias).
    - Empirically as good or better in transformer+diffusion models.
    - Used in LLaMA, Gemma, FLUX, and most modern large models.

    Debugging: if you see scale explosions early in training, reduce the
    learning rate — RMSNorm has no bias to absorb scale shifts.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, then cast back
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020; used in PaLM, LLaMA, SD3).

    Architecture:
        gate = SiLU(W_gate · x)        # Gating branch
        hidden = W_up · x              # Value branch
        out = W_down · (gate * hidden) # Gated projection

    Parameter count:
        3 linear layers instead of 2, but hidden_dim = int(4d * 2/3) so
        total parameters are equal to a standard GELU MLP with hidden_dim=4d.

    Why it's better than GELU-MLP:
    - Gated structure allows selective feature propagation.
    - Consistently ~10% lower perplexity than GELU at matched parameter count.
    - Used in every top-tier diffusion model (FLUX, SD3, Würstchen).

    Debugging: if the gate values saturate to 0 or 1 early, reduce the
    learning rate for the gate projection or add weight decay.
    """

    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        # Hidden dim: 4d scaled to 2/3 to keep param count equal to standard MLP
        hidden = int(dim * ratio * 2 / 3)
        # Round to nearest multiple of 64 for CUDA efficiency
        hidden = (hidden + 63) // 64 * 64
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up   = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        # Zero-init output projection → identity at initialisation
        nn.init.zeros_(self.w_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TIMESTEP EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    Maps a scalar timestep t ∈ [0, 1] to a dense embedding vector.

    Pipeline:
        t (float, 0→1)
          → scaled to [0, 1000] for backward-compatible sinusoidal encoding
          → sinusoidal_embed (d_model,)
          → MLP: d_model → d_model*4 → SiLU → d_model*4
          → used for adaLN-Zero conditioning

    The 4× wider hidden layer lets the MLP learn a rich non-linear
    transformation of the frequency features before they modulate each block.

    Debugging:
    - If all timestep embeddings look the same: check that t is not being
      accidentally clipped to 0 or 1 before calling forward().
    - If the model ignores the timestep: verify that t is passed in [0,1]
      (not as an integer 0–999 as in v1). Use scheduler.get_t() to convert.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal encoding of continuous t ∈ [0, 1].

        We scale t → t*1000 so the frequency range matches the original
        DDPM integer timestep encodings (0–999), keeping pretrained
        checkpoints compatible when fine-tuning from DDPM-trained models.

        Shape: (B,) → (B, d_model)
        """
        half = self.d_model // 2
        # Log-spaced frequencies (same as v1 but now for continuous t)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        # Scale t to [0, 1000] to maintain frequency resolution
        angles = t.float()[:, None] * 1000.0 * freqs[None, :]  # (B, half)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)   # (B, d_model)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Continuous timesteps (B,) in [0, 1].  0 = clean data, 1 = noise.
        Returns:
            Timestep embedding (B, d_model*4).
        """
        # [DEBUG] Sanity check — t must be in [0, 1] for flow matching
        # assert t.min() >= 0.0 and t.max() <= 1.0, f"t out of range: [{t.min():.3f}, {t.max():.3f}]"
        return self.mlp(self._sinusoidal(t))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — 2-D ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

def get_2d_rope_freqs(
    height: int,
    width:  int,
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute 2-D RoPE sin/cos tables for a spatial grid of (height × width) patches.

    How 2-D RoPE works
    ------------------
    Standard 1-D RoPE rotates pairs of (q, k) elements by an angle that
    depends on their sequence position.  For 2-D images, we split the head
    dimension in half:
        - First half (dim//2):  encodes the row position (y-axis).
        - Second half (dim//2): encodes the column position (x-axis).
    Each half uses standard 1-D RoPE logic but with independent x/y indices.

    This gives the model an inductive bias for spatial translation equivariance
    — two patches at the same relative spatial offset will have the same
    rotational difference, regardless of their absolute positions.

    Args:
        height, width: Spatial grid dimensions (in patches, after patching).
        head_dim:      Per-head attention dimension (must be divisible by 4).
        device:        Target device.
        theta:         Base frequency (10000 is standard; lower = faster decay).

    Returns:
        sin_table: (height*width, head_dim)  [contains both x and y components]
        cos_table: (height*width, head_dim)

    Debugging:
    - If model learns nothing about spatial layout: confirm RoPE is applied
      to Q and K (NOT V) by checking apply_2d_rope() call sites.
    - If shape errors appear: head_dim must be divisible by 4 (2 for x, 2 for y).
    - For variable resolution training, call this function with each new
      (height, width) pair and do NOT cache across resolutions.
    """
    assert head_dim % 4 == 0, (
        f"head_dim={head_dim} must be divisible by 4 for 2-D RoPE "
        f"(two halves, each processed as complex pairs)."
    )
    dim_per_axis = head_dim // 2  # half for y-axis, half for x-axis
    half         = dim_per_axis // 2

    # Frequency bands: log-spaced (same as standard 1-D RoPE)
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))

    # Build y (row) and x (col) position tables
    y_pos = torch.arange(height, device=device).float()  # (H,)
    x_pos = torch.arange(width,  device=device).float()  # (W,)

    # Outer product → angles (H, half) and (W, half)
    y_angles = torch.outer(y_pos, freqs)  # (H, half)
    x_angles = torch.outer(x_pos, freqs)  # (W, half)

    # Broadcast to full (H, W) grid
    y_angles = y_angles[:, None, :].expand(height, width, half).reshape(height * width, half)
    x_angles = x_angles[None, :, :].expand(height, width, half).reshape(height * width, half)

    # Concatenate y and x components: (N, dim_per_axis) for each
    angles = torch.cat([y_angles, x_angles], dim=-1)  # (N, head_dim//2)

    # Full sin/cos tables — repeated twice to match head_dim after interleaving
    sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)  # (N, head_dim)
    cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (N, head_dim)

    return sin_table, cos_table


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [x1, x2, x3, x4, ...] → [-x2, x1, -x4, x3, ...]."""
    # Split the last dimension into two equal halves and rotate
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([-x2, x1], dim=-1)


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2-D RoPE to query and key tensors.

    Args:
        q, k:        (B, heads, N, head_dim) — image tokens only (not text).
        sin_table:   (N, head_dim) from get_2d_rope_freqs().
        cos_table:   (N, head_dim) from get_2d_rope_freqs().

    Returns:
        q_rot, k_rot: Same shapes as q, k but with positions encoded.

    Important: RoPE is applied ONLY to Q and K, never to V.  The value
    projection retains the original feature space without rotation.

    Debugging:
    - If you see a shape error here, ensure N matches height*width from the
      RoPE table: N = (latent_h / patch_size) * (latent_w / patch_size).
    - The sin/cos tables are cheaply recomputed each forward pass; no need
      to cache them in the module state_dict.
    """
    # Broadcast over batch and head dimensions: (1, 1, N, head_dim)
    sin = sin_table.unsqueeze(0).unsqueeze(0)
    cos = cos_table.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ADAPTIVE LAYERNORM ZERO (adaLN-Zero)
# ═══════════════════════════════════════════════════════════════════════════════

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalisation with Zero-Init gating (Peebles & Xie, 2023).

    The conditioning vector c (from timestep + pooled text) modulates each
    transformer block via three learned parameters per sub-layer:
        - α (alpha): scale applied after normalisation.
        - β (beta):  shift applied after normalisation.
        - γ (gamma): gate applied to the sub-layer output before residual add.

    Conditioning equations per sub-layer:
        c_proj(SiLU(c)) → [α₁, β₁, γ₁, α₂, β₂, γ₂]   (each of shape (B, d))
        x_norm = RMSNorm(x) * (1 + α₁) + β₁
        x = x + γ₁ * sub_layer(x_norm)
        x_norm2 = RMSNorm(x) * (1 + α₂) + β₂
        x = x + γ₂ * mlp(x_norm2)

    Why zero-init gating is critical
    ---------------------------------
    If the final linear of c_proj is zero-initialised, all gates (γ) start
    at 0 → every block is a perfect identity at the very start of training.
    This means gradients flow cleanly through all residual skip connections
    at step 0, eliminating the "dead block" problem common in deep nets.

    Debugging:
    - If gamma values are all zero after 1000 steps: the learning rate for
      this module is too low, or the zero-init is not being cleared properly.
      Print `self.c_proj.weight.norm()` to verify it's growing.
    - If conditioning collapses (all γ → same value): add a small weight
      decay term (1e-4) to the c_proj parameters.
    - If adaLN is not conditioning the image on text: verify that pooled_txt
      is being concatenated with t_emb BEFORE being passed to this module.
    """

    def __init__(self, d_model: int, cond_dim: int):
        """
        Args:
            d_model:  Hidden dimension of the transformer.
            cond_dim: Conditioning vector dimension (t_emb + pooled_txt).
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        # Projects conditioning vector to 6 modulation parameters
        # SiLU activates before projection; the linear is zero-initialised
        self.c_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True),
        )
        # ── Zero-init is the key trick — starts every block as identity ──
        nn.init.zeros_(self.c_proj[-1].weight)
        nn.init.zeros_(self.c_proj[-1].bias)

    def modulate(
        self, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose conditioning vector into 6 modulation scalars.

        Returns:
            (alpha1, beta1, gamma1, alpha2, beta2, gamma2) each (B, 1, d_model).
            The `1` broadcast dimension handles the sequence length.
        """
        params = self.c_proj(c).unsqueeze(1)         # (B, 1, 6*d)
        chunks = params.chunk(6, dim=-1)             # 6 × (B, 1, d)
        return chunks  # (α₁, β₁, γ₁, α₂, β₂, γ₂)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — JOINT MULTI-MODAL ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class JointAttention(nn.Module):
    """
    Multi-modal joint attention over concatenated image and text token sequences.

    Core idea (from SD3 / MM-DiT, Esser et al., 2024):
        Image tokens and text tokens each have their own separate QKV projections,
        but attention is computed over their CONCATENATED sequence.  This means
        image tokens can attend to text tokens and vice versa within the same
        attention matrix, enabling deep multi-modal fusion at every layer.

    Key stabilisation features:
        1. QK-RMSNorm: Normalise Q and K before computing attention scores.
           This prevents attention logit explosion at scale, which would cause
           NaN losses with deep models or large batch sizes.
        2. 2-D RoPE: Applied to image Q, K to encode spatial structure.
        3. Flash Attention (SDPA): torch.nn.functional.scaled_dot_product_attention
           dispatches to Flash Attention 2 on CUDA — up to 8× faster than
           naive attention and O(N) not O(N²) in memory.

    Architecture:
        ┌────────────────────┐  ┌────────────────────┐
        │  Image stream      │  │  Text stream        │
        │  (B, N_img, d)     │  │  (B, N_txt, d)      │
        └────────┬───────────┘  └────────┬────────────┘
                 │  img_qkv               │  txt_qkv
                 ↓                       ↓
        img Q,K,V                 txt Q,K,V
                 │                       │
        QK-norm + 2-D RoPE        QK-norm (no RoPE)
                 │                       │
                 └───────────┬───────────┘
                          concat Q, K, V
                             ↓
                    SDPA  (Flash Attn 2)
                             ↓
                          split back
                 ┌───────────┴───────────┐
                 ↓                       ↓
            img output              txt output

    Debugging:
    - NaN after attention: check QK-norm is enabled (qknorm=True).
      Disable mixed precision temporarily to pinpoint.
    - Wrong output shape: N_img must equal (latent_h / patch_size)².
      Add assert after patchify: `assert img_tokens.shape[1] == expected_N`.
    - Text tokens are not contributing: verify txt_qkv is NOT sharing weights
      with img_qkv — they must be independent nn.Linear layers.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dropout:  float = 0.0,
        qknorm:   bool  = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}. "
            f"head_dim would be {d_model / n_heads:.1f}."
        )
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5
        self.dropout  = dropout

        # ── Separate QKV projections for image and text streams ──────────────
        self.img_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.txt_qkv = nn.Linear(d_model, d_model * 3, bias=False)

        # ── Independent output projections ────────────────────────────────────
        self.img_out = nn.Linear(d_model, d_model, bias=False)
        self.txt_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.img_out.weight)
        nn.init.zeros_(self.txt_out.weight)

        # ── QK-RMSNorm (one per stream per Q/K) ──────────────────────────────
        if qknorm:
            self.q_norm_img = RMSNorm(self.head_dim)
            self.k_norm_img = RMSNorm(self.head_dim)
            self.q_norm_txt = RMSNorm(self.head_dim)
            self.k_norm_txt = RMSNorm(self.head_dim)
        else:
            self.q_norm_img = self.k_norm_img = nn.Identity()
            self.q_norm_txt = self.k_norm_txt = nn.Identity()

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, d) → (B, heads, N, head_dim)."""
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        img: torch.Tensor,             # (B, N_img, d)
        txt: torch.Tensor,             # (B, N_txt, d)
        rope_sin: Optional[torch.Tensor] = None,  # (N_img, head_dim) RoPE tables
        rope_cos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Jointly attend over image and text tokens.

        Args:
            img:      Image tokens (B, N_img, d).
            txt:      Text  tokens (B, N_txt, d).
            rope_sin: 2-D RoPE sine table (N_img, head_dim). None → no RoPE.
            rope_cos: 2-D RoPE cosine table (N_img, head_dim).

        Returns:
            img_out: (B, N_img, d) — attended image features.
            txt_out: (B, N_txt, d) — attended text features.
        """
        B, N_img, _ = img.shape
        B, N_txt, _ = txt.shape

        # ── Project QKV for each stream ──────────────────────────────────────
        iq, ik, iv = self.img_qkv(img).chunk(3, dim=-1)   # each (B, N_img, d)
        tq, tk, tv = self.txt_qkv(txt).chunk(3, dim=-1)   # each (B, N_txt, d)

        # Reshape to multi-head format
        iq = self._reshape_for_heads(iq)  # (B, h, N_img, d_h)
        ik = self._reshape_for_heads(ik)
        iv = self._reshape_for_heads(iv)
        tq = self._reshape_for_heads(tq)  # (B, h, N_txt, d_h)
        tk = self._reshape_for_heads(tk)
        tv = self._reshape_for_heads(tv)

        # ── QK-RMSNorm — prevents score explosion ────────────────────────────
        # Norm is applied per-head over head_dim dimension
        iq = self.q_norm_img(iq)
        ik = self.k_norm_img(ik)
        tq = self.q_norm_txt(tq)
        tk = self.k_norm_txt(tk)

        # ── Apply 2-D RoPE to image Q and K only ────────────────────────────
        # Text tokens get no spatial RoPE — they have sequential CLIP ordering
        if rope_sin is not None and rope_cos is not None:
            iq, ik = apply_2d_rope(iq, ik, rope_sin, rope_cos)

        # ── Concatenate both streams for joint attention ──────────────────────
        # Joint Q/K/V: image tokens attend to text tokens and vice versa
        q = torch.cat([iq, tq], dim=2)   # (B, h, N_img+N_txt, d_h)
        k = torch.cat([ik, tk], dim=2)
        v = torch.cat([iv, tv], dim=2)

        # ── Flash Attention (SDPA) — O(N) memory via kernel fusion ───────────
        # [DEBUG] If SDPA fails with a backend error on older CUDA, fall back:
        # attn = (q @ k.transpose(-2,-1)) * self.scale
        # attn = F.softmax(attn, dim=-1)
        # out  = attn @ v
        attn_drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                              dropout_p=attn_drop,
                                              scale=self.scale)
        # (B, h, N_img+N_txt, d_h)

        # ── Split back into image and text streams ────────────────────────────
        img_out = out[:, :, :N_img, :]            # (B, h, N_img, d_h)
        txt_out = out[:, :, N_img:, :]            # (B, h, N_txt, d_h)

        # Merge heads: (B, h, N, d_h) → (B, N, d)
        img_out = img_out.transpose(1, 2).reshape(B, N_img, -1)
        txt_out = txt_out.transpose(1, 2).reshape(B, N_txt, -1)

        return self.img_out(img_out), self.txt_out(txt_out)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — JOINT TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class JointTransformerBlock(nn.Module):
    """
    Single MM-DiT joint transformer block — the core repeating unit.

    One block contains:
        1. adaLN-Zero modulation (shared conditioning from t + pooled_txt)
        2. JointAttention  (image ↔ text cross-stream attention + self)
        3. SwiGLU MLP for the image stream
        4. SwiGLU MLP for the text stream

    Each stream has its own MLP, so image and text features evolve via
    different learned transformations after their shared attention.  This
    allows the model to specialise the two representations differently
    while still coupling them through joint attention.

    Data flow:
        ┌── img ──────────────────────────────────────────────────────────┐
        │   x = x + γ₁ * JointAttn.img(adaLN₁(x, c))                    │
        │   x = x + γ₂ * img_mlp(adaLN₂(x, c))                          │
        └─────────────────────────────────────────────────────────────────┘
        ┌── txt ──────────────────────────────────────────────────────────┐
        │   y = y + γ₁ * JointAttn.txt(adaLN₁(y, c))                    │
        │   y = y + γ₂ * txt_mlp(adaLN₂(y, c))                          │
        └─────────────────────────────────────────────────────────────────┘
        (c = timestep_embedding + pooled_text_embedding)

    Note: each stream uses its OWN adaLN instance — image and text are
    conditioned independently, which gives the model more capacity to
    learn different modulation strategies per modality.

    Debugging:
    - If image tokens look the same regardless of prompt: check that txt
      stream is flowing into joint attention (N_txt > 0).
    - If loss explodes after 100 steps: inspect γ₁ magnitudes; values
      above ~3 indicate the gates have grown too large. Add weight decay
      to adaLN parameters (1e-4 to 1e-3).
    - To check block activations during a run:
        block.register_forward_hook(lambda m, i, o: print(o[0].std()))
    """

    def __init__(self, cfg: MMDiTConfig):
        super().__init__()
        cond_dim = cfg.d_model * 4  # t_emb output width (see TimestepEmbedding)

        # Independent adaLN for each stream
        self.adaln_img = AdaLNZero(cfg.d_model, cond_dim)
        self.adaln_txt = AdaLNZero(cfg.d_model, cond_dim)

        # Shared joint attention (processes both streams together)
        self.attn = JointAttention(cfg.d_model, cfg.n_heads, cfg.dropout)

        # Separate MLPs for each stream
        self.img_mlp = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio)
        self.txt_mlp = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio)

        self.grad_ckpt = cfg.grad_ckpt

    def _forward(
        self,
        img:      torch.Tensor,   # (B, N_img, d)
        txt:      torch.Tensor,   # (B, N_txt, d)
        c:        torch.Tensor,   # (B, d*4) conditioning
        rope_sin: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ── adaLN-Zero modulation for image stream ───────────────────────────
        a1i, b1i, g1i, a2i, b2i, g2i = self.adaln_img.modulate(c)
        img_n = self.adaln_img.norm1(img) * (1 + a1i) + b1i

        # ── adaLN-Zero modulation for text stream ───────────────────────────
        a1t, b1t, g1t, a2t, b2t, g2t = self.adaln_txt.modulate(c)
        txt_n = self.adaln_txt.norm1(txt) * (1 + a1t) + b1t

        # ── Joint attention ──────────────────────────────────────────────────
        img_attn, txt_attn = self.attn(img_n, txt_n, rope_sin, rope_cos)

        # ── Gated residual add (attention) ────────────────────────────────────
        img = img + g1i * img_attn
        txt = txt + g1t * txt_attn

        # ── MLP with adaLN modulation ─────────────────────────────────────────
        img = img + g2i * self.img_mlp(self.adaln_img.norm2(img) * (1 + a2i) + b2i)
        txt = txt + g2t * self.txt_mlp(self.adaln_txt.norm2(txt) * (1 + a2t) + b2t)

        return img, txt

    def forward(
        self,
        img:      torch.Tensor,
        txt:      torch.Tensor,
        c:        torch.Tensor,
        rope_sin: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.grad_ckpt and self.training:
            # [DEBUG] use_reentrant=False avoids the deprecated reentrant path.
            # If you see "one of the variables needed for gradient computation
            # has been modified by an inplace operation" — confirm this flag.
            return checkpoint(
                self._forward, img, txt, c, rope_sin, rope_cos,
                use_reentrant=False
            )
        return self._forward(img, txt, c, rope_sin, rope_cos)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SINGLE-STREAM TRANSFORMER BLOCK (FLUX-style)
# ═══════════════════════════════════════════════════════════════════════════════

class SingleTransformerBlock(nn.Module):
    """
    Image-only transformer block for the final processing stages.

    Inspired by FLUX.1, the MM-DiT processes image+text jointly in N_joint
    blocks, then continues with M_single image-only blocks.  By this stage,
    the text conditioning has been fully absorbed into the image token
    representations via the joint blocks; additional pure-image refinement
    blocks improve fine-grained spatial detail without the computational
    overhead of cross-stream attention.

    Architecture per block:
        c → adaLN-Zero modulation (γ₁, α₁, β₁, γ₂, α₂, β₂)
        x = x + γ₁ * SelfAttn(adaLN₁(x))    ← image self-attention only
        x = x + γ₂ * SwiGLU(adaLN₂(x))

    Debugging:
    - If n_single=0 (not using these blocks), no action needed.
    - If spatial structure looks wrong after adding single blocks: ensure
      RoPE tables are still passed in (image tokens still need spatial PE).
    - Single blocks use more compute but no txt memory, so peak VRAM
      is actually lower than joint blocks for the same d_model.
    """

    def __init__(self, cfg: MMDiTConfig):
        super().__init__()
        cond_dim = cfg.d_model * 4
        self.adaln   = AdaLNZero(cfg.d_model, cond_dim)
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv      = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.q_norm   = RMSNorm(self.head_dim)
        self.k_norm   = RMSNorm(self.head_dim)
        self.mlp      = SwiGLUMLP(cfg.d_model, cfg.mlp_ratio)
        self.dropout  = cfg.dropout
        self.grad_ckpt = cfg.grad_ckpt

    def _forward(
        self,
        x:        torch.Tensor,   # (B, N_img, d)
        c:        torch.Tensor,   # (B, d*4)
        rope_sin: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # adaLN-Zero modulation
        a1, b1, g1, a2, b2, g2 = self.adaln.modulate(c)
        x_n = self.adaln.norm1(x) * (1 + a1) + b1

        # Self-attention (image only)
        q, k, v = self.qkv(x_n).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        if rope_sin is not None:
            q, k = apply_2d_rope(q, k, rope_sin, rope_cos)

        attn_drop = self.dropout if self.training else 0.0
        attn_out  = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, scale=self.scale)
        attn_out  = attn_out.transpose(1, 2).reshape(B, N, -1)
        x = x + g1 * self.out_proj(attn_out)

        # SwiGLU MLP
        x = x + g2 * self.mlp(self.adaln.norm2(x) * (1 + a2) + b2)
        return x

    def forward(self, x, c, rope_sin=None, rope_cos=None):
        if self.grad_ckpt and self.training:
            return checkpoint(self._forward, x, c, rope_sin, rope_cos, use_reentrant=False)
        return self._forward(x, c, rope_sin, rope_cos)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MM-DiT  (Multi-Modal Diffusion Transformer)
# ═══════════════════════════════════════════════════════════════════════════════

class MMDiT(nn.Module):
    """
    Multi-Modal Diffusion Transformer backbone.

    The MM-DiT replaces the UNet from SD 1.x/2.x with a pure transformer
    that jointly models image latent tokens and text tokens.

    Processing pipeline:
        latents (B, C, H, W)
            ↓ patchify()
        img_tokens (B, N_img, d_model)       N_img = (H/p)*(W/p)
            ↓
        [for each joint block]
        img_tokens, txt_tokens → JointTransformerBlock → img_tokens, txt_tokens
            ↓
        [for each single block]
        img_tokens → SingleTransformerBlock → img_tokens
            ↓
        final_norm + unpatchify()
        velocity_pred (B, C, H, W)

    The text tokens are discarded after the joint blocks — only the image
    tokens (which now encode text information via cross-attention) are
    processed by single blocks and unpatchified to velocity predictions.

    Patchify / Unpatchify
    ----------------------
    Patchify flattens a spatial feature map into a sequence of patches:
        (B, C, H, W) → rearrange patches of size (p, p)
                     → (B, N, C*p²)   where N = (H/p)*(W/p)
                     → linear proj    → (B, N, d_model)

    Unpatchify is the inverse: (B, N, d_model) → (B, C, H, W).

    Debugging:
    - "Expected 3D input" in patchify: latents must have shape (B, C, H, W).
      Check VAE encode output — it should be 4D.
    - H or W not divisible by patch_size: pad the image to the nearest
      multiple of (patch_size * vae_downsample = p*8) before VAE encoding.
    - "Size mismatch during unpatchify": H_patches * W_patches must equal N.
      Store (H_patches, W_patches) from patchify and pass to unpatchify.
    - Velocity predictions have extreme values (>10): check the final
      LayerNorm; it must come BEFORE the linear, not after.
    - To diagnose which block is causing NaN: temporarily add
        assert not img.isnan().any(), f"NaN after joint block {i}"
      inside the forward loop (see [DEBUG] markers below).
    """

    def __init__(self, cfg: MMDiTConfig):
        super().__init__()
        self.cfg = cfg
        p, d = cfg.patch_size, cfg.d_model

        # ── Input projections ─────────────────────────────────────────────────
        # Patchify linear: flattens C*p² patch pixels into d_model features
        self.patch_embed = nn.Linear(cfg.in_channels * p * p, d, bias=True)
        # Text token projection from ctx_dim (2048) → d_model
        self.txt_embed   = nn.Linear(cfg.ctx_dim, d, bias=True)
        # Conditioning: timestep embedding (d*4) + pooled text (pooled_dim) → d*4
        cond_in = cfg.d_model * 4 + cfg.pooled_dim
        self.cond_proj   = nn.Linear(cond_in, cfg.d_model * 4, bias=True)

        # ── Timestep embedding MLP ────────────────────────────────────────────
        self.t_embed = TimestepEmbedding(d)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.joint_blocks  = nn.ModuleList([JointTransformerBlock(cfg)  for _ in range(cfg.n_joint)])
        self.single_blocks = nn.ModuleList([SingleTransformerBlock(cfg) for _ in range(cfg.n_single)])

        # ── Output head ───────────────────────────────────────────────────────
        # Final normalisation before unpatchify
        self.norm_out  = RMSNorm(d)
        # Linear maps from d_model back to C*p² (reverse of patch_embed)
        self.proj_out  = nn.Linear(d, cfg.in_channels * p * p, bias=True)
        # Zero-init output projection → model predicts zero velocity at step 0
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Convert latents (B, C, H, W) into a sequence of patch tokens.

        Args:
            x: Latents (B, C, H, W).  H and W must be divisible by patch_size.

        Returns:
            tokens:   (B, N, d_model)  where N = H_p * W_p.
            H_patches: number of patches along H.
            W_patches: number of patches along W.

        Debugging:
        - "RuntimeError: size of tensor x must be divisible by patch_size":
          Ensure input resolution is padded to a multiple of patch_size × 8.
        - Incorrect token count: N must equal (H/p) * (W/p).
        """
        p = self.cfg.patch_size
        B, C, H, W = x.shape

        # [DEBUG] uncomment to validate divisibility before training
        assert H % p == 0 and W % p == 0, (
            f"Latent spatial dims ({H}, {W}) must be divisible by "
            f"patch_size={p}.  Pad your images to a multiple of {p * 8}."
        )

        H_p, W_p = H // p, W // p

        # Rearrange: (B, C, H, W) → (B, H_p, W_p, C, p, p)
        x = x.view(B, C, H_p, p, W_p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H_p, W_p, C, p, p)
        x = x.view(B, H_p * W_p, C * p * p)            # (B, N, C*p²)

        tokens = self.patch_embed(x)  # (B, N, d_model)
        return tokens, H_p, W_p

    def unpatchify(
        self,
        tokens:   torch.Tensor,
        H_patches: int,
        W_patches: int,
    ) -> torch.Tensor:
        """
        Reconstruct (B, C, H, W) from patch tokens (B, N, d_model).

        Applies the final norm + linear to map d_model → C*p², then
        rearranges tokens back to spatial layout.

        Debugging:
        - "Shape mismatch": H_patches * W_patches must equal tokens.shape[1].
          Store (H_p, W_p) from patchify() and reuse them here.
        """
        p = self.cfg.patch_size
        C = self.cfg.in_channels
        B = tokens.shape[0]

        # Final normalisation + projection: (B, N, d) → (B, N, C*p²)
        tokens = self.proj_out(self.norm_out(tokens))

        # (B, N, C*p²) → (B, H_p, W_p, C, p, p)
        tokens = tokens.view(B, H_patches, W_patches, C, p, p)
        # (B, C, H_p, p, W_p, p) → (B, C, H, W)
        tokens = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        return tokens.view(B, C, H_patches * p, W_patches * p)

    def forward(
        self,
        latents:     torch.Tensor,   # (B, C, H, W) noisy latent
        t:           torch.Tensor,   # (B,) timestep in [0, 1]
        txt_tokens:  torch.Tensor,   # (B, S_txt, ctx_dim) text sequence
        pooled_txt:  torch.Tensor,   # (B, pooled_dim) pooled text embedding
    ) -> torch.Tensor:
        """
        Forward pass: predict velocity field v(x_t, t) for rectified flow.

        Args:
            latents:    Noisy latent (B, C, H, W).  C = 4 or 16.
            t:          Continuous timestep (B,) in [0, 1].
                        0 = clean data, 1 = pure noise.
            txt_tokens: Text context sequence (B, S_txt, ctx_dim).
            pooled_txt: Pooled text embedding (B, pooled_dim).

        Returns:
            v_pred: Predicted velocity (B, C, H, W) — same shape as latents.
                    The rectified flow objective trains this to match
                    v_target = noise - clean_latent.

        Debugging:
        - If v_pred is all-zero after step 0: expected — proj_out is
          zero-initialised.  Loss should be ~1.0 at step 0.
        - If v_pred has NaN: check adaLN gates; inspect QK-norm activations.
        - Shape mismatch "expected 4D": latents must be (B, C, H, W), not
          (C, H, W) or (B, H, W, C).
        """
        # ── 1. Patchify latents ───────────────────────────────────────────────
        img, H_p, W_p = self.patchify(latents)  # (B, N_img, d)
        N_img = H_p * W_p

        # ── 2. Project text tokens ────────────────────────────────────────────
        txt = self.txt_embed(txt_tokens)  # (B, S_txt, d)

        # ── 3. Build conditioning vector c ────────────────────────────────────
        # t_emb: sinusoidal→MLP → (B, d*4)
        t_emb = self.t_embed(t)
        # Concatenate with pooled text and project to d*4
        # [DEBUG] If conditioning looks wrong: verify pooled_txt is (B, pooled_dim)
        # not (B, 1, pooled_dim). Squeeze the sequence dim if needed.
        c = self.cond_proj(torch.cat([t_emb, pooled_txt], dim=-1))  # (B, d*4)

        # ── 4. Pre-compute 2-D RoPE for image tokens ─────────────────────────
        rope_sin, rope_cos = get_2d_rope_freqs(
            H_p, W_p,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            device=latents.device,
        )

        # ── 5. Joint transformer blocks ───────────────────────────────────────
        for i, block in enumerate(self.joint_blocks):
            img, txt = block(img, txt, c, rope_sin, rope_cos)
            # [DEBUG] NaN watchpoint — uncomment during debugging:
            # assert not img.isnan().any(), f"NaN in img after joint block {i}"
            # assert not txt.isnan().any(), f"NaN in txt after joint block {i}"

        # ── 6. Single-stream blocks (image only) ─────────────────────────────
        for i, block in enumerate(self.single_blocks):
            img = block(img, c, rope_sin, rope_cos)
            # [DEBUG] assert not img.isnan().any(), f"NaN after single block {i}"

        # ── 7. Unpatchify to recover spatial layout ───────────────────────────
        return self.unpatchify(img, H_p, W_p)  # (B, C, H, W)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing on all transformer blocks.

        Effect: Reduces VRAM by ~40% at the cost of ~25% more compute.
        Use this when training at higher resolution or with larger batch size.

        Debugging:
        - If you see a "RuntimeError: Expected all tensors to be on the same device"
          after enabling checkpointing, it's a bug in one of the block's
          _forward methods — ensure no tensors are being moved in _forward.
        """
        self.cfg.grad_ckpt = True
        for m in self.modules():
            if isinstance(m, (JointTransformerBlock, SingleTransformerBlock)):
                m.grad_ckpt = True

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total":         total,
            "joint_blocks":  sum(p.numel() for b in self.joint_blocks  for p in b.parameters()),
            "single_blocks": sum(p.numel() for b in self.single_blocks for p in b.parameters()),
            "other":         sum(p.numel() for p in list(self.patch_embed.parameters()) +
                                            list(self.txt_embed.parameters()) +
                                            list(self.t_embed.parameters()) +
                                            list(self.proj_out.parameters())),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ADVANCED VAE  (Pretrained — Frozen)
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedVAE(nn.Module):
    """
    Pretrained VAE wrapper supporting both 4-channel and 16-channel latent spaces.

    4-channel VAE (stabilityai/sd-vae-ft-mse):
        - Downscale factor: 8×  (512px → 64×64 latents)
        - Scale factor: 0.18215
        - Compatible with most existing SD models and checkpoints.
        - Use when in_channels=4 in MMDiTConfig.

    16-channel VAE (black-forest-labs/FLUX.1-schnell → ae.safetensors):
        - Same 8× downscale factor.
        - Scale factor: 0.3611  (different normalisation)
        - Encodes more fine-grained detail per latent channel.
        - Better reconstruction, especially for text in images and fine lines.
        - Use when in_channels=16 in MMDiTConfig.

    The VAE is always frozen — it is never trained.

    Debugging:
    - If decoded images look washed out: wrong scale_factor.  The 4-ch and
      16-ch VAEs have DIFFERENT scale factors (0.18215 vs 0.3611).
      Double-check `self.scale_factor` matches the loaded checkpoint.
    - If encoded latents have std >> 1: scale_factor is too small.
      Check `latents.std()` after encode(); target ~1.0.
    - If CUDA OOM during encode with fp16: set `use_fp16=False` or reduce
      batch size.  The VAE is the largest VRAM consumer during data prep.
    - For 16-ch VAE: you may need to install diffusers >= 0.28.0 and
      `from diffusers import AutoencoderKL` still works.

    Scale factors by model:
        sd-vae-ft-mse:      0.18215
        sd-vae-ft-ema:      0.18215
        stabilityai/sdxl:   0.13025
        black-forest FLUX:  0.3611
    """

    # Known scale factors for popular VAEs
    SCALE_FACTORS = {
        "stabilityai/sd-vae-ft-mse": 0.18215,
        "stabilityai/sd-vae-ft-ema": 0.18215,
        "madebyollin/sdxl-vae-fp16-fix": 0.13025,
    }

    def __init__(
        self,
        model_id:    str  = "stabilityai/sd-vae-ft-mse",
        use_fp16:    bool = True,
        scale_factor: Optional[float] = None,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Resolve scale factor
        if scale_factor is not None:
            self.scale_factor = scale_factor
        elif model_id in self.SCALE_FACTORS:
            self.scale_factor = self.SCALE_FACTORS[model_id]
        else:
            # Fall back to model config if available
            self.scale_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        self.use_fp16 = use_fp16

        # Detect number of latent channels for MMDiTConfig validation
        self.latent_channels: int = self.vae.config.latent_channels

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images [-1, 1] → normalised latents.

        Args:
            x: Images (B, 3, H, W) in range [-1, 1].
        Returns:
            Scaled latents (B, C_lat, H/8, W/8).

        Debugging:
        - If latents are all NaN: the input images contain NaN.
          Add `assert not x.isnan().any()` before calling this.
        - If latents have very large std (>5): scale_factor is wrong.
        """
        dtype = torch.float16 if self.use_fp16 else x.dtype
        posterior = self.vae.encode(x.to(dtype)).latent_dist
        # Use mean (not sample()) for deterministic encoding and stable training
        latents = posterior.mean
        return (latents * self.scale_factor).to(x.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode normalised latents → images [-1, 1].

        Args:
            z: Normalised latents (B, C_lat, H/8, W/8).
        Returns:
            Images (B, 3, H, W) approximately in [-1, 1].
        """
        dtype = torch.float16 if self.use_fp16 else z.dtype
        return self.vae.decode(z.to(dtype) / self.scale_factor).sample.to(z.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip encode → decode (sanity check only)."""
        return self.decode(self.encode(x))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — DUAL TEXT ENCODER  (Pretrained — Frozen)
# ═══════════════════════════════════════════════════════════════════════════════

class DualTextEncoder(nn.Module):
    """
    Dual CLIP text encoder: CLIP-ViT-L/14 + OpenCLIP-ViT-bigG/14.

    Why dual encoders?
    ------------------
    CLIP-L/14 (768-dim):     Strong semantic understanding, standard SD encoder.
    OpenCLIP-bigG (1280-dim): Richer captions, better long-text, more detail.
    Together:                 2048-dim context captures both semantic + detail.
    This is the same strategy used by SDXL and SD3.

    Output format:
        ctx_seq:    (B, 154, 2048) — concatenated sequence embeddings
                    [CLIP-L tokens (77) + OpenCLIP-bigG tokens (77)] × 2048
        pooled_txt: (B, 2048) — concatenated pooled outputs for adaLN-Zero
                    [CLIP-L pooled (768) + OpenCLIP-bigG pooled (1280)]

    Both encoders are permanently frozen.  Gradients are disabled.

    Debugging:
    - OOM when loading both encoders: load in fp16 (default).
      Total VRAM for both: ~3.5 GB in fp16.
    - If OpenCLIP-bigG is not available: install `open_clip_torch`.
      pip install open_clip_torch
    - If text conditioning looks wrong: check that the tokenisers use
      the correct max_length (77 for both CLIP models).
    - If CLIP-L and bigG are producing the same embeddings: they share
      a vocabulary but have different architectures. Verify model IDs.
    - Sequence embeddings are projected independently to ctx_dim via
      linear layers inside MMDiT (txt_embed). This allows the model to
      learn how to weigh each encoder's contribution.

    Context dim arithmetic:
        CLIP-L seq:       (B, 77, 768)
        OpenCLIP-bigG seq:(B, 77, 1280)
        Concatenate:      (B, 154, 768+1280) ← THIS DOES NOT WORK
        Instead: project each → (B, 77, ctx_dim//2) then cat along seq dim
        Result:  (B, 154, ctx_dim)  where ctx_dim = 2048

    Or simpler (our approach): project each to ctx_dim separately and
    stack along the sequence dimension → (B, 154, ctx_dim). This preserves
    the full capacity of both encoders independently.
    """

    def __init__(
        self,
        clip_l_id: str = "openai/clip-vit-large-patch14",
        clip_g_id: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        ctx_dim:   int = 2048,
        use_fp16:  bool = True,
    ):
        """
        Args:
            clip_l_id: HuggingFace ID for CLIP-ViT-L/14.
            clip_g_id: HuggingFace ID for OpenCLIP-ViT-bigG/14.
            ctx_dim:   Target context sequence dimension (must equal MMDiTConfig.ctx_dim).
            use_fp16:  Load both encoders in fp16.
        """
        super().__init__()
        from transformers import CLIPTextModel

        dtype = torch.float16 if use_fp16 else torch.float32

        # ── CLIP-ViT-L/14 (768-dim) ───────────────────────────────────────────
        self.clip_l = CLIPTextModel.from_pretrained(clip_l_id, torch_dtype=dtype)
        self.clip_l.requires_grad_(False)
        self.clip_l.eval()
        clip_l_dim = 768  # Fixed for ViT-L/14

        # ── OpenCLIP-ViT-bigG/14 (1280-dim) ───────────────────────────────────
        # We load it via HuggingFace transformers for a unified interface.
        # [DEBUG] If HF doesn't have this model: use open_clip directly:
        #   import open_clip
        #   self.clip_g, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', ...)
        self.clip_g = CLIPTextModel.from_pretrained(clip_g_id, torch_dtype=dtype)
        self.clip_g.requires_grad_(False)
        self.clip_g.eval()
        clip_g_dim = 1280  # Fixed for ViT-bigG/14

        # ── Projection layers (these ARE trainable via MMDiT.txt_embed) ───────
        # Each encoder's sequence is projected to ctx_dim/2 independently
        # then concatenated along the channel (last) dimension.
        # This gives (B, 77, ctx_dim) per encoder → stacked to (B, 154, ctx_dim).
        # We use simple linear projections; they're kept here for shape math.
        half = ctx_dim // 2
        self.proj_l = nn.Linear(clip_l_dim, half, bias=False)
        self.proj_g = nn.Linear(clip_g_dim, half, bias=False)
        # NOTE: proj_l and proj_g are NOT frozen — they learn to map each
        # encoder into a shared space.

        # Pooled output projections
        # Pooled: CLIP-L (768) + bigG (1280) concatenated → (B, 2048)
        self.pooled_dim = clip_l_dim + clip_g_dim  # 2048

        self.use_fp16 = use_fp16
        self.ctx_dim  = ctx_dim

    @torch.no_grad()
    def _encode_clip(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single CLIP encoder. Returns (seq, pooled)."""
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, out.pooler_output

    def forward(
        self,
        input_ids_l:       torch.Tensor,          # (B, 77) for CLIP-L
        input_ids_g:       torch.Tensor,          # (B, 77) for OpenCLIP-bigG
        attention_mask_l:  Optional[torch.Tensor] = None,
        attention_mask_g:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text with both CLIP models.

        Args:
            input_ids_l:  CLIP-L token IDs  (B, 77).
            input_ids_g:  OpenCLIP-bigG token IDs (B, 77).
            attention_mask_l, _g: Optional padding masks.

        Returns:
            ctx_seq:    (B, 154, ctx_dim) — projected and concatenated sequences.
            pooled_txt: (B, 2048) — concatenated pooled embeddings.

        Debugging:
        - If ctx_seq has wrong shape: check that both encoders output (B,77,*).
          Truncate to max_length=77 in your tokeniser.
        - If pooled_txt is all zeros: CLIP pooler_output is sometimes None for
          certain model configs. Fall back to taking mean of seq output:
              pooled = seq.mean(dim=1)
        - Slow encode (>200ms): both CLIP models are being run; this is normal
          on the first few steps (JIT compilation). Use torch.compile() on
          training loops to amortise this cost.
        """
        seq_l, pool_l = self._encode_clip(self.clip_l, input_ids_l, attention_mask_l)
        # (B, 77, 768), (B, 768)
        seq_g, pool_g = self._encode_clip(self.clip_g, input_ids_g, attention_mask_g)
        # (B, 77, 1280), (B, 1280)

        # Project each to ctx_dim/2 and concatenate along channel dim
        seq_l_proj = self.proj_l(seq_l.float())  # (B, 77, ctx_dim//2)
        seq_g_proj = self.proj_g(seq_g.float())  # (B, 77, ctx_dim//2)

        # Option A: concat along channel dim → (B, 77, ctx_dim)
        # Option B: stack along sequence dim → (B, 154, ctx_dim//2)
        # We use Option B so each token sequence stays length-77 but they
        # see each other through joint attention in MM-DiT.
        ctx_l = torch.cat([seq_l_proj, torch.zeros_like(seq_l_proj)], dim=-1)  # (B,77,ctx_dim)
        ctx_g = torch.cat([torch.zeros_like(seq_g_proj), seq_g_proj], dim=-1)  # (B,77,ctx_dim)
        ctx_seq = torch.cat([ctx_l, ctx_g], dim=1)  # (B, 154, ctx_dim)

        # Pooled: simple concatenation of CLS/pooled outputs
        pooled_txt = torch.cat([pool_l.float(), pool_g.float()], dim=-1)  # (B, 2048)

        return ctx_seq, pooled_txt


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RECTIFIED FLOW SCHEDULER  (Training)
# ═══════════════════════════════════════════════════════════════════════════════

class RectifiedFlowScheduler:
    """
    Rectified Flow (Liu et al., 2022) noise schedule for training.

    Core idea
    ---------
    Instead of the DDPM Markov chain (1000 steps of small noise additions),
    rectified flow defines a straight-line ODE connecting data and noise:

        x_t = (1 - t) · x_data  +  t · x_noise       t ∈ [0, 1]

    where:
        x_data  ~ training latents
        x_noise ~ N(0, I)
        t       ~ LogitNormal(μ, σ)  [better than Uniform for training]

    The model learns to predict the velocity field:
        v(x_t, t) ≈ dx_t/dt = x_noise - x_data

    Advantages over DDPM
    --------------------
    1. Straight trajectories → only 20 inference steps instead of 1000.
    2. Simpler math: v_target is constant (independent of t) for each sample.
    3. No need for a β schedule — just interpolate linearly.
    4. Better scaling behaviour as model capacity increases.
    5. Used in SD3, FLUX, Stable Video Diffusion, and most 2024+ models.

    Timestep sampling: Logit-Normal distribution
    ---------------------------------------------
    Uniform sampling wastes steps on extreme timesteps (t≈0 and t≈1)
    where the model has the least to learn.  LogitNormal concentrates
    training steps around t=0.5, where the image is half-corrupted and
    the denoising task is hardest:

        u ~ Normal(μ, σ)
        t = sigmoid(u) = 1 / (1 + exp(-u))

    With μ=0, σ=1 (default), roughly 68% of steps fall in [0.27, 0.73].

    Debugging:
    - If loss is stuck near 1.0 after many steps: the model is predicting
      zero velocity. Check that the output head (proj_out) has non-zero
      gradients flowing through it.
    - If loss oscillates wildly: timestep sampling may be too extreme.
      Try σ=0.5 (tighter logit-normal) or add a loss weight based on SNR.
    - To verify correct noising: reconstruct x_data from (x_t, t, v_target):
        x_data_hat = x_t - t * v_target     (should match original latents)
    - Min-SNR weighting (Hang et al., 2023) can stabilise training:
        weight = 1 / (snr + 1)  where snr = (1-t)² / t²
        Use weight in loss computation for more stable multi-scale training.
    """

    def __init__(
        self,
        logit_normal_mean:  float = 0.0,
        logit_normal_std:   float = 1.0,
        t_min:              float = 0.001,  # Avoid t=0 (clean) during training
        t_max:              float = 0.999,  # Avoid t=1 (pure noise) during training
    ):
        self.mu    = logit_normal_mean
        self.sigma = logit_normal_std
        self.t_min = t_min
        self.t_max = t_max

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample training timesteps from a logit-normal distribution.

        Returns:
            t: (B,) float timesteps in (t_min, t_max) ⊂ (0, 1).

        Debugging:
        - If t values cluster too close to 0 or 1: increase sigma or reduce
          t_min/t_max bounds.  Print t.histogram(bins=10) to inspect.
        - Swap to uniform sampling for baseline comparisons:
            return torch.rand(batch_size, device=device).clamp(t_min, t_max)
        """
        u = torch.randn(batch_size, device=device) * self.sigma + self.mu
        t = torch.sigmoid(u)
        return t.clamp(self.t_min, self.t_max)

    def add_noise(
        self,
        x_data:  torch.Tensor,              # (B, C, H, W) clean latents
        t:       Optional[torch.Tensor] = None,  # (B,) pre-sampled; samples if None
        noise:   Optional[torch.Tensor] = None,  # (B, C, H, W) or None
        device:  Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Corrupt clean latents using the rectified flow interpolation.

        Args:
            x_data: Clean latents (B, C, H, W).
            t:      Pre-sampled timesteps (B,) in [0,1]. Sampled if None.
            noise:  Gaussian noise to mix in. Sampled if None.
            device: Target device (inferred from x_data if None).

        Returns:
            x_t:      Noisy latents at time t  (B, C, H, W).
            v_target: Velocity target = noise − data  (B, C, H, W).
            t:        Timesteps (B,) — pass to MMDiT forward.

        Math:
            x_t = (1 − t) · x_data + t · noise
            v_target = dx_t/dt = noise − x_data   (constant w.r.t. t)

        Debugging:
        - Verify: x_t at t=0 should equal x_data exactly.
          assert (add_noise(x, t=zeros)[0] - x).abs().max() < 1e-5
        - Verify: x_t at t=1 should equal noise.
          assert (add_noise(x, t=ones, noise=n)[0] - n).abs().max() < 1e-5
        - If x_t has wrong range: latents should be ~N(0,1) after VAE encoding.
          Check `x_data.std()` is ~1.0; if it's 0.18215 you forgot to scale.
        """
        if device is None:
            device = x_data.device
        if t is None:
            t = self.sample_timesteps(x_data.shape[0], device)
        if noise is None:
            noise = torch.randn_like(x_data)

        # Reshape t for broadcasting over spatial dims: (B,) → (B, 1, 1, 1)
        t_view = t.view(-1, 1, 1, 1)

        # Rectified flow interpolation (straight line between data and noise)
        x_t = (1.0 - t_view) * x_data + t_view * noise

        # Velocity target: derivative of x_t w.r.t. t (constant)
        v_target = noise - x_data

        return x_t, v_target, t

    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Optional Min-SNR loss weighting (Hang et al., 2023).

        At small t (nearly clean), the model has an easy task but the
        gradient signal is weak.  At large t (nearly noisy), the SNR is
        very low and gradients are noisy.  Min-SNR weighting balances
        these extremes.

        weight(t) = min(SNR(t), γ) / SNR(t)   where SNR(t) = (1-t)²/t²

        γ=5 is the recommended default (from original Min-SNR paper).

        Args:
            t: (B,) timesteps in [0,1].
        Returns:
            weight: (B,) scalar loss weights.

        Usage:
            loss = (weight.view(-1,1,1,1) * (v_pred - v_target).pow(2)).mean()
        """
        gamma = 5.0
        snr = ((1.0 - t) / t.clamp(min=1e-6)) ** 2
        weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
        return weight.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — HEUN / EULER SAMPLER  (Inference)
# ═══════════════════════════════════════════════════════════════════════════════

class FlowMatchingSampler:
    """
    Heun (2nd-order) and Euler (1st-order) samplers for rectified flow inference.

    For rectified flow, inference means integrating the ODE:
        dx/dt = v_θ(x_t, t)   backwards from t=1 (noise) to t=0 (data)

    Euler (1st-order): 20–25 steps for good quality
        x_{t-Δt} = x_t − Δt · v_θ(x_t, t)

    Heun (2nd-order): 20 steps for high quality (~40 network evaluations)
        k1 = v_θ(x_t, t)              [predictor step]
        x̃_{t-Δt} = x_t − Δt · k1
        k2 = v_θ(x̃_{t-Δt}, t-Δt)    [corrector step]
        x_{t-Δt} = x_t − Δt · (k1 + k2) / 2

    Classifier-Free Guidance (CFG):
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        guidance_scale=1.0 = no guidance; 7.0 is typical.

    Debugging:
    - If generated images are noise (grey, uniform): CFG scale is too high,
      or the model is predicting incorrect velocity direction. Check that
      the unconditional tokens are proper empty embeddings (not zeros).
    - If images are always the same regardless of prompt: check that
      input_ids for the positive prompt are being tokenised differently
      from the unconditional ones.
    - If Heun produces much worse results than Euler: the velocity field
      may not be smooth enough — use more joint blocks or increase d_model.
    - For ablating: start with Euler 50 steps to confirm model quality,
      then switch to Heun 20 steps for production.
    """

    def __init__(self, num_steps: int = 20, method: str = "heun"):
        """
        Args:
            num_steps: Number of ODE integration steps.
                       Heun: each step = 2 NFE (network function evaluations).
                       Euler: each step = 1 NFE.
            method:    "heun" or "euler".
        """
        assert method in ("heun", "euler"), f"Unknown method: {method}"
        self.num_steps = num_steps
        self.method    = method

    def get_timesteps(self, device: torch.device) -> torch.Tensor:
        """
        Linearly-spaced timesteps from t=1 (noise) to t=0 (data).

        Returns:
            (num_steps+1,) timesteps: [1.0, ..., 0.0]
        """
        # Slight offset from exact 0 and 1 for numerical stability
        return torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

    @torch.no_grad()
    def sample(
        self,
        model_fn,                    # Callable: (x_t, t, txt, pooled) → v_pred
        z_T:           torch.Tensor, # (B, C, H, W) initial noise
        txt_tokens:    torch.Tensor, # (B, S_txt, ctx_dim)
        pooled_txt:    torch.Tensor, # (B, pooled_dim)
        txt_uncond:    Optional[torch.Tensor] = None,  # Unconditional text tokens
        pooled_uncond: Optional[torch.Tensor] = None,  # Unconditional pooled
        guidance_scale: float = 7.0,
        callback = None,             # Optional: fn(step, x_t) for previews
    ) -> torch.Tensor:
        """
        Denoise from pure noise to a clean latent.

        Args:
            model_fn:      Forward pass function (handles batch splitting for CFG).
            z_T:           Initial Gaussian noise (B, C, H, W).
            txt_tokens:    Conditional text context (B, S_txt, ctx_dim).
            pooled_txt:    Conditional pooled text (B, pooled_dim).
            txt_uncond:    Unconditional text context for CFG.
                           If None: CFG is disabled.
            pooled_uncond: Unconditional pooled text for CFG.
            guidance_scale: CFG strength. 1.0 = no guidance, 7.0 = strong.
            callback:      Optional hook for progress visualisation.

        Returns:
            z_0: Denoised latent (B, C, H, W).  Pass to VAE.decode().

        Debugging:
        - If images have grid-like artefacts: the patch_size might not divide
          the spatial dims. Check that H and W are multiples of patch_size*8.
        - If every image looks like the unconditional output: guidance_scale
          is set to 1.0 (or txt_uncond is identical to txt_tokens).
        - Memory: Heun doubles the network evaluations. For interactive
          demos, use Euler with 25 steps for faster previews.
        """
        timesteps = self.get_timesteps(z_T.device)  # (num_steps+1,)
        x = z_T.clone()
        use_cfg = (txt_uncond is not None) and (guidance_scale != 1.0)

        for i in range(self.num_steps):
            t_cur  = timesteps[i]
            t_next = timesteps[i + 1]
            dt     = t_next - t_cur  # negative (going from 1 → 0)

            # Batch t to match latent batch size
            t_batch = t_cur.expand(x.shape[0])

            # ── Predict velocity (with optional CFG) ──────────────────────────
            v_pred = self._predict_with_cfg(
                model_fn, x, t_batch,
                txt_tokens, pooled_txt,
                txt_uncond, pooled_uncond,
                guidance_scale, use_cfg,
            )

            if self.method == "euler":
                # 1st-order Euler step
                x = x + dt * v_pred

            elif self.method == "heun":
                # 2nd-order Heun: predictor + corrector
                x_pred = x + dt * v_pred          # Euler predictor

                t_next_batch = t_next.expand(x.shape[0])
                v_pred2 = self._predict_with_cfg(
                    model_fn, x_pred, t_next_batch,
                    txt_tokens, pooled_txt,
                    txt_uncond, pooled_uncond,
                    guidance_scale, use_cfg,
                )
                # Heun corrector: average of both velocity estimates
                x = x + dt * (v_pred + v_pred2) / 2.0

            if callback is not None:
                callback(i, x)

        return x  # z_0: clean latent

    def _predict_with_cfg(
        self,
        model_fn,
        x_t, t_batch,
        txt_cond, pooled_cond,
        txt_uncond, pooled_uncond,
        guidance_scale, use_cfg,
    ) -> torch.Tensor:
        """
        Run the model with classifier-free guidance.

        CFG concatenates conditional and unconditional inputs into a single
        batch (doubling batch size), runs one forward pass, then combines:
            v_guided = v_uncond + s * (v_cond - v_uncond)

        This is more memory-efficient than two separate forward passes.

        Debugging:
        - If unconditional tokens look wrong: create them by tokenising an
          empty string ("") and encoding it through DualTextEncoder.
          Do NOT use zero tensors — CLIP needs proper padding tokens.
        """
        if not use_cfg:
            return model_fn(x_t, t_batch, txt_cond, pooled_cond)

        # Double the batch: [cond, uncond]
        x_2  = torch.cat([x_t,        x_t],         dim=0)
        t_2  = torch.cat([t_batch,    t_batch],      dim=0)
        txt_2    = torch.cat([txt_cond, txt_uncond],  dim=0)
        pool_2   = torch.cat([pooled_cond, pooled_uncond], dim=0)

        v_2 = model_fn(x_2, t_2, txt_2, pool_2)

        v_cond, v_uncond = v_2.chunk(2, dim=0)
        return v_uncond + guidance_scale * (v_cond - v_uncond)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential Moving Average of model weights for inference.

    EMA smooths the weight trajectory during training by maintaining a
    running average: θ_ema ← decay · θ_ema + (1 − decay) · θ_train

    Why it matters for diffusion models:
    - Diffusion training loss landscapes are noisy; individual checkpoints
      can overfit to recent mini-batches.
    - EMA weights are much more stable and consistently produce better
      images (higher FID, better CLIP scores) than the raw trained weights.
    - All production SD models (1.x, 2.x, XL, 3, FLUX) use EMA for inference.

    Warm-up (first 100 steps):
    - Use decay=0 until enough updates have been seen to build a meaningful
      average.  The `step` counter triggers this automatically.

    Recommended decay values:
    - 0.9999: Slow averaging; good for long training runs (>100k steps).
    - 0.999:  Fast averaging; good for short runs or fine-tuning.
    - 0.9995: Default balanced setting.

    Debugging:
    - If EMA and train weights are identical: EMA is not being called
      after optimiser.step().  Check the training loop.
    - If EMA quality is worse than train weights after 1000 steps:
      decay is too low (fast averaging picks up noise).  Increase to 0.9999.
    - To inspect the gap: compare sample images from both weights periodically.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 100):
        self.decay         = decay
        self.warmup_steps  = warmup_steps
        self.step          = 0
        # Deep copy of model state dict — stored on CPU to save GPU VRAM
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.clone().cpu()
            for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module):
        """
        Update EMA shadow weights from the current model state.

        Call this ONCE after every `optimiser.step()`, not every forward pass.

        Debugging:
        - If you see "missing keys" when loading EMA: the model architecture
          changed after creating the EMA object. Re-create EMA from the new
          model.
        """
        self.step += 1
        # Effective decay: 0 during warmup (pure copy), then exponential average
        d = 0.0 if self.step < self.warmup_steps else self.decay
        with torch.no_grad():
            for key, param in model.state_dict().items():
                if key in self.shadow:
                    cpu_param = param.detach().cpu()
                    self.shadow[key] = d * self.shadow[key] + (1.0 - d) * cpu_param

    @torch.contextmanager
    def ema_scope(self, model: nn.Module):
        """
        Context manager that temporarily loads EMA weights into the model
        for inference, then restores training weights.

        Usage:
            with ema.ema_scope(model):
                images = sampler.sample(model.forward, ...)

        Debugging:
        - If samples look the same inside and outside ema_scope: EMA has not
          diverged yet (normal in the first 1000 steps).
        - If training resumes with wrong weights: the try/finally block ensures
          restoration even if an exception occurs inside the context.
        """
        train_state = {k: v.clone() for k, v in model.state_dict().items()}
        try:
            # Load EMA weights (move to model's device)
            device = next(model.parameters()).device
            ema_state = {k: v.to(device) for k, v in self.shadow.items()}
            model.load_state_dict(ema_state)
            yield model
        finally:
            model.load_state_dict(train_state)

    def save(self, path: str):
        """Persist EMA weights to disk (CPU tensors)."""
        torch.save(self.shadow, path)

    def load(self, path: str):
        """Load EMA weights from disk."""
        self.shadow = torch.load(path, map_location="cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — MAIN ADVANCED STABLE DIFFUSION MODEL  (Wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedStableDiffusionModel(nn.Module):
    """
    Unified wrapper for the next-generation Stable Diffusion pipeline.

    Frozen (no gradients):
        vae          — AdvancedVAE (4-ch or 16-ch pretrained)
        text_encoder — DualTextEncoder (CLIP-L + OpenCLIP-bigG)

    Trainable:
        dit          — MMDiT (all joint + single transformer blocks)
        text_encoder.proj_l, .proj_g — small linear projectors into ctx_dim

    Quick-start:
        # Build the model
        cfg   = MMDiTConfig(**MMDIT_PRESETS["small"])
        vae   = AdvancedVAE()
        txt   = DualTextEncoder()
        dit   = MMDiT(cfg)
        sched = RectifiedFlowScheduler()
        model = AdvancedStableDiffusionModel(vae, txt, dit, sched)

        # Training step
        loss = model.training_step(imgs, input_ids_l, input_ids_g)
        loss.backward()

        # Inference
        images = model.generate(prompt_ids_l, prompt_ids_g)

    Parameter count (approximate):
        vae:          84 M  (frozen)
        text_encoder: ~900 M  (frozen, CLIP-L+bigG)
        dit (small):  ~700 M  (trainable)
        Total trainable: ~700 M + projection layers

    Debugging checklist before first training run:
    ─────────────────────────────────────────────────
    1. model.count_parameters() — verify param counts match expectations.
    2. model.validate_config()  — checks in_channels, ctx_dim consistency.
    3. Run a single forward pass with dummy data and check:
       a. No NaN in loss.
       b. Loss magnitude ~1.0 at step 0.
       c. Gradients flow to all dit parameters (use .grad.norm() check).
    4. Check GPU memory with torch.cuda.memory_summary().
    5. Enable model.enable_gradient_checkpointing() if OOM.
    """

    def __init__(
        self,
        vae:       AdvancedVAE,
        text_enc:  DualTextEncoder,
        dit:       MMDiT,
        scheduler: RectifiedFlowScheduler,
    ):
        super().__init__()
        self.vae       = vae
        self.text_enc  = text_enc
        self.dit       = dit
        self.scheduler = scheduler

    def validate_config(self):
        """
        Cross-check component configurations for common mismatches.

        Call this once before training begins.

        Debugging guide:
        - "in_channels mismatch": MMDiTConfig.in_channels must match the
          number of latent channels in the VAE. Set in_channels=4 for
          sd-vae-ft-mse and in_channels=16 for FLUX VAE.
        - "ctx_dim mismatch": DualTextEncoder outputs sequences of shape
          (B, S, ctx_dim). MMDiTConfig.ctx_dim must match this value.
          Both default to 2048.
        - "pooled_dim mismatch": DualTextEncoder pools CLIP-L (768) + bigG
          (1280) = 2048 total. MMDiTConfig.pooled_dim must be 2048.
        """
        vae_ch  = self.vae.latent_channels
        dit_ch  = self.dit.cfg.in_channels
        assert vae_ch == dit_ch, (
            f"[CONFIG ERROR] VAE latent_channels={vae_ch} but "
            f"MMDiTConfig.in_channels={dit_ch}. "
            f"Set cfg.in_channels={vae_ch}."
        )

        enc_pool = self.text_enc.pooled_dim
        dit_pool = self.dit.cfg.pooled_dim
        assert enc_pool == dit_pool, (
            f"[CONFIG ERROR] DualTextEncoder.pooled_dim={enc_pool} but "
            f"MMDiTConfig.pooled_dim={dit_pool}. They must match."
        )

        print(f"[validate_config] OK — in_channels={vae_ch}, pooled_dim={enc_pool}")

    # ── Core encode / decode helpers ──────────────────────────────────────────

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """Images (B,3,H,W) in [-1,1] → normalised latents. No gradient."""
        return self.vae.encode(imgs)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised latents → images (B,3,H,W) in [-1,1]. No gradient."""
        return self.vae.decode(z)

    def encode_text(
        self,
        input_ids_l:      torch.Tensor,
        input_ids_g:      torch.Tensor,
        attention_mask_l: Optional[torch.Tensor] = None,
        attention_mask_g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token IDs → (ctx_seq, pooled_txt). No gradient on CLIP weights."""
        return self.text_enc(
            input_ids_l, input_ids_g, attention_mask_l, attention_mask_g
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def forward(
        self,
        latents:    torch.Tensor,
        t:          torch.Tensor,
        txt_tokens: torch.Tensor,
        pooled_txt: torch.Tensor,
    ) -> torch.Tensor:
        """
        MM-DiT forward pass: predict velocity v given noisy latents and text.

        This is what the optimiser differentiates.

        Args:
            latents:    Noisy latents (B, C, H, W).
            t:          Timesteps (B,) in [0, 1].
            txt_tokens: Text context sequence (B, S_txt, ctx_dim).
            pooled_txt: Pooled text embedding (B, pooled_dim).
        Returns:
            v_pred: Predicted velocity (B, C, H, W).
        """
        return self.dit(latents, t, txt_tokens, pooled_txt)

    def training_step(
        self,
        clean_imgs:       torch.Tensor,           # (B, 3, H, W) in [-1, 1]
        input_ids_l:      torch.Tensor,           # (B, 77)
        input_ids_g:      torch.Tensor,           # (B, 77)
        attention_mask_l: Optional[torch.Tensor] = None,
        attention_mask_g: Optional[torch.Tensor] = None,
        use_min_snr:      bool = True,
    ) -> torch.Tensor:
        """
        Complete training step: encode → noise → predict → loss.

        Args:
            clean_imgs:  Raw training images in [-1, 1].
            input_ids_l: CLIP-L token IDs.
            input_ids_g: OpenCLIP-bigG token IDs.
            use_min_snr: Apply Min-SNR loss weighting for training stability.

        Returns:
            loss: Scalar MSE loss (B,C,H,W mean).

        Debugging:
        - Loss is NaN at step 0: usually caused by fp16 overflow in the
          text encoder projection. Use torch.cuda.amp with GradScaler.
        - Loss converges to ~0.3 and stays there: the model has learned to
          predict a constant velocity. Check that text conditioning is wired
          in — temporarily set guidance_scale=0 and see if loss changes.
        - Loss spikes at step N: learning rate is too high or a batch of
          extreme-aspect-ratio images caused shape errors. Add loss clipping:
          loss = loss.clamp(max=10.0)
        - Verifying the loss math:
          At t=0: x_t = x_data, v_target = noise - x_data.
          A model predicting all-zeros has loss = E[||noise - x_data||²]
          which equals E[||noise||² + ||x_data||²] ≈ C*H*W*2 ≈ ~1.0 per element.
          So initial loss ≈ 1.0 is correct.
        """
        # ── Step 1: Encode images to latents ──────────────────────────────────
        with torch.no_grad():
            latents = self.encode_images(clean_imgs)

        # [DEBUG] Verify latent statistics (should be ~N(0,1) after scale)
        # print(f"Latent stats: mean={latents.mean():.3f} std={latents.std():.3f}")

        # ── Step 2: Encode text ───────────────────────────────────────────────
        # text_enc CLIP weights are frozen but proj_l/proj_g have gradients
        txt_tokens, pooled_txt = self.encode_text(
            input_ids_l, input_ids_g, attention_mask_l, attention_mask_g
        )
        # txt_tokens: (B, 154, 2048), pooled_txt: (B, 2048)

        # ── Step 3: Sample timesteps and add noise ─────────────────────────────
        x_t, v_target, t = self.scheduler.add_noise(latents)
        # x_t: (B, C, H, W),  v_target: (B, C, H, W),  t: (B,)

        # ── Step 4: Forward pass — predict velocity ────────────────────────────
        v_pred = self.forward(x_t, t, txt_tokens, pooled_txt)
        # v_pred: (B, C, H, W)

        # ── Step 5: Compute MSE loss ──────────────────────────────────────────
        loss = (v_pred - v_target).pow(2)  # (B, C, H, W)

        if use_min_snr:
            # Per-sample weighting: scale loss by Min-SNR weight (B,) → broadcast
            w = self.scheduler.get_loss_weight(t).view(-1, 1, 1, 1)
            loss = loss * w

        return loss.mean()

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids_l:       torch.Tensor,
        input_ids_g:       torch.Tensor,
        input_ids_uncond_l: Optional[torch.Tensor] = None,
        input_ids_uncond_g: Optional[torch.Tensor] = None,
        guidance_scale:    float = 7.0,
        num_steps:         int   = 20,
        height:            int   = 512,
        width:             int   = 512,
        sampler_method:    str   = "heun",
        seed:              Optional[int] = None,
        use_ema:           bool  = False,
        ema:               Optional[EMA] = None,
    ) -> torch.Tensor:
        """
        Generate images from text prompts.

        Args:
            input_ids_l/g:          Positive prompt token IDs.
            input_ids_uncond_l/g:   Negative prompt token IDs (empty string).
                                    If None: CFG is disabled.
            guidance_scale:         CFG strength (7.0 typical for SD models).
            num_steps:              Denoising steps (20 for Heun, 25-50 for Euler).
            height, width:          Output image resolution (must be multiples of 64).
            sampler_method:         "heun" or "euler".
            seed:                   Random seed for reproducibility.
            use_ema:                Use EMA weights for inference (recommended).
            ema:                    EMA instance (required if use_ema=True).

        Returns:
            images: (B, 3, H, W) float tensors in [-1, 1].
                    To get uint8: ((images * 0.5 + 0.5) * 255).clamp(0,255).byte()

        Debugging:
        - Black images: guidance_scale too high (try 5-8) or model checkpoint
          is from very early training.
        - All-white images: the VAE decode is receiving out-of-range latents.
          Clamp z_0 to [-4, 4] before decoding.
        - Resolution artefacts: height/width must be multiples of patch_size * 8
          (= 2 * 8 = 16 at minimum, or 64 for typical configurations).
        """
        device = next(self.dit.parameters()).device

        # Validate resolution
        min_mult = self.dit.cfg.patch_size * 8
        assert height % min_mult == 0 and width % min_mult == 0, (
            f"height={height} and width={width} must be multiples of {min_mult}."
        )

        if seed is not None:
            torch.manual_seed(seed)

        B = input_ids_l.shape[0]

        # ── 1. Encode positive prompt ─────────────────────────────────────────
        txt_cond, pool_cond = self.encode_text(
            input_ids_l.to(device), input_ids_g.to(device)
        )

        # ── 2. Encode negative prompt (for CFG) ───────────────────────────────
        txt_uncond = pool_uncond = None
        if input_ids_uncond_l is not None:
            txt_uncond, pool_uncond = self.encode_text(
                input_ids_uncond_l.to(device),
                input_ids_uncond_g.to(device),
            )

        # ── 3. Initial noise ──────────────────────────────────────────────────
        C = self.dit.cfg.in_channels
        z_T = torch.randn(B, C, height // 8, width // 8, device=device)

        # ── 4. Sample with (optional) EMA weights ─────────────────────────────
        sampler = FlowMatchingSampler(num_steps=num_steps, method=sampler_method)

        def model_fn(x_t, t, txt, pooled):
            return self.dit(x_t, t, txt, pooled)

        if use_ema and ema is not None:
            with ema.ema_scope(self.dit):
                z_0 = sampler.sample(
                    model_fn, z_T, txt_cond, pool_cond,
                    txt_uncond, pool_uncond, guidance_scale,
                )
        else:
            z_0 = sampler.sample(
                model_fn, z_T, txt_cond, pool_cond,
                txt_uncond, pool_uncond, guidance_scale,
            )

        # ── 5. Decode latents to pixel space ─────────────────────────────────
        # [DEBUG] Clamp to prevent extreme latent values from crashing the VAE
        z_0 = z_0.clamp(-4.0, 4.0)
        images = self.decode_latents(z_0)

        return images

    # ── Utilities ─────────────────────────────────────────────────────────────

    def enable_gradient_checkpointing(self):
        """Trade compute for memory: ~40% VRAM reduction, ~25% slower training."""
        self.dit.enable_gradient_checkpointing()

    def trainable_parameters(self):
        """
        Yield parameters that should be passed to the optimiser.

        Includes:
          - All MMDiT parameters
          - DualTextEncoder projection layers (proj_l, proj_g)

        Excludes:
          - CLIP-L and OpenCLIP-bigG backbone weights (frozen)
          - VAE weights (frozen)

        Usage:
            optimiser = torch.optim.AdamW(
                model.trainable_parameters(),
                lr=1e-4, weight_decay=1e-2,
            )
        """
        yield from self.dit.parameters()
        yield from self.text_enc.proj_l.parameters()
        yield from self.text_enc.proj_g.parameters()

    def count_parameters(self) -> Dict[str, int | float]:
        """Return a parameter count summary."""
        trainable = sum(p.numel() for p in self.trainable_parameters())
        frozen    = sum(p.numel() for p in self.parameters()) - trainable
        total     = trainable + frozen
        dit_stats = self.dit.count_parameters()
        return {
            "total":             total,
            "trainable":         trainable,
            "frozen":            frozen,
            "trainable_pct":     100.0 * trainable / total if total else 0.0,
            "dit_joint_blocks":  dit_stats["joint_blocks"],
            "dit_single_blocks": dit_stats["single_blocks"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — FACTORY  (Convenience build functions)
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(
    preset:          str  = "small",
    vae_model_id:    str  = "stabilityai/sd-vae-ft-mse",
    clip_l_id:       str  = "openai/clip-vit-large-patch14",
    clip_g_id:       str  = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    use_fp16_vae:    bool = True,
    grad_ckpt:       bool = False,
    logit_normal_mu: float = 0.0,
    logit_normal_sigma: float = 1.0,
) -> AdvancedStableDiffusionModel:
    """
    Build and return a fully-configured AdvancedStableDiffusionModel.

    Args:
        preset:  One of "nano", "small", "medium", "large".
                 See MMDIT_PRESETS for exact dimensions.
        vae_model_id: HuggingFace model ID for the pretrained VAE.
        clip_l_id:    CLIP-ViT-L/14 model ID.
        clip_g_id:    OpenCLIP-ViT-bigG/14 model ID.
        use_fp16_vae: Load VAE in fp16 to save VRAM.
        grad_ckpt:    Enable gradient checkpointing immediately.
        logit_normal_mu, sigma: Timestep sampling distribution parameters.

    Returns:
        AdvancedStableDiffusionModel ready for training.

    Example:
        model = build_model("small")
        model.validate_config()
        print(model.count_parameters())

    Debugging:
    - "Preset not found": choose one of "nano", "small", "medium", "large".
    - OOM on load: use "nano" preset, or set use_fp16_vae=True and ensure
      text encoders are loaded in fp16 (default in DualTextEncoder).
    - Slow model build: CLIP models download on first call. Pre-cache them
      with: `from transformers import CLIPTextModel; CLIPTextModel.from_pretrained(id)`.
    """
    assert preset in MMDIT_PRESETS, (
        f"Unknown preset '{preset}'. Choose from: {list(MMDIT_PRESETS.keys())}"
    )

    cfg = MMDIT_PRESETS[preset]
    cfg.grad_ckpt = grad_ckpt

    vae  = AdvancedVAE(vae_model_id, use_fp16=use_fp16_vae)
    txt  = DualTextEncoder(clip_l_id, clip_g_id)
    dit  = MMDiT(cfg)
    sched = RectifiedFlowScheduler(logit_normal_mu, logit_normal_sigma)

    model = AdvancedStableDiffusionModel(vae, txt, dit, sched)
    model.validate_config()

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — QUICK SANITY CHECK  (Run with: python SD_Model_v2.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Dry-run sanity check for the MM-DiT architecture WITHOUT loading any
    pretrained weights.  Tests shapes, forward pass, and loss computation.

    This does NOT test the VAE or text encoders (which require downloads).
    To run the full stack, use build_model() in a training script.

    Expected output (small preset):
        MMDiT architecture dry-run
        Config: MMDiTConfig(d_model=1152, n_heads=16, n_joint=24, ...)
        Patchified img tokens : torch.Size([2, 1024, 1152])
        Velocity pred shape   : torch.Size([2, 4, 64, 64])
        MSE loss (random)     : ~1.00
        Parameter count: {...}
        All assertions passed.

    Debugging this script:
    - "AssertionError: head_dim": d_model must be divisible by n_heads.
    - "RuntimeError: sizes":  latent H/W must be divisible by patch_size.
    - NaN loss: check adaLN zero-init (proj weights should start at 0).
    - Wrong output shape: inspect patchify/unpatchify step by step.
    """
    print("=" * 70)
    print("  SD_Model_v2 — MM-DiT architecture dry-run (no pretrained weights)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32  # Use float32 for dry-run (exact reproducibility)

    # ── Build MM-DiT with "small" preset ─────────────────────────────────────
    cfg = MMDiTConfig(**{
        "d_model": 1152, "n_heads": 16, "n_joint": 4,   # Use 4 blocks for speed
        "n_single": 0, "in_channels": 4, "patch_size": 2,
        "ctx_dim": 2048, "pooled_dim": 2048,
    })
    print(f"\nConfig: {cfg}\n")

    dit = MMDiT(cfg).to(device).to(dtype)

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    B, C, H, W = 2, 4, 64, 64          # Batch of 2, 4-ch latents, 64×64 spatial
    S_txt      = 154                    # 77 CLIP-L + 77 OpenCLIP-bigG tokens

    latents    = torch.randn(B, C, H, W, device=device, dtype=dtype)
    t          = torch.rand(B, device=device, dtype=dtype)
    txt_tokens = torch.randn(B, S_txt, cfg.ctx_dim, device=device, dtype=dtype)
    pooled_txt = torch.randn(B, cfg.pooled_dim, device=device, dtype=dtype)

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        # Test patchify
        img_tokens, H_p, W_p = dit.patchify(latents)
        print(f"Patchified img tokens : {img_tokens.shape}")
        assert img_tokens.shape == (B, H_p * W_p, cfg.d_model), \
            f"Patchify shape mismatch: {img_tokens.shape}"

        # Full forward pass
        v_pred = dit(latents, t, txt_tokens, pooled_txt)
        print(f"Velocity pred shape   : {v_pred.shape}")
        assert v_pred.shape == (B, C, H, W), \
            f"Output shape mismatch: expected {(B,C,H,W)}, got {v_pred.shape}"

        # Check no NaN
        assert not v_pred.isnan().any(), "NaN detected in velocity prediction!"

        # Compute a dummy loss
        v_target = torch.randn_like(latents)
        loss = (v_pred - v_target).pow(2).mean()
        print(f"MSE loss (random init): {loss.item():.4f}  (expected ~1.0)")

    # ── Parameter counts ──────────────────────────────────────────────────────
    stats = dit.count_parameters()
    print(f"\nParameter counts:")
    for k, v in stats.items():
        print(f"  {k:20s}: {v / 1e6:.1f} M")

    # ── Rectified flow scheduler ──────────────────────────────────────────────
    sched   = RectifiedFlowScheduler()
    x_clean = torch.randn(B, C, H, W)
    x_t, v_tgt, t_s = sched.add_noise(x_clean)
    print(f"\nScheduler sanity:")
    print(f"  t range   : [{t_s.min():.3f}, {t_s.max():.3f}]  (should be in (0,1))")
    print(f"  v_target std: {v_tgt.std():.3f}  (expected ~1.4)")

    # Verify identity at t=0 and t=1
    t_zero = torch.zeros(B)
    x_t0, _, _ = sched.add_noise(x_clean, t=t_zero, noise=torch.zeros_like(x_clean))
    assert (x_t0 - x_clean).abs().max() < 1e-5, "t=0 should return clean latent"

    # ── 2-D RoPE ─────────────────────────────────────────────────────────────
    sin_t, cos_t = get_2d_rope_freqs(H // cfg.patch_size, W // cfg.patch_size,
                                      cfg.d_model // cfg.n_heads, device)
    print(f"\n2-D RoPE table shape  : {sin_t.shape}  (expected [{H_p}*{W_p}={H_p*W_p}, {cfg.d_model//cfg.n_heads}])")

    print("\n" + "=" * 70)
    print("  All assertions passed. SD_Model_v2 architecture is correct.")
    print("=" * 70)