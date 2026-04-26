"""
SD_Model.py
===========
Core model architecture for Stable Diffusion training on 2× RTX 5090.

Components
----------
PretrainedVAE
    Frozen pretrained VAE (stabilityai/sd-vae-ft-mse).
    Encodes (B, 3, H, W) images → (B, 4, H/8, W/8) scaled latents and back.

PretrainedCLIPTextEncoder
    Frozen pretrained CLIP text encoder (openai/clip-vit-large-patch14).
    Produces (B, 77, 768) sequence embeddings for UNet cross-attention.

UNetModel
    Trainable denoising UNet operating entirely in latent space.
    Input:  noisy latent (B, 4, 64, 64) + timestep (B,) + text context (B, 77, 768)
    Output: predicted noise ε (B, 4, 64, 64)

DDPMScheduler
    DDPM (Ho et al., 2020) forward-diffusion noise schedule used during training.

DDIMScheduler
    DDIM (Song et al., 2020) deterministic sampler used during inference.

StableDiffusionModel
    Unified wrapper that wires VAE, CLIP, UNet, and DDPM scheduler together.
    Only the UNet is trained; VAE and CLIP are always frozen.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ═══════════════════════════════════════════════════════════════════════════════
# VAE WRAPPER  (Pretrained — Frozen)
# ═══════════════════════════════════════════════════════════════════════════════

class PretrainedVAE(nn.Module):
    """
    Optimized wrapper around a pretrained HuggingFace AutoencoderKL.

    The VAE is permanently frozen — its weights are never updated.
    The scale factor (0.18215) is the empirical std of the LAION-2B latent
    distribution and normalises latents to unit variance for stable training.

    Optimizations:
    - Direct encoder access (bypasses latent_dist.sample() overhead)
    - Uses mean instead of sampling for deterministic, faster inference
    - fp16 support for faster computation
    """

    SCALE_FACTOR = 0.18215

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", use_fp16: bool = True):
        super().__init__()
        from diffusers import AutoencoderKL

        # Load with fp16 for faster inference if requested
        if use_fp16:
            self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            self.vae = AutoencoderKL.from_pretrained(model_id)

        self.vae.requires_grad_(False)
        self.vae.eval()
        self.use_fp16 = use_fp16
        self.scale_factor = self.SCALE_FACTOR

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to scaled latents using the HF AutoencoderKL API.

        Args:
            x: Float tensor in [-1, 1], shape (B, 3, H, W).
        Returns:
            Scaled latents (B, 4, H//8, W//8).
        """
        dtype = torch.float16 if self.use_fp16 else x.dtype
        mean = self.vae.encode(x.to(dtype=dtype)).latent_dist.mean
        return mean * self.scale_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode scaled latents back to pixel images.

        Args:
            z: Scaled latents (B, 4, H//8, W//8).
        Returns:
            Float tensor in [-1, 1], shape (B, 3, H, W).
        """
        dtype = torch.float16 if self.use_fp16 else z.dtype
        z = z.to(dtype=dtype) / self.scale_factor
        return self.vae.decode(z).sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode → Decode round-trip (useful for sanity-checking only)."""
        return self.decode(self.encode(x))


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP TEXT ENCODER WRAPPER  (Pretrained — Frozen)
# ═══════════════════════════════════════════════════════════════════════════════

class PretrainedCLIPTextEncoder(nn.Module):
    """
    Thin wrapper around a pretrained HuggingFace CLIPTextModel.

    The CLIP encoder is permanently frozen.  Only the sequence output
    (last_hidden_state) is used by the UNet cross-attention layers;
    the pooled CLS output is also returned for optional global conditioning.

    Output shapes:
        last_hidden_state : (B, 77, 768) — per-token embeddings
        pooler_output     : (B, 768)     — CLS token projection
    """

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPTextModel
        self.model = CLIPTextModel.from_pretrained(model_id)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      (B, 77) tokenised text.
            attention_mask: (B, 77) — 1 for real tokens, 0 for padding.
        Returns:
            sequence_out: (B, 77, 768) used by UNet cross-attention.
            pooled_out:   (B, 768)    CLS token projection.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, out.pooler_output


# ═══════════════════════════════════════════════════════════════════════════════
# UNET BUILDING BLOCKS  (Trainable)
# ═══════════════════════════════════════════════════════════════════════════════

class ResNetBlock(nn.Module):
    """
    Standard residual block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv,
    with a 1×1 projection on the skip path when channel counts change.

    The final conv is zero-initialised so the block is an identity at
    the start of training, which stabilises early gradient flow.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for 2D spatial feature maps using PyTorch SDPA.

    The (B, C, H, W) feature map is reshaped to a sequence of H*W tokens,
    attention is computed, and the result is projected back to (B, C, H, W).
    A fused QKV projection keeps memory access patterns cache-friendly.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels={channels} must be divisible by num_heads={num_heads}"
        )
        self.norm      = nn.GroupNorm(32, channels)
        self.qkv       = nn.Conv2d(channels, channels * 3, 1)   # fused Q, K, V projection
        self.proj      = nn.Conv2d(channels, channels, 1)       # output projection
        self.num_heads = num_heads
        self.scale     = (channels // num_heads) ** -0.5

        # Zero-init output projection → block starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Project and split into Q, K, V; reshape to (B, heads, seq=h*w, head_dim)
        qkv = (
            self.qkv(self.norm(x))
            .view(b, 3, self.num_heads, c // self.num_heads, h * w)
            .permute(1, 0, 2, 4, 3)  # → (3, B, heads, seq, head_dim)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, h*w, head_dim)

        # Flash / memory-efficient attention via torch's SDPA dispatcher
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # Merge heads back: (B, heads, head_dim, h*w) → (B, C, H, W)
        out = attn.transpose(2, 3).reshape(b, c, h, w)
        return self.proj(out) + x  # residual


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention between spatial features (query) and a
    context sequence such as CLIP text embeddings (key / value).

    When *ctx* is None the module falls back to self-attention, which is
    how `TransformerBlock` reuses this class for its self-attention layer.
    """

    def __init__(self, q_dim: int, c_dim: int, heads: int = 8):
        super().__init__()
        assert q_dim % heads == 0, f"q_dim={q_dim} must be divisible by heads={heads}"
        self.heads    = heads
        self.head_dim = q_dim // heads
        self.scale    = self.head_dim ** -0.5
        self.to_q   = nn.Linear(q_dim, q_dim, bias=False)
        self.to_k   = nn.Linear(c_dim, q_dim, bias=False)
        self.to_v   = nn.Linear(c_dim, q_dim, bias=False)
        self.to_out = nn.Linear(q_dim, q_dim)

        # Zero-init output for identity initialisation
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        h = self.heads
        ctx = x if ctx is None else ctx  # self-attention when no context given

        # Project and reshape to (B, heads, seq, head_dim)
        q = self.to_q(x).view(b, n, h, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).view(b, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).view(b, -1, h, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm:
        x → self-attention → cross-attention (text conditioning) → MLP

    Pre-norm (LayerNorm before each sub-layer) is more stable than post-norm
    at large model scales and is standard in modern diffusion transformers.
    """

    def __init__(self, dim: int, ctx_dim: int, heads: int = 8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.self_attn  = CrossAttention(dim, dim,     heads)   # self-attn (ctx=x)
        self.cross_attn = CrossAttention(dim, ctx_dim, heads)   # text cross-attn
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # Zero-init last MLP layer → block starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))           # self-attention
        if ctx is not None:
            x = x + self.cross_attn(self.ln2(x), ctx)  # text cross-attention
        return x + self.mlp(self.ln3(x))              # feed-forward MLP


class SpatialTransformer(nn.Module):
    """
    Applies a `TransformerBlock` to a 2D spatial feature map.

    Layout:
        (B, C, H, W)
            → GroupNorm + 1×1 proj
            → flatten to sequence (B, H*W, C)
            → TransformerBlock (self-attn + cross-attn + MLP)
            → reshape back (B, C, H, W)
            → 1×1 proj + residual

    The output projection is zero-initialised so the block starts as a
    residual identity, which keeps early training stable.
    """

    def __init__(self, ch: int, ctx_dim: int, heads: int = 8):
        super().__init__()
        self.norm     = nn.GroupNorm(32, ch)
        self.proj_in  = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)
        self.tfm      = TransformerBlock(ch, ctx_dim, heads)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        # Flatten spatial dims into a sequence of tokens
        x = self.proj_in(self.norm(x)).view(b, c, h * w).transpose(1, 2)  # (B, h*w, C)
        x = self.tfm(x, ctx)
        # Restore spatial shape and apply output projection
        x = self.proj_out(x.transpose(1, 2).view(b, c, h, w))
        return x + res


class UNetResBlock(nn.Module):
    """
    Core UNet building block: two conv layers conditioned on timestep embedding,
    optionally followed by a SpatialTransformer for text conditioning.

    Timestep conditioning is added as a channel-wise bias after the first conv:
        h = conv1(x) + time_mlp(t_emb)[:, :, None, None]
    which broadcasts the (B, out_ch) timestep vector over spatial dimensions.
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        t_dim:    int,
        ctx_dim:  int,
        heads:    int  = 8,
        use_attn: bool = True,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.use_attn  = use_attn
        self.grad_ckpt = grad_ckpt

        # SiLU activation before the linear keeps the projection non-linear
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.norm1    = nn.GroupNorm(32, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(32, out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if use_attn:
            self.tfm = SpatialTransformer(out_ch, ctx_dim, heads)

        # Zero-init final conv → identity residual at initialisation
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _forward(
        self,
        x:     torch.Tensor,
        t_emb: torch.Tensor,
        ctx:   Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Timestep bias is broadcast over (H, W) via [:, :, None, None]
        h = self.conv1(F.silu(self.norm1(x))) + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.use_attn and ctx is not None:
            h = self.tfm(h, ctx)
        return h + self.skip(x)

    def forward(
        self,
        x:     torch.Tensor,
        t_emb: torch.Tensor,
        ctx:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.grad_ckpt and self.training:
            # use_reentrant=False avoids the deprecated reentrant autograd path
            return checkpoint(self._forward, x, t_emb, ctx, use_reentrant=False)
        return self._forward(x, t_emb, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# UNET MODEL  (Trainable — The Heart of Stable Diffusion)
# ═══════════════════════════════════════════════════════════════════════════════

class UNetModel(nn.Module):
    """
    Denoising UNet that operates entirely in VAE latent space.

    Architecture overview
    ---------------------
    Encoder (4 stages, downsampling):
        stage 0: ch×1  = 320  ch — no attention (highest res, 64×64 spatial)
        stage 1: ch×2  = 640  ch — attention
        stage 2: ch×4  = 1280 ch — attention
        stage 3: ch×4  = 1280 ch — attention

    Bottleneck:
        Two UNetResBlocks at 1280 ch; first has attention, second does not.

    Decoder (4 stages, upsampling):
        Mirror of encoder with skip connections concatenated from encoder outputs.

    Skip connections carry feature maps from each encoder ResBlock output to the
    corresponding decoder ResBlock input (concatenated along the channel dim).

    Default config (ch=320, res_blks=2, ch_mults=(1,2,4,4)):
        Total parameters: ~860 M  (matches SD 1.x UNet)
    """

    def __init__(
        self,
        in_ch:     int   = 4,          # VAE latent channels
        out_ch:    int   = 4,          # Must match in_ch for noise prediction
        ch:        int   = 320,        # Base channel width
        res_blks:  int   = 2,          # ResBlocks per encoder / decoder stage
        attn_lvls: tuple = (1, 2, 3),  # Stages that include spatial attention
        ch_mults:  tuple = (1, 2, 4, 4),
        heads:     int   = 8,
        t_dim:     int   = 320,        # Time embedding MLP width
        ctx_dim:   int   = 768,        # CLIP sequence embedding dimension
        grad_ckpt: bool  = False,
    ):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.t_dim = t_dim

        # Sinusoidal time embedding → 2-layer MLP projection
        # Input dimension equals t_dim; hidden dimension is t_dim * 4
        self.t_emb = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        # ── Encoder (Downsampling) ──────────────────────────────────────────
        self.downs = nn.ModuleList()
        # ch_list stores output channel counts of every ResBlock for skip connections
        ch_list = []
        cur_ch  = ch

        for i, mult in enumerate(ch_mults):
            next_ch = ch * mult
            for _ in range(res_blks):
                blk = UNetResBlock(
                    cur_ch, next_ch, t_dim, ctx_dim, heads,
                    use_attn=(i in attn_lvls), grad_ckpt=grad_ckpt,
                )
                self.downs.append(blk)
                cur_ch = next_ch
                ch_list.append(cur_ch)
            if i < len(ch_mults) - 1:
                # Strided 3×3 conv halves spatial resolution; no skip saved here
                self.downs.append(nn.Conv2d(cur_ch, cur_ch, 3, stride=2, padding=1))

        # ── Bottleneck ──────────────────────────────────────────────────────
        self.mid = nn.ModuleList([
            UNetResBlock(cur_ch, cur_ch, t_dim, ctx_dim, heads, use_attn=True,  grad_ckpt=grad_ckpt),
            UNetResBlock(cur_ch, cur_ch, t_dim, ctx_dim, heads, use_attn=False, grad_ckpt=grad_ckpt),
        ])

        # ── Decoder (Upsampling) ────────────────────────────────────────────
        self.ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(ch_mults))):
            next_ch = ch * mult
            for _ in range(res_blks):
                skip_ch = ch_list.pop()  # channels of the matching encoder skip
                blk = UNetResBlock(
                    cur_ch + skip_ch, next_ch, t_dim, ctx_dim, heads,
                    use_attn=(i in attn_lvls), grad_ckpt=grad_ckpt,
                )
                self.ups.append(blk)
                cur_ch = next_ch
            if i > 0:
                # Transposed conv doubles spatial resolution
                self.ups.append(nn.ConvTranspose2d(cur_ch, cur_ch, 4, stride=2, padding=1))

        # Output normalisation + projection back to latent channel count
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

        # Zero-init → network starts by predicting zero noise (stable init)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all UNetResBlocks to trade compute for VRAM."""
        self.grad_ckpt = True
        for module in self.modules():
            if isinstance(module, UNetResBlock):
                module.grad_ckpt = True

    def _sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map scalar timestep indices to sinusoidal embedding vectors.

        Uses the standard formulation from Vaswani et al. (2017):
            PE(t, 2i)   = sin(t / 10000^(2i/d))
            PE(t, 2i+1) = cos(t / 10000^(2i/d))

        Args:
            t: Timestep indices (B,) in [0, num_train_timesteps - 1].
        Returns:
            Embedding vectors (B, t_dim).
        """
        half = self.t_dim // 2
        # Logarithmically-spaced frequencies, one per embedding dimension
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(math.log(10000) / (half - 1))
        )
        # Outer product: (B, 1) × (1, half) → (B, half)
        angles = t.float()[:, None] * freq[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(
        self,
        x:   torch.Tensor,
        t:   torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:   Noisy latent (B, 4, H, W).
            t:   Timestep indices (B,) — long integers in [0, 999].
            ctx: CLIP text embeddings (B, 77, 768); None disables text conditioning.
        Returns:
            Predicted noise ε (B, 4, H, W).
        """
        # 1. Time embedding: sinusoidal → MLP projection → (B, t_dim)
        t_emb = self.t_emb(self._sinusoidal_time_embedding(t))

        # 2. Initial spatial projection: (B, 4, H, W) → (B, ch, H, W)
        h = self.conv_in(x)

        # 3. Encoder — save ResBlock outputs as skip connections
        skips = []
        for module in self.downs:
            if isinstance(module, UNetResBlock):
                h = module(h, t_emb, ctx)
                skips.append(h)
            else:
                h = module(h)   # strided conv downsampling — no skip saved

        # 4. Bottleneck
        for module in self.mid:
            h = module(h, t_emb, ctx)

        # 5. Decoder — pop and concatenate skips along the channel dimension
        for module in self.ups:
            if isinstance(module, UNetResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = module(h, t_emb, ctx)
            else:
                h = module(h)   # transposed conv upsampling

        # 6. Output projection: normalise → SiLU → conv → (B, 4, H, W)
        return self.conv_out(F.silu(self.norm_out(h)))


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """
    DDPM (Ho et al., 2020) noise schedule for training.

    Forward (noising) process at timestep t:
        z_t = √ᾱ_t · z_0 + √(1 − ᾱ_t) · ε,    ε ~ N(0, I)

    where ᾱ_t = ∏ᵢ₌₁ᵗ (1 − βᵢ) is the cumulative noise factor.

    The UNet is trained to predict ε (epsilon-prediction objective).
    Scaled-linear beta schedule is used by default as it empirically
    produces better image quality than linear for latent diffusion models.
    """

    def __init__(
        self,
        steps:      int   = 1000,
        beta_start: float = 0.00085,
        beta_end:   float = 0.012,
        schedule:   str   = "scaled_linear",
    ):
        self.num_train_timesteps = steps

        if schedule == "scaled_linear":
            # SD 1.x schedule: interpolate √β linearly, then square.
            # This concentrates small beta values near t=0 for fine details.
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, steps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'. Choose 'scaled_linear' or 'linear'.")

        self.betas                        = betas
        self.alphas                       = 1.0 - betas
        self.alphas_cumprod               = torch.cumprod(self.alphas, 0)
        self.sqrt_alphas_cumprod          = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

    def add_noise(
        self,
        x:     torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt clean latents *x* at diffusion timestep *t*.

        Args:
            x:     Clean latents (B, 4, 64, 64).
            t:     Timestep indices (B,) in [0, steps − 1].
            noise: Optional pre-generated Gaussian noise; sampled if None.
        Returns:
            (noisy_x, noise) — both (B, 4, 64, 64).
            The returned *noise* is the supervision target for MSE training.
        """
        if noise is None:
            noise = torch.randn_like(x)
        device      = x.device
        sqrt_alpha  = self.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        sqrt_sigma  = self.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_sigma * noise, noise

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep *t*.
        SNR(t) = ᾱ_t / (1 − ᾱ_t).  Useful for Min-SNR loss weighting.
        """
        acp = self.alphas_cumprod.to(t.device)[t]
        return acp / torch.clamp(1.0 - acp, min=1e-6)

    def get_velocity(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity target for v-prediction (Salimans & Ho, 2022).
        v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0
        """
        device = x_0.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        sqrt_sigma = self.sqrt_one_minus_alphas_cumprod.to(device)[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_sigma * x_0

    def to(self, device):
        """Move all schedule tensors to *device* for zero-copy indexing."""
        self.betas                         = self.betas.to(device)
        self.alphas                        = self.alphas.to(device)
        self.alphas_cumprod                = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod           = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self


class DDIMScheduler:
    """
    DDIM (Song et al., 2020) deterministic sampler for inference.

    DDIM skips timesteps, requiring only 25–50 forward passes instead of 1000,
    while producing comparable quality.  Setting eta=0 makes sampling fully
    deterministic (same noise → same image every run).

    Denoising update at each step:
        1. Predict x̂_0 from x_t:  x̂_0 = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t
        2. x_{t−1} = √ᾱ_{t−1}·x̂_0 + √(1−ᾱ_{t−1})·ε_θ  (eta=0)
    """

    def __init__(
        self,
        steps:      int   = 1000,
        beta_start: float = 0.00085,
        beta_end:   float = 0.012,
        schedule:   str   = "scaled_linear",
    ):
        self.num_train_timesteps = steps

        if schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, steps)

        alphas = 1.0 - betas
        # Kept on CPU; values are moved to device inside step()
        self.alphas_cumprod      = torch.cumprod(alphas, 0)
        self.timesteps:          Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int]         = None

    def set_timesteps(self, num_steps: int, device: torch.device):
        """
        Compute the subset of evenly-spaced timesteps used for inference.
        Timesteps are ordered high → low (noisy → clean).

        Args:
            num_steps: Number of denoising steps (e.g. 25, 50).
            device:    Target device for the timestep tensor.
        """
        self.num_inference_steps = num_steps
        step_ratio = self.num_train_timesteps // num_steps
        # Count up from 0, scale, then flip to get descending order (999 → 0)
        self.timesteps = (
            torch.arange(0, num_steps) * step_ratio
        ).flip(0).long().to(device)

    def step(
        self,
        noise_pred: torch.Tensor,
        t:          torch.Tensor,
        x_t:        torch.Tensor,
        eta:        float = 0.0,
    ) -> torch.Tensor:
        """
        Single DDIM denoising step: x_t → x_{t−1}.

        Args:
            noise_pred: UNet-predicted noise ε_θ(x_t, t), shape (B, 4, H, W).
            t:          Current timestep (scalar tensor or int).
            x_t:        Current noisy latent (B, 4, H, W).
            eta:        Stochasticity level; 0.0 = deterministic DDIM,
                        1.0 = DDPM-like stochastic sampling.
        Returns:
            x_{t−1}: Less-noisy latent (B, 4, H, W).
        """
        assert self.num_inference_steps is not None, "Call set_timesteps() before step()."
        device = x_t.device

        t_int     = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        step_size = self.num_train_timesteps // self.num_inference_steps
        prev_t    = t_int - step_size

        alpha_t    = self.alphas_cumprod[t_int].to(device)
        alpha_prev = (
            self.alphas_cumprod[prev_t].to(device)
            if prev_t >= 0
            else torch.ones(1, device=device)
        )

        # Step 1: Estimate clean latent x̂_0 from noisy x_t
        pred_x0 = (x_t - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)  # prevent extreme values propagating

        # Step 2: Direction from x̂_0 towards x_t (diffusion "velocity")
        dir_xt = (1.0 - alpha_prev).sqrt() * noise_pred

        # Step 3: Interpolate between clean prediction and diffusion direction
        x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt

        # Optional stochastic noise injection (DDPM-like when eta = 1)
        if eta > 0.0:
            sigma_t = eta * (
                (1.0 - alpha_prev) / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev)
            ).clamp(min=0.0).sqrt()
            x_prev = x_prev + sigma_t * torch.randn_like(x_t)

        return x_prev

    def to(self, device):
        """Move schedule tensors to *device*."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        if self.timesteps is not None:
            self.timesteps = self.timesteps.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class StableDiffusionModel(nn.Module):
    """
    Unified Stable Diffusion model that wires all components together.

    Frozen  : VAE, CLIP text encoder (no gradient, no optimiser update)
    Trainable: UNet only

    Pass `model.trainable_parameters()` to the optimiser — never `model.parameters()`.
    """

    def __init__(
        self,
        vae:          PretrainedVAE,
        text_encoder: PretrainedCLIPTextEncoder,
        unet:         UNetModel,
        scheduler:    DDPMScheduler,
    ):
        super().__init__()
        self.vae          = vae
        self.text_encoder = text_encoder
        self.unet         = unet
        self.scheduler    = scheduler

    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """Images [-1, 1] → scaled latents (B, 4, H/8, W/8). No gradient."""
        return self.vae.encode(imgs)

    def encode_text(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token IDs → CLIP sequence embeddings (B, 77, 768). No gradient."""
        seq_emb, _ = self.text_encoder(input_ids, attention_mask)
        return seq_emb

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scaled latents → images [-1, 1] (B, 3, H, W). No gradient."""
        return self.vae.decode(latents)

    def forward(
        self,
        latents: torch.Tensor,
        t:       torch.Tensor,
        ctx:     torch.Tensor,
    ) -> torch.Tensor:
        """
        UNet forward pass: predict the noise added to *latents* at timestep *t*
        conditioned on text context *ctx*.

        Args:
            latents: Noisy latent (B, 4, H, W).
            t:       Timestep indices (B,).
            ctx:     CLIP text embeddings (B, 77, 768).
        Returns:
            Predicted noise ε (B, 4, H, W).
        """
        return self.unet(latents, t, ctx)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the UNet to reduce VRAM at the cost of compute."""
        self.unet.enable_gradient_checkpointing()

    def trainable_parameters(self):
        """Yield only UNet parameters — VAE and CLIP are always frozen."""
        return self.unet.parameters()

    def count_parameters(self) -> dict:
        """
        Return a summary dict with total, trainable, and frozen parameter counts.
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        frozen    = total - trainable
        return {
            "total":          total,
            "trainable":      trainable,
            "frozen":         frozen,
            "trainable_pct":  100.0 * trainable / total if total > 0 else 0.0,
        }
