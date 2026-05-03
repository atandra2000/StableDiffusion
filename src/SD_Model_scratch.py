import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class ResNetBlock(nn.Module):
    """Residual block with group normalization and SiLU activation"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)
        x = F.silu(self.norm1(x))
        x = self.conv1(x)
        x = F.silu(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class AttentionBlock(nn.Module):
    """Self-attention block for VAE with Flash Attention"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = 8
        self.head_dim = channels // self.num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        residual = x
        x = self.norm(x)
        b, c, h, w = x.shape
        
        q = self.q(x).view(b, c, h * w).transpose(1, 2).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, c, h * w).transpose(1, 2).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, c, h * w).transpose(1, 2).view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        # Use Flash Attention if available
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
                )
        except:
            # Fallback to standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(b, h * w, c).transpose(1, 2).view(b, c, h, w)
        out = self.proj_out(out)
        return out + residual

class VAEEncoder(nn.Module):
    """VAE Encoder for compressing images to latent space"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(128, 128),
                ResNetBlock(128, 128),
                nn.Conv2d(128, 128, 3, stride=2, padding=1)
            ]),
            nn.ModuleList([
                ResNetBlock(128, 256),
                ResNetBlock(256, 256),
                nn.Conv2d(256, 256, 3, stride=2, padding=1)
            ]),
            nn.ModuleList([
                ResNetBlock(256, 512),
                ResNetBlock(512, 512),
                nn.Conv2d(512, 512, 3, stride=2, padding=1)
            ])
        ])
        
        # Middle block with attention
        self.mid_block = nn.ModuleList([
            ResNetBlock(512, 512),
            AttentionBlock(512),
            ResNetBlock(512, 512)
        ])
        
        # Final projection to latent space
        self.norm_out = nn.GroupNorm(32, 512)
        self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        
        # Downsampling
        for down_block in self.down_blocks:
            x = down_block[0](x)
            x = down_block[1](x)
            x = down_block[2](x)
        
        # Middle processing
        for layer in self.mid_block:
            x = layer(x)
        
        # Final output
        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        
        # Split into mean and logvar for reparameterization
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar

class VAEDecoder(nn.Module):
    """VAE Decoder for reconstructing images from latent space"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, 3, padding=1)
        
        # Middle block
        self.mid_block = nn.ModuleList([
            ResNetBlock(512, 512),
            AttentionBlock(512),
            ResNetBlock(512, 512)
        ])
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(512, 512),
                ResNetBlock(512, 512),
                nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
            ]),
            nn.ModuleList([
                ResNetBlock(512, 256),
                ResNetBlock(256, 256),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
            ]),
            nn.ModuleList([
                ResNetBlock(256, 128),
                ResNetBlock(128, 128),
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
            ])
        ])
        
        # Final output
        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        
        # Middle processing
        for layer in self.mid_block:
            x = layer(x)
        
        # Upsampling
        for up_block in self.up_blocks:
            x = up_block[0](x)
            x = up_block[1](x)
            x = up_block[2](x)
        
        # Final output
        x = F.silu(self.norm_out(x))
        x = torch.tanh(self.conv_out(x))
        return x

class CLIPTextEmbedding(nn.Module):
    """Text embedding layer for CLIP encoder"""
    def __init__(self, vocab_size: int, embed_dim: int, max_position: int = 77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_position, embed_dim))

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding[:seq_len]
        return token_emb + pos_emb

class CLIPAttention(nn.Module):
    """Multi-head self-attention for CLIP transformer"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attention_mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=self.scale
                )
        except:
            # Fallback to standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

class CLIPTransformerBlock(nn.Module):
    """CLIP transformer block with attention and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CLIPAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class CLIPTextEncoder(nn.Module):
    """CLIP text encoder for conditioning"""
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 768,
                 num_layers: int = 12, num_heads: int = 12, max_position: int = 77):
        super().__init__()
        self.embeddings = CLIPTextEmbedding(vocab_size, embed_dim, max_position)
        self.transformer = nn.ModuleList([
            CLIPTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.transformer:
            x = layer(x, attention_mask)
        x = self.ln_final(x)
        
        # Use EOS token representation
        if attention_mask is not None:
            eos_indices = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(x.size(0)), eos_indices]
        else:
            pooled = x[:, -1]
        
        # Apply text projection
        pooled = pooled @ self.text_projection
        return x, pooled

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings for diffusion process"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class CrossAttention(nn.Module):
    """Cross-attention mechanism for text-image conditioning"""
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context=None):
        b, n, c = x.shape
        h = self.num_heads
        
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        
        if context is None:
            context = x
        
        k = self.to_k(context).view(b, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, -1, h, self.head_dim).transpose(1, 2)

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
                )
        except:
            # Fallback to standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """Transformer block with self and cross attention"""
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context=None):
        # Self attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross attention with text conditioning
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # Feed forward
        x = x + self.mlp(self.norm3(x))
        return x

class SpatialTransformer(nn.Module):
    """Spatial transformer for applying cross-attention to feature maps"""
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_block = TransformerBlock(channels, context_dim, num_heads)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x

        # Prepare for transformer
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.view(b, c, h * w).transpose(1, 2)

        # Apply transformer
        x = self.transformer_block(x, context)

        # Reshape back
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.proj_out(x)
        return x + residual

class ResNetBlockWithAttention(nn.Module):
    """ResNet block integrated with transformer attention"""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 context_dim: int, num_heads: int = 8, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )

        # ResNet layers
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Spatial transformer for cross-attention
        if use_attention:
            self.spatial_transformer = SpatialTransformer(out_channels, context_dim, num_heads)

    def forward(self, x, time_emb, context=None):
        residual = self.skip_connection(x)

        # First conv block
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb_proj

        # Second conv block
        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        # Apply attention if enabled
        if self.use_attention and context is not None:
            h = self.spatial_transformer(h, context)

        return h + residual

class UNetModel(nn.Module):
    """The full UNet model for noise prediction in latent diffusion"""
    def __init__(self, in_channels: int = 4, out_channels: int = 4, channels: int = 320,
                 num_res_blocks: int = 2, attention_levels: Tuple[int, ...] = (4, 2, 1),
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                 num_heads: int = 8, time_embed_dim: int = 320, context_dim: int = 768):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim)

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Encoder (Down-sampling path)
        self.down_blocks = nn.ModuleList()
        current_channels = channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResNetBlockWithAttention(
                        in_channels=current_channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        context_dim=context_dim,
                        num_heads=num_heads,
                        use_attention=(i in attention_levels),
                    )
                )
                current_channels = out_ch
            
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                )

        # Middle block
        self.middle_block = nn.ModuleList([
            ResNetBlockWithAttention(
                in_channels=current_channels,
                out_channels=current_channels,
                time_embed_dim=time_embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                use_attention=True,
            ),
            ResNetBlockWithAttention(
                in_channels=current_channels,
                out_channels=current_channels,
                time_embed_dim=time_embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                use_attention=False,
            ),
        ])

        # Decoder (Up-sampling path)
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = channels * mult
            for j in range(num_res_blocks + 1):
                in_ch = current_channels + out_ch if j == 0 else out_ch
                self.up_blocks.append(
                    ResNetBlockWithAttention(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        context_dim=context_dim,
                        num_heads=num_heads,
                        use_attention=(i in attention_levels),
                    )
                )
                current_channels = out_ch

            if i != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(current_channels, current_channels, kernel_size=4, stride=2, padding=1)
                )

        # Output projection
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Timestep embedding
        t_emb = self.time_embedding(timesteps)
        t_emb = self.time_proj(t_emb)

        # Input convolution
        h = self.conv_in(x)

        # Store hidden states for skip connections
        hidden_states = []

        # Encoder
        for block in self.down_blocks:
            if isinstance(block, ResNetBlockWithAttention):
                h = block(h, t_emb, context)
                hidden_states.append(h)
            else:
                h = block(h)

        # Middle block
        for block in self.middle_block:
            h = block(h, t_emb, context)

        # Decoder
        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            else:
                skip_h = hidden_states.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = block(h, t_emb, context)

        # Output
        h = F.silu(self.norm_out(h))
        output = self.conv_out(h)
        return output

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models scheduler"""
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn_like(original_samples)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples, noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        t = timestep
        
        # Get parameters
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Predict original sample
        pred_original_sample = (sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output) / self.sqrt_alphas_cumprod[t]
        
        # Compute mean of previous sample
        alpha_cumprod_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        mean_pred = (
            (alpha_cumprod_t_prev.sqrt() * beta_t) / (1 - alpha_cumprod_t) * pred_original_sample +
            (alpha_t.sqrt() * (1 - alpha_cumprod_t_prev)) / (1 - alpha_cumprod_t) * sample
        )
        
        # Add noise
        variance = self._get_variance(t)
        noise = torch.randn_like(model_output)
        prev_sample = mean_pred + variance.sqrt() * noise
        
        return prev_sample

    def _get_variance(self, t):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

class DDIMScheduler:
    """Denoising Diffusion Implicit Models scheduler"""
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = self.alphas_cumprod[-1]
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        # Get the index of the previous timestep
        prev_timestep = timestep - (self.num_train_timesteps // self.num_inference_steps)
        
        # Get alpha and beta values for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict original sample (x_0) from the model output (epsilon)
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance.sqrt()
        
        # Compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * model_output
        
        # Compute x_{t-1}
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample += std_dev_t * noise
        
        return prev_sample

class StableDiffusionModel(nn.Module):
    """The complete Stable Diffusion model"""
    def __init__(self, vae_encoder: VAEEncoder, vae_decoder: VAEDecoder,
                 text_encoder: CLIPTextEncoder, unet: UNetModel,
                 scheduler: DDPMScheduler, latent_scale_factor: float = 0.18215):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.latent_scale_factor = latent_scale_factor

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes images from pixel space to latent space"""
        mean, logvar = self.vae_encoder(images)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latents = mean + eps * std
        return latents * self.latent_scale_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents from latent space back to pixel space"""
        latents = 1 / self.latent_scale_factor * latents
        images = self.vae_decoder(latents)
        return images

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """The forward pass for the UNet. Predicts the noise in the latents."""
        return self.unet(latents, timesteps, context)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.unet.enable_gradient_checkpointing()
        self.text_encoder.enable_gradient_checkpointing()
