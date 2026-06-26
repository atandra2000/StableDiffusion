# Model Architecture

## Overview

The model implements the full latent diffusion pipeline from Rombach et al. (2022), composed of four learned components and two schedulers.

## Components

### 1. VAE (Frozen) — `stabilityai/sd-vae-ft-mse`

- **Encoder:** Down-convolves 512×512 RGB images to 64×64 latents (4 channels)
- **Decoder:** Up-convolves latents back to 512×512 RGB
- **Scale factor:** 0.18215 (multiply latents by this before UNet, divide after)
- **Parameters:** ~83M, frozen during training
- **Integration:** Loaded from diffusers `AutoencoderKL` via `PretrainedVAE` wrapper in `model.py`

### 2. CLIP Text Encoder (Frozen) — `openai/clip-vit-large-patch14`

- **Input:** 77 BPE tokens (truncated/padded)
- **Output:** `(B, 77, 768)` hidden states
- **Parameters:** ~123M, frozen during training
- **Integration:** Loaded from transformers `CLIPTextModel` via `PretrainedCLIPTextEncoder` wrapper

### 3. UNet (Trainable) — ~860M Parameters

The denoising backbone is a U-Net with spatial self-attention and cross-attention for text conditioning:

```mermaid
flowchart TB
    X["z_t  noisy latent<br/>(B, 4, 64, 64)"]:::in
    T_EMB["timestep t<br/>sinusoidal → MLP → 1280-d<br/>injected into every ResBlock"]:::te
    CTX["text context from CLIP<br/>(B, 77, 768)"]:::ctx

    subgraph DOWN["Encoder (Down path)"]
        direction TB
        D0["Stage 0 — 64×64<br/>320 ch · 2× ResBlock<br/><i>no attention</i>"]:::down0
        D1["Stage 1 — 32×32<br/>640 ch · 2× ResBlock<br/>+ SpatialTransformer<br/>(self + cross-attn)"]:::down1
        D2["Stage 2 — 16×16<br/>1280 ch · 2× ResBlock<br/>+ SpatialTransformer"]:::down1
        D3["Stage 3 — 8×8<br/>1280 ch · 2× ResBlock<br/>+ SpatialTransformer"]:::down1
        BN["Bottleneck — 8×8<br/>1280 ch · ResBlock + Attn"]:::btn
    end

    subgraph UP["Decoder (Up path, mirror of encoder)"]
        direction TB
        U3["Stage 3 — 8×8<br/>1280 ch · + skip from D3<br/>+ SpatialTransformer"]:::up1
        U2["Stage 2 — 16×16<br/>1280 ch · + skip from D2<br/>+ SpatialTransformer"]:::up1
        U1["Stage 1 — 32×32<br/>640 ch · + skip from D1<br/>+ SpatialTransformer"]:::up1
        U0["Stage 0 — 64×64<br/>320 ch · + skip from D0<br/><i>no attention</i>"]:::up0
    end

    OUT["predicted noise ε̂<br/>(B, 4, 64, 64)"]:::out

    X --> D0 --> D1 --> D2 --> D3 --> BN --> U3 --> U2 --> U1 --> U0 --> OUT
    T_EMB -. FiLM .-> D0
    T_EMB -. FiLM .-> D1
    T_EMB -. FiLM .-> D2
    T_EMB -. FiLM .-> D3
    T_EMB -. FiLM .-> BN
    T_EMB -. FiLM .-> U3
    T_EMB -. FiLM .-> U2
    T_EMB -. FiLM .-> U1
    T_EMB -. FiLM .-> U0
    CTX --> D1
    CTX --> D2
    CTX --> D3
    CTX --> BN
    CTX --> U3
    CTX --> U2
    CTX --> U1

    classDef in fill:#e0e7ff,stroke:#3730a3,color:#000
    classDef te fill:#fef3c7,stroke:#92400e,color:#000
    classDef ctx fill:#fed7aa,stroke:#9a3412,color:#000
    classDef down0 fill:#f3f4f6,stroke:#374151,color:#000
    classDef down1 fill:#dbeafe,stroke:#1d4ed8,color:#000
    classDef btn fill:#fde68a,stroke:#b45309,color:#000
    classDef up0 fill:#f3f4f6,stroke:#374151,color:#000
    classDef up1 fill:#bbf7d0,stroke:#15803d,color:#000
    classDef out fill:#bbf7d0,stroke:#15803d,color:#000
```

#### Block internals: ResBlock + SpatialTransformer

```mermaid
flowchart LR
    H["h"]:::in --> RB1["ResBlock<br/>GN → SiLU → Conv3×3<br/>+ t-embed (FiLM)<br/>─────────────<br/>GN → SiLU → Conv3×3"]:::rb --> RB2["ResBlock<br/>(identical)"]:::rb --> NORM["GroupNorm"]:::norm
    NORM --> ST["SpatialTransformer<br/>LN → self-attn<br/>LN → cross-attn (ctx)<br/>LN → FFN"]:::st --> OUT["h'"]:::out
    H -. residual .-> RB2
    RB2 -. residual .-> OUT

    ST_IN["text ctx (B, 77, 768)"]:::ctx --> ST
    T_IN["t (1280-d)"]:::te --> RB1

    classDef in fill:#e0e7ff,stroke:#3730a3,color:#000
    classDef out fill:#bbf7d0,stroke:#15803d,color:#000
    classDef rb fill:#dbeafe,stroke:#1d4ed8,color:#000
    classDef norm fill:#f3f4f6,stroke:#374151,color:#000
    classDef st fill:#fce7f3,stroke:#9d174d,color:#000
    classDef ctx fill:#fed7aa,stroke:#9a3412,color:#000
    classDef te fill:#fef3c7,stroke:#92400e,color:#000
```

#### Key implementation details:

#### Key implementation details:

- **SpatialTransformer:** GroupNorm → Conv 1×1 → multi-head self-attention → cross-attention (text as K,V) → Conv 1×1 → residual
- **ResNetBlock:** GroupNorm → SiLU → Conv 3×3 → GroupNorm → SiLU → dropout → Conv 3×3 + residual
- **Time conditioning:** Sinusoidal timestep embedding → MLP → added to each ResNetBlock (FiLM-style scale+shift)
- **Flash Attention:** Enabled via `torch.backends.cuda.enable_flash_sdp(True)` on Blackwell (cc ≥ 8.0)
- **Gradient checkpointing:** Saves ~40% activation memory by recomputing activations during backward (configurable via `--grad_ckpt`)

### 4. DDPM Scheduler (Training)

Defines the forward noising process:

```
z_t = √(ᾱ_t) · z_0 + √(1 − ᾱ_t) · ε    where ε ~ N(0, I)
```

- **Beta schedule:** `scaled_linear` (β linearly spaced after square-root transformation)
- **Steps:** 1000
- **Range:** β₁ = 0.00085, β₁₀₀₀ = 0.012

### 5. DDIM Scheduler (Inference)

Deterministic reverse process (Song et al., 2020). Only 25–50 steps needed:

```
x̂_0     = (x_t − √(1−ᾱ_t) · ε_θ) / √(ᾱ_t)     (clean latent estimate)
x_{t−1} = √(ᾱ_{t−1}) · x̂_0 + √(1 − ᾱ_{t−1}) · ε_θ   (eta=0, deterministic)
```

**Note on `pred_x0.clamp(-1.0, 1.0)`:** The original 1000-step DDPM training code clamps the clean latent estimate to `[-1, 1]`. This is incorrect for inference — SD latents have a standard deviation of ~4, and clamping destroys signal quality. `model.py` now makes this clamp opt-in via `DDIMScheduler(clamp_pred_x0=False)` (default). The `inference.py` script and `SD_ImageGen.py` both set this to `False`.

## Parameter Count Breakdown

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| CLIP Text Encoder | 123M | ❌ |
| VAE (Encoder + Decoder) | 83M | ❌ |
| UNet | ~860M | ✅ |
| **Total pipeline** | **~1.07B** | **~860M** |

## Memory Format

The model uses `channels_last` memory format (NHWC) on CUDA for optimal convolution performance on Blackwell. For Apple Silicon (MPS) or AMD GPUs, pass `--memory_format contiguous` at the command line to disable this.
