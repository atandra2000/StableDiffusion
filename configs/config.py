"""
Training configuration for Stable Diffusion on LAION-2B-en-aesthetic.
Hardware: 2× RTX 5090 (Blackwell, cc 10.x, 32 GB VRAM each) on RunPod.
Multi-GPU: DistributedDataParallel (DDP) via torchrun --nproc_per_node=2.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    # LAION-2B-en-aesthetic paths (RunPod workspace)
    metadata_dir:    str = "/workspace/StableDiffusion/laion_metadata"
    filtered_dir:    str = "/workspace/StableDiffusion/laion_filtered"
    shards_dir:      str = "/workspace/StableDiffusion/laion_shards"
    cache_dir:       str = "/workspace/StableDiffusion/laion_cache"
    hf_dataset_dir:  str = "/workspace/StableDiffusion/laion_hf_dataset"
    latent_dir:      str = "/workspace/StableDiffusion/laion_latents"

    # Filtering
    aesthetic_min:   float = 6.5    # Top ~2% of LAION-2B-en ≈ 12M images
    clip_sim_min:    float = 0.28
    min_resolution:  int   = 512
    max_watermark:   float = 0.5
    allow_nsfw:      bool  = False

    # Dataset split
    val_size:        int = 5_000
    image_size:      int = 512


@dataclass
class ModelConfig:
    # VAE (frozen)
    vae_model_id:    str   = "stabilityai/sd-vae-ft-mse"
    vae_scale:       float = 0.18215

    # CLIP text encoder (frozen)
    clip_model_id:   str = "openai/clip-vit-large-patch14"
    max_token_len:   int = 77
    text_dim:        int = 768

    # UNet (trainable, ~860M params)
    in_ch:     int          = 4
    out_ch:    int          = 4
    ch:        int          = 320
    res_blks:  int          = 2
    attn_lvls: Tuple[int, ...] = field(default_factory=lambda: (1, 2, 3))
    ch_mults:  Tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 4))
    heads:     int          = 8
    t_dim:     int          = 320
    ctx_dim:   int          = 768

    # Latent space
    latent_ch: int = 4
    latent_res: int = 64          # 512 / 8 = 64 (VAE 8× downsample)


@dataclass
class SchedulerConfig:
    # DDPM (training)
    num_train_steps: int   = 1000
    beta_start:      float = 0.00085
    beta_end:        float = 0.012
    schedule:        str   = "scaled_linear"

    # DDIM (inference)
    num_inference_steps: int   = 30
    guidance_scale:      float = 7.5


@dataclass
class TrainingConfig:
    # Hardware (2× RTX 5090, DDP via torchrun)
    num_gpus:    int = 2
    vram_per_gpu: str = "32GB"

    # Batch
    batch_size_per_gpu: int = 24
    grad_accum:         int = 2
    # Effective batch = 24 × 2 GPUs × 2 accum = 96

    # Optimiser
    lr:            float = 1e-4
    weight_decay:  float = 1e-2
    betas:         Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    eps:           float = 1e-8

    # LR schedule
    warmup_steps:  int = 500

    # EMA
    ema_decay:     float = 0.9999

    # Epochs / checkpointing
    epochs:        int = 10
    save_every:    int = 1
    val_every:     int = 2

    # Precision / compilation (Blackwell: no GradScaler, max-autotune compile)
    dtype:         str  = "bfloat16"
    use_compile:   bool = True
    compile_mode:  str  = "max-autotune"
    grad_ckpt:     bool = False     # Enable if VRAM is tight

    # Paths
    ckpt_dir:      str = "/workspace/checkpoints"
    output_dir:    str = "/workspace/outputs"


@dataclass
class SDConfig:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training:  TrainingConfig  = field(default_factory=TrainingConfig)
