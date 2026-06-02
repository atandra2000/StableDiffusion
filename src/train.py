"""
SD_Train.py — 2× RTX 5090 (32 GB VRAM each) | DDP + BF16 Training
====================================================================
Optimization stack for dual RTX 5090 (Blackwell):

  1. DDP (DistributedDataParallel) — True multi-GPU with NCCL backend.
     Each GPU processes its own micro-batch; gradients are all-reduced.
  2. BF16 native (torch.autocast)  — Blackwell has native BF16 support.
     No GradScaler needed (BF16 doesn't suffer from FP16 underflow).
  3. Flash Attention (FlashSDP)    — Enabled on Blackwell (cc ≥ 8.0).
     ~2-4× faster attention, O(N) memory vs O(N²).
  4. Full-size UNet (ch=320)       — ~860M params, the real SD 1.x size.
     Fits comfortably in 32 GB with BF16 + gradient checkpointing.
  5. Gradient checkpointing        — Saves ~40% activation memory.
     Recomputes activations on backward instead of storing them.
  6. Fused AdamW                   — torch.optim.AdamW with fused=True.
     Single kernel for the entire optimizer step (Blackwell CUDA support).
  7. channels_last memory format   — Optimal for convolutions on Blackwell.
  8. pin_memory + high prefetch    — NVLink/PCIe 5.0 bandwidth is huge.
  9. CUDA memory config            — Expandable segments for less fragmentation.
 10. EMA on GPU                    — 32 GB is plenty; no CPU round-trips.
 11. Min-SNR loss weighting        — Better gradient signal across timesteps.
 12. Large effective batch         — 24 × 2 GPUs × 2 accum = 96 effective.

NOTE: torch.compile is intentionally disabled. Compiling individual module
.forward() methods conflicts with gradient checkpointing — during backward,
checkpoint recomputes the forward inside a dynamo-disabled context, which
causes the compiled wrapper to crash with AssertionError. Eager mode runs
cleanly with Flash SDP and BF16 on Blackwell.
"""

import gc
import os
import argparse
import logging
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
import wandb

from datasets import load_from_disk
from transformers import CLIPTokenizer
from tqdm import tqdm

from SD_Model import (
    StableDiffusionModel, PretrainedVAE, PretrainedCLIPTextEncoder,
    UNetModel, DDPMScheduler, DDIMScheduler,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] | %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── GPU flags — tuned for Blackwell (RTX 5090) ────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False
torch.set_float32_matmul_precision("high")
# Blackwell supports Flash SDP natively — enable all fast paths
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# ═══════════════════════════════════════════════════════════════════════════════
# DDP UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def setup_ddp(rank, world_size):
    """Initialize the DDP process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def _unwrap_module(module: nn.Module) -> nn.Module:
    """Strip DDP wrapper to get the raw nn.Module."""
    return module.module if isinstance(module, DDP) else module

def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """
    Remove common wrapper prefixes so checkpoints can be restored across
    DDP and plain-module saves.
    """
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    cleaned = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        cleaned[k] = v
    return cleaned

def _resolve_resume_path(resume_arg: str | None, ckpt_dir: str) -> str | None:
    """
    Resolve a checkpoint path. Priority:
      1) explicit --resume value (file or dir)
      2) ckpt_dir/sd_latest.pt
      3) newest ckpt_dir/sd_epoch_*.pt
    """
    def _best_in_dir(d: Path) -> str | None:
        if (d / "sd_latest.pt").exists():
            return str(d / "sd_latest.pt")
        epochs = sorted(d.glob("sd_epoch_*.pt"))
        return str(epochs[-1]) if epochs else None

    if resume_arg:
        p = Path(resume_arg).expanduser()
        return _best_in_dir(p) if p.is_dir() else str(p)
    return _best_in_dir(Path(ckpt_dir).expanduser())


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _min_snr_weight(t: torch.Tensor, sched, gamma: float) -> torch.Tensor:
    """
    Min-SNR loss weighting (Hang et al., 2023).
    Downweights high-noise timesteps where vanilla MSE signal is poor.
    Weight = min(SNR, γ) / SNR — clips to 1.0 at low noise, <1 at high noise.
    """
    acp = sched.alphas_cumprod.to(t.device)[t]
    snr = torch.clamp(acp / torch.clamp(1.0 - acp, min=1e-6), min=1e-6).float()
    return torch.nan_to_num(torch.clamp(snr, max=gamma) / snr, nan=1.0, posinf=1.0, neginf=1.0)

def _apply_cfg_dropout(text_emb, uncond_emb, dropout_p: float) -> torch.Tensor:
    """
    Classifier-free guidance dropout: randomly replace some text embeddings
    with the unconditional embedding during training so the model learns
    to denoise with and without text guidance (enables cfg at inference).
    """
    if uncond_emb is None or dropout_p <= 0.0:
        return text_emb
    drop = torch.rand(text_emb.shape[0], device=text_emb.device) < dropout_p
    if not drop.any():
        return text_emb
    uncond_batch = uncond_emb.unsqueeze(0).expand(text_emb.shape[0], -1, -1)
    return torch.where(drop[:, None, None], uncond_batch, text_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# EMA — GPU-resident (32 GB is plenty, no CPU round-trips needed)
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential moving average of UNet weights.
    Shadow weights are kept on-GPU for zero-latency updates.
    The warmup formula d = min(decay, (1+step)/(10+step)) ramps EMA
    gently at the start so early noisy updates don't pollute the shadow.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.step_count = 0
        self.shadow = {n: p.detach().clone().to(self.device)
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].lerp_(p.detach().to(self.device), 1.0 - d)

    def apply_shadow(self, model: nn.Module) -> dict:
        """Swap live weights → shadow. Returns backup to restore later."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self):
        # Move to CPU for serialization
        return {"shadow": {k: v.cpu() for k, v in self.shadow.items()},
                "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = {k: v.to(self.device) for k, v in state["shadow"].items()}
        self.step_count = state["step_count"]
        self.decay = state.get("decay", self.decay)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — RAM-cached latents (.npy)
# ═══════════════════════════════════════════════════════════════════════════════

LATENT_DIR: Path = Path("laion_latents")
_LATENT_CACHE: dict = {}

def key_to_filename(image_key: str) -> str:
    return image_key.replace("/", "_").replace("::", "__") + ".npy"

LATENT_FRACTION = 1.0   # Load ALL latents — RunPod pods have ample RAM

def load_latent_cache(latent_dir: Path):
    """Load all .npy latent files into RAM using 16 threads for fast I/O."""
    all_files = sorted(latent_dir.glob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy latent files found in {latent_dir}")
    files = all_files[:max(1, int(len(all_files) * LATENT_FRACTION))]
    if is_main_process():
        logger.info(f"Latent folder: {len(all_files):,} files — loading {len(files):,} ({LATENT_FRACTION:.0%})")

    def _load(f):
        try:    return f.name, torch.from_numpy(np.load(f).copy())
        except: return f.name, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_load, f): f for f in files}
        it = tqdm(as_completed(futs), total=len(files), desc="Caching latents",
                  unit="file", dynamic_ncols=True, smoothing=0.02) if is_main_process() else as_completed(futs)
        for fut in it:
            name, tensor = fut.result()
            if tensor is not None:
                _LATENT_CACHE[name] = tensor
    if is_main_process():
        logger.info(f"RAM cache ready: {len(_LATENT_CACHE):,} tensors")


def build_dataset(cache_path: str, val_size: int = 500):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found: '{cache_path}'")
    if is_main_process():
        logger.info(f"Loading HuggingFace dataset from {cache_path}")
    ds = load_from_disk(cache_path)
    actual_val = max(1, min(val_size, len(ds) // 10))
    split = ds.train_test_split(test_size=actual_val / len(ds), seed=42)
    if is_main_process():
        logger.info(f"Train: {len(split['train']):,} | Val: {len(split['test']):,}")
    return split["train"], split["test"]


class LatentDistributedSampler(DistributedSampler):
    """
    DistributedSampler that skips samples without a cached latent.
    Ensures each GPU sees a disjoint, evenly-sized subset of valid samples.
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=42, drop_last=True)
        keys = dataset["image_key"]
        self._valid_indices = [i for i, k in enumerate(keys) if key_to_filename(k) in _LATENT_CACHE]
        if not self._valid_indices:
            raise ValueError("No dataset samples matched cached latent files. "
                             "Check --latent_dir and the dataset's image_key formatting.")
        if is_main_process():
            logger.info(f"LatentDistributedSampler: {len(self._valid_indices):,} valid "
                        f"({len(keys) - len(self._valid_indices):,} missing)")

    def __iter__(self):
        # Deterministic shuffle per epoch, then shard evenly across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = [self._valid_indices[i] for i in torch.randperm(len(self._valid_indices), generator=g)]
        total = len(indices) - (len(indices) % self.num_replicas)   # drop remainder so shards are equal
        per_rank = total // self.num_replicas
        return iter(indices[self.rank * per_rank: (self.rank + 1) * per_rank])

    def __len__(self):
        total = len(self._valid_indices)
        return (total - total % self.num_replicas) // self.num_replicas


def collate_fn(batch: list) -> dict | None:
    """Build a batch dict, skipping any samples whose latent is missing from cache."""
    items = [(torch.as_tensor(b["input_ids"]), torch.as_tensor(b["attention_mask"]),
              _LATENT_CACHE.get(key_to_filename(b["image_key"])))
             for b in batch if b is not None]
    items = [(ids, mask, lat) for ids, mask, lat in items if lat is not None]
    if not items:
        return None
    ids, masks, lats = zip(*items)
    return {"pixel_values": torch.stack(lats), "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks)}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model, ddp_unet, loader, optimizer, lr_scheduler,
    device, epoch, noise_scheduler, grad_accum, ema, global_step,
    uncond_text_emb=None, use_wandb=False, use_min_snr=True,
    min_snr_gamma=5.0, cfg_dropout=0.0,
    save_steps=0, ckpt_dir="checkpoints", best_loss=float("inf"),
    memory_format: str = "channels_last",
) -> tuple[float, int]:
    ddp_unet.train()
    model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)

    total_loss, step_count, accum_loss = 0.0, 0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True) if is_main_process() else loader

    for step, batch in enumerate(pbar):
        if batch is None:
            continue
        try:
            latents = batch["pixel_values"].to(device, dtype=torch.bfloat16, non_blocking=True)
            if memory_format == "channels_last":
                latents = latents.contiguous(memory_format=torch.channels_last)
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            # Text encoding is frozen — no grad needed, compute outside autocast for speed
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                text_emb = model.encode_text(ids, mask)
                # CFG dropout: randomly condition on empty text so model learns unconditional denoising
                text_emb = _apply_cfg_dropout(text_emb, uncond_text_emb, cfg_dropout)
                t = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.size(0),), device=device, dtype=torch.long)
                noisy_latents, noise = noise_scheduler.add_noise(latents, t)

            # UNet forward + loss
            with torch.autocast("cuda", dtype=torch.bfloat16):
                noise_pred = ddp_unet(noisy_latents, t, text_emb)
                # Per-sample MSE, then optionally reweight by Min-SNR
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean(dim=[1, 2, 3])
                if use_min_snr:
                    loss = loss * _min_snr_weight(t, noise_scheduler, min_snr_gamma)
                loss = loss.mean() / grad_accum

            # Skip DDP all-reduce on accumulation steps — only sync on the actual optimizer step
            with ddp_unet.no_sync() if (step + 1) % grad_accum != 0 else nullcontext():
                loss.backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(ddp_unet.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(_unwrap_module(model.unet))
                total_loss += accum_loss
                step_count += 1
                accum_loss = 0.0
                global_step += 1

                if save_steps > 0 and global_step % save_steps == 0 and is_main_process():
                    _save_step_checkpoint(model, optimizer, lr_scheduler, ema,
                                         epoch, global_step, best_loss, ckpt_dir)

            if is_main_process() and step_count > 0:
                pbar.set_postfix(loss=f"{total_loss/step_count:.4f}",
                                 lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                                 vram=f"{torch.cuda.memory_reserved(device)/1e9:.1f}GB")
            if use_wandb and is_main_process() and step % 100 == 0 and step_count > 0:
                try:
                    wandb.log({"train/loss": total_loss/step_count,
                               "train/lr": lr_scheduler.get_last_lr()[0],
                               "train/global_step": global_step,
                               "train/vram_gb": torch.cuda.memory_reserved(device)/1e9})
                except Exception:
                    pass

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM at step {step} on rank {device}. Reduce --batch_size or --ch.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
            logger.warning(f"Step {step} skipped: {e}")
            optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(step_count, 1)
    if is_main_process():
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Steps: {step_count:,}")
    return avg_loss, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION — rank 0 only, uses EMA weights
# ═══════════════════════════════════════════════════════════════════════════════

FIXED_VAL_PROMPTS = [
    "epic mountain landscape at golden hour, photorealistic, detailed, 8k",
    "portrait of a young woman, soft lighting, detailed face, photorealistic",
    "cyberpunk city street at night, neon lights, rain reflections, cinematic",
    "beautiful flower meadow, sunlight, sharp focus, nature photography",
]

@torch.no_grad()
def validate(model, ema, tokenizer, device, epoch, output_dir,
             num_steps=25, guidance_scale=7.5):
    if not is_main_process():
        return None

    model.eval()
    sched = DDIMScheduler().to(device)
    sched.set_timesteps(num_steps, device=device)

    cond_tok   = tokenizer(FIXED_VAL_PROMPTS, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").to(device)
    uncond_tok = tokenizer([""] * len(FIXED_VAL_PROMPTS), padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").to(device)

    unet_raw = _unwrap_module(model.unet)
    backup = ema.apply_shadow(unet_raw)   # swap in EMA weights for inference
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond_emb   = model.encode_text(cond_tok.input_ids,   cond_tok.attention_mask)
            uncond_emb = model.encode_text(uncond_tok.input_ids, uncond_tok.attention_mask)
            ctx = torch.cat([uncond_emb, cond_emb], dim=0)   # batched for CFG

            # Fixed seed → same noise every epoch so progress is visually comparable
            latents = torch.randn(len(FIXED_VAL_PROMPTS), 4, 64, 64, device=device,
                                  generator=torch.Generator(device).manual_seed(42))
            for t in sched.timesteps:
                latent_in = torch.cat([latents, latents], dim=0)
                t_batch   = torch.full((latent_in.shape[0],), t, dtype=torch.long, device=device)
                noise_pred = model.unet(latent_in, t_batch, ctx)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided  = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = sched.step(guided, t, latents)

            images = (model.decode_latents(latents).clamp(-1.0, 1.0) + 1.0) / 2.0
    finally:
        ema.restore(unet_raw, backup)   # always restore live weights

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"val_epoch_{epoch:03d}.png")
    save_image(images.float().cpu(), path, nrow=4, normalize=False)
    logger.info(f"Validation grid saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def _save_step_checkpoint(model, optimizer, lr_scheduler, ema,
                          epoch, global_step, best_loss, ckpt_dir):
    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet_raw = _unwrap_module(model.unet)
    ckpt = {
        "epoch":                   epoch,
        "global_step":             global_step,
        "best_loss":               best_loss,
        "unet_state_dict":         _strip_state_dict_prefixes(unet_raw.state_dict()),
        "optimizer_state_dict":    optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "ema_state_dict":          ema.state_dict(),
    }
    path = ckpt_dir / f"sd_step_{global_step:07d}.pt"
    torch.save(ckpt, path)
    torch.save(ckpt, ckpt_dir / "sd_latest.pt")
    logger.info(f"Step checkpoint saved: {path}")


def save_checkpoint(model, optimizer, lr_scheduler, ema,
                    epoch, global_step, best_loss, ckpt_dir):
    if not is_main_process():
        return
    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw underlying module so the file can be restored across DDP and plain-module runs.
    unet_raw = _unwrap_module(model.unet)
    ckpt = {
        "epoch":                   epoch,
        "global_step":             global_step,
        "best_loss":               best_loss,
        "unet_state_dict":         _strip_state_dict_prefixes(unet_raw.state_dict()),
        "optimizer_state_dict":    optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "ema_state_dict":          ema.state_dict(),
    }
    path = ckpt_dir / f"sd_epoch_{epoch:03d}.pt"
    tmp  = ckpt_dir / f".sd_epoch_{epoch:03d}.pt.tmp"
    # Atomic write: save to .tmp then rename — prevents a partial file if killed mid-save
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    torch.save(ckpt, ckpt_dir / "sd_latest.pt")
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, lr_scheduler, ema, ckpt_path, device="cuda"):
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if is_main_process():
        logger.info(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Load UNet — support both new-style ("unet_state_dict") and legacy key names
    unet_raw = _unwrap_module(model.unet)
    if "unet_state_dict" in ckpt:
        sd = _strip_state_dict_prefixes(ckpt["unet_state_dict"])
    elif "model_state_dict" in ckpt:
        sd = {k[5:]: v for k, v in _strip_state_dict_prefixes(ckpt["model_state_dict"]).items() if k.startswith("unet.")}
    elif "state_dict" in ckpt:
        sd = _strip_state_dict_prefixes(ckpt["state_dict"])
    else:
        raise KeyError(f"No recognized state dict key in checkpoint. Keys: {sorted(ckpt.keys())}")
    missing = unet_raw.load_state_dict(sd, strict=False)
    if is_main_process() and (missing.missing_keys or missing.unexpected_keys):
        logger.warning(f"UNet key mismatches — missing: {len(missing.missing_keys)}, unexpected: {len(missing.unexpected_keys)}")

    # Restore optimizer, scheduler, EMA — log but don't crash on mismatch
    for key, obj, label in [
        ("optimizer_state_dict", optimizer, "optimizer"),
        ("lr_scheduler_state_dict", lr_scheduler, "scheduler"),
        ("ema_state_dict", ema, "EMA"),
    ]:
        if key in ckpt:
            try:
                obj.load_state_dict(ckpt[key])
            except Exception as e:
                logger.warning(f"Could not load {label}: {e}")

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_loss   = ckpt.get("best_loss", float("inf"))
    if is_main_process():
        logger.info(f"Resumed from epoch {start_epoch} | step {global_step} | best loss {best_loss:.4f}")
    return start_epoch, global_step, best_loss


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (per-rank entry point)
# ═══════════════════════════════════════════════════════════════════════════════

def main(rank, world_size, args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process():
        logger.info("=" * 70)
        logger.info("STABLE DIFFUSION — 2× RTX 5090 DDP TRAINING")
        logger.info("=" * 70)
        for i in range(world_size):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | cc {p.major}.{p.minor}")
        logger.info(f"Effective batch: {args.batch_size} × {world_size} GPUs × {args.grad_accum} accum = "
                    f"{args.batch_size * world_size * args.grad_accum}")
        logger.info("=" * 70)

    if args.use_wandb and is_main_process():
        try:
            wandb.init(project="stable-diffusion", config=vars(args),
                       name=f"2x5090-ep{args.epochs}-{datetime.now().strftime('%m%d_%H%M')}",
                       resume="allow", settings=wandb.Settings(init_timeout=120))
            logger.info("wandb initialized successfully")
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            args.use_wandb = False

    # ── Load latents into RAM ──────────────────────────────────────────────────
    global LATENT_DIR
    LATENT_DIR = Path(args.latent_dir)
    if not LATENT_DIR.exists():
        raise FileNotFoundError(f"Latent directory not found: {LATENT_DIR}")
    load_latent_cache(LATENT_DIR)

    # ── Build model ───────────────────────────────────────────────────────────
    if is_main_process(): logger.info("Loading VAE (frozen, stays on GPU)...")
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse", use_fp16=False)
    vae.vae = vae.vae.to(dtype=torch.bfloat16)   # BF16 VAE on Blackwell
    vae.use_fp16 = False

    if is_main_process(): logger.info("Loading CLIP text encoder (frozen, stays on GPU)...")
    text_enc = PretrainedCLIPTextEncoder(model_id="openai/clip-vit-large-patch14")

    if is_main_process(): logger.info(f"Building UNet (ch={args.ch}, heads={args.heads})...")
    unet = UNetModel(in_ch=4, out_ch=4, ch=args.ch, res_blks=2, attn_lvls=(1, 2, 3),
                     ch_mults=(1, 2, 4, 4), heads=args.heads, t_dim=args.ch,
                     ctx_dim=768, grad_ckpt=args.grad_ckpt)

    noise_scheduler = DDPMScheduler(steps=1000, beta_start=0.00085, beta_end=0.012, schedule="scaled_linear")
    model = StableDiffusionModel(vae, text_enc, unet, noise_scheduler).to(device)
    if args.memory_format == "channels_last":
        model.unet = model.unet.to(memory_format=torch.channels_last)
    noise_scheduler.to(device)

    # ── Dataset + unconditional embedding (precomputed once) ──────────────────
    if is_main_process(): logger.info("Loading dataset...")
    train_ds, _ = build_dataset(args.cache_path, args.val_size)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    empty_tok = tokenizer([""], padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").to(device)
    # Pre-encode the empty/unconditional embedding once — reused for CFG dropout every step
    uncond_text_emb = model.encode_text(empty_tok.input_ids, empty_tok.attention_mask).squeeze(0).detach()

    # ── Optimizer — fused AdamW (single CUDA kernel) ──────────────────────────
    # Falls back to standard AdamW if fused is unavailable on this platform
    try:    optimizer = AdamW(model.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8, fused=True)
    except: optimizer = AdamW(model.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    # ── LR Scheduler — warmup + cosine ────────────────────────────────────────
    _tmp_sampler = LatentDistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    batches_per_epoch = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=_tmp_sampler, drop_last=True))
    total_steps  = max(1, batches_per_epoch * args.epochs // max(1, args.grad_accum))
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    if warmup_steps < 1:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 1e-2)
        if is_main_process(): logger.info(f"LR: cosine over {total_steps:,} steps (no warmup)")
    else:
        lr_scheduler = SequentialLR(optimizer, milestones=[warmup_steps], schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=args.lr * 1e-2),
        ])
        if is_main_process(): logger.info(f"LR: {warmup_steps} warmup + cosine over {total_steps:,} steps")

    # ── EMA + resume ──────────────────────────────────────────────────────────
    ema = EMA(_unwrap_module(model.unet), decay=0.9999, device=device)
    start_epoch, global_step, best_loss = 1, 0, float("inf")
    resume_path = _resolve_resume_path(args.resume, args.ckpt_dir)
    if resume_path:
        try:
            start_epoch, global_step, best_loss = load_checkpoint(
                model, optimizer, lr_scheduler, ema, resume_path, device=str(device))
        except Exception as e:
            logger.error(f"Resume failed: {e}. Starting from scratch.")
    elif is_main_process():
        logger.info("No checkpoint found. Starting from scratch.")

    # Wrap UNet in DDP — only the UNet is trained
    ddp_unet = DDP(model.unet, device_ids=[rank], output_device=rank,
                   find_unused_parameters=False, gradient_as_bucket_view=True)

    if is_main_process():
        s = model.count_parameters()
        logger.info(f"UNet: {s['trainable']:,} trainable | {s['frozen']:,} frozen")
        logger.info(f"VRAM after model: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    # ── DataLoader ────────────────────────────────────────────────────────────
    sampler = LatentDistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None, drop_last=True)
    if is_main_process():
        logger.info(f"Train loader: {len(train_loader):,} batches/epoch/rank")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        if is_main_process():
            logger.info("\n" + "─" * 70 + f"\nEPOCH {epoch}/{args.epochs}\n" + "─" * 70)

        sampler.set_epoch(epoch)
        avg_loss, global_step = train_epoch(
            model=model, ddp_unet=ddp_unet, loader=train_loader,
            optimizer=optimizer, lr_scheduler=lr_scheduler, device=device,
            epoch=epoch, noise_scheduler=noise_scheduler, grad_accum=args.grad_accum,
            ema=ema, global_step=global_step, uncond_text_emb=uncond_text_emb,
            use_wandb=args.use_wandb, use_min_snr=args.min_snr, min_snr_gamma=args.min_snr_gamma,
            cfg_dropout=args.cfg_dropout,
            save_steps=args.save_steps, ckpt_dir=args.ckpt_dir, best_loss=best_loss,
            memory_format=args.memory_format,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main_process(): logger.info(f"New best loss: {best_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, lr_scheduler, ema, epoch, global_step, best_loss, args.ckpt_dir)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            validate(model=model, ema=ema, tokenizer=tokenizer, device=device,
                     epoch=epoch, output_dir=args.output_dir)

        if args.use_wandb and is_main_process():
            try: wandb.log({"epoch": epoch, "train/loss": avg_loss, "train/best_loss": best_loss})
            except Exception: pass

        if dist.is_initialized():
            dist.barrier()   # keep ranks in sync between epochs

    if is_main_process():
        logger.info("=" * 70)
        logger.info(f"TRAINING COMPLETE — Best Loss: {best_loss:.4f}")
        logger.info("=" * 70)
        if args.use_wandb:
            wandb.finish()

    cleanup_ddp()


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER + LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Training — 2× RTX 5090 DDP")

    # Data
    parser.add_argument("--cache_path",   type=str,   default="laion_hf_dataset/train")
    parser.add_argument("--latent_dir",   type=str,   default="laion_latents")
    parser.add_argument("--val_size",     type=int,   default=500)

    # Model
    parser.add_argument("--ch",    type=int, default=320,  help="UNet base channels (320 = full SD 1.x, ~860M params).")
    parser.add_argument("--heads", type=int, default=8,    help="Attention heads.")

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=24,   help="Per-GPU batch size.")
    parser.add_argument("--grad_accum",   type=int,   default=2,    help="Gradient accumulation steps.")
    parser.add_argument("--lr",           type=float, default=1e-5, help="Peak LR.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--num_workers",  type=int,   default=16,   help="DataLoader workers per rank.")

    # Optimizations
    parser.add_argument("--grad_ckpt",     action="store_true",  default=True, help="Gradient checkpointing (~40%% VRAM saving).")
    parser.add_argument("--no-grad-ckpt",  dest="grad_ckpt",     action="store_false")
    parser.add_argument("--min_snr",       action="store_true",  default=True, help="Min-SNR loss weighting.")
    parser.add_argument("--no-min-snr",    dest="min_snr",       action="store_false")
    parser.add_argument("--min_snr_gamma", type=float,           default=5.0)
    parser.add_argument("--cfg_dropout",   type=float,           default=0.05, help="CFG dropout probability.")
    parser.add_argument("--memory_format", type=str, default="channels_last",
                        choices=("channels_last", "contiguous"),
                        help="Memory format for UNet and latents. channels_last speeds up convs on NVIDIA GPUs. "
                             "Use 'contiguous' for AMD or Apple Silicon.")

    # Checkpointing
    parser.add_argument("--save_every",  type=int, default=1)
    parser.add_argument("--save_steps",  type=int, default=0, help="Save a mid-epoch checkpoint every N global steps (0 = disabled).")
    parser.add_argument("--val_every",   type=int, default=1)
    parser.add_argument("--ckpt_dir",    type=str, default="checkpoints")
    parser.add_argument("--output_dir",  type=str, default="outputs")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--use_wandb",   action="store_true")

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(f"Only {world_size} GPU(s) detected. DDP works best with 2+.")

    # Prefer torchrun; fall back to mp.spawn for direct execution
    if "RANK" in os.environ:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
