"""
SD_Train.py — Optimized for 2x RTX PRO 4500 | Latent Pre-encoding Pipeline
========================================================================
"""

import gc
import os
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Sampler
from torchvision.utils import save_image
import wandb

from datasets import load_from_disk
from transformers import CLIPTokenizer
from tqdm import tqdm

from model import (
    StableDiffusionModel, PretrainedVAE, PretrainedCLIPTextEncoder,
    UNetModel, DDPMScheduler, DDIMScheduler
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Global GPU flags ───────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True
torch.backends.cudnn.deterministic     = False
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# ═══════════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        # Unwrap DataParallel if needed
        self.model = model.module if isinstance(model, nn.DataParallel) else model
        self.decay = decay
        self.shadow = {
            n: p.detach().cpu().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.step_count = 0

    @torch.no_grad()
    def update(self):
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().cpu(), alpha=1.0 - d)

    def apply_shadow(self):
        backup = {}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n].to(p.device))
        return backup

    def restore(self, backup: dict):
        for n, p in self.model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self):
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow     = state["shadow"]
        self.step_count = state["step_count"]
        self.decay      = state.get("decay", self.decay)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — RAM-cached latents
# ═══════════════════════════════════════════════════════════════════════════════

LATENT_DIR: Path = Path("/workspace/StableDiffusion/laion_latents")

_LATENT_CACHE: dict = {}

def key_to_filename(image_key: str) -> str:
    return image_key.replace("/", "_").replace("::", "__") + ".pt"

def load_latent_cache(latent_dir: Path):
    """
    Load all pre-encoded latent .pt files into RAM using parallel threads.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    files = list(latent_dir.glob("*.pt"))
    logger.info(f"Loading {len(files):,} latents into RAM (~42 GB) with 16 threads...")

    def _load(f):
        try:
            return f.name, torch.load(f, weights_only=True)
        except Exception:
            return f.name, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_load, f): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(files),
                        desc="Caching latents", unit="file",
                        dynamic_ncols=True, smoothing=0.02):
            name, tensor = fut.result()
            if tensor is not None:
                _LATENT_CACHE[name] = tensor

    logger.info(f"✅ RAM cache ready: {len(_LATENT_CACHE):,} tensors")

def build_dataset(cache_path: str, val_size: int = 5000):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found: '{cache_path}'")
    logger.info(f"Loading HuggingFace dataset from {cache_path}")
    ds = load_from_disk(cache_path)
    actual_val = min(val_size, len(ds) // 10)
    split = ds.train_test_split(test_size=actual_val / len(ds), seed=42)
    logger.info(f"📊 Train: {len(split['train']):,} | Val: {len(split['test']):,}")
    return split["train"], split["test"]


class LatentSampler(Sampler):
    """
    Shuffle at the sample level (not shard level) since latents are individual
    files already on disk — no need for shard-sorted access.
    """
    def __init__(self, dataset, epoch: int = 0):
        import random
        keys = dataset["image_key"]
        # Keep only indices that have latents in RAM cache
        self._indices = [
            i for i, k in enumerate(keys)
            if key_to_filename(k) in _LATENT_CACHE
        ]
        rng = random.Random(epoch * 1337)
        rng.shuffle(self._indices)
        logger.info(f"LatentSampler: {len(self._indices):,} valid samples "
                    f"({len(keys) - len(self._indices):,} missing latents skipped)")

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


def collate_fn(batch: list) -> dict | None:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    latents, input_ids, attention_masks = [], [], []
    for item in batch:
        t = _LATENT_CACHE.get(key_to_filename(item["image_key"]))
        if t is None:
            continue
        latents.append(t.float())
        input_ids.append(torch.as_tensor(item["input_ids"]))
        attention_masks.append(torch.as_tensor(item["attention_mask"]))
    if not latents:
        return None
    return {
        "pixel_values":   torch.stack(latents),          # (B, 4, 64, 64) float32
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    lr_scheduler,
    device: torch.device,
    epoch: int,
    noise_scheduler: DDPMScheduler,
    grad_accum: int,
    ema: EMA,
    global_step: int,
    use_wandb: bool = False,
) -> tuple[float, int]:
    """
    One epoch. Latents arrive pre-encoded — no VAE forward pass here.
    Loss: MSE between UNet-predicted noise and actual noise added to latents.
    """
    model.train()
    # VAE and CLIP are frozen — keep in eval even if model is in train mode
    if hasattr(model, "module"):
        model.module.vae.eval()
        model.module.text_encoder.eval()
    else:
        model.vae.eval()
        model.text_encoder.eval()

    total_loss = 0.0
    step_count = 0
    accum_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        if batch is None:
            continue
        try:
            # Latents are already encoded — skip VAE entirely
            latents = batch["pixel_values"].to(device, non_blocking=True)
            ids     = batch["input_ids"].to(device, non_blocking=True)
            mask    = batch["attention_mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Sample timesteps
                t = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (latents.size(0),), device=device, dtype=torch.long
                )
                # Forward diffusion: add noise to clean latents
                noisy_latents, noise = noise_scheduler.add_noise(latents, t)

                # Text conditioning
                unet = model.module if isinstance(model, nn.DataParallel) else model
                text_emb = unet.encode_text(ids, mask)

                # UNet predicts noise
                noise_pred = model(noisy_latents, t, text_emb)

                # MSE loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (model.module if isinstance(model, nn.DataParallel)
                     else model).trainable_parameters(),
                    max_norm=1.0, error_if_nonfinite=False
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()

                total_loss += accum_loss
                step_count += 1
                accum_loss  = 0.0
                global_step += 1

            if step_count > 0:
                avg = total_loss / step_count
                vram = torch.cuda.memory_reserved(device) / 1e9
                pbar.set_postfix(loss=f"{avg:.4f}",
                                 lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                                 vram=f"{vram:.1f}GB")

            if use_wandb and step % 100 == 0 and step_count > 0:
                try:
                    wandb.log({
                        "train/loss": total_loss / step_count,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/global_step": global_step,
                        "train/vram_gb": torch.cuda.memory_reserved(device) / 1e9,
                    })
                except Exception:
                    pass

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM at step {step}. Reduce batch_size.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
            logger.warning(f"Step {step} skipped: {e}")
            optimizer.zero_grad(set_to_none=True)
            continue

    avg_loss = total_loss / max(step_count, 1)
    logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Steps: {step_count:,}")
    return avg_loss, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

FIXED_VAL_PROMPTS = [
    "a beautiful sunset over mountain peaks",
    "a photorealistic portrait of a woman with blue eyes",
    "a dog playing in a field of flowers",
    "a futuristic city skyline at night with neon lights",
    "a wooden cabin in a snowy forest",
    "a close-up of a red rose with water droplets",
    "a bowl of fresh fruit on a wooden table",
    "a sailing boat on a calm ocean at golden hour",
]


@torch.no_grad()
def validate(
    model,
    ema: EMA,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    epoch: int,
    output_dir: str,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
):
    # Use single GPU (GPU 0) for validation — unwrap DataParallel
    unet = model.module if isinstance(model, nn.DataParallel) else model
    unet.eval()

    sched = DDIMScheduler()
    sched.set_timesteps(num_steps, device=device)

    cond_tok = tokenizer(
        FIXED_VAL_PROMPTS, padding="max_length",
        max_length=77, truncation=True, return_tensors="pt"
    ).to(device)
    uncond_tok = tokenizer(
        [""] * len(FIXED_VAL_PROMPTS), padding="max_length",
        max_length=77, truncation=True, return_tensors="pt"
    ).to(device)

    backup = ema.apply_shadow()
    unet.eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond_emb   = unet.encode_text(cond_tok.input_ids,   cond_tok.attention_mask)
            uncond_emb = unet.encode_text(uncond_tok.input_ids, uncond_tok.attention_mask)
            ctx = torch.cat([uncond_emb, cond_emb], dim=0)

            latents = torch.randn(
                len(FIXED_VAL_PROMPTS), 4, 64, 64, device=device,
                generator=torch.Generator(device).manual_seed(42)
            )
            for t in sched.timesteps:
                latent_in = torch.cat([latents] * 2)
                t_batch   = torch.full((latent_in.shape[0],), t, dtype=torch.long, device=device)
                noise_pred = unet(latent_in, t_batch, ctx)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latents = sched.step(guided, t, latents)

            images = unet.decode_latents(latents).clamp(-1.0, 1.0)
            images = (images + 1.0) / 2.0
    finally:
        ema.restore(backup)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"val_epoch_{epoch:03d}.png")
    save_image(images.float().cpu(), path, nrow=4, normalize=False)
    logger.info(f"✅ Validation grid saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, lr_scheduler, ema, epoch, global_step, best_loss, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    unet = model.module.unet if isinstance(model, nn.DataParallel) else model.unet
    ckpt = {
        "epoch": epoch, "global_step": global_step, "best_loss": best_loss,
        "unet_state_dict":        unet.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "ema_state_dict":         ema.state_dict(),
    }
    path = os.path.join(ckpt_dir, f"sd_epoch_{epoch:03d}.pt")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(ckpt_dir, "sd_latest.pt"))
    logger.info(f"✅ Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, lr_scheduler, ema, ckpt_path, device="cuda"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    unet = model.module.unet if isinstance(model, nn.DataParallel) else model.unet

    if "unet_state_dict" in ckpt:
        unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    elif "model_state_dict" in ckpt:
        full_sd = ckpt["model_state_dict"]
        unet_sd = {}
        for k, v in full_sd.items():
            for prefix in ["_fsdp_wrapped_module.", "module.", "_orig_mod."]:
                if k.startswith(prefix): k = k[len(prefix):]
            if k.startswith("unet."): unet_sd[k[5:]] = v
        unet.load_state_dict(unet_sd, strict=False)

    if "optimizer_state_dict" in ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e: logger.warning(f"Could not load optimizer: {e}")
    if "lr_scheduler_state_dict" in ckpt:
        try: lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        except Exception as e: logger.warning(f"Could not load scheduler: {e}")
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_loss   = ckpt.get("best_loss", float("inf"))
    logger.info(f"✅ Resumed from epoch {start_epoch} | step {global_step} | loss {best_loss:.4f}")
    return start_epoch, global_step, best_loss


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    gc.collect()

    # Primary device — GPU 0
    device = torch.device("cuda:0")
    n_gpus = torch.cuda.device_count()

    logger.info("=" * 70)
    logger.info("🚀 STABLE DIFFUSION — RTX PRO 4500 OPTIMIZED TRAINING")
    logger.info("=" * 70)
    logger.info(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name} | {props.total_memory/1e9:.1f} GB VRAM")
    logger.info(f"Effective batch: {args.batch_size} × {n_gpus} GPUs × {args.grad_accum} accum "
                f"= {args.batch_size * n_gpus * args.grad_accum}")
    logger.info("=" * 70)

    if args.use_wandb:
        wandb.init(
            project="stable-diffusion",
            config=vars(args),
            name=f"4500x{n_gpus}-ep{args.epochs}-{datetime.now().strftime('%m%d_%H%M')}",
            resume="allow"
        )

    # ── Load latents into RAM first ───────────────────────────────────────────
    global LATENT_DIR
    LATENT_DIR = Path(args.latent_dir)
    if not LATENT_DIR.exists():
        raise FileNotFoundError(
            f"Latent directory not found: {LATENT_DIR}\n"
            f"Run encode_latents_fast.py first."
        )
    load_latent_cache(LATENT_DIR)

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Loading pretrained VAE (frozen)...")
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)

    logger.info("Loading pretrained CLIP text encoder (frozen)...")
    text_enc = PretrainedCLIPTextEncoder(model_id="openai/clip-vit-large-patch14").to(device)

    logger.info("Building UNet (trainable)...")
    unet = UNetModel(
        in_ch=4, out_ch=4, ch=320,
        res_blks=2, attn_lvls=(1, 2, 3),
        ch_mults=(1, 2, 4, 4), heads=8,
        t_dim=320, ctx_dim=768,
        grad_ckpt=args.grad_ckpt
    )

    noise_scheduler = DDPMScheduler(
        steps=1000, beta_start=0.00085, beta_end=0.012, schedule="scaled_linear"
    )

    model = StableDiffusionModel(vae, text_enc, unet, noise_scheduler).to(device)

    # channels_last for conv efficiency
    model.unet = model.unet.to(memory_format=torch.channels_last)

    # Wrap in DataParallel if multiple GPUs available
    if n_gpus > 1:
        gpu_ids = list(range(n_gpus))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"✅ DataParallel across GPUs: {gpu_ids}")
    else:
        logger.info("✅ Single GPU mode")

    # torch.compile on UNet — 20-40% speedup on Blackwell after ~60s warmup
    # fullgraph=False allows unsupported ops to fall back gracefully
    logger.info("Compiling UNet with torch.compile (reduce-overhead)...")
    inner = model.module if isinstance(model, nn.DataParallel) else model
    inner.unet = torch.compile(inner.unet, mode="reduce-overhead", fullgraph=False)
    logger.info("✅ UNet compiled")

    if args.grad_ckpt:
        inner = model.module if isinstance(model, nn.DataParallel) else model
        inner.enable_gradient_checkpointing()
        logger.info("✅ Gradient checkpointing enabled")

    inner = model.module if isinstance(model, nn.DataParallel) else model
    stats = inner.count_parameters()
    logger.info(f"✅ UNet: {stats['trainable']:,} trainable params")

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=0.9999)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    logger.info("Loading dataset...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    train_ds, val_ds = build_dataset(args.cache_path, args.val_size)

    # LatentSampler filters to only samples with cached latents
    sampler = LatentSampler(train_ds, epoch=0)

    # With latents in RAM, num_workers > 0 speeds up collation
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size * n_gpus,  # DataParallel splits this across GPUs
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True
    )
    logger.info(f"✅ Train loader: {len(train_loader):,} batches/epoch")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    inner = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = AdamW(
        inner.trainable_parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8, fused=True
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1),
                              eta_min=args.lr * 1e-2),
        ],
        milestones=[warmup_steps]
    )
    logger.info(f"✅ LR: {warmup_steps} warmup + cosine over {total_steps:,} steps")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    best_loss   = float("inf")

    if args.resume:
        try:
            start_epoch, global_step, best_loss = load_checkpoint(
                model, optimizer, lr_scheduler, ema,
                args.resume, device=str(device)
            )
        except Exception as e:
            logger.error(f"Resume failed: {e}. Starting from scratch.")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"\n🎯 Training epochs {start_epoch}–{args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'─'*70}\nEPOCH {epoch}/{args.epochs}\n{'─'*70}")

        # Rebuild sampler with new epoch seed for shuffling variety
        train_loader.sampler.__init__(train_ds, epoch=epoch)

        avg_loss, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch,
            noise_scheduler=noise_scheduler,
            grad_accum=args.grad_accum,
            ema=ema,
            global_step=global_step,
            use_wandb=args.use_wandb,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"🌟 New best loss: {best_loss:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, lr_scheduler, ema,
                epoch, global_step, best_loss, args.ckpt_dir
            )

        if epoch % args.val_every == 0 or epoch == args.epochs:
            validate(
                model=model, ema=ema, tokenizer=tokenizer,
                device=device, epoch=epoch, output_dir=args.output_dir,
            )

        if args.use_wandb:
            try:
                wandb.log({"epoch": epoch, "train/loss": avg_loss, "train/best_loss": best_loss})
            except Exception:
                pass

        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ TRAINING COMPLETE — Best Loss: {best_loss:.4f}")
    logger.info(f"{'='*70}")
    if args.use_wandb:
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Training — 2x RTX PRO 4500 Optimized")

    # Data
    parser.add_argument("--cache_path",   type=str, required=True)
    parser.add_argument("--latent_dir",   type=str,
                        default="/workspace/StableDiffusion/laion_latents")
    parser.add_argument("--val_size",     type=int, default=5000)

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=128,
                        help="Per-GPU batch size. Total = batch_size × num_gpus × grad_accum")
    parser.add_argument("--grad_accum",   type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--num_workers",  type=int,   default=8,
                        help="DataLoader workers. Safe with latents in RAM.")
    parser.add_argument("--grad_ckpt",    action="store_true")

    # Checkpointing
    parser.add_argument("--save_every",   type=int, default=1)
    parser.add_argument("--val_every",    type=int, default=2)
    parser.add_argument("--ckpt_dir",     type=str, default="/workspace/checkpoints")
    parser.add_argument("--output_dir",   type=str, default="/workspace/outputs")
    parser.add_argument("--resume",       type=str, default=None)

    # Logging
    parser.add_argument("--use_wandb",    action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    main(args)