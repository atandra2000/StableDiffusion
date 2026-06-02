"""
SD_Train_v2.py — MM-DiT + Rectified Flow | 2× RTX 5090 (32 GB each) | DDP + BF16
====================================================================================
Training script for SD_Model_v2.py.  Fully optimised for Blackwell dual-GPU setup.

Optimisation stack (identical hardware approach as SD_Train.py v1, redesigned for MM-DiT)
------------------------------------------------------------------------------------------
  1. DDP (DistributedDataParallel, NCCL)  — Both GPUs train the DiT + text projections.
     Frozen CLIP-L and bigG are run locally on each rank (no DDP overhead on ~1.1B frozen params).
  2. BF16 native autocast              — Blackwell BF16: no GradScaler, no fp16 underflow.
  3. Flash Attention (FlashSDP)        — PyTorch SDPA dispatches to Flash Attention 2 on cc≥8.0.
     JointAttention processes (N_img+N_txt)=1178-token sequences in O(N) memory.
  4. Fused AdamW                       — Single CUDA kernel per optimiser step.
  5. Gradient checkpointing            — ~40% VRAM saving on DiT transformer blocks.
  6. Pre-cached latents in RAM         — Zero-copy .npy reads; latents served from numpy cache.
  7. Dual-tokeniser reuse              — CLIP-L and OpenCLIP-bigG share the same BPE tokeniser;
     v1 `input_ids` column works for BOTH encoders without re-preprocessing.
  8. Logit-Normal timestep sampling    — Concentrates training near t=0.5 (hardest region).
  9. GPU-resident EMA                  — 32 GB VRAM: no CPU round-trips needed.
 10. Atomic checkpoints                — Write-to-tmp, then rename; safe against kill-mid-save.
 11. DDP no_sync accumulation          — Suppresses all-reduce on accumulation micro-steps.
 12. Pin memory + prefetch_factor=4    — Maximises PCIe 5.0 / NVLink bandwidth utilisation.

Architecture differences from SD_Train.py (v1)
-----------------------------------------------
  v1:  UNet + single CLIP + DDPM (ε-prediction, 1000 int timesteps, DDIM inference)
  v2:  MM-DiT + dual CLIP + Rectified Flow (v-prediction, float t∈[0,1], Heun inference)

  Trainable params:
    v1: UNet only (~860 M)
    v2: MMDiT (~700 M "small") + DualTextEncoder.proj_l + .proj_g (~2 M)
    All wrapped together in a single DDP module (TrainableWrapper) for clean gradient sync.

Dataset compatibility
---------------------
  The v2 training pipeline is BACKWARD COMPATIBLE with v1 pre-cached latents:
    - Same VAE (sd-vae-ft-mse), same .npy latent format → no re-encoding needed.
    - Same `input_ids` (CLIP tokeniser, max_len=77) used for BOTH CLIP-L and bigG:
      both models share the GPT-2 BPE vocabulary, so one tokenised column is enough.
    - If your dataset has a `caption` column, v2 will tokenise on-the-fly instead.

Launch commands
---------------
  # Recommended — torchrun (handles rank/world_size automatically)
  torchrun --nproc_per_node=2 SD_Train_v2.py --preset small --epochs 20 --use_wandb

  # Fallback — mp.spawn (direct execution)
  python SD_Train_v2.py --preset small --epochs 20

NOTE: torch.compile is disabled intentionally (same reason as v1).
Compiling conflicts with gradient checkpointing's backward recomputation path
on the dynamo-disabled reentrant context. BF16 + Flash SDP runs near-optimally
without compilation on Blackwell.
"""

from __future__ import annotations

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
from typing import Optional, Tuple

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

from SD_Model_v2 import (
    AdvancedStableDiffusionModel,
    AdvancedVAE,
    DualTextEncoder,
    MMDiT,
    MMDiTConfig,
    MMDIT_PRESETS,
    RectifiedFlowScheduler,
    FlowMatchingSampler,
    build_model,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] | %(message)s",
    handlers=[
        logging.FileHandler("training_v2.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── GPU flags — identical to v1; tuned for Blackwell (RTX 5090) ───────────────
# TF32: lets matmuls use TF32 mantissa on Tensor Cores (faster, nearly lossless)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False
torch.set_float32_matmul_precision("high")
# Flash SDP: SDPA dispatcher → Flash Attention 2 on cc≥8.0 (Blackwell = cc 10.0)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


# ═══════════════════════════════════════════════════════════════════════════════
# DDP UTILITIES  (unchanged from v1 — battle-tested patterns)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_ddp(rank: int, world_size: int):
    """
    Initialise the NCCL process group.

    Debugging:
    - "Connection refused on MASTER_PORT": another process owns port 29500.
      Set MASTER_PORT=29501 (or any free port) in the environment.
    - "Timeout waiting for group": one rank is hanging on model/data loading.
      Add barrier() calls after each heavy load to synchronise ranks.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: Optional[int] = None) -> bool:
    """True only on rank 0 (or if DDP is not initialised)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _unwrap(module: nn.Module) -> nn.Module:
    """Strip DDP wrapper to get the underlying nn.Module."""
    return module.module if isinstance(module, DDP) else module


def _strip_prefixes(state_dict: dict) -> dict:
    """
    Remove wrapper prefixes so checkpoints restore across DDP / plain module runs.

    Handles: "module.", "_orig_mod.", "_fsdp_wrapped_module."

    Debugging:
    - If you see unexpected_keys after load_state_dict: the prefix stripping
      didn't catch a wrapper. Print state_dict.keys()[:5] to identify the prefix.
    """
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    out = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def _resolve_resume_path(resume_arg: Optional[str], ckpt_dir: str) -> Optional[str]:
    """
    Auto-resolve latest checkpoint in ckpt_dir when --resume is not given.

    Priority:
      1. Explicit --resume path (file or directory).
      2. ckpt_dir/dit_latest.pt  (latest symlink written after every save).
      3. Newest ckpt_dir/dit_epoch_NNN.pt file.
    """
    def _best_in_dir(d: Path) -> Optional[str]:
        if (d / "dit_latest.pt").exists():
            return str(d / "dit_latest.pt")
        candidates = sorted(d.glob("dit_epoch_*.pt"))
        return str(candidates[-1]) if candidates else None

    if resume_arg:
        p = Path(resume_arg).expanduser()
        return _best_in_dir(p) if p.is_dir() else str(p)
    return _best_in_dir(Path(ckpt_dir).expanduser())


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINABLE WRAPPER  (DiT + text projections → single DDP module)
# ═══════════════════════════════════════════════════════════════════════════════

class TrainableWrapper(nn.Module):
    """
    Bundles MMDiT + DualTextEncoder projection layers into one module.

    Why this exists
    ---------------
    DDP requires a single nn.Module to manage gradient communication.
    In v2, the trainable parameters are split across two objects:
      - model.dit          (~700M for "small" preset)
      - model.text_enc.proj_l  (~1M)
      - model.text_enc.proj_g  (~1.6M)
    Placing all three inside this wrapper means a single DDP(TrainableWrapper)
    call handles all-reduce for every gradient correctly.

    Forward pass responsibility
    ---------------------------
    The wrapper receives RAW (detached) sequences from the frozen CLIP encoders
    (called outside DDP), projects them with proj_l/proj_g, assembles ctx_seq,
    then calls MMDiT.forward().

    Debugging:
    - "unused parameters in DDP": means find_unused_parameters=False is set but
      some params don't get gradients. If n_single=0, single_blocks params are
      never used → set find_unused_parameters=True or remove single_blocks.
    - NaN gradient on proj_l/proj_g early in training: very normal — the CLIP
      sequences arrive as BF16 and small-magnitude projections can produce NaN
      in the first few steps. Enable GradScaler or switch proj layers to fp32.
    """

    def __init__(self, dit: MMDiT, proj_l: nn.Linear, proj_g: nn.Linear, ctx_dim: int):
        super().__init__()
        self.dit     = dit
        self.proj_l  = proj_l   # (768  → ctx_dim//2)
        self.proj_g  = proj_g   # (1280 → ctx_dim//2)
        self.ctx_dim = ctx_dim

    def forward(
        self,
        x_t:       torch.Tensor,   # (B, C, H, W) noisy latent
        t:         torch.Tensor,   # (B,) float in [0,1]
        seq_l_raw: torch.Tensor,   # (B, 77, 768)  detached from CLIP-L
        seq_g_raw: torch.Tensor,   # (B, 77, 1280) detached from OpenCLIP-bigG
        pooled_txt: torch.Tensor,  # (B, 2048) detached pooled concat
    ) -> torch.Tensor:
        """
        Project raw CLIP sequences, assemble context, forward DiT.

        Context assembly (mirrors DualTextEncoder.forward):
            proj_l(seq_l) → (B, 77, ctx_dim//2)  → zero-pad right half
            proj_g(seq_g) → (B, 77, ctx_dim//2)  → zero-pad left half
            stack along seq dim → (B, 154, ctx_dim)

        Debugging:
        - If ctx_seq has NaN: seq_l_raw or seq_g_raw contains NaN.
          Check that CLIP encoders are not producing NaN on degenerate inputs
          (empty captions or very short prompts). Add input_ids.clamp(0, vocab_size-1).
        - If the DiT ignores text (images always look unconditional): verify that
          pooled_txt is non-zero and being passed through to adaLN-Zero. Print
          pooled_txt.abs().mean() — should be ~0.3–1.0 after a few steps.
        """
        half = self.ctx_dim // 2

        # Project each encoder's sequence to ctx_dim/2 in float32 for stability
        p_l = self.proj_l(seq_l_raw.float())   # (B, 77, half)
        p_g = self.proj_g(seq_g_raw.float())   # (B, 77, half)

        # Each encoder occupies its own "slot" in the ctx_dim channel space
        # CLIP-L fills the first half; bigG fills the second half.
        zeros_half = torch.zeros_like(p_l)
        ctx_l = torch.cat([p_l,        zeros_half], dim=-1)   # (B, 77, ctx_dim)
        ctx_g = torch.cat([zeros_half, p_g],        dim=-1)   # (B, 77, ctx_dim)
        ctx_seq = torch.cat([ctx_l, ctx_g], dim=1)            # (B, 154, ctx_dim)

        # DiT forward: velocity prediction
        return self.dit(x_t, t, ctx_seq, pooled_txt)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU-RESIDENT EMA  (kept on VRAM — 32 GB per card is plenty)
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential Moving Average of TrainableWrapper parameters.

    Warm-up formula (same as v1):
        d_eff = min(decay, (1 + step) / (10 + step))
    This ramps EMA gently for the first ~10 steps so early noisy updates
    don't pollute the shadow weights.

    Debugging:
    - Shadow weights diverge from train weights after many steps: expected.
      If they diverge drastically (>10% difference), decay might be too high.
      Inspect: max(|(shadow - train).abs()|) periodically.
    - OOM when creating EMA: you forgot to call .to(device) first. The shadow
      is copied from model.named_parameters() which should already be on-device.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        self.decay      = decay
        self.device     = device or next(model.parameters()).device
        self.step_count = 0
        self.shadow     = {
            n: p.detach().clone().to(self.device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Call once per optimiser step (NOT per accumulation micro-step)."""
        self.step_count += 1
        d = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].lerp_(p.detach().to(self.device), 1.0 - d)

    def apply_shadow(self, model: nn.Module) -> dict:
        """Swap live weights → EMA shadow.  Returns backup dict to restore later."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        """Restore live training weights from backup."""
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {
            "shadow":     {k: v.cpu() for k, v in self.shadow.items()},
            "step_count": self.step_count,
            "decay":      self.decay,
        }

    def load_state_dict(self, state: dict):
        self.shadow     = {k: v.to(self.device) for k, v in state["shadow"].items()}
        self.step_count = state["step_count"]
        self.decay      = state.get("decay", self.decay)


# ═══════════════════════════════════════════════════════════════════════════════
# CFG DROPOUT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_cfg_dropout_v2(
    seq_l:        torch.Tensor,   # (B, 77, 768)
    seq_g:        torch.Tensor,   # (B, 77, 1280)
    pooled:       torch.Tensor,   # (B, 2048)
    uncond_seq_l: torch.Tensor,   # (77, 768)
    uncond_seq_g: torch.Tensor,   # (77, 1280)
    uncond_pooled: torch.Tensor,  # (2048,)
    dropout_p:    float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly replace a fraction of samples with unconditional embeddings.

    This is the v2 equivalent of v1's _apply_cfg_dropout, extended to handle
    the dual-encoder context.  All three tensors are replaced simultaneously
    so the model never sees a partial conditioning (e.g. CLIP-L uncond + bigG cond).

    Args:
        seq_l, seq_g, pooled:  Conditional embeddings for the batch.
        uncond_*:              Pre-encoded embeddings for the empty string.
        dropout_p:             Probability of replacing each sample (e.g. 0.1).

    Returns:
        Modified (seq_l, seq_g, pooled) with some samples replaced.

    Debugging:
    - If CFG doesn't work at inference: the model didn't see unconditional inputs
      during training. Verify dropout_p > 0 and check `drop.sum()` > 0 per batch.
    - If images look unconditional at any guidance_scale: uncond_seq_l/g are the
      same as cond. Make sure you tokenised an EMPTY STRING (""), not None.
    - Memory: we expand uncond tensors to batch size inline; no extra allocation
      is held between calls.
    """
    if dropout_p <= 0.0:
        return seq_l, seq_g, pooled

    B = seq_l.shape[0]
    drop = torch.rand(B, device=seq_l.device) < dropout_p

    if not drop.any():
        return seq_l, seq_g, pooled

    # Expand unconditional tensors to full batch size before masking
    uncond_l_batch = uncond_seq_l.unsqueeze(0).expand(B, -1, -1)   # (B, 77, 768)
    uncond_g_batch = uncond_seq_g.unsqueeze(0).expand(B, -1, -1)   # (B, 77, 1280)
    uncond_p_batch = uncond_pooled.unsqueeze(0).expand(B, -1)       # (B, 2048)

    # Replace dropped samples with unconditional embeddings
    seq_l  = torch.where(drop[:, None, None], uncond_l_batch, seq_l)
    seq_g  = torch.where(drop[:, None, None], uncond_g_batch, seq_g)
    pooled = torch.where(drop[:, None],        uncond_p_batch, pooled)

    return seq_l, seq_g, pooled


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — RAM-cached latents (.npy) + dual tokeniser
# ═══════════════════════════════════════════════════════════════════════════════

LATENT_DIR: Path = Path("laion_latents")
_LATENT_CACHE: dict = {}


def key_to_filename(image_key: str) -> str:
    """Same filename encoding as v1 — ensures latent cache reuse."""
    return image_key.replace("/", "_").replace("::", "__") + ".npy"


def load_latent_cache(latent_dir: Path, fraction: float = 1.0):
    """
    Load pre-computed .npy latents into RAM using 16 I/O threads.

    Latents encoded with the 4-channel sd-vae-ft-mse VAE are fully reusable
    from the v1 pipeline — no re-encoding needed when switching to v2.

    Debugging:
    - "No .npy files found": check latent_dir path and confirm the latent
      preprocessing script (04_preprocess_to_cache.py) ran successfully.
    - RAM OOM: reduce `fraction` to e.g. 0.5 to load only half the dataset.
    - Mismatched filenames: ensure key_to_filename() here matches the one
      used in the latent preprocessing script.
    """
    all_files = sorted(latent_dir.glob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy latent files found in {latent_dir}")

    files = all_files[: max(1, int(len(all_files) * fraction))]
    if is_main():
        logger.info(
            f"Latent folder: {len(all_files):,} files → "
            f"loading {len(files):,} ({fraction:.0%})"
        )

    def _load(f):
        try:
            return f.name, torch.from_numpy(np.load(f).copy())
        except Exception:
            return f.name, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_load, f): f for f in files}
        it = (
            tqdm(as_completed(futs), total=len(files),
                 desc="Caching latents", unit="file",
                 dynamic_ncols=True, smoothing=0.02)
            if is_main() else as_completed(futs)
        )
        for fut in it:
            name, tensor = fut.result()
            if tensor is not None:
                _LATENT_CACHE[name] = tensor

    if is_main():
        logger.info(f"RAM cache ready: {len(_LATENT_CACHE):,} tensors")


def build_dataset(cache_path: str, val_size: int = 500):
    """
    Load the HuggingFace dataset from disk.

    Expected columns (minimum):
      - image_key  : str  — used to look up the .npy latent file
      - input_ids  : list — CLIP/bigG token IDs (shared tokeniser, max_len=77)
        OR caption : str  — raw text; tokenised on-the-fly if input_ids absent
      - attention_mask : list  (optional if input_ids present)

    v1 datasets are directly compatible — no preprocessing needed.

    Debugging:
    - "Dataset not found": run 05_build_hf_dataset.py first.
    - If using a caption column: tokenisation happens inside collate_fn and is
      the bottleneck at batch assembly; increase --num_workers to parallelise.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache not found: '{cache_path}'")
    if is_main():
        logger.info(f"Loading HuggingFace dataset from {cache_path}")

    ds = load_from_disk(cache_path)
    actual_val = max(1, min(val_size, len(ds) // 10))
    split = ds.train_test_split(test_size=actual_val / len(ds), seed=42)

    if is_main():
        logger.info(f"Train: {len(split['train']):,} | Val: {len(split['test']):,}")
    return split["train"], split["test"]


class V2DistributedSampler(DistributedSampler):
    """
    DistributedSampler that pre-filters to samples with a cached latent.

    Identical in spirit to v1's LatentDistributedSampler but with a more
    descriptive name for clarity when diagnosing issues.

    Debugging:
    - "No valid samples": every dataset entry is missing a latent file.
      Confirm _LATENT_CACHE is populated and key_to_filename() matches
      the preprocessing output.
    - Unequal shard sizes: the sampler drops remainder samples so every rank
      sees the same number of batches. This is intentional (drop_last=True).
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=42, drop_last=True)
        keys = dataset["image_key"]
        self._valid = [
            i for i, k in enumerate(keys)
            if key_to_filename(k) in _LATENT_CACHE
        ]
        if not self._valid:
            raise ValueError(
                "No dataset samples matched cached latent files. "
                "Check --latent_dir and the image_key column format."
            )
        if is_main():
            logger.info(
                f"V2DistributedSampler: {len(self._valid):,} valid "
                f"({len(keys) - len(self._valid):,} missing latents)"
            )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        shuffled = [self._valid[i] for i in torch.randperm(len(self._valid), generator=g)]
        total    = len(shuffled) - (len(shuffled) % self.num_replicas)
        per_rank = total // self.num_replicas
        return iter(shuffled[self.rank * per_rank: (self.rank + 1) * per_rank])

    def __len__(self):
        n = len(self._valid)
        return (n - n % self.num_replicas) // self.num_replicas


def build_collate_fn(tokenizer: CLIPTokenizer):
    """
    Build a collate function that handles both pre-tokenised and caption-only datasets.

    Behaviour:
      - If sample has `input_ids`: use directly (v1 compatibility).
      - If sample has `caption`:   tokenise on-the-fly (slower but more flexible).
      - Samples with missing latents are silently skipped.

    Debugging:
    - Batches are sometimes None: all samples in the batch had missing latents.
      Training loop skips None batches. If this is frequent, your latent cache
      is incomplete — check the ratio from V2DistributedSampler logs.
    - Wrong input_ids shape: should be (B, 77). If your dataset has max_length=70,
      pad to 77 here or re-preprocess with max_length=77.
    """

    def collate_fn(batch: list) -> Optional[dict]:
        items = []
        for b in batch:
            if b is None:
                continue
            latent = _LATENT_CACHE.get(key_to_filename(b["image_key"]))
            if latent is None:
                continue

            # Token IDs — prefer pre-tokenised; fall back to on-the-fly
            if "input_ids" in b and b["input_ids"] is not None:
                ids  = torch.as_tensor(b["input_ids"],        dtype=torch.long)
                mask = torch.as_tensor(b["attention_mask"],   dtype=torch.long) \
                    if "attention_mask" in b else torch.ones(77, dtype=torch.long)
            elif "caption" in b:
                enc = tokenizer(
                    b["caption"], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                )
                ids  = enc.input_ids.squeeze(0)
                mask = enc.attention_mask.squeeze(0)
            else:
                continue  # no text available

            # Pad / trim to exactly 77 tokens
            if ids.shape[0] != 77:
                ids  = ids[:77] if ids.shape[0] > 77 else F.pad(ids,  (0, 77 - ids.shape[0]))
                mask = mask[:77] if mask.shape[0] > 77 else F.pad(mask,(0, 77 - mask.shape[0]))

            items.append((ids, mask, latent))

        if not items:
            return None

        ids_stack, masks_stack, latents_stack = zip(*items)
        return {
            "latents":        torch.stack(latents_stack),    # (B, C, H, W)
            "input_ids":      torch.stack(ids_stack),        # (B, 77) — used for BOTH encoders
            "attention_mask": torch.stack(masks_stack),      # (B, 77)
        }

    return collate_fn


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    trainable:     nn.Module,    # TrainableWrapper (unwrapped from DDP)
    optimizer:     torch.optim.Optimizer,
    lr_scheduler,
    ema:           EMA,
    epoch:         int,
    global_step:   int,
    best_loss:     float,
    ckpt_dir:      str,
    tag:           str = "",
):
    """
    Atomic epoch checkpoint save.

    Saves: trainable weights (stripped of DDP prefix), optimizer, LR scheduler,
    EMA shadow, and training metadata.

    Atomic write pattern: write to .tmp file then os.replace() → safe on kill.

    Debugging:
    - "No space left on device": checkpoints are large (~1.4 GB for "small").
      Reduce --save_every or add a rotation policy (keep last N checkpoints).
    - "Checkpoint too slow": use save_steps=0 and save_every=5 to reduce I/O.
    - After resuming: if loss immediately spikes, the optimizer state failed to
      load (e.g. param group mismatch). Check warning logs after load.
    """
    if not is_main():
        return

    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = f"dit_epoch_{epoch:03d}{tag}.pt"
    ckpt = {
        "epoch":                   epoch,
        "global_step":             global_step,
        "best_loss":               best_loss,
        "trainable_state_dict":    _strip_prefixes(trainable.state_dict()),
        "optimizer_state_dict":    optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "ema_state_dict":          ema.state_dict(),
    }

    tmp  = ckpt_dir / f".{filename}.tmp"
    path = ckpt_dir / filename
    torch.save(ckpt, tmp)
    os.replace(tmp, path)                          # atomic rename
    torch.save(ckpt, ckpt_dir / "dit_latest.pt")  # latest symlink
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    trainable:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    lr_scheduler,
    ema:        EMA,
    ckpt_path:  str,
    device:     str = "cuda",
) -> Tuple[int, int, float]:
    """
    Load a checkpoint and restore all training state.

    Robust to:
      - DDP prefix in saved state dicts ("module.*").
      - Missing optimizer / EMA keys (logs warning, doesn't crash).
      - Strict=False for model weights (logs mismatch count, continues).

    Returns:
        (start_epoch, global_step, best_loss)

    Debugging:
    - Large number of missing_keys: the checkpoint is from a different model
      preset. You can still fine-tune with strict=False but quality will be lower.
    - "Optimizer state size mismatch": you changed the preset between runs.
      Delete the optimizer state from the checkpoint to reset it:
          ckpt.pop("optimizer_state_dict"); torch.save(ckpt, path)
    - Loss spikes after resume: EMA loaded with wrong decay. Check ema.decay.
    """
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if is_main():
        logger.info(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # ── Model weights ─────────────────────────────────────────────────────────
    # Support both new ("trainable_state_dict") and legacy key names
    for key in ("trainable_state_dict", "unet_state_dict", "model_state_dict", "state_dict"):
        if key in ckpt:
            sd = _strip_prefixes(ckpt[key])
            result = trainable.load_state_dict(sd, strict=False)
            if is_main() and (result.missing_keys or result.unexpected_keys):
                logger.warning(
                    f"Trainable key mismatches — "
                    f"missing: {len(result.missing_keys)}, "
                    f"unexpected: {len(result.unexpected_keys)}"
                )
            break

    # ── Optimizer, scheduler, EMA ──────────────────────────────────────────
    for key, obj, label in [
        ("optimizer_state_dict",    optimizer,    "optimizer"),
        ("lr_scheduler_state_dict", lr_scheduler, "LR scheduler"),
        ("ema_state_dict",          ema,          "EMA"),
    ]:
        if key in ckpt:
            try:
                obj.load_state_dict(ckpt[key])
            except Exception as e:
                logger.warning(f"Could not restore {label}: {e}")

    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_loss   = ckpt.get("best_loss", float("inf"))

    if is_main():
        logger.info(
            f"Resumed from epoch {start_epoch} | "
            f"step {global_step:,} | best loss {best_loss:.4f}"
        )
    return start_epoch, global_step, best_loss


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING EPOCH
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:          AdvancedStableDiffusionModel,
    ddp_trainable:  DDP,               # DDP-wrapped TrainableWrapper
    loader:         DataLoader,
    optimizer:      torch.optim.Optimizer,
    lr_scheduler,
    device:         torch.device,
    epoch:          int,
    scheduler:      RectifiedFlowScheduler,
    grad_accum:     int,
    ema:            EMA,
    global_step:    int,
    # Unconditional embeddings (pre-computed, on device, for CFG dropout)
    uncond_seq_l:   torch.Tensor,   # (77, 768)
    uncond_seq_g:   torch.Tensor,   # (77, 1280)
    uncond_pooled:  torch.Tensor,   # (2048,)
    # Options
    cfg_dropout:    float = 0.1,
    use_wandb:      bool  = False,
    save_steps:     int   = 0,
    ckpt_dir:       str   = "checkpoints_v2",
    best_loss:      float = float("inf"),
) -> Tuple[float, int]:
    """
    Single training epoch over all batches.

    Key differences from v1 train_epoch:
      1. Velocity prediction loss (not ε-prediction).
      2. Float timestep t∈[0,1] via logit-normal sampling (not randint 0–999).
      3. Min-SNR weighting from RectifiedFlowScheduler (not DDPM alphas).
      4. Dual CLIP encoding outside DDP (frozen, no grad sync overhead).
      5. CFG dropout on all three context tensors simultaneously.

    Debugging:
    - Loss is always exactly 1.0: model is outputting zero velocity.
      Check that proj_out in MMDiT is NOT zero throughout training — its
      gradients should be non-zero from step 2 onward.
    - Loss oscillates wildly between steps: learning rate is too high.
      Add gradient norm logging: after clip_grad_norm, print grad_norm value.
    - DDP all-reduce is very slow: make sure NCCL is using NVLink (not PCIe).
      Set NCCL_P2P_DISABLE=0 and verify with nvidia-smi nvlink --status.
    - Step checkpoints are slow: use save_steps=500 or larger. The I/O blocks
      the training loop on rank 0 only; other ranks continue to next step.
    """
    _unwrap(ddp_trainable).train()
    model.vae.eval()
    model.text_enc.clip_l.eval()
    model.text_enc.clip_g.eval()
    optimizer.zero_grad(set_to_none=True)

    total_loss    = 0.0
    step_count    = 0
    accum_loss    = 0.0
    last_ckpt_step = 0

    pbar = (
        tqdm(loader, desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True)
        if is_main() else loader
    )

    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        try:
            # ── Move data to GPU ──────────────────────────────────────────────
            latents  = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
            ids      = batch["input_ids"].to(device, non_blocking=True)       # (B, 77)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)  # (B, 77)

            # ── Encode text — frozen encoders, no gradient ────────────────────
            # Both CLIP-L and bigG use the same input_ids (shared BPE vocab).
            # Run OUTSIDE DDP wrapper to avoid DDP overhead on frozen params.
            # [DEBUG] If text encoding is very slow: pre-cache text embeddings
            #         with a separate script and load them as .npy, similar to latents.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                seq_l_raw, pool_l = model.text_enc._encode_clip(
                    model.text_enc.clip_l, ids, attn_mask
                )  # (B,77,768), (B,768)
                seq_g_raw, pool_g = model.text_enc._encode_clip(
                    model.text_enc.clip_g, ids, attn_mask
                )  # (B,77,1280), (B,1280)

            # Detach from encoder computation graph; gradients will flow through
            # proj_l / proj_g inside the DDP wrapper instead.
            seq_l_raw  = seq_l_raw.detach()
            seq_g_raw  = seq_g_raw.detach()
            pooled_txt = torch.cat([pool_l, pool_g], dim=-1).detach()  # (B, 2048)

            # ── CFG dropout — replace some samples with uncond embeddings ──────
            # All three tensors are replaced together to prevent partial conditioning.
            seq_l_raw, seq_g_raw, pooled_txt = apply_cfg_dropout_v2(
                seq_l_raw, seq_g_raw, pooled_txt,
                uncond_seq_l, uncond_seq_g, uncond_pooled,
                cfg_dropout,
            )

            # ── Sample timesteps + add noise (rectified flow) ─────────────────
            with torch.no_grad():
                x_t, v_target, t = scheduler.add_noise(latents)
                # [DEBUG] Verify shapes:
                # assert x_t.shape == latents.shape, f"x_t shape mismatch: {x_t.shape}"
                # assert v_target.shape == latents.shape
                # assert t.shape == (latents.shape[0],)

            # ── DiT forward + loss ────────────────────────────────────────────
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v_pred = ddp_trainable(x_t, t, seq_l_raw, seq_g_raw, pooled_txt)
                # v_pred: (B, C, H, W) — predicted velocity field

                # Per-sample MSE, then Min-SNR reweighting
                loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="none")
                loss = loss.mean(dim=[1, 2, 3])  # (B,) — one scalar per sample

                # Min-SNR weighting: downweights extreme-t samples
                snr_weight = scheduler.get_loss_weight(t)   # (B,)
                loss       = (loss * snr_weight).mean() / grad_accum

            # ── Backward — suppress DDP all-reduce on accumulation micro-steps ─
            # [DEBUG] The no_sync() context suppresses gradient all-reduce inside
            # the accumulation window. Only the FINAL step triggers sync.
            # If you see "RuntimeError: Expected to have finished reduction":
            # ensure every rank takes the same number of accumulation steps.
            sync_ctx = (
                ddp_trainable.no_sync()
                if (step + 1) % grad_accum != 0
                else nullcontext()
            )
            with sync_ctx:
                loss.backward()

            accum_loss += loss.item()

            # ── Optimiser step (every grad_accum micro-steps) ─────────────────
            if (step + 1) % grad_accum == 0:
                # Gradient clipping — essential for transformer stability
                # [DEBUG] If grad_norm is consistently >10: lower learning rate.
                # Print grad_norm to diagnose: logger.debug(f"grad_norm={grad_norm:.2f}")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    ddp_trainable.parameters(), max_norm=1.0
                )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update on the raw (unwrapped) trainable module
                ema.update(_unwrap(ddp_trainable))

                total_loss += accum_loss
                step_count += 1
                global_step += 1
                accum_loss  = 0.0

                # ── Step-level checkpoint ─────────────────────────────────────
                if (
                    save_steps > 0
                    and is_main()
                    and (global_step - last_ckpt_step) >= save_steps
                ):
                    _avg   = total_loss / max(step_count, 1)
                    _best  = min(best_loss, _avg)
                    save_checkpoint(
                        _unwrap(ddp_trainable), optimizer, lr_scheduler, ema,
                        epoch, global_step, _best, ckpt_dir,
                        tag=f"_step{global_step:07d}",
                    )
                    last_ckpt_step = global_step

            # ── Progress bar update ───────────────────────────────────────────
            if is_main() and step_count > 0:
                pbar.set_postfix(
                    loss=f"{total_loss / step_count:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    grad=f"{grad_norm.item():.2f}" if step_count > 0 else "—",
                    vram=f"{torch.cuda.memory_reserved(device) / 1e9:.1f}GB",
                )

            # ── W&B step logging ──────────────────────────────────────────────
            if use_wandb and is_main() and step % 100 == 0 and step_count > 0:
                try:
                    wandb.log({
                        "train/loss":        total_loss / step_count,
                        "train/lr":          lr_scheduler.get_last_lr()[0],
                        "train/grad_norm":   grad_norm.item(),
                        "train/global_step": global_step,
                        "train/vram_gb":     torch.cuda.memory_reserved(device) / 1e9,
                    })
                except Exception:
                    pass

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"OOM on rank {device} at step {step}. "
                    f"Reduce --batch_size or switch to a smaller --preset."
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
            # Non-OOM error: log and skip the step
            logger.warning(f"Step {step} skipped — {e}")
            optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(step_count, 1)
    if is_main():
        logger.info(f"Epoch {epoch} | avg loss: {avg_loss:.4f} | steps: {step_count:,}")
    return avg_loss, global_step


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION  (rank 0 only — uses EMA weights, Heun sampler)
# ═══════════════════════════════════════════════════════════════════════════════

FIXED_VAL_PROMPTS = [
    "a majestic mountain landscape at golden hour, photorealistic",
    "a futuristic cityscape at night with neon reflections on wet streets",
    "a close-up portrait of a fox in a forest, bokeh background",
    "abstract art in the style of impressionism, vivid colours and brushstrokes",
    "an astronaut floating in space with Earth visible in the background",
    "a cozy wooden cabin interior with fireplace and bookshelves, warm lighting",
    "a serene Japanese garden with cherry blossoms and a koi pond",
    "a detailed illustration of a dragon in flight over medieval castle",
]


@torch.no_grad()
def validate(
    model:        AdvancedStableDiffusionModel,
    ddp_trainable: DDP,
    ema:          EMA,
    tokenizer:    CLIPTokenizer,
    device:       torch.device,
    epoch:        int,
    output_dir:   str,
    guidance_scale: float = 7.0,
    val_steps:    int   = 20,
    height:       int   = 512,
    width:        int   = 512,
):
    """
    Generate a validation image grid using EMA weights and Heun sampling.

    Uses the same 8 fixed prompts every epoch so visual progress is directly
    comparable.  Seed is fixed to 42 so the same noise is denoised each epoch,
    making improvements (or regressions) clearly visible.

    Differences from v1 validate():
      - Uses FlowMatchingSampler (Heun, 20 steps) instead of DDIMScheduler.
      - CFG computed inside FlowMatchingSampler._predict_with_cfg().
      - EMA weights swapped via ema.apply_shadow / ema.restore instead of
        the context-manager form (for compatibility with DDP inner module).

    Debugging:
    - All images are identical regardless of prompt: CFG is not working.
      Verify uncond_seq_l/g are properly empty-string encodings.
    - Images look like noise after epoch 1: model hasn't converged yet.
      Expected — flow matching models typically need 5-10 epochs to stabilise.
    - "RuntimeError: expected 4D input": latent shape is wrong.
      Check height/width are multiples of patch_size * 8 = 16.
    - VRAM OOM during validation: generating 8 images × CFG doubles batch.
      Reduce len(FIXED_VAL_PROMPTS) or set guidance_scale=1.0 (no CFG).
    """
    if not is_main():
        return None

    trainable_raw = _unwrap(ddp_trainable)
    trainable_raw.eval()
    model.vae.eval()

    os.makedirs(output_dir, exist_ok=True)

    sampler = FlowMatchingSampler(num_steps=val_steps, method="heun")
    B       = len(FIXED_VAL_PROMPTS)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # ── Encode positive prompts ──────────────────────────────────────────
        cond_tok = tokenizer(
            FIXED_VAL_PROMPTS, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        seq_l_raw, pool_l = model.text_enc._encode_clip(
            model.text_enc.clip_l, cond_tok.input_ids, cond_tok.attention_mask
        )
        seq_g_raw, pool_g = model.text_enc._encode_clip(
            model.text_enc.clip_g, cond_tok.input_ids, cond_tok.attention_mask
        )
        pooled_cond = torch.cat([pool_l, pool_g], dim=-1)
        p_l = trainable_raw.proj_l(seq_l_raw.float())
        p_g = trainable_raw.proj_g(seq_g_raw.float())
        half = p_l.shape[-1]
        ctx_cond = torch.cat([
            torch.cat([p_l, torch.zeros_like(p_l)], dim=-1),
            torch.cat([torch.zeros_like(p_g), p_g], dim=-1),
        ], dim=1)  # (B, 154, ctx_dim)

        # ── Encode negative (unconditional) prompts ──────────────────────────
        uncond_tok = tokenizer(
            [""] * B, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        seq_l_u, pool_l_u = model.text_enc._encode_clip(
            model.text_enc.clip_l, uncond_tok.input_ids, uncond_tok.attention_mask
        )
        seq_g_u, pool_g_u = model.text_enc._encode_clip(
            model.text_enc.clip_g, uncond_tok.input_ids, uncond_tok.attention_mask
        )
        pooled_uncond = torch.cat([pool_l_u, pool_g_u], dim=-1)
        p_l_u = trainable_raw.proj_l(seq_l_u.float())
        p_g_u = trainable_raw.proj_g(seq_g_u.float())
        ctx_uncond = torch.cat([
            torch.cat([p_l_u, torch.zeros_like(p_l_u)], dim=-1),
            torch.cat([torch.zeros_like(p_g_u), p_g_u], dim=-1),
        ], dim=1)  # (B, 154, ctx_dim)

    # ── Swap in EMA weights for inference ────────────────────────────────────
    backup = ema.apply_shadow(trainable_raw)
    try:
        # Fixed seed → same noise every epoch for direct visual comparison
        gen    = torch.Generator(device=device).manual_seed(42)
        C      = model.dit.cfg.in_channels
        z_T    = torch.randn(B, C, height // 8, width // 8,
                             device=device, generator=gen)

        def model_fn(x_t, t_batch, txt_ctx, pooled):
            """Adapter: calls trainable_raw.dit directly with assembled context."""
            return trainable_raw.dit(x_t, t_batch, txt_ctx, pooled)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_0 = sampler.sample(
                model_fn, z_T,
                ctx_cond,   pooled_cond,
                ctx_uncond, pooled_uncond,
                guidance_scale=guidance_scale,
            )

        z_0    = z_0.clamp(-4.0, 4.0)
        images = model.decode_latents(z_0).float().clamp(-1.0, 1.0)
        images = (images + 1.0) / 2.0   # [-1,1] → [0,1]

    finally:
        # Always restore training weights — even if an exception occurred
        ema.restore(trainable_raw, backup)

    path = os.path.join(output_dir, f"val_epoch_{epoch:03d}.png")
    save_image(images.cpu(), path, nrow=4, normalize=False)
    logger.info(f"Validation grid saved → {path}")

    trainable_raw.train()
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (per-rank entry point)
# ═══════════════════════════════════════════════════════════════════════════════

def main(rank: int, world_size: int, args: argparse.Namespace):
    """
    Per-GPU training process.  Launched once per GPU by torchrun or mp.spawn.

    Initialisation order:
      1. DDP setup + CUDA memory config.
      2. W&B init (rank 0 only).
      3. Latent cache load (all ranks, RAM).
      4. Model construction + device placement.
      5. Dataset + DataLoader.
      6. Optimizer + LR scheduler.
      7. EMA + checkpoint resume.
      8. DDP wrapping of TrainableWrapper.
      9. Training loop.

    Debugging:
    - Ranks get out of sync: add dist.barrier() after heavy init steps.
    - VRAM usage uneven between GPUs: ensure both GPUs are receiving equal
      batch sizes. Check that DistributedSampler is used (not plain Sampler).
    - NaN loss on first step: usually BF16 overflow in proj_l/proj_g.
      Temporarily run with dtype=torch.float32 in autocast to isolate.
    """
    # Expandable memory segments — reduces CUDA allocator fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.empty_cache()
    gc.collect()

    # ── Startup banner (rank 0 only) ──────────────────────────────────────────
    if is_main():
        logger.info("=" * 72)
        logger.info("  SD v2 — MM-DiT TRAINING  |  2× RTX 5090 DDP + BF16")
        logger.info("=" * 72)
        for i in range(world_size):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | cc {p.major}.{p.minor}")
        logger.info(f"  Preset: {args.preset} | Effective batch: "
                    f"{args.batch_size} × {world_size} GPUs × {args.grad_accum} accum = "
                    f"{args.batch_size * world_size * args.grad_accum}")
        logger.info("=" * 72)

    # ── W&B (rank 0 only) ─────────────────────────────────────────────────────
    if args.use_wandb and is_main():
        try:
            wandb.init(
                project="stable-diffusion-v2",
                config=vars(args),
                name=(f"{args.preset}-2x5090-ep{args.epochs}-"
                      f"{datetime.now().strftime('%m%d_%H%M')}"),
                resume="allow",
                settings=wandb.Settings(init_timeout=120),
            )
            logger.info("W&B initialized successfully")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            args.use_wandb = False

    # ── Load latents into RAM ─────────────────────────────────────────────────
    global LATENT_DIR
    LATENT_DIR = Path(args.latent_dir)
    if not LATENT_DIR.exists():
        raise FileNotFoundError(f"Latent directory not found: {LATENT_DIR}")
    load_latent_cache(LATENT_DIR, fraction=args.latent_fraction)
    # Wait for all ranks to finish loading before proceeding
    if dist.is_initialized():
        dist.barrier()

    # ── Build model ───────────────────────────────────────────────────────────
    if is_main():
        logger.info(f"Building model (preset={args.preset}) ...")

    # VAE — frozen, BF16 for speed
    vae = AdvancedVAE(
        model_id=args.vae_model_id,
        use_fp16=False,        # use BF16 explicitly below
    )
    vae.vae = vae.vae.to(dtype=torch.bfloat16).to(device)

    # Dual text encoders — frozen, BF16
    # [DEBUG] If bigG download hangs: manually download with
    #   huggingface-cli download laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
    text_enc = DualTextEncoder(
        clip_l_id=args.clip_l_id,
        clip_g_id=args.clip_g_id,
        ctx_dim=MMDIT_PRESETS[args.preset].ctx_dim,
        use_fp16=False,   # BF16 below
    )
    text_enc.clip_l = text_enc.clip_l.to(dtype=torch.bfloat16).to(device)
    text_enc.clip_g = text_enc.clip_g.to(dtype=torch.bfloat16).to(device)
    # proj_l and proj_g stay in float32 for gradient stability
    text_enc.proj_l = text_enc.proj_l.to(device)
    text_enc.proj_g = text_enc.proj_g.to(device)

    # MM-DiT — trainable, BF16 during autocast
    cfg = MMDIT_PRESETS[args.preset]
    if args.grad_ckpt:
        cfg.grad_ckpt = True

    dit = MMDiT(cfg).to(device)

    # Rectified flow scheduler
    noise_scheduler = RectifiedFlowScheduler(
        logit_normal_mean=args.logit_normal_mu,
        logit_normal_std=args.logit_normal_sigma,
    )

    # Assemble the full model object (for encode/decode helpers)
    model = AdvancedStableDiffusionModel(vae, text_enc, dit, noise_scheduler)
    model.validate_config()

    # ── TrainableWrapper — bundles DiT + projections for single DDP call ──────
    trainable = TrainableWrapper(
        dit=dit,
        proj_l=text_enc.proj_l,
        proj_g=text_enc.proj_g,
        ctx_dim=cfg.ctx_dim,
    ).to(device)

    if is_main():
        param_counts = {
            "dit_total":    sum(p.numel() for p in dit.parameters()),
            "proj_l":       sum(p.numel() for p in text_enc.proj_l.parameters()),
            "proj_g":       sum(p.numel() for p in text_enc.proj_g.parameters()),
            "clip_l_frozen": sum(p.numel() for p in text_enc.clip_l.parameters()),
            "clip_g_frozen": sum(p.numel() for p in text_enc.clip_g.parameters()),
            "vae_frozen":    sum(p.numel() for p in vae.vae.parameters()),
        }
        total_trainable = param_counts["dit_total"] + param_counts["proj_l"] + param_counts["proj_g"]
        logger.info("Parameter summary:")
        for k, v in param_counts.items():
            logger.info(f"  {k:20s}: {v/1e6:.1f} M")
        logger.info(f"  {'TOTAL TRAINABLE':20s}: {total_trainable/1e6:.1f} M")
        logger.info(f"  VRAM after model load: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    # ── Precompute unconditional embeddings for CFG dropout ───────────────────
    # Encode the empty string once; reuse every step.
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_l_id)
    empty_tok = tokenizer(
        [""], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        seq_l_u, pool_l_u = model.text_enc._encode_clip(
            model.text_enc.clip_l, empty_tok.input_ids, empty_tok.attention_mask
        )
        seq_g_u, pool_g_u = model.text_enc._encode_clip(
            model.text_enc.clip_g, empty_tok.input_ids, empty_tok.attention_mask
        )
    # Remove batch dim → (77, 768), (77, 1280), (2048,)
    uncond_seq_l  = seq_l_u.squeeze(0).detach()
    uncond_seq_g  = seq_g_u.squeeze(0).detach()
    uncond_pooled = torch.cat([pool_l_u, pool_g_u], dim=-1).squeeze(0).detach()

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    if is_main():
        logger.info("Loading dataset ...")
    train_ds, _ = build_dataset(args.cache_path, args.val_size)
    collate_fn  = build_collate_fn(tokenizer)
    sampler     = V2DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    if is_main():
        logger.info(f"DataLoader: {len(train_loader):,} batches/epoch/rank")

    # ── Optimizer — fused AdamW ───────────────────────────────────────────────
    # Higher base LR than v1 (1e-4 vs 1e-5) because flow matching is more robust.
    # Separate param groups: DiT uses standard LR; projections use 10× higher LR
    # to converge the small projection layers faster.
    #
    # [DEBUG] If proj_l/proj_g don't converge: increase their LR multiplier
    #         or verify gradients are flowing (proj_l.weight.grad.norm()).
    proj_params = (
        list(trainable.proj_l.parameters()) +
        list(trainable.proj_g.parameters())
    )
    dit_params = list(trainable.dit.parameters())

    param_groups = [
        {"params": dit_params,  "lr": args.lr,           "name": "dit"},
        {"params": proj_params, "lr": args.lr * 10.0,    "name": "projections"},
    ]
    try:
        optimizer = AdamW(
            param_groups,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,   # Single CUDA kernel — Blackwell CUDA support
        )
        if is_main():
            logger.info("Using fused AdamW")
    except TypeError:
        optimizer = AdamW(
            param_groups,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        if is_main():
            logger.warning("Fused AdamW unavailable; using standard AdamW")

    # ── LR Scheduler — linear warmup → cosine decay ──────────────────────────
    batches_per_epoch = len(train_loader)
    total_steps  = max(1, batches_per_epoch * args.epochs // max(1, args.grad_accum))
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    if warmup_steps < 1:
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 1e-2
        )
    else:
        lr_scheduler = SequentialLR(
            optimizer,
            milestones=[warmup_steps],
            schedulers=[
                LinearLR(optimizer, start_factor=1e-2, end_factor=1.0,
                         total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1),
                                  eta_min=args.lr * 1e-2),
            ],
        )
    if is_main():
        logger.info(
            f"LR scheduler: {warmup_steps} warmup steps + "
            f"cosine over {total_steps:,} total steps"
        )

    # ── EMA + checkpoint resume ───────────────────────────────────────────────
    ema         = EMA(trainable, decay=args.ema_decay, device=device)
    start_epoch = 1
    global_step = 0
    best_loss   = float("inf")

    resume_path = _resolve_resume_path(args.resume, args.ckpt_dir)
    if resume_path:
        try:
            start_epoch, global_step, best_loss = load_checkpoint(
                trainable, optimizer, lr_scheduler, ema,
                resume_path, device=str(device),
            )
        except Exception as e:
            logger.error(f"Resume failed: {e}. Starting from scratch.")
    elif is_main():
        logger.info("No checkpoint found — starting from scratch.")

    # ── DDP wrapping — after resume so weights are correct before syncing ─────
    # find_unused_parameters=False: all trainable params participate in every
    # forward pass. If you set n_single > 0 but the single blocks happen to
    # not be called (e.g. a bug), flip to True to diagnose.
    ddp_trainable = DDP(
        trainable,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=(cfg.n_single > 0),  # True only if single blocks exist
        gradient_as_bucket_view=True,   # reduces peak DDP memory
    )

    if is_main():
        logger.info(
            f"DDP ready | "
            f"grad_ckpt={'ON' if args.grad_ckpt else 'OFF'} | "
            f"VRAM: {torch.cuda.memory_reserved(device)/1e9:.1f} GB"
        )

    if dist.is_initialized():
        dist.barrier()   # all ranks ready before training begins

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        if is_main():
            logger.info("\n" + "─" * 72)
            logger.info(f"  EPOCH {epoch}/{args.epochs}")
            logger.info("─" * 72)

        sampler.set_epoch(epoch)

        avg_loss, global_step = train_epoch(
            model=model,
            ddp_trainable=ddp_trainable,
            loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch,
            scheduler=noise_scheduler,
            grad_accum=args.grad_accum,
            ema=ema,
            global_step=global_step,
            uncond_seq_l=uncond_seq_l,
            uncond_seq_g=uncond_seq_g,
            uncond_pooled=uncond_pooled,
            cfg_dropout=args.cfg_dropout,
            use_wandb=args.use_wandb,
            save_steps=args.save_steps,
            ckpt_dir=args.ckpt_dir,
            best_loss=best_loss,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main():
                logger.info(f"New best loss: {best_loss:.4f}")

        # ── Epoch checkpoint ──────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                _unwrap(ddp_trainable), optimizer, lr_scheduler, ema,
                epoch, global_step, best_loss, args.ckpt_dir,
            )

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.epochs:
            validate(
                model=model,
                ddp_trainable=ddp_trainable,
                ema=ema,
                tokenizer=tokenizer,
                device=device,
                epoch=epoch,
                output_dir=args.output_dir,
                guidance_scale=args.guidance_scale,
                val_steps=args.val_steps,
                height=args.height,
                width=args.width,
            )

        # ── W&B epoch log ─────────────────────────────────────────────────────
        if args.use_wandb and is_main():
            try:
                wandb.log({
                    "epoch":            epoch,
                    "train/epoch_loss": avg_loss,
                    "train/best_loss":  best_loss,
                })
            except Exception:
                pass

        # Barrier: keep all ranks aligned between epochs
        if dist.is_initialized():
            dist.barrier()

    # ── Training complete ─────────────────────────────────────────────────────
    if is_main():
        logger.info("=" * 72)
        logger.info(f"TRAINING COMPLETE — Best Loss: {best_loss:.4f}")
        logger.info(f"Total steps: {global_step:,}")
        logger.info("=" * 72)
        if args.use_wandb:
            wandb.finish()

    cleanup_ddp()


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER + LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SD v2 MM-DiT Training — 2× RTX 5090 DDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--preset", type=str, default="small",
                        choices=["nano", "small", "medium", "large"],
                        help="MM-DiT size preset. nano≈250M, small≈700M, medium≈2B, large≈8B.")
    parser.add_argument("--vae_model_id", type=str,
                        default="stabilityai/sd-vae-ft-mse",
                        help="Pretrained VAE model ID. 4-ch for mse, 16-ch for FLUX.")
    parser.add_argument("--clip_l_id",   type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_g_id",   type=str,
                        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--cache_path",      type=str, default="laion_hf_dataset/train",
                        help="Path to HuggingFace dataset (Arrow format from 05_build_hf_dataset.py).")
    parser.add_argument("--latent_dir",      type=str, default="laion_latents",
                        help="Directory of pre-cached .npy latent files (v1 latents are reusable).")
    parser.add_argument("--val_size",        type=int, default=500)
    parser.add_argument("--latent_fraction", type=float, default=1.0,
                        help="Fraction of latents to load into RAM (reduce if RAM-limited).")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=16,
                        help="Per-GPU batch size. small preset fits 16–24 on 32 GB BF16.")
    parser.add_argument("--grad_accum", type=int,   default=2,
                        help="Gradient accumulation steps. Effective batch = bs × gpus × accum.")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Peak learning rate for DiT. Projections get 10× this value.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int,   default=1000)
    parser.add_argument("--num_workers",  type=int,   default=16,
                        help="DataLoader worker processes per rank.")
    parser.add_argument("--ema_decay",    type=float, default=0.9999)

    # ── Rectified flow / timestep ─────────────────────────────────────────────
    parser.add_argument("--logit_normal_mu",    type=float, default=0.0,
                        help="Mean of logit-normal timestep distribution.")
    parser.add_argument("--logit_normal_sigma", type=float, default=1.0,
                        help="Std of logit-normal timestep distribution. Higher = more spread.")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Prob of replacing conditioning with uncond during training.")

    # ── Optimisations ─────────────────────────────────────────────────────────
    parser.add_argument("--grad_ckpt",    action="store_true",  default=True,
                        help="Gradient checkpointing (~40%% VRAM saving, ~25%% slower).")
    parser.add_argument("--no-grad-ckpt", dest="grad_ckpt",     action="store_false")

    # ── Validation / inference ────────────────────────────────────────────────
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--val_steps",      type=int,   default=20,
                        help="Heun sampler steps for validation. 20 is fast + high quality.")
    parser.add_argument("--height",         type=int,   default=512)
    parser.add_argument("--width",          type=int,   default=512)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument("--save_every",  type=int, default=1)
    parser.add_argument("--save_steps",  type=int, default=0,
                        help="Save step checkpoint every N global steps (0 = disabled).")
    parser.add_argument("--val_every",   type=int, default=1)
    parser.add_argument("--ckpt_dir",    type=str, default="checkpoints_v2")
    parser.add_argument("--output_dir",  type=str, default="outputs_v2")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint or directory. Auto-resolves latest if omitted.")
    parser.add_argument("--use_wandb",   action="store_true")

    args = parser.parse_args()

    # ── Sanity checks before launching ────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No GPU detected.")

    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(
            f"Only {world_size} GPU(s) detected. "
            "This script is optimised for 2 GPUs. Training will still work."
        )

    resolution_min = MMDIT_PRESETS[args.preset].patch_size * 8
    if args.height % resolution_min != 0 or args.width % resolution_min != 0:
        raise ValueError(
            f"--height and --width must be multiples of {resolution_min} "
            f"(patch_size={MMDIT_PRESETS[args.preset].patch_size} × 8). "
            f"Got: {args.height}×{args.width}."
        )

    logger.info(
        f"Launching: preset={args.preset} | "
        f"{world_size} GPU(s) | "
        f"effective batch = {args.batch_size}×{world_size}×{args.grad_accum} = "
        f"{args.batch_size * world_size * args.grad_accum}"
    )

    # ── Launch: torchrun (preferred) or mp.spawn (fallback) ───────────────────
    if "RANK" in os.environ:
        # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)