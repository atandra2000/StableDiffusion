# Stable Diffusion 1.x — From-Scratch Training Summary

**Project:** Custom Stable Diffusion model trained entirely from scratch  
**Hardware:** 2× RTX 5090 (33.7 GB VRAM each), DDP + NCCL  
**Platform:** RunPod  
**Total training epochs:** 48 (in progress)  
**Best loss achieved:** 0.0947 (epoch 16)

---

## Model Architecture

- **UNet:** 860M parameters, ch=320, 8 attention heads, 4-stage encoder/decoder (ch_mults=(1,2,4,4) → 320/640/1280/1280), attn_lvls=(1,2,3), 2 res_blks/stage, t_dim=320, ctx_dim=768
- **Init:** Zero-initialized output projections (ResNet conv2, attention proj, cross-attn `to_out`, MLP final, SpatialTransformer `proj_out`, UNet `conv_out`)
- **Attention:** Self + cross-attention via PyTorch SDPA (Flash + mem-efficient, math kernel disabled)
- **VAE:** Frozen `stabilityai/sd-vae-ft-mse` (BF16 during training; uses `posterior.mean`, not sampled)
- **Text encoder:** Frozen `openai/clip-vit-large-patch14` (`last_hidden_state` → 77×768)
- **Scheduler:** DDPM scaled_linear betas (0.00085 → 0.012, 1000 steps) for training; DDIM for inference
- **Precision:** BF16 autocast, no GradScaler (Blackwell has native BF16, no FP16 underflow)
- **Memory format:** Currently `channels_last` in training (`SD_Train.py:341,606`); during Phase 4 (VGGFace2) `contiguous_format` was required on sm_120 — reverted back to `channels_last` once the build stabilized

---

## Training Stack

| Component | Detail |
|-----------|--------|
| Distributed | DDP + NCCL, 2 GPUs, `gradient_as_bucket_view=True`, `find_unused_parameters=False` |
| Optimizer | Fused AdamW (betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2; falls back to non-fused if unavailable) |
| LR schedule | `SequentialLR`: LinearLR warmup (start_factor=1e-2 → 1.0) + CosineAnnealingLR (eta_min = lr·1e-2) |
| Loss weighting | Min-SNR (default γ=5.0) |
| CFG training | Dropout (0.05 → 0.15 across phases) — uses precomputed unconditional embedding |
| Gradient checkpointing | Enabled on every `UNetResBlock` (~40% VRAM saving) via `torch.utils.checkpoint` (use_reentrant=False) |
| Gradient clipping | `clip_grad_norm_` max_norm=1.0 |
| `no_sync()` on accum | DDP all-reduce skipped on non-optimizer steps |
| EMA | GPU-resident shadow, decay=0.9999 with warmup `d = min(decay, (1+step)/(10+step))` |
| Batch size | 24/GPU × 2 GPUs × 2 grad_accum = 96 effective |
| Data | Latents loaded fully into RAM (`LATENT_FRACTION=1.0`, 16-thread `.npy` loader) |
| TF32 / SDP | TF32 enabled; Flash + mem-efficient SDP enabled, math SDP disabled |
| CUDA alloc | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| torch.compile | **Disabled** — conflicts with gradient checkpointing's dynamo-disabled forward |
| Monitoring | WandB (`stable-diffusion` project) |
| Fault tolerance | Atomic `.tmp` → `os.replace` saves; optional `--save_steps` mid-epoch checkpoints (e.g. every ~1500 steps) |

---

## Training Phases

### Phase 1 — LAION 1.3M Broad (Epochs 1–10)
- **Dataset:** LAION-2B-en aesthetic ≥6.5, 1,315,411 images
- **Filters:** aesthetic ≥6.5, CLIP sim ≥0.28, 512×512+, no watermark/NSFW
- **LR:** 1e-5 peak, 500 warmup steps
- **Epoch time:** ~3 hrs
- **Loss:** 0.22 → ~0.149 (epoch 7: 0.1507, epoch 8: 0.1508, epoch 9: 0.1249, epoch 10: 0.1247)
- **Interrupts:** Epochs 4, 9 stopped early via deliberate `KeyboardInterrupt` (manual termination — *not* VRAM/OOM failures). Peak VRAM observed ~25.3 GB on a 33.7 GB card.
- **Key learning:** Coarse structure, color statistics, spatial frequency

### Phase 2 — LAION 400k High-Quality (Epochs 11–17)
- **Dataset:** LAION filtered aesthetic ≥7.5, watermark <0.15 (script default; ≤0.25 in blog), CLIP sim ≥0.30 → 213,458 images
- **LR:** 1e-5 peak (fresh restart), 500 warmup
- **Epoch time:** ~30 min
- **Loss:** 0.1260 (ep 11) → 0.1254 → 0.1254 → 0.1248 → 0.1026 (ep 15) → **0.0947 (epoch 16, best ever)** → 0.1083 (ep 17)
- **Interrupts:** Epochs 15 and 17 were stopped early via deliberate `KeyboardInterrupt` (manual termination — *not* OOM).
- **Key learning:** This was the single biggest quality jump in the entire project. High-aesthetic filtering > raw scale.
- **Note:** Epoch 16 used Min-SNR gamma=2.0 (too low, face geometry issues). Epoch 15 EMA weights recommended for inference.

### Phase 3 — DiffusionDB + JourneyDB Mixed (Epochs 18–22)
- **Dataset:** ~482k images from DiffusionDB (500 shards) + JourneyDB (10 archives) → ~705k latents
- **LR:** 1e-5 (restart)
- **Epoch time:** ~1 hr
- **Loss:** ~0.120 → 0.1191
- **Note:** Loss jumped from 0.0947 → 0.12 on domain shift (synthetic → real). Expected behavior. Epoch 22 used for inference.

### Phase 4 — VGGFace2 Face Fine-Tuning (Epochs 23–29+)
- **Dataset:** VGGFace2, 51,786 images @ 512×512, template captions
- **LR:** 2e-6 (surgical fine-tune)
- **Warmup:** 200 steps
- **CFG dropout:** 0.15 (↑ from 0.05)
- **Epoch time:** ~2.3 hrs (999k samples)
- **Key fix needed:** `channels_last` → `contiguous_format` for sm_120 Blackwell
- **Result:** Face anatomy dramatically improved — bilateral eye symmetry, correct nose/mouth, skin texture

### Phase 5 — COCO Full-Body Fine-Tuning (Epochs 30–38)
- **Dataset:** COCO detection-datasets, person bbox ≥55% height filter → 59,494 images
- **LR:** 1.5e-6
- **Warmup:** 150 steps
- **Epoch time:** ~17 min
- **Result:** Background integration improved, body proportions corrected. Face remained slightly elongated.

### Phase 6 — Mixed Consolidation (Epochs 39–42)
- **Dataset:** LAION 150k + VGGFace2 50k + COCO 58k = 250k mixed (60/20/20 ratio)
- **LR:** 1e-6
- **Warmup:** 100 steps
- **Epoch time:** ~17 min
- **Loss:** ~0.12 at epoch 42
- **Result:** Scene quality restored (forest/landscape/city near-perfect). Face and body gains preserved.

### Phase 7 — Final Comprehensive Consolidation (Epochs 43–48, in progress)
- **Dataset:** LAION 213k + DM 250k + VGGFace2 51k + COCO 58k = 572k (37/44/9/10%)
- **LR:** 1e-6
- **Warmup:** 150 steps
- **Epoch time:** ~40 min
- **Status:** Epochs 43–44 completed (loss 0.1202/0.1193). Resumed from epoch 044. Pod CUDA issues causing restart difficulties.

---

## Loss History

| Epoch | Loss | Phase | Notes |
|-------|------|-------|-------|
| 1 | ~0.220 | P1 | Start |
| 2 | 0.1583 | P1 | Large initial drop |
| 7 | 0.1507 | P1 | |
| 8 | 0.1508 | P1 | |
| 9 | 0.1249 | P1 | Resumed after crash |
| 10 | 0.1247 | P1 | End of broad phase |
| 11 | 0.1260 | P2 | Dataset switch (filtered LAION) |
| 12 | 0.1254 | P2 | |
| 13 | 0.1254 | P2 | |
| 14 | 0.1248 | P2 | |
| 15 | 0.1026 | P2 | (parallel run: 0.1030) |
| 16 | **0.0947** | P2 | **Best ever** (γ=2.0) |
| 17 | 0.1083 | P2 | γ back to 3.0 |
| 18 | 0.1207 | P3 | Domain shift jump (early DM mix) |
| 19 | 0.1037–0.1201 | P3 | branched runs |
| 20 | 0.1029–0.1201 | P3 | branched runs |
| 21 | 0.1030–0.1191 | P3 | |
| 22 | 0.1191 | P3 | End of DM phase |
| 38 | ~0.119 | P5 | |
| 42 | ~0.115 | P6 | Released as `sd_epoch_042.pt` |
| 43 | 0.1202 | P7 | |
| 44 | 0.1193 | P7 | |

---

## Data Pipeline

```
01_download_metadata.py         → snapshot_download LAION-2B-en aesthetic parquet shards
01b_download_diffusiondb.py     → 500 zipped shards of DiffusionDB → 512×512 JPEG tars
01c_download_journeydb_images.py→ 10 JourneyDB tgz archives → 512×512 JPEG tars
02_filter_metadata.py           → Quality filter (aesthetic, CLIP sim, dimensions, watermark, NSFW, dedup)
03_download_images.py           → img2dataset → WebDataset tar shards (LAION pipeline)
03_build_hf_dataset.py          → DiffusionDB/JourneyDB → Arrow HF dataset
04_preprocess_to_cache.py       → Tar shards → image_key + 77-token CLIP IDs → parquet batches
05_build_hf_dataset.py          → Parquet batches → Arrow HF dataset (train/val split)
encode_latents.py               → 4-stage pipeline: shard prefetch → tar extract →
                                  cv2 decode (16 workers, ~3× PIL) → DMA via pinned buffer →
                                  VAE encode (BF16, torch.compile, ~20–30% boost) → fp16 .npy (32 KB each)
SD_Train.py                     → DDP training loop
inference.py                    → DDIM sampling with EMA + monkey-patched no-clamp `step()`
SD_ImageGen.py                  → CUDA inference with proper negative-prompt support
```

---

## Inference Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| DDIM steps | 100 | 50 is insufficient for faces |
| CFG scale | 7.5 (scenes) / 8.5 (portraits) | 9.0+ causes artifacts |
| EMA weights | Required | Live weights are lower quality |
| DDIM clamp | **Removed at inference** | Original `pred_x0.clamp(-1,1)` still in `SD_Model.py:736`; `inference.py` monkey-patches `DDIMScheduler.step` to skip it (SD latents std ≈ 4.0) |
| Negative prompt | Use `SD_ImageGen.py` | `inference.py --negative` is parsed but not wired into `generate()` (still uses empty string) |
| Inference device | CUDA or MPS (Apple Silicon) | `inference.py` auto-detects via `get_device()` |

---

## Key Bugs Fixed

| Bug | Fix |
|-----|-----|
| `total_steps` used dataset length instead of loader length | Broke cosine LR scheduler |
| LR default was 1e-5 not 5e-5 | Corrected |
| V-prediction validation used DDPM alphas | Changed to DDIM alphas |
| DDIM latent clamp `[-1,1]` | Removed at inference via monkey-patch (training-time `SD_Model.py:736` still clamps) |
| EMA prefix stripping | Handles `module.`, `unet.`, `_orig_mod.`, `_fsdp_wrapped_module.` prefixes |
| `channels_last` on sm_120 | Temporarily switched to `contiguous_format` during Phase 4; current `SD_Train.py` uses `channels_last` again |
| Negative prompt `--negative` arg in `inference.py` | Argparsed but not threaded into `generate()`; use `SD_ImageGen.py` for true negative-prompt CFG |
| Validation grid quality | Still uses live UNet weights, not EMA |

---

## Known Issues / Still Unfixed

- `validate()` in `SD_Train.py` *does* call `ema.apply_shadow()` for the validation pass, but the saved `val_epoch_*.png` grids were generated before that fix and still reflect live-weight quality
- Training-time `DDIMScheduler.step` in `SD_Model.py:736` still clamps `pred_x0` to [-1, 1]; only inference scripts patch it out
- `inference.py --negative` parsed but never plumbed into `generate()`; use `SD_ImageGen.py` for negative prompts
- Face shape slightly elongated, eyes slightly too large (Phase 7 should fix)
- Left arm anatomy in full-body (partially fixed in Phase 5)
- `LATENT_DIR` default is `laion_latents/laion_latents` (nested) — verify before launching new phases

---

## Infrastructure Lessons

| Issue | Lesson |
|-------|--------|
| RunPod network bandwidth | ~100 KB/s public, ~300 MB/s datacenter (HuggingFace) |
| pip re-downloading torch | Always use `--no-deps` for package installs |
| HF cache on container disk | Set `HF_DATASETS_CACHE` and `HF_HOME` to `/workspace` |
| PyTorch + sm_120 (RTX 5090) | Requires PyTorch 2.6+ cu124, `channels_last` breaks |
| CUDA 13.2 pods | Incompatible with PyTorch 2.6+cu124 — use CUDA 12.4 image |
| Checkpoint saving | Atomic write (`.tmp` then `os.replace`) prevents corruption |
| Mid-epoch crashes | `save_steps` parameter + batch fast-forward for fault tolerance (also covers deliberate `KeyboardInterrupt` resumes) |
| Latent storage | Pre-encode to .npy (32 KB each), load all to RAM at start |

---

## Dataset Sources Used

| Dataset | Images | Used For |
|---------|--------|---------|
| LAION-2B-en aesthetic | 1.3M / 213k filtered | Phases 1, 2, 6, 7 |
| DiffusionDB | ~205k | Phase 3, 7 |
| JourneyDB | ~277k | Phase 3, 7 |
| VGGFace2 | 51–159k | Phases 4, 6, 7 |
| COCO (detection-datasets) | 59k | Phases 5, 6, 7 |

---

## Output Quality at Epoch 42

| Category | Quality | Status |
|----------|---------|--------|
| Forest / nature | 9.5/10 | ✅ Excellent |
| Landscapes | 9/10 | ✅ Excellent |
| Cyberpunk city | 8.5/10 | ✅ Very good |
| Vehicles (car) | 7/10 | ✅ Good |
| Portrait faces | 7/10 | 🔧 Eyes slightly large |
| Full body | 6/10 | 🔧 Arm anatomy |
| Animals | 6/10 | ⚠️ Face anatomy weak |
| Architecture | 7/10 | ✅ Mostly good |
| Food | 5/10 | ⚠️ Category confusion |

---

## Recommended Next Steps (Post Phase 7)

1. **Rectified flow fine-tuning** — highest effort-to-impact for field relevance
2. **LCM distillation** — enables 1–4 step generation
3. **ControlNet** — adds spatial conditioning
4. **SD_Model_v2.py** — MM-DiT backbone, dual CLIP-L + OpenCLIP-bigG, rectified flow (already designed)

---

## File Structure

### Repo layout (`/Users/atandrabharati/Desktop/Computer Vision/Stable Diffusion/`)
```
├── SD_Model.py                 # UNet + VAE/CLIP wrappers + DDPM/DDIM schedulers
├── SD_Train.py                 # 2× RTX 5090 DDP + BF16 training loop
├── SD_ImageGen.py              # CUDA inference (full negative-prompt CFG support)
├── inference.py                # Apple-Silicon / CUDA inference (DDIM clamp monkey-patched)
├── encode_latents.py           # 4-stage VAE → .npy pipeline
├── 01_download_metadata.py     # LAION parquet snapshot
├── 01b_download_diffusiondb.py # DiffusionDB shards → 512×512 tar
├── 01c_download_journeydb_images.py # JourneyDB archives → 512×512 tar
├── 02_filter_metadata.py       # aesthetic / CLIP / watermark / NSFW / dedup filter
├── 03_download_images.py       # img2dataset LAION downloader
├── 03_build_hf_dataset.py      # DiffusionDB/JourneyDB HF dataset
├── 04_preprocess_to_cache.py   # Tars → parquet (image_key + tokens)
├── 05_build_hf_dataset.py      # Parquet → Arrow HF dataset
├── sd_epoch_042.pt             # Released checkpoint (~12.5 GB)
├── blog_post.md                # Public write-up
├── summary.md                  # (this file)
├── sd-val-imgs/                # val_epoch_001..043.png + ad-hoc test grids
├── sd-logs/                    # Captured `training.log` / `output*.log` runs
├── sd-test-imgs/               # Inference smoke tests
├── generated_images/           # Curated final renders (epoch 42)
├── Training Screenshots/       # WandB / terminal screenshots + claude notes
├── diffusion/                  # local venv (Python 3.12)
├── .vscode/                    # Pyright off, autoimport on
└── __pycache__/                # cached SD_Model bytecode
```

### Training environment (`/workspace/StableDiffusion/` on RunPod)
```
/workspace/StableDiffusion/
├── SD_Model.py / SD_Train.py / inference.py / encode_latents.py / 01-05_*.py
├── checkpoints/             # Epoch checkpoints (~12 GB each)
├── outputs/                 # Validation grids per epoch
├── laion_latents/           # 213k LAION .npy latents
├── vggface_latents/         # 51k VGGFace2 .npy latents
├── coco_latents/            # 59k COCO .npy latents
├── dm_latents/              # 705k DiffusionDB+JourneyDB .npy latents
├── p7_latents/              # 572k mixed symlinks for Phase 7
├── laion_hf_dataset/        # Arrow dataset
├── vggface_hf_dataset_v2/   # Arrow dataset (corrected keys)
├── coco_hf_dataset/         # Arrow dataset
├── dm_hf_dataset/           # Arrow dataset
└── p7_hf_dataset/           # Mixed Arrow dataset for Phase 7
```
