# I Trained Stable Diffusion From Scratch on 2× RTX 5090s — Here's What Actually Matters

*A 48-epoch deep-dive into 860M parameters, 1.3M filtered images, and every footgun the textbooks don't warn you about.*

---

> **TL;DR**
>
> Over a few months I trained a Latent Diffusion Model from scratch:
>
> - **860M-parameter UNet** (ch=320, ch_mults=(1,2,4,4)) — the full SD 1.x recipe
> - **2× RTX 5090** (Blackwell, 33.7 GB VRAM each), DDP + NCCL on RunPod
> - **48 epochs across 7 phases**, ~1.3M → 213k → 572k mixed dataset
> - **BF16 native, gradient checkpointing, Min-SNR γ, EMA decay 0.9999**
> - **Best loss: 0.0947 (epoch 16)**. Best images: epoch 42.
>
> The hardest problems had nothing to do with transformers, attention layers, or diffusion maths. They were:
>
> - corrupt JPEGs
> - VAE throughput
> - GPU DMA bottlenecks
> - catastrophic forgetting from sequential fine-tuning
> - broken latent distributions
> - and one line of code that silently destroyed image quality for days
>
> This article is everything I wish someone had told me before I started.

*(Insert Image: Hero shot — side-by-side `val_epoch_002.png` (multicolored noise) vs `val_epoch_042.png` (photorealistic scenes).)*

---

## 1. Why "From Scratch" Is Different

Fine-tuning a pretrained SD is a weekend project. Training the UNet from random init is a different sport. You inherit every numerical instability, every dataset wart, and every CUDA quirk that the original authors solved silently before they shipped weights.

If you're going to do this:

- **Plan in phases**, not epochs.
- **Treat the loss curve with deep suspicion.** Visual coherence and MSE are only loosely correlated.
- **Budget 80 % of your time for data, 20 % for the model.** I wrote the UNet in two days. Building, filtering, and encoding the dataset took two weeks.

---

## 2. The Architecture — Every Layer Matters

Diffusion models don't operate on pixels; they operate in a compressed latent space. That single design choice is what makes an 860M-parameter model trainable on a consumer-grade pair of GPUs.

### 2.1 The Big Picture

Three frozen-or-trained components in a tight loop:

1. **VAE** (frozen) — compresses 512×512×3 → 64×64×4 latents (8× spatial compression).
2. **CLIP text encoder** (frozen) — turns a prompt into a (77, 768) sequence.
3. **UNet** (trained) — given a noisy latent, a timestep, and the text embedding, predicts the noise.

### 2.2 VAE: The Latent Translator

I used `stabilityai/sd-vae-ft-mse`, kept in BF16 on Blackwell, and used `posterior.mean` rather than `posterior.sample()` for deterministic, cheaper encoding.

The single most important constant in the entire codebase is the **VAE scale factor of 0.18215**:

```python
latents = posterior.mean * 0.18215   # encode
decoded = vae.decode(latents / 0.18215)
```

This is the empirical standard deviation of the LAION-2B latent distribution. Skip it and your UNet trains on data with the wrong variance — the loss looks vaguely sensible but the model never converges.

### 2.3 CLIP: The Language Bridge

I used `openai/clip-vit-large-patch14`, also frozen. The UNet's cross-attention consumes the full `last_hidden_state` of shape **(B, 77, 768)** — not the pooled CLS vector. That sequence-level view is what allows cross-attention to "look at the word *red* when it draws the hat."

### 2.4 The UNet — 860M Parameters of Denoising

The exact SD 1.x topology:

| Stage | Channels | Spatial | Self-attn | Cross-attn |
|---|---|---|---|---|
| 0 (in) | 320 | 64×64 | — | — |
| 1 | 640 | 32×32 | yes | yes |
| 2 | 1280 | 16×16 | yes | yes |
| 3 | 1280 | 8×8 | yes | yes |
| Bottleneck | 1280 | 8×8 | yes (1st only) | yes |
| Decoder | mirror | mirror | mirror | mirror |

Concretely:

```python
unet = UNetModel(
    in_ch=4, out_ch=4,
    ch=320,
    res_blks=2,
    ch_mults=(1, 2, 4, 4),
    attn_lvls=(1, 2, 3),    # no attention at full 64×64
    heads=8,
    t_dim=320, ctx_dim=768,
)
```

Attention runs through PyTorch's `scaled_dot_product_attention` with **Flash SDP** and the **memory-efficient kernel** both enabled; the math fallback is disabled to make sure I don't silently fall off the fast path.

### 2.5 Zero-Initialisation: The Calm Start

Every residual block's final conv, every attention output projection, every MLP's last linear, and the UNet's `conv_out` are **zero-initialised**:

```python
nn.init.zeros_(self.conv2.weight)
nn.init.zeros_(self.conv2.bias)
```

At step 0, the network predicts zero noise. The first gradient updates start the model in a region where activations are well-conditioned. Without this, large-channel residual blocks blow up in the first few hundred steps.

### 2.6 Schedulers — DDPM for Training, DDIM for Inference

I use the SD 1.x **scaled-linear** beta schedule:

```python
betas = torch.linspace(0.00085**0.5, 0.012**0.5, 1000) ** 2
```

This concentrates small betas near `t=0` (fine-detail noise) and grows them quickly toward `t=999`.

- **Training:** DDPM, 1000 steps, ε-prediction objective.
- **Inference:** DDIM (η=0, deterministic). 25 steps is fine for scenes; **100 steps is required to keep facial detail crisp**.

*(Insert Asset: Architecture diagram — VAE → UNet → CLIP wiring with shapes.)*

> ### Engineering Takeaways
> - The **VAE scale factor (0.18215) is non-negotiable**.
> - **Zero-init** every output projection — your training stability comes from boring places.
> - Use `torch.nn.functional.scaled_dot_product_attention` and **explicitly enable Flash + mem-efficient SDP**.
> - Train DDPM, sample DDIM.

---

## 3. The Data Pipeline — The Real Work

If you take one lesson from this whole project, take this: **brutal filtering beats raw scale**. Every "ugly" image in your batch yanks the gradient sideways.

### 3.1 Two-Stage Filtering

I started from `laion/laion2B-en-aesthetic` (~2 B URL/caption rows) and ran two filtering passes.

**Stage 1 — Broad pretraining (~1.3 M images kept)**

| Filter | Threshold |
|---|---|
| Aesthetic score | ≥ 6.5 |
| CLIP similarity | ≥ 0.28 |
| Min resolution | 512×512 |
| Aspect ratio | 0.5 – 2.0 |
| Watermark prob. | < 0.15 (script default) |
| NSFW | "UNLIKELY" only |
| Caption length | 20 – 300 chars |
| Dedup | URL-level |

The script (`02_filter_metadata.py`) reads, normalises column names across LAION's inconsistent schemas, then writes filtered parquet files. Survival rate from LAION-2B-en: roughly **0.065 %**.

**Stage 2 — Rigorous refinement (213,458 images)**

After Phase 1, I cranked the thresholds:

- Aesthetic ≥ 7.5
- CLIP similarity ≥ 0.30
- Watermark probability < 0.15 (kept tight)

It feels obscene to throw away 99.9 % of your data. Do it anyway. This subset produced the single biggest visual jump of the entire project.

### 3.2 Other Datasets

The full data mix:

| Dataset | Images | Purpose | Phases |
|---|---|---|---|
| LAION-2B-en aesthetic | 1.3 M / 213k filtered | Broad pretraining + refinement | 1, 2, 6, 7 |
| DiffusionDB | ~205k | Synthetic prompt diversity | 3, 7 |
| JourneyDB | ~277k | Midjourney-style aesthetics | 3, 7 |
| VGGFace2 | 51k | Face anatomy | 4, 6, 7 |
| COCO (detection-datasets) | 59k | Full-body / scene integration | 5, 6, 7 |

DiffusionDB and JourneyDB came as zipped image dumps — I rebuilt them into standard `*.tar` WebDataset shards with 512×512 centre-cropped JPEGs and matching `*.txt` captions (`01b_*.py` and `01c_*.py`) so they could feed the same downstream pipeline.

### 3.3 The VAE Latent Encoding Pipeline (the real performance work)

If you encode latents on the fly during training, your GPUs spend most of their time waiting for the VAE. The fix is to **pre-compute every latent once and stream them as `.npy` files** during training.

`encode_latents.py` is a 4-stage concurrent pipeline:

1. **Shard prefetch thread** — copies tar shards from the network mount (RunPod MFS) to local NVMe. A bounded queue (`SHARD_PREFETCH=3`) provides backpressure so `/tmp` never overflows.
2. **Extractor + decode submit thread** — opens each local tar, indexes JPEG members by stem, and fans decode tasks into a worker pool.
3. **Decode pool (16 threads, OpenCV)** — `cv2.imdecode` is ~3× faster than PIL. Each worker decodes, resizes preserving aspect ratio, centre-crops to 512×512, and normalises to `[-1, 1]`.
4. **GPU main thread** — packs results into a **pre-allocated pinned-memory staging buffer**, kicks off an async H2D copy on a **dedicated CUDA stream**, then runs `vae.encode()` under BF16 autocast.

```python
staging = torch.empty(BATCH_SIZE, 3, 512, 512, dtype=torch.float32, pin_memory=True)
h2d_stream = torch.cuda.Stream(device=device)

with torch.cuda.stream(h2d_stream):
    pixel = staging[:n].to(device, non_blocking=True)
torch.cuda.current_stream(device).wait_stream(h2d_stream)

with torch.autocast("cuda", dtype=torch.bfloat16):
    latents = vae.encode(pixel)        # (n, 4, 64, 64), fp16-saved
```

Page-locked memory means the H2D copy goes through **DMA**, bypassing the kernel's software bounce buffer. The dedicated stream overlaps that copy with the next batch's decode futures resolving on the CPU.

I also `torch.compile`'d the VAE in `reduce-overhead` mode, which gave another 20–30 % throughput after the first batch's graph capture. End result: **~1.3 M images → ~48 GB of fp16 latents** (32 KB per file), saved as one `.npy` per image. Resume is `O(1)` via `os.listdir`.

*(Insert Asset: Pipeline diagram — Shard Prefetch → Tar Extract → Parallel Decode → Pinned DMA → VAE → fp16 .npy.)*

> ### Engineering Takeaways
> - **Quality dominates quantity.** Filter ruthlessly.
> - **Pre-compute latents.** Don't pay the VAE cost in your training hot loop.
> - **Pinned memory + dedicated CUDA stream + DMA** is how you actually saturate PCIe 5.0.
> - `torch.compile` is safe **outside** the training loop (encoding, evaluation). Inside the loop it's a trap (see §7).

---

## 4. The Training Loop — Engineering for Scale

Training 860M parameters is mostly plumbing. Get the plumbing right and the model trains. Get it wrong and you'll lose a week to non-deterministic crashes.

### 4.1 Hardware & Distribution

Two RTX 5090s (sm_120, 33.7 GB each), connected by PCIe 5.0, talking via NCCL. I use **DistributedDataParallel** rather than DeepSpeed/FSDP — for a model that fits in a single card's VRAM, DDP's all-reduce is the simplest correct choice.

Key DDP knobs that mattered:

```python
ddp_unet = DDP(
    model.unet,
    device_ids=[rank],
    output_device=rank,
    find_unused_parameters=False,        # every param sees a gradient
    gradient_as_bucket_view=True,        # zero-copy bucket views
)
```

And on gradient accumulation steps, **skip the all-reduce**:

```python
with ddp_unet.no_sync() if (step + 1) % grad_accum != 0 else nullcontext():
    loss.backward()
```

That single line saves ~50 % of inter-GPU traffic.

### 4.2 VRAM Survival Guide

Effective batch = **24 / GPU × 2 GPUs × 2 grad-accum = 96**. To fit it:

1. **BF16 native autocast.** Blackwell does BF16 natively; no `GradScaler` needed because BF16's dynamic range matches FP32.
   ```python
   with torch.autocast("cuda", dtype=torch.bfloat16):
       noise_pred = ddp_unet(noisy_latents, t, text_emb)
   ```
2. **Gradient checkpointing on every UNet residual block** (`use_reentrant=False`). Trades ~30 % compute for ~40 % VRAM. Without it, ch=320 won't fit at this batch size.
3. **TF32 + Flash + mem-efficient SDP enabled, math SDP disabled.** Saves both VRAM and compute.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce fragmentation across long runs.
5. **Channels-last memory format** for the UNet. (Important caveat below — it briefly stopped working on sm_120 during Phase 4.)

I observed peak reserved VRAM around **25.3 GB** during the heaviest epochs — comfortable on a 33.7 GB card.

### 4.3 The "Zero I/O" Training Loop

Because every latent is a 32 KB `.npy`, the entire 213k LAION cache (~7 GB) fits in RAM trivially. At startup, a 16-thread loader streams every file into a `dict[str, torch.Tensor]`:

```python
LATENT_FRACTION = 1.0
with ThreadPoolExecutor(max_workers=16) as pool:
    for fut in as_completed(...):
        _LATENT_CACHE[name] = torch.from_numpy(np.load(f).copy())
```

The custom `LatentDistributedSampler` then yields **only indices whose latent is actually in cache**, shards them evenly across ranks per epoch, and drops the remainder so every GPU sees an equal-sized slice. Result: **zero disk I/O inside the epoch**, GPUs at >95 % utilisation.

### 4.4 Loss Weighting — Min-SNR γ

Vanilla MSE treats every timestep equally. At high noise (large `t`), the signal in `ε` is weak and the gradients are noisy. Min-SNR (Hang et al., 2023) downweights those steps:

```python
def _min_snr_weight(t, sched, gamma):
    acp = sched.alphas_cumprod[t]
    snr = acp / (1.0 - acp).clamp_min(1e-6)
    return (snr.clamp(max=gamma) / snr).float()
```

I trained with **γ=5.0** for most phases. Dropping to γ=2.0 at epoch 16 produced my best loss ever (0.0947) — and silently broke face geometry. More on that in §7. γ=3.0 turned out to be the practical sweet spot.

### 4.5 Classifier-Free Guidance Dropout

For CFG to work at inference, the model has to know how to denoise *without* a prompt. During training I randomly replace the text embedding with a precomputed unconditional embedding:

```python
empty_tok = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt").to(device)
uncond_text_emb = model.encode_text(empty_tok.input_ids, empty_tok.attention_mask).squeeze(0).detach()
# ... reuse on every step
```

CFG dropout ramps from **0.05 (broad pretraining) → 0.15 (fine-tuning)**.

### 4.6 EMA — The Secret Weapon

The Exponential Moving Average of UNet weights is the single largest "free" quality win in the entire project. I keep it **GPU-resident** (no CPU round-trip per step) with decay 0.9999 and a warmup schedule:

```python
d = min(decay, (1 + step) / (10 + step))
self.shadow[n].lerp_(p.detach(), 1.0 - d)
```

The warmup formula prevents early noisy updates from dominating the shadow. At validation time, I swap live weights for the EMA shadow, sample, then restore — `ema.apply_shadow()` / `ema.restore()`.

The visual gap between live and EMA weights is enormous:

- **Live weights:** noisy, colour-leaky, jittery composition.
- **EMA weights:** stable, photorealistic, coherent.

### 4.7 The Optimiser & Scheduler

```python
optimizer = AdamW(
    model.unet.parameters(),
    lr=args.lr,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=True,                          # single CUDA kernel; falls back if unavailable
)
```

LR schedule is **`SequentialLR(LinearLR warmup → CosineAnnealingLR)`** with `eta_min = lr * 1e-2`. Warmup is `min(args.warmup_steps, total_steps // 10)` so short fine-tunes don't get a 500-step warmup out of a 1500-step run.

Gradients are clipped with `clip_grad_norm_(..., max_norm=1.0)` immediately before each optimiser step.

### 4.8 Fault Tolerance

I lost more time to "the pod died at 80 % of an epoch" than to anything else. Two patches:

- **Atomic checkpoint writes.** Save to `*.tmp`, then `os.replace`. A killed process never leaves a half-written `.pt`.
- **`--save_steps`.** A mid-epoch step checkpoint (e.g. every 1500 global steps) lets a crash cost minutes instead of an hour.

*(Insert Asset: Training step flow — Latent batch → BF16 forward → Min-SNR-weighted MSE → DDP all-reduce → AdamW step → EMA update.)*

> ### Engineering Takeaways
> - **BF16 + Flash SDP + gradient checkpointing** is the right Blackwell triple.
> - **`no_sync()` on accumulation steps** halves your DDP traffic.
> - **Pre-encode and RAM-cache** to get a zero-I/O hot loop.
> - **EMA is not optional.** Decay 0.9999, GPU-resident, with warmup.
> - **Atomic saves + step checkpoints** are the cheapest insurance you'll ever buy.

---

## 5. The Multi-Phase Journey — From Noise to Coherence

Training was seven distinct phases, each with its own dataset, learning rate, and tactical goal. Total **48 epochs**, best loss **0.0947** (epoch 16), best images **epoch 42**.

### Phase 1 — LAION Broad Pretraining (Epochs 1–10)

- **Data:** 1,315,411 LAION images, aesthetic ≥ 6.5.
- **LR:** 1e-5 peak, 500-step warmup.
- **Epoch time:** ~3 hours.
- **Loss:** 0.220 → 0.1247.

By **epoch 3 the outputs were literally multicolored static** — ghost-shapes if you squinted. I almost killed the run. By epoch 6 vague blobs started cohering; by epoch 10 the model could roughly distinguish "sunset" from "person."

> **Footnote on the "crashes" at epochs 4 and 9:** these were **not** OOMs. They were deliberate `KeyboardInterrupt`s on my end (config tweaks). Peak reserved VRAM that whole phase was ~25.3 GB — well under the 33.7 GB headroom. The earlier draft of this post called them OOM crashes; that was wrong.

### Phase 2 — LAION Rigorous Refinement (Epochs 11–17)

- **Data:** 213,458 filtered LAION images, aesthetic ≥ 7.5, CLIP sim ≥ 0.30.
- **LR:** 1e-5 (fresh restart), 500-step warmup.
- **Epoch time:** ~30 minutes.
- **Loss:** 0.1260 → **0.0947 (epoch 16, best ever)** → 0.1083 (epoch 17).

This phase produced the **single biggest visual quality jump** of the project. The lesson is uncomfortable: throwing away 84 % of your already-filtered data made the model dramatically better.

At epoch 16 I tried Min-SNR γ=2.0 — got a beautiful loss number, watched faces start melting, reverted to γ=3.0 at epoch 17. (See §7.4.)

Epochs 15 and 17 were stopped early via deliberate `KeyboardInterrupt`. The **epoch 15 EMA checkpoint is still my recommended inference base for pure image quality** before the synthetic-data domain shift.

### Phase 3 — DiffusionDB + JourneyDB (Epochs 18–22)

- **Data:** ~482k synthetic/curated images (DiffusionDB 500 shards + JourneyDB 10 archives → ~705k latents with mirroring).
- **LR:** 1e-5 (restart).
- **Epoch time:** ~1 hour.
- **Loss:** 0.0947 → 0.1207 → 0.1191.

The loss **jumped** when I introduced the new mix. That's expected — the domain shifted from photographic LAION to a heavier synthetic distribution and the model had to re-calibrate. By epoch 22 it had stabilised.

### Phase 4 — VGGFace2 Face Fine-Tuning (Epochs 23–29+)

- **Data:** 51,786 VGGFace2 images @ 512×512 with templated captions ("photorealistic portrait of a person, soft studio lighting").
- **LR:** 2e-6, 200-step warmup.
- **CFG dropout:** 0.05 → 0.15.
- **Epoch time:** ~2.3 hours.

Face anatomy improved dramatically — bilateral eye symmetry, correct nose/mouth ratios, plausible skin texture.

This phase needed a runtime fix: **`channels_last` briefly broke on sm_120** in my PyTorch 2.6+cu124 build. I switched to `contiguous_format` for the duration of Phase 4, then **reverted to `channels_last` once the environment stabilised** — that's what the current `SD_Train.py` ships with. (The earlier blog draft implied the switch was permanent; it wasn't.)

### Phase 5 — COCO Full-Body Fine-Tuning (Epochs 30–38)

- **Data:** COCO `detection-datasets`, filtered to person bbox ≥ 55 % image height → 59,494 images.
- **LR:** 1.5e-6.

Background integration and body proportions improved. But **faces regressed** — the model overwrote some of Phase 4's gains with COCO's utilitarian framing. **This was my first hard lesson in catastrophic forgetting.**

### Phase 6 — Mixed Consolidation (Epochs 39–42)

- **Data:** LAION 150k + VGGFace2 50k + COCO 58k mixed 60/20/20.
- **LR:** 1e-6.

Scene quality snapped back to near-perfect, face/body gains were preserved. At epoch 42 the model hit its visual "sweet spot." That checkpoint became **`sd_epoch_042.pt`** — the file I now use as my reference base.

### Phase 7 — Final Comprehensive Consolidation (Epochs 43–48)

- **Data:** LAION 213k + DiffusionDB/JourneyDB 250k + VGGFace2 51k + COCO 58k ≈ 572k total (37 / 44 / 9 / 10 %).
- **LR:** 1e-6.

Epochs 43–44 came in at losses 0.1202 / 0.1193 — flat, healthy, no signs of divergence. Training stayed in progress while I battled pod-level CUDA restart issues on RunPod (cu124 vs cu13.2 image mismatches).

*(Insert Asset: Validation time-lapse grid — epoch 1 → 10 → 17 → 22 → 42 → 48.)*

### Loss History at a Glance

| Epoch | Loss | Phase | Note |
|---|---|---|---|
| 1 | ~0.220 | P1 | Start |
| 2 | 0.1583 | P1 | First big drop |
| 7–8 | 0.1507 / 0.1508 | P1 | Plateau |
| 9–10 | 0.1249 / 0.1247 | P1 | End of broad |
| 11–14 | 0.126 → 0.1248 | P2 | Filtered LAION |
| 15 | 0.1026 | P2 | |
| **16** | **0.0947** | P2 | **Best ever (γ=2.0)** |
| 17 | 0.1083 | P2 | γ back to 3.0 |
| 18 | 0.1207 | P3 | Domain shift |
| 22 | 0.1191 | P3 | Stabilised |
| 42 | ~0.115 | P6 | Released: `sd_epoch_042.pt` |
| 43–44 | 0.1202 / 0.1193 | P7 | Final consolidation |

> ### Engineering Takeaways
> - **Stop using the loss curve to decide when to stop.** Use a fixed-seed visual grid.
> - **The best loss number happened at epoch 16; the best images happened at epoch 42.**
> - **Sequential fine-tuning is a trap.** Mix datasets in every batch.
> - **Patience is a hyperparameter.** Diffusion models look catastrophic for the first 5–8 epochs. Trust the process.

---

## 6. Inference — From Checkpoint to Canvas

A `.pt` is not a picture. Several pieces have to slot together.

### 6.1 DDIM Sampling

Training uses 1000 DDPM steps; inference uses **DDIM** (deterministic, η=0). My rules of thumb:

| Use case | Steps |
|---|---|
| Quick exploration | 25 |
| Production scenes | 50 |
| Faces, fine detail | **100** |

Below 50 steps, faces look smudged. Above 100, diminishing returns.

### 6.2 Classifier-Free Guidance

Run the UNet twice each step — once with the prompt embedding, once with the unconditional/empty embedding — then combine:

```python
guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
latents = scheduler.step(guided, t, latents)
```

| Subject | CFG scale |
|---|---|
| Scenes / landscapes | 7.5 |
| Portraits | 8.5 |
| Anything | **stop at 9.0** |

Higher than 9.0 introduces oversaturation and CFG artefacts (waxy skin, blown highlights).

### 6.3 Negative Prompts

Negative prompts replace the unconditional embedding with something like `"blurry, low quality, distorted, deformed"`. They're the fastest way to prune common generative failure modes.

**Implementation note for the repo:** I have two inference scripts.

- `SD_ImageGen.py` (CUDA) — properly wires negative prompts through `generate(..., negative_prompts=...)`.
- `inference.py` (CUDA/MPS, Apple Silicon friendly) — currently parses `--negative` but doesn't thread it into `generate()`; the unconditional branch still uses the empty string. **If you want true negative-prompt CFG, use `SD_ImageGen.py` until I fix that.**

### 6.4 Live vs EMA — The A/B Test That Mattered Most

Same prompt, same seed, same steps, same CFG, same UNet weights but different snapshot:

| Snapshot | Result |
|---|---|
| Live UNet weights | Noisy, jittery composition, "colour leakage" between subject and background |
| EMA shadow (decay 0.9999) | Stable, photorealistic, coherent |

**EMA at inference is non-negotiable.**

> **Honest caveat about the saved `val_epoch_*.png` grids.** My current `SD_Train.py:validate()` does call `ema.apply_shadow()` before sampling — so going forward, validation grids reflect EMA quality. But the grids saved earlier in the run (everything you see in `sd-val-imgs/`) were produced before that fix was wired in and reflect *live* weights. Mentally add ~15 % of perceived quality when looking at them.

*(Insert Asset: Side-by-side — live-weight portrait vs EMA portrait, identical seed.)*

---

## 7. Pitfalls — The Hard-Won Lessons

The most valuable part of any project is the failures.

### 7.1 The Latent Clamp Bug — Two Days Lost to One Line

Training looked great. Inference output was greyish, washed-out, incoherent. I spent two days convinced the UNet was broken before I found this line in `DDIMScheduler.step()`:

```python
pred_x0 = pred_x0.clamp(-1.0, 1.0)   # ← THIS
```

It looks reasonable. It is catastrophically wrong. **SD latents are not in `[-1, 1]`** — their standard deviation is ≈ 4.0. Clamping decapitates the signal.

I removed it at inference and the images snapped into focus.

```python
# in inference.py: monkey-patch the scheduler before sampling
SD_Model.DDIMScheduler.step = _fixed_ddim_step   # the no-clamp version
```

**Caveat:** the clamp is still present in `SD_Model.py:736` because removing it changes behaviour for any old script that imports the class directly. The inference scripts patch it out at runtime. The proper long-term fix is to delete that line.

**Lesson:** Never assume latent distributions match pixel distributions.

### 7.2 `torch.compile` × Gradient Checkpointing — Two Great Tastes That Don't Mix

I tried wrapping the UNet in `torch.compile` to squeeze more speed. It worked — until I enabled gradient checkpointing. Immediate `AssertionError` deep inside Dynamo.

The reason: `checkpoint(use_reentrant=False)` re-runs the forward pass inside a dynamo-disabled context during backward. The compiled wrapper sees a forward whose Dynamo state has been pulled out from under it.

I stuck with eager mode. With BF16 + Flash SDP the throughput penalty was negligible. `torch.compile` is still great outside the loop — I used it on the VAE during latent encoding (§3.3).

### 7.3 `channels_last` on sm_120

When I first enabled `torch.channels_last` for the UNet on the 5090s, my PyTorch 2.6+cu124 build threw shape-mismatch errors inside cuDNN. The fix that worked for Phase 4 was switching to `contiguous_format`. After a CUDA / driver update I switched back to `channels_last` and it now runs cleanly — the current `SD_Train.py` ships with `channels_last`.

**Lesson:** Optimisations that "everyone uses" can break on bleeding-edge hardware. Have a fallback ready.

### 7.4 Min-SNR γ = 2.0 — The Beautiful Wrong Number

Chasing better gradients, I dropped Min-SNR γ from 5.0 to 2.0 in Phase 2. The loss looked gorgeous — **0.0947, my best ever** — but the *images* started melting. Faces lost geometry because high-noise timesteps were being over-weighted relative to the low-noise timesteps that actually carry facial detail.

I reverted to γ=3.0 and the metric got worse while the images got better.

**Lesson:** lower MSE ≠ better images. Visual validation is the only ground truth.

### 7.5 The Patience Problem

In the first three epochs I almost gave up. The loss looked fine but the validation grids were noise. I kept the fixed-seed grids anyway. By epoch 5 vague shapes appeared. By epoch 8 they became things.

**Lesson:** the "aha" moment in diffusion happens late. Don't kill a run because the first few epochs look like television static.

*(Insert Asset: Side-by-side — "broken" clamp-bug output vs "fixed" output after removing the clamp.)*

---

## 8. What I'd Do Differently

If I started over tomorrow:

1. **Mixed batching from day one.** I wasted weeks in a sequential mindset. Catastrophic forgetting between Phases 4 and 5 cost me a week. Always sample across datasets in every batch.
2. **Start with a lower Min-SNR γ (2.0–2.5) carefully.** For aesthetic-heavy data, lower γ can speed convergence — but test it visually, not just by loss.
3. **Test inference every single epoch.** I'd have caught the latent clamp bug in week one instead of week six.
4. **Validate with EMA from the very first epoch.** Don't ship a fix you forgot to backfill into your visualisations.
5. **Delete the training-time `pred_x0.clamp`** instead of monkey-patching around it. Make the wrong version unreachable.
6. **Wire `--negative` through `inference.py:generate()` properly** so both inference scripts behave identically.

---

## 9. What's Next

`sd_epoch_042.pt` is a solid base. The roadmap from here:

1. **Rectified flow fine-tuning.** Current frontier for diffusion image quality and the natural successor to ε-prediction.
2. **LCM (Latent Consistency Model) distillation.** Enables 1–4 step generation — real-time inference.
3. **ControlNet.** Spatial conditioning — pose, depth, edge maps — without retraining the base.
4. **SD_Model_v2.** Already designed: MM-DiT backbone, dual CLIP-L + OpenCLIP-bigG text encoders, native rectified flow. This is where I want to live next.

---

## 10. The Repo at a Glance

```
StableDiffusion/
├── SD_Model.py                       # UNet + VAE/CLIP wrappers + DDPM/DDIM schedulers
├── SD_Train.py                       # 2× RTX 5090 DDP + BF16 training loop
├── SD_ImageGen.py                    # CUDA inference (full negative-prompt CFG)
├── inference.py                      # CUDA/MPS inference (DDIM clamp monkey-patched)
├── encode_latents.py                 # 4-stage VAE → fp16 .npy pipeline
├── 01_download_metadata.py           # LAION parquet snapshot
├── 01b_download_diffusiondb.py       # DiffusionDB → 512×512 tar shards
├── 01c_download_journeydb_images.py  # JourneyDB → 512×512 tar shards
├── 02_filter_metadata.py             # aesthetic / CLIP / watermark / NSFW / dedup
├── 03_download_images.py             # img2dataset LAION downloader
├── 03_build_hf_dataset.py            # DiffusionDB/JourneyDB → Arrow HF dataset
├── 04_preprocess_to_cache.py         # Tars → parquet (image_key + CLIP tokens)
├── 05_build_hf_dataset.py            # Parquet → Arrow HF dataset
├── sd_epoch_042.pt                   # Released checkpoint (~12.5 GB)
├── sd-val-imgs/                      # val_epoch_001..043.png (live-weight grids)
├── sd-logs/                          # captured training.log / output*.log
└── generated_images/                 # curated epoch-42 renders
```

---

## Final Thoughts

Training a Stable Diffusion model from scratch was one of the most rewarding engineering projects I've worked on — and the "AI" was the easy part. The real difficulty is in:

- treating data quality as a hyperparameter,
- moving bytes between disk, host RAM, pinned buffers and VRAM without ever blocking the GPU,
- knowing which optimisations compose and which ones explode when you stack them,
- and trusting visual validation over a loss number that lies to you.

If you're considering this: do it. But go in with your eyes open. The transformers will be fine. It's the JPEG decoder, the pinned buffer, the EMA decay and the one clamp at line 736 that will decide whether your model converges.

---

## Resources

- **Core frameworks:** [PyTorch](https://pytorch.org), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers)
- **Models & data:** [LAION](https://laion.ai), [DiffusionDB](https://github.com/poloclub/diffusiondb), [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB), [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/), [COCO](https://cocodataset.org/)
- **Performance:** [Flash Attention](https://github.com/Dao-AILab/flash-attention), [NCCL](https://developer.nvidia.com/nccl)
- **Reference papers:** Ho et al. 2020 (DDPM), Song et al. 2020 (DDIM), Hang et al. 2023 (Min-SNR), Rombach et al. 2022 (Latent Diffusion)

**Tags:** #MachineLearning #DeepLearning #StableDiffusion #PyTorch #GenerativeAI #GPUComputing #AITraining #RTX5090 #Blackwell
