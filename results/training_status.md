# Training Status

## Summary

| | |
|---|---|
| **Total epochs** | 21 |
| **Total global steps** | 181,177 |
| **Best loss** | 0.0924 |
| **Final loss (ep 21)** | 0.1030 |
| **Hardware** | 2× RTX 5090 (Blackwell, 32 GB each) |
| **Multi-GPU** | DistributedDataParallel (DDP) + NCCL |
| **W&B project** | [atandrabharati-self/stable-diffusion](https://wandb.ai/atandrabharati-self/stable-diffusion) |

---

## Pipeline Completion

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 01 | `01_download_metadata.py` | ✅ Complete | `laion_metadata/` — parquet shards |
| 02 | `02_filter_metadata.py` | ✅ Complete | `laion_filtered/` — ~12M URLs |
| 03 | `03_download_images.py` | ✅ Complete | `laion_shards/` — WebDataset tar |
| 04 | `04_preprocess_to_cache.py` | ✅ Complete | `laion_cache/` — tokenised cache |
| 05 | `05_build_hf_dataset.py` | ✅ Complete | `laion_hf_dataset/` — HF Dataset |
| — | `encode_pipeline.py` | ✅ Complete | `laion_latents/` — 41.4 GB .npy |
| — | `train.py` (pre-training) | ✅ Complete | Epochs 1–10, steps 1–136K |
| — | `train.py` (fine-tuning) | ✅ Complete | Epochs 11–21, steps 136K–181K |

---

## Phase 1 — Pre-training (Epochs 1–10)

**Dataset:** `laion_hf_dataset` / `laion_latents` (full LAION-2B-en-aesthetic, ~12M images)
**Learning rate:** 1e-4 (ep-1, warmup) → 1e-5 (ep-3 onwards, cosine decay)
**Min-SNR γ:** 5 (eps 1–9) → 2.5 (ep-9–10)

| W&B Run | Epochs | End Step | End Loss | Best Loss | Notes |
|---------|--------|----------|----------|-----------|-------|
| ep-1 | ~1 | 13,600 | 0.1658 | — | LR=1e-4, crashed |
| ep-3 | 1 | 15,031 | 0.1604 | — | LR=1e-5 fresh start, crashed |
| ep-3-res | 3 | 41,643 | 0.1548 | 0.1555 | Resumed ep-3 |
| ep-4 | 6 | 82,936 | 0.1513 | 0.1515 | +cfg_dropout=0.05 |
| ep-7 | 8 | 109,098 | 0.1504 | 0.1507 | |
| ep-9 | 10 | 136,279 | **0.1247** | **0.1247** | γ→2.5, **pre-train complete** |

---

## Phase 2 — Fine-tuning (Epochs 11–21)

**Dataset:** `laion_hf_dataset_ft` / `laion_latents_ft` (curated higher-quality subset)
**Learning rate:** 3e-6 → 8e-6 (cosine, warmup=100 steps)
**Min-SNR γ:** 2–5 (varied per run)
**cfg_dropout:** 0.05–0.10

| W&B Run | Epochs | End Step | End Loss | Best Loss | Notes |
|---------|--------|----------|----------|-----------|-------|
| ft-11 | 14 | 145,282 | 0.1257 | 0.1247 | FT dataset switch, LR=3e-6 |
| ft-15 | 15 | 147,382 | **0.1026** | **0.1026** | |
| ft-16 | 16 | 149,718 | **0.0936** | **0.0947** | Best single-epoch loss |
| ft-17 | 17 | 151,818 | 0.1083 | 0.0947 | |
| ft-18-p | 18 | 159,883 | 0.1189 | 0.0947 | |
| ft-19 | 20 | 175,327 | 0.1033 | 0.0947 | LR bump to 8e-6 |
| ft-20 | 21 | 181,177 | 0.1030 | 0.0947 | **Training complete** |

---

## Loss Curve

Full step-by-step loss data: [`results/loss_curve.csv`](loss_curve.csv) (2,450 points, 181K steps)

```
Loss
0.90 ┤╮
0.80 ┤ ╰─╮
0.70 ┤   ╰╮
0.60 ┤    ╰─╮
0.50 ┤      ╰─╮
0.40 ┤        ╰╮
0.30 ┤         ╰──╮
0.20 ┤            ╰──╮
0.16 ┤               ╰────────────────╮  [Pre-training plateau ~0.15]
0.12 ┤                                ╰──────╮  [ep-9: 0.1247]
0.10 ┤                                       ╰──╮  [ft-16: 0.0947 best]
0.09 ┤                                          ╰──
     └──────────────────────────────────────────────→ step
     0       50K      100K     136K     150K    181K
     [────────── Pre-training ──────────][── Fine-tune ──]
```

---

## Checkpoints (Google Drive)

| Epoch | Phase | Step | Drive |
|-------|-------|------|-------|
| 10 | Pre-training | 136,279 | [Checkpoints folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa) |
| 14 | Fine-tuning | 145,282 | [Checkpoints folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa) |
| 17 | Fine-tuning | 151,818 | [Checkpoints folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBow6KRoa) |
| 21 | Fine-tuning | 181,177 | [Checkpoints folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa) |

Each checkpoint (~11.6 GB) contains: `unet_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`, `best_loss`.

---

## Hardware & Training Config

| Parameter | Pre-training | Fine-tuning |
|-----------|-------------|-------------|
| Hardware | 2× RTX 5090 (32 GB) | 2× RTX 5090 (32 GB) |
| Multi-GPU | DDP + NCCL | DDP + NCCL |
| Dataset | laion_hf_dataset (~12M) | laion_hf_dataset_ft (curated) |
| Batch/GPU | 24 | 24 |
| Grad accum | 2 | 2 |
| Effective batch | 96 | 96 |
| LR | 1e-4 → 1e-5 | 3e-6 → 8e-6 |
| Warmup | 500 steps | 100 steps |
| Min-SNR γ | 5 → 2.5 | 2 – 5 |
| cfg_dropout | 0 → 0.05 | 0.05 – 0.10 |
| Precision | bfloat16 | bfloat16 |
| Compile | max-autotune | max-autotune |
| EMA decay | 0.9999 (on GPU) | 0.9999 (on GPU) |
