# Training Loop

## Optimization Stack

The training loop in `src/train.py` is optimized for dual RTX 5090 (Blackwell) GPUs:

1. **DDP (DistributedDataParallel):** True multi-GPU with NCCL backend. Each GPU processes its own micro-batch; gradients are all-reduced across both GPUs.
2. **BF16 native (torch.autocast):** Blackwell has native BF16 support — no GradScaler needed.
3. **Flash Attention (FlashSDP):** Enabled on Blackwell (cc ≥ 8.0), ~2–4× faster attention with O(N) memory vs O(N²).
4. **Gradient checkpointing:** Saves ~40% activation memory by recomputing activations on backward.
5. **Fused AdamW:** `torch.optim.AdamW` with `fused=True` — single kernel for the entire optimizer step.
6. **`channels_last` memory format:** Optimal for convolutions on Blackwell (configurable via `--memory_format`).
7. **EMA on GPU:** 32 GB VRAM per GPU is plenty — no CPU round-trips for the EMA shadow copy.
8. **Min-SNR loss weighting:** Better gradient signal across timesteps (Hang et al., 2023).

## Training Procedure

### Epoch Structure

```
for each epoch:
    sampler.set_epoch(epoch)              # shuffle DDP sampler
    for each batch:
        latents = batch["pixel_values"]   # pre-encoded VAE latents from disk
        ids, mask = batch["input_ids"]    # tokenized text
        t = sample_timesteps(batch)       # uniform [0, 1000)
        noise = randn_like(latents)
        z_t = noise_scheduler.add_noise(latents, noise, t)

        # CFG dropout: randomly replace conditioning with empty text
        if random() < cfg_dropout:
            ids, mask = empty_tokens()

        with autocast("cuda", bf16):
            noise_pred = ddp_unet(z_t, t, context)
            loss = min_snr_weighted_mse(noise_pred, noise, t)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.update(unet_raw)
```

### Loss Function

Standard MSE loss between predicted and actual noise, weighted by Min-SNR (Hang et al., 2023):

```python
def min_snr_weighted_mse(noise_pred, noise, t, gamma=5.0):
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    weights = torch.clamp(snr, max=gamma)
    loss = F.mse_loss(noise_pred, noise, reduction="none")
    return (loss * weights.view(-1, 1, 1, 1)).mean()
```

### Learning Rate Schedule

1. **Linear warmup** for 500 steps from 0 → peak LR
2. **Cosine annealing** from peak LR → 0 over remaining steps
3. **Peak LR:** 1e-4 for pre-training, 1e-5 for fine-tuning

### Hyperparameters

| Parameter | Pre-training (ep 1–10) | Fine-tuning (ep 11–42) |
|---|---|---|
| Batch size per GPU | 24 | 24 |
| Gradient accumulation | 2 | 2 |
| Effective batch | 96 | 96 |
| Peak LR | 1e-4 (ep 1), 1e-5 (ep 2–10) | 1e-5 |
| Min-SNR γ | 5.0 | 2.5 |
| CFG dropout | 0.05 | 0.05 |
| Weight decay | 0.01 | 0.01 |
| EMA decay | 0.9999 | 0.9999 |
| Warmup steps | 500 | 500 |

## EMA (Exponential Moving Average)

Polyak-style EMA maintains a shadow copy of all UNet parameters:

```python
shadow[n].lerp_(p.detach(), 1.0 - decay)
```

The shadow weights are updated after every optimizer step with an effective decay that increases to 0.9999 over the first 10 steps:

```python
d = min(decay, (1 + step) / (10 + step))
```

At evaluation and checkpoint time, EMA weights are swapped in for inference (producing noticeably better samples than the live weights).

## Checkpoint Format

Each checkpoint saved to `checkpoints/sd_epoch_NNN.pt` contains:

```python
{
    "unet_state_dict": ...,       # Raw UNet weights (for strict loading)
    "ema_state_dict": {           # EMA shadow copy
        "shadow": {...},           # Parameter name → tensor mapping (with prefix stripping)
        "step_count": 232235,
        "decay": 0.9999,
    },
    "optimizer_state_dict": ...,  # Full AdamW state (for resume)
    "lr_scheduler_state_dict": ...,
    "epoch": 42,
    "global_step": 232235,
    "best_loss": 0.0947,
    "config": {...},              # Training configuration snapshot
}
```

The checkpoint is ~12.5 GB (BF16 weights are stored as FP32 for CPU loading stability).
