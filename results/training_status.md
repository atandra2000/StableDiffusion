# Training Status

## Current Stage: Training — Epoch 21 Complete

Training is actively running on RunPod. The full data pipeline (Steps 01–05) and VAE latent
pre-encoding are both complete. The model has been trained for **21 epochs** with checkpoints
saved at epochs 10, 14, 17, and 21.

---

### Pipeline Completion

| Step | Script                          | Status       | Output                              |
|------|---------------------------------|--------------|-------------------------------------|
| 01   | `01_download_metadata.py`       | ✅ Complete   | `laion_metadata/`  — parquet shards |
| 02   | `02_filter_metadata.py`         | ✅ Complete   | `laion_filtered/`  — ~12M URLs      |
| 03   | `03_download_images.py`         | ✅ Complete   | `laion_shards/`    — WebDataset tar |
| 04   | `04_preprocess_to_cache.py`     | ✅ Complete   | `laion_cache/`     — tokenised cache|
| 05   | `05_build_hf_dataset.py`        | ✅ Complete   | `laion_hf_dataset/`— HF Dataset     |
| —    | `encode_pipeline.py`            | ✅ Complete   | `laion_latents/`   — 41.4 GB .npy  |
| —    | `train.py`                      | ▶ In Progress | Epoch 21 / ~30 complete             |

---

### Hardware Configuration

| Parameter         | Value                                          |
|-------------------|------------------------------------------------|
| Hardware          | 2× RTX 5090 (Blackwell, cc 10.x, 32 GB each)  |
| Multi-GPU         | DistributedDataParallel (DDP) + NCCL backend   |
| VAE model         | stabilityai/sd-vae-ft-mse                      |
| Batch size/GPU    | 24                                             |
| Gradient accum    | 2                                              |
| Effective batch   | 24 × 2 GPUs × 2 grad accum = 96               |
| Target resolution | 512×512                                        |
| Output format     | `.npy` (float16) latents                       |

---

### Checkpoints

| Epoch | Size     | Location (Google Drive)                                                                                          |
|-------|----------|------------------------------------------------------------------------------------------------------------------|
| 10    | ~11.6 GB | [Drive folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa)                      |
| 14    | ~11.6 GB | [Drive folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa)                      |
| 17    | ~11.6 GB | [Drive folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa)                      |
| 21    | ~11.6 GB | [Drive folder](https://drive.google.com/drive/folders/1_BFLxvZHLaU9HZhZmQXEHGtCPBw6KRoa)                      |

---

### Training Configuration

| Hyperparameter    | Value                                           |
|-------------------|-------------------------------------------------|
| Learning rate     | 1e-4                                            |
| LR schedule       | 500-step warmup + CosineAnnealing               |
| Precision         | bfloat16 (native Blackwell, no GradScaler)      |
| Compilation       | torch.compile (max-autotune)                    |
| EMA               | decay=0.9999, maintained on GPU                 |
| Loss              | ε-prediction MSE + Min-SNR weighting (γ=5)     |
| Attention         | Flash Attention via SDPA (native on Blackwell)  |
| Validation        | DDIM 30 steps, CFG=7.5, 8 prompts              |
| Tracking          | Weights & Biases (atandrabharati-self/stable-diffusion) |

---

*Updated: April 2026. Fine-tuning pipeline also started (laion_hf_dataset_ft, laion_latents_ft).*
