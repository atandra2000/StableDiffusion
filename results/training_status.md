# Training Status

## Current Stage: Latent Encoding

The full data pipeline (Steps 01–05) has been completed on RunPod.
The model is currently in the **VAE latent pre-encoding stage** using `src/encode_pipeline.py`.

---

### Pipeline Completion

| Step | Script                          | Status       | Output                              |
|------|---------------------------------|--------------|-------------------------------------|
| 01   | `01_download_metadata.py`       | ✅ Complete   | `laion_metadata/`  — parquet shards |
| 02   | `02_filter_metadata.py`         | ✅ Complete   | `laion_filtered/`  — ~12M URLs      |
| 03   | `03_download_images.py`         | ✅ Complete   | `laion_shards/`    — WebDataset tar |
| 04   | `04_preprocess_to_cache.py`     | ✅ Complete   | `laion_cache/`     — tokenised cache|
| 05   | `05_build_hf_dataset.py`        | ✅ Complete   | `laion_hf_dataset/`— HF Dataset     |
| —    | `encode_pipeline.py`            | ▶ In Progress | `laion_latents/`   — .npy files     |
| —    | `train.py`                      | ⏳ Pending    | —                                   |

---

### Encoding Configuration

| Parameter         | Value                        |
|-------------------|------------------------------|
| Hardware          | 2× RTX PRO 4500 (32 GB each) |
| VAE model         | stabilityai/sd-vae-ft-mse    |
| Batch size/GPU    | 32                           |
| Target resolution | 512×512                      |
| Output format     | `.npy` (float16)             |
| Parallelism       | 2 processes, process isolation (CUDA_VISIBLE_DEVICES) |
| Expected RAM      | ~42 GB for full latent cache |

---

### Expected Training Configuration

| Hyperparameter    | Value                               |
|-------------------|-------------------------------------|
| Effective batch   | 128 × 2 GPUs × 2 grad accum = 512  |
| Learning rate     | 1e-4                                |
| LR schedule       | 500-step warmup + CosineAnnealing   |
| Precision         | bfloat16                            |
| Compilation       | torch.compile (reduce-overhead)     |
| EMA               | decay=0.9999                        |
| Loss              | ε-prediction MSE                    |
| Validation        | DDIM 30 steps, CFG=7.5, 8 prompts  |
| Tracking          | Weights & Biases                    |

---

*This file will be updated with loss curves and sample images once training begins.*
