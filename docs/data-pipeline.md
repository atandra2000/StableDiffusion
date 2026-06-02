# Data Pipeline

The data pipeline processes LAION-2B-en-aesthetic (~12 million images filtered to aesthetic score ≥ 6.5) into a format suitable for training. Each script is a self-contained step.

## Pipeline Steps

### Step 1: Download Metadata

```bash
python data_pipeline/01_download_metadata.py
```

Downloads LAION-2B-en-aesthetic parquet shards from the Hugging Face Hub. The dataset contains ~230 million image-URL–caption pairs with precomputed aesthetic scores, CLIP similarity scores, and watermark probabilities.

**Output:** `laion_metadata/*.parquet` (multiple shards)

### Step 1b-c: Alternative Sources (Optional)

```bash
python data_pipeline/01b_download_diffusiondb.py
python data_pipeline/01c_download_journeydb_images.py
```

Alternative or supplementary datasets — not used in the final training run.

### Step 2: Filter Metadata

```bash
python data_pipeline/02_filter_metadata.py
```

Applies quality filters to the raw metadata:

| Filter | Threshold | Rationale |
|---|---|---|
| Aesthetic score | ≥ 6.5 | Top ~5% of LAION-2B, high visual quality |
| CLIP similarity | ≥ 0.28 | Text–image alignment |
| Resolution | ≥ 512 px | Minimum for 512×512 training |
| Watermark probability | ≤ 0.5 | Reduce watermarked images |
| NSFW | False | Safety filter |

**Output:** `laion_filtered/*.parquet` (~12M entries, ~9 GB)

### Step 3: Download Images

```bash
python data_pipeline/03_download_images.py
```

Downloads images from the filtered URLs using `img2dataset`. Handles:
- HTTP timeouts and retries
- Image format validation (JPEG, PNG)
- Resolution checks (skip images < 512 px in either dimension)
- WebDataset tar packaging for efficient I/O

**Output:** `laion_shards/` — WebDataset tar files (~6 TB raw images)

### Step 4: Preprocess to Cache

```bash
python data_pipeline/04_preprocess_to_cache.py
```

Tokenizes captions with CLIP tokenizer and applies basic augmentations:

- BPE tokenization (77 tokens, truncation/padding)
- CLIP attention mask computation
- Image resizing to 512×512 (center crop if necessary)

**Output:** `laion_cache/` — Tokenized cache shards

### Step 5: Build Hugging Face Dataset

```bash
python data_pipeline/05_build_hf_dataset.py
```

Assembles tokenized data and raw image paths into a Hugging Face `datasets.Dataset` (memory-mapped Arrow format). Splits into train/validation (5K val samples).

**Output:** `laion_hf_dataset/` (HF Dataset on disk, ~50 GB)

### Step 5b: Fine-tuning Subset

```bash
python data_pipeline/05_build_hf_dataset.py --output laion_hf_dataset_ft
```

Builds a finer-quality subset for fine-tuning — stricter aesthetic thresholds (≥ 7.0) and additional deduplication.

### Step 6: Filter Dataset (Optional)

```bash
python data_pipeline/06_filter_dataset.py
```

Additional filtering after dataset construction (e.g., CLIP score re-ranking, near-dedup).

## VAE Pre-encoding

Latent pre-encoding is the last preprocessing step before training. It converts all images in the dataset to VAE latents, which are saved as `.npy` files. During training, the DataLoader reads pre-encoded latents directly — no VAE forward pass needed.

```bash
# Single GPU encoding
python src/encode_latents.py

# Dual-GPU parallel encoding
python src/encode_pipeline.py
```

**Output:** `laion_latents/*.npy` (41.4 GB total, 4×64×64 per image)

### Why Pre-encode?

1. **VRAM savings:** The VAE forward pass requires ~2 GB per image at 512×512. Pre-encoding frees this memory for the UNet.
2. **Speed:** Latent loading from `.npy` is ~10× faster than VAE encoding per image.
3. **Dataset augmentation:** Pre-encoding is deterministic — all training epochs see identical latents from the same images, which is actually desired for reproducibility.

## Production Training Pipeline

For training, the DataLoader reads from two sources simultaneously:

1. **Latents:** Memory-mapped from `laion_latents/*.npy` via `numpy.load` + `torch.from_numpy`
2. **Tokenized text:** From the HF Dataset `input_ids` and `attention_mask` columns

```python
class LatentDataset(Dataset):
    def __getitem__(self, idx):
        latent = np.load(self.latent_files[idx])        # (4, 64, 64) float32
        ids     = self.dataset[idx]["input_ids"]         # (77,) int64
        mask    = self.dataset[idx]["attention_mask"]    # (77,) int64
        return {"pixel_values": latent, "input_ids": ids, "attention_mask": mask}
```

The DataLoader uses `pin_memory=True`, `prefetch_factor=4`, and `num_workers=16` (per GPU) to saturate the 2× RTX 5090 setup.
