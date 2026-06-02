# Inference Guide

## Quick Start

```bash
# Download checkpoint first
python scripts/download_checkpoint.py

# Basic inference
python src/inference.py \
  --prompt "a beautiful sunset over mountain peaks" \
  --checkpoint checkpoints/sd_epoch_042.pt

# Apple Silicon (MPS) — auto-detected
python src/inference.py \
  --prompt "a cat wearing a spacesuit" \
  --checkpoint checkpoints/sd_epoch_042.pt
```

## Usage Options

### Basic Parameters

| Flag | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text prompt |
| `--batch` | — | `.txt` file with one prompt per line (overrides `--prompt`) |
| `--negative` | `""` | Negative prompt for CFG |
| `--checkpoint` | `sd_epoch_042.pt` | Path to checkpoint |
| `--steps` | 50 | DDIM steps: 25 (fast), 50 (good), 100 (best) |
| `--guidance` | 7.5 | CFG scale: 1.0 (no guidance), 7.5 (balanced), 15+ (high) |
| `--seed` | 42 | Random seed for reproducibility |
| `--width` / `--height` | 512 | Output dimensions (must be multiples of 8) |
| `--batch_size` | 1 | Images per generation call (for batch prompts) |
| `--output` | `output.png` | Output filename (single prompt) |
| `--output_dir` | `./outputs` | Output directory (batch mode) |

### Examples

```bash
# High quality, deterministic
python src/inference.py \
  --prompt "a cinematic shot of a mountain lake at sunrise, professional photography" \
  --steps 100 --guidance 7.5 --seed 42

# With negative prompt
python src/inference.py \
  --prompt "a portrait of a woman" \
  --negative "blurry, low quality, deformed hands, extra fingers" \
  --steps 50 --guidance 9.0

# Batch mode — generate from 100 prompts
python src/inference.py \
  --batch prompts.txt \
  --output_dir ./generated \
  --steps 50 --guidance 7.5

# Stochastic sampling (DDPM-like, more variety)
python src/inference.py \
  --prompt "fantasy landscape" \
  --eta 1.0 --seed 999
```

## Scripts

### `src/inference.py` — Apple Silicon + CUDA

Designed for MacBook (MPS) and single-GPU inference. Automatically detects MPS, CUDA, or CPU.

Features:
- Automatic device selection
- Negative prompt support
- Batch processing from text file
- DDIM with optional stochastic (eta)
- EMA weight loading from checkpoint

### `src/SD_ImageGen.py` — Alternative CLI

Full-featured CLI with additional options:
- Supports both raw UNet weights and EMA shadow weights
- Negative prompt per-sample broadcasting
- autocast BF16 on CUDA
- Image grid generation for multiple outputs

### `src/generate.py` — Programmatic API

```python
from generate import generate_images

images = generate_images(
    prompts=["a cosmic nebula with vibrant colors"],
    checkpoint_path="checkpoints/sd_epoch_042.pt",
    num_steps=50,
    guidance_scale=7.5,
    seed=42,
    device="cuda",           # or "mps", "cpu"
)
images[0].save("nebula.png")
```

## DDIM Parameters

### Steps

| Steps | Quality | Speed |
|---|---|---|
| 25 | Good | 2× faster |
| 50 | Recommended | Baseline |
| 100 | Excellent | 2× slower |
| 200+ | Diminishing | Not worth it |

### Eta (Stochasticity)

| Eta | Behavior |
|---|---|
| 0.0 | Deterministic — same seed always produces the same image |
| 0.5 | Moderate stochasticity — small variations |
| 1.0 | DDPM-like — maximum variety, but may lose fidelity |

### CFG Scale

| Scale | Effect |
|---|---|
| 1.0 | No guidance — pure model prior, often blurry/unrelated |
| 5.0–7.5 | Balanced — recommended range |
| 9.0–12.0 | Strong guidance — more prompt alignment, may oversaturate |
| 15.0+ | Excessive — often produces artifacts, burned-in look |

## Output Quality Tips

1. **Use descriptive prompts:** "a cinematic shot of..." works better than "a photo of..."
2. **Negative prompts help:** Common negatives: "blurry, low quality, deformed, extra limbs, bad anatomy, ugly, text, watermark"
3. **Seed selection:** For a given prompt, try seeds 0–20 and pick the best
4. **Steps vs. quality:** 50 steps is usually sufficient; 100+ gives marginal gains
5. **CFG tuning:** Start at 7.5, adjust ±2 based on output character
