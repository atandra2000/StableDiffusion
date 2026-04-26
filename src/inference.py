"""
inference.py — SD Inference on Apple Silicon (M1/M2/M3)
========================================================
Loads EMA weights from a training checkpoint and runs DDIM sampling.

Requirements:
    pip install torch torchvision diffusers transformers Pillow

Usage:
    python3 inference.py --prompt "a beautiful sunset over mountain peaks"
    python3 inference.py --prompt "your prompt" --steps 50 --guidance 7.5 --seed 42
    python3 inference.py --prompt "your prompt" --output my_image.png
    python3 inference.py --batch "prompts.txt" --output_dir ./outputs

Setup:
    1. Copy SD_Model.py to the same directory as this script
    2. Copy your checkpoint (sd_epoch_021.pt) to the same directory
    3. Run: python3 inference.py --prompt "your prompt"
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPTokenizer


# ── Device setup ──────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("Warning: no GPU found, running on CPU — will be slow")
        return torch.device("cpu")


# ── EMA weight extraction ──────────────────────────────────────────────────────

def load_ema_unet(checkpoint_path: str, unet, device: torch.device):
    """
    Load EMA shadow weights from checkpoint into the UNet.
    EMA weights produce better inference quality than the live training weights.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema_state_dict" not in ckpt:
        raise KeyError(
            "No 'ema_state_dict' found in checkpoint. "
            "Make sure this is a checkpoint saved by SD_Train.py."
        )

    ema_state = ckpt["ema_state_dict"]
    shadow = ema_state["shadow"]

    epoch = ckpt.get("epoch", "?")
    global_step = ckpt.get("global_step", "?")
    best_loss = ckpt.get("best_loss", "?")
    print(f"  Epoch: {epoch} | Step: {global_step} | Best loss: {best_loss}")
    print(f"  EMA shadow keys: {len(shadow):,}")

    # Strip any DDP or module prefix from shadow keys
    cleaned = {}
    for k, v in shadow.items():
        key = k
        for prefix in ("module.", "unet.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    missing, unexpected = unet.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys in UNet")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys ignored")

    print("  EMA weights loaded successfully")
    unet.to(device)
    unet.eval()
    return unet


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Build the full SD model and load EMA UNet weights."""
    sys.path.insert(0, str(Path(__file__).parent))
    from model import (
        PretrainedVAE,
        PretrainedCLIPTextEncoder,
        UNetModel,
        DDIMScheduler,
    )

    print("Loading VAE...")
    # MPS doesn't support fp16 well — use fp32
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse", use_fp16=False)
    vae.to(device)
    vae.eval()

    print("Loading CLIP text encoder...")
    text_encoder = PretrainedCLIPTextEncoder(model_id="openai/clip-vit-large-patch14")
    text_encoder.to(device)
    text_encoder.eval()

    print("Building UNet...")
    unet = UNetModel(in_ch=4, out_ch=4, ch=320, ch_mults=(1, 2, 4, 4), res_blks=2,
                     attn_lvls=(1, 2, 3), heads=8, ctx_dim=768)

    unet = load_ema_unet(checkpoint_path, unet, device)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DDIMScheduler()
    scheduler.to(device)

    return vae, text_encoder, unet, tokenizer, scheduler


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    prompts:        list[str],
    vae,
    text_encoder,
    unet,
    tokenizer,
    scheduler,
    device:         torch.device,
    num_steps:      int = 50,
    guidance_scale: float = 7.5,
    seed:           int = 42,
    height:         int = 512,
    width:          int = 512,
) -> list[Image.Image]:
    """
    Run DDIM sampling for a list of prompts.
    Returns a list of PIL Images.
    """
    assert height % 8 == 0 and width % 8 == 0, "Height and width must be divisible by 8"
    batch_size = len(prompts)
    latent_h = height // 8
    latent_w = width // 8

    # ── Encode prompts ────────────────────────────────────────────────────────
    cond_tok = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    uncond_tok = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    cond_emb,   _ = text_encoder(cond_tok.input_ids,   cond_tok.attention_mask)
    uncond_emb, _ = text_encoder(uncond_tok.input_ids, uncond_tok.attention_mask)
    ctx = torch.cat([uncond_emb, cond_emb], dim=0)

    # ── Initial noise ─────────────────────────────────────────────────────────
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (batch_size, 4, latent_h, latent_w),
        generator=generator,
    ).to(device)

    # ── DDIM denoising loop ───────────────────────────────────────────────────
    scheduler.set_timesteps(num_steps, device)

    for i, t in enumerate(scheduler.timesteps):
        latent_in = torch.cat([latents, latents], dim=0)
        t_batch = torch.full((latent_in.shape[0],), t, dtype=torch.long, device=device)

        # UNet forward — use float32 on MPS for stability
        with torch.autocast("cpu") if device.type == "cpu" else torch.no_grad():
            noise_pred = unet(latent_in.float(), t_batch, ctx.float())

        noise_uncond, noise_cond = noise_pred.chunk(2)
        guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(guided, t, latents)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Step {i+1:3d}/{num_steps}", end="\r")

    print()

    # ── Decode latents ────────────────────────────────────────────────────────
    images_tensor = vae.decode(latents).clamp(-1.0, 1.0)
    images_tensor = (images_tensor + 1.0) / 2.0

    pil_images = []
    for i in range(batch_size):
        img_np = (images_tensor[i].cpu().float().permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_images.append(Image.fromarray(img_np))

    return pil_images


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SD Inference — Apple Silicon")

    parser.add_argument("--prompt",     type=str,   default=None,
                        help="Text prompt for generation")
    parser.add_argument("--batch",      type=str,   default=None,
                        help="Path to .txt file with one prompt per line")
    parser.add_argument("--checkpoint", type=str,   default="sd_epoch_017.pt",
                        help="Path to training checkpoint (.pt file)")
    parser.add_argument("--output",     type=str,   default="output.png",
                        help="Output image path (for single prompt)")
    parser.add_argument("--output_dir", type=str,   default="./outputs",
                        help="Output directory (for batch mode)")
    parser.add_argument("--steps",      type=int,   default=50,
                        help="Number of DDIM denoising steps (25=fast, 50=balanced, 100=quality)")
    parser.add_argument("--guidance",   type=float, default=7.5,
                        help="Classifier-free guidance scale (7.5 default, higher=more prompt adherence)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--width",      type=int,   default=512)
    parser.add_argument("--height",     type=int,   default=512)
    parser.add_argument("--batch_size", type=int,   default=1,
                        help="Images to generate per forward pass (reduce if out of memory)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.prompt is None and args.batch is None:
        print("Error: provide --prompt or --batch")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        print("Make sure sd_epoch_021.pt is in the same directory as this script")
        sys.exit(1)

    if not Path("SD_Model.py").exists():
        print("Error: SD_Model.py not found in current directory")
        print("Copy SD_Model.py from your training directory to here")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    t0 = time.time()
    vae, text_encoder, unet, tokenizer, scheduler = load_model(args.checkpoint, device)
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # Collect prompts
    if args.batch:
        prompts = Path(args.batch).read_text().strip().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        print(f"Batch mode: {len(prompts)} prompts")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        prompts = [args.prompt]

    # Generate in batches
    all_images = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        print(f"Generating: {batch_prompts[0][:80]}{'...' if len(batch_prompts[0]) > 80 else ''}")

        t0 = time.time()
        images = generate(
            prompts=batch_prompts,
            vae=vae,
            text_encoder=text_encoder,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            device=device,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed + i,
            height=args.height,
            width=args.width,
        )
        elapsed = time.time() - t0
        print(f"  Generated in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)")

        all_images.extend(zip(batch_prompts, images))

    # Save outputs
    if args.batch:
        for j, (prompt, img) in enumerate(all_images):
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:50]).strip()
            path = Path(args.output_dir) / f"{j:04d}_{safe_name}.png"
            img.save(path)
            print(f"Saved: {path}")
    else:
        all_images[0][1].save(args.output)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
