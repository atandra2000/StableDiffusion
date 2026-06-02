"""
STABLE DIFFUSION INFERENCE
===========================================

This script loads a trained Stable Diffusion model checkpoint and generates images from text prompts.
Usage:
    python SD_ImageGen.py --checkpoint checkpoints/sd_latest.pt \
                          --prompts "a cat sitting on a windowsill" "a futuristic cityscape at sunset" \
                          --negative "blurry, low quality" \
                          --steps 50 \
                          --guidance 7.5 \
                          --eta 0.0 \
                          --output generated.png \
                          --height 512 \
                          --width 512 \
                          --seed 42
"""

import torch
import logging
import argparse
from pathlib import Path
from PIL import Image
from transformers import CLIPTokenizer

from SD_Model import (
    StableDiffusionModel, PretrainedVAE, PretrainedCLIPTextEncoder,
    UNetModel, DDPMScheduler, DDIMScheduler
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> StableDiffusionModel:
    """
    Rebuild the full model and load UNet weights from checkpoint.

    The VAE and CLIP are always loaded from their pretrained hub models —
    we never saved them in checkpoints because they never change.
    """
    logger.info(f"🔧 Loading pretrained VAE...")
    vae = PretrainedVAE(model_id="stabilityai/sd-vae-ft-mse").to(device)

    logger.info(f"🔧 Loading pretrained CLIP text encoder...")
    text_enc = PretrainedCLIPTextEncoder(model_id="openai/clip-vit-large-patch14").to(device)

    logger.info(f"🔧 Building UNet architecture...")
    unet = UNetModel(
        in_ch=4, out_ch=4, ch=320,
        res_blks=2,
        attn_lvls=(1, 2, 3),
        ch_mults=(1, 2, 4, 4),
        heads=8, t_dim=320, ctx_dim=768
    )

    noise_scheduler = DDPMScheduler(
        steps=1000, beta_start=0.00085, beta_end=0.012, schedule="scaled_linear"
    )

    model = StableDiffusionModel(vae, text_enc, unet, noise_scheduler).to(device)

    # ── Load UNet weights from checkpoint ────────────────────────────────────
    logger.info(f"📥 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "unet_state_dict" in ckpt:
        # New format (from the rewritten trainer)
        state_dict = ckpt["unet_state_dict"]
        missing, unexpected = model.unet.load_state_dict(state_dict, strict=True)
        epoch = ckpt.get("epoch", "?")
        best_loss = ckpt.get("best_loss", "?")
        logger.info(f"✅ UNet loaded from epoch {epoch} | best_loss: {best_loss}")

        # Load EMA weights if present (they produce better images)
        if "ema_state_dict" in ckpt:
            from SD_Train import EMA
            ema = EMA(model.unet, decay=0.9999)
            ema.load_state_dict(ckpt["ema_state_dict"])
            # Apply EMA shadow weights to the model
            ema.apply_shadow(model.unet)
            logger.info(f"✅ EMA weights applied (step {ema.step_count}) — better image quality")

    elif "model_state_dict" in ckpt:
        # Old format — extract UNet sub-keys
        logger.warning("Old checkpoint format. Extracting UNet weights...")
        full_sd = ckpt["model_state_dict"]
        unet_sd = {}
        for k, v in full_sd.items():
            for prefix in ["_fsdp_wrapped_module.", "module.", "_orig_mod."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            if k.startswith("unet."):
                unet_sd[k[len("unet."):]] = v
        missing, unexpected = model.unet.load_state_dict(unet_sd, strict=False)
        if missing:
            logger.warning(f"⚠️  Missing keys: {len(missing)}")

    else:
        # Bare state dict
        logger.warning("No recognized format. Attempting direct load...")
        missing, unexpected = model.unet.load_state_dict(ckpt, strict=False)
        if missing:
            logger.warning(f"⚠️  Missing keys: {len(missing)}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model: StableDiffusionModel,
    tokenizer: CLIPTokenizer,
    prompts: list,
    negative_prompts: list = None,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    output_path: str = "generated.png",
    height: int = 512,
    width: int = 512,
    seed: int = 42
) -> list:
    """
    Generate images from text prompts using DDIM sampling with Classifier-Free Guidance.

    Args:
        prompts:          List of text prompts to generate images for
        negative_prompts: List of negative prompts (default: empty strings)
                          These describe what you DON'T want: "blurry, low quality, ..."
        num_steps:        DDIM denoising steps (25-50 is good; 100 is diminishing returns)
        guidance_scale:   CFG scale:
                            1.0 = no guidance (pure model prior)
                            7.5 = balanced (recommended)
                            15+ = over-saturated, often degrades quality
        eta:              DDIM eta:
                            0.0 = deterministic (same seed → same image)
                            1.0 = DDPM-like stochastic (more variety)
        seed:             Random seed for reproducibility
    """
    device = next(model.parameters()).device

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_size = len(prompts)
    if negative_prompts is None:
        negative_prompts = [""] * batch_size
    assert len(negative_prompts) == batch_size, \
        "negative_prompts length must match prompts length"

    logger.info(f"🎨 Generating {batch_size} image(s) | steps={num_steps} | cfg={guidance_scale}")

    # ── Encode text prompts ──────────────────────────────────────────────────
    def encode_prompts(texts: list) -> torch.Tensor:
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        return model.encode_text(tokens.input_ids, tokens.attention_mask)

    cond_emb = encode_prompts(prompts)           # (B, 77, 768)
    uncond_emb = encode_prompts(negative_prompts) # (B, 77, 768)

    # Concatenate for batched CFG computation: [uncond, cond] → split after UNet
    ctx = torch.cat([uncond_emb, cond_emb], dim=0)  # (2B, 77, 768)

    # ── Initialize latents ───────────────────────────────────────────────────
    sched = DDIMScheduler(steps=1000, beta_start=0.00085, beta_end=0.012, schedule="scaled_linear")
    sched.set_timesteps(num_steps, device=device)

    latent_h = height // 8  # VAE compresses by 8×
    latent_w = width // 8

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        batch_size, 4, latent_h, latent_w,
        device=device,
        generator=generator
    )

    # ── DDIM Denoising Loop ──────────────────────────────────────────────────
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, t in enumerate(sched.timesteps):

            # Double batch for CFG: process conditional and unconditional together
            latent_input = torch.cat([latents] * 2, dim=0)   # (2B, 4, H, W)
            t_batch = torch.full(
                (latent_input.shape[0],), t, dtype=torch.long, device=device
            )

            # UNet predicts noise for both uncond and cond
            noise_pred = model(latent_input, t_batch, ctx)    # (2B, 4, H, W)

            # Split and apply Classifier-Free Guidance
            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            # CFG formula: guided = uncond + scale * (cond - uncond)
            # = moves prediction in the direction of the prompt
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # DDIM step: x_t → x_{t-1}
            latents = sched.step(guided_noise, t, latents, eta=eta)

            if (i + 1) % 10 == 0 or i == len(sched.timesteps) - 1:
                logger.info(f"  Step {i+1}/{len(sched.timesteps)}")

    # ── Decode Latents → Pixel Images ────────────────────────────────────────
    logger.info("🖼️  Decoding latents to images...")
    image_tensors = model.decode_latents(latents).clamp(-1.0, 1.0)
    images = (image_tensors + 1.0) / 2.0  # [-1, 1] → [0, 1]
    images_np = (images.float().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images_np]

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if len(pil_images) == 1:
        pil_images[0].save(output_path)
    else:
        # Arrange into a grid
        cols = min(4, len(pil_images))
        rows = (len(pil_images) + cols - 1) // cols
        W, H = pil_images[0].size
        grid = Image.new("RGB", (W * cols, H * rows))
        for idx, img in enumerate(pil_images):
            grid.paste(img, ((idx % cols) * W, (idx // cols) * H))
        grid.save(output_path)

    logger.info(f"✅ Saved to: {output_path}")
    return pil_images


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate images with your trained Stable Diffusion model")

    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to .pt checkpoint file (e.g. checkpoints/sd_latest.pt)")
    parser.add_argument("--prompts",     type=str, nargs="+", required=True,
                        help="One or more text prompts")
    parser.add_argument("--negative",    type=str, nargs="+", default=None,
                        help="Negative prompts (must match count of --prompts, or provide one to reuse for all)")
    parser.add_argument("--steps",       type=int, default=50,
                        help="DDIM steps (25-50 recommended; more = slower, diminishing returns)")
    parser.add_argument("--guidance",    type=float, default=7.5,
                        help="Classifier-Free Guidance scale (7-9 recommended)")
    parser.add_argument("--eta",         type=float, default=0.0,
                        help="DDIM eta: 0=deterministic, 1=stochastic (DDPM-like)")
    parser.add_argument("--output",      type=str, default="generated.png")
    parser.add_argument("--height",      type=int, default=512)
    parser.add_argument("--width",       type=int, default=512)
    parser.add_argument("--seed",        type=int, default=42)

    args = parser.parse_args()

    # Validate dimensions
    assert args.height % 8 == 0 and args.width % 8 == 0, \
        "Height and width must be multiples of 8 (VAE downsamples by 8×)"

    # Handle negative prompts
    negative_prompts = None
    if args.negative:
        if len(args.negative) == 1 and len(args.prompts) > 1:
            # Broadcast single negative prompt to all
            negative_prompts = args.negative * len(args.prompts)
        else:
            negative_prompts = args.negative
            assert len(negative_prompts) == len(args.prompts), \
                "Number of --negative prompts must match --prompts (or provide exactly 1)"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("⚠️  Running on CPU — generation will be VERY slow. GPU strongly recommended.")
    else:
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer + model
    logger.info("📚 Loading CLIP tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    model = load_model(args.checkpoint, device)

    # Generate
    logger.info(f"💭 Prompts: {args.prompts}")
    if negative_prompts:
        logger.info(f"🚫 Negative: {negative_prompts}")

    generate(
        model=model,
        tokenizer=tokenizer,
        prompts=args.prompts,
        negative_prompts=negative_prompts,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        eta=args.eta,
        output_path=args.output,
        height=args.height,
        width=args.width,
        seed=args.seed
    )


if __name__ == "__main__":
    main()